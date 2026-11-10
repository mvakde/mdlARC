"""Communicating vessels: route-local liquid conservation and a common head.

No parent code is imported. All semantic information used by solve is visible
in its one input: closed tank shells, exterior pipes, and liquid inside tanks.
"""

from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _neighbors(cells, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _components(cells, steps=D4):
    remaining = set(cells)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, queue = {p}, [p]
        for r, c in queue:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _holes(a, color):
    h, w = a.shape
    return [s for s in _components(zip(*np.where(a != color)))
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or a.dtype.kind not in "iu"
            or not all(1 <= v <= 30 for v in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _analyze(input_grid):
    """Reconstruct tanks, pipe attachments, source colors, and volumes."""
    a = _array(input_grid)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if len(np.unique(border)) != 1:
        raise ValueError("the exterior border must be uniformly background")
    bg = int(border[0])
    enclosing = [(int(c), _holes(a, int(c))) for c in np.unique(a) if c != bg]
    enclosing = [(c, holes) for c, holes in enclosing if holes]
    if len(enclosing) != 1:
        raise ValueError("exactly one color must form closed tank shells")
    wall, tanks = enclosing[0]
    shells = [_neighbors(s, D4) for s in tanks]
    all_inside = set().union(*tanks)
    all_shell = set().union(*shells)
    if set(zip(*np.where(a == wall))) != all_shell:
        raise ValueError("wall cells must be the thin tank shells")
    outside = {(int(r), int(c)) for r, c in zip(*np.where((a != bg) & (a != wall)))} - all_inside
    pipe_colors = {int(a[p]) for p in outside}
    if len(pipe_colors) != 1:
        raise ValueError("one exterior pipe color is required")
    pipe = pipe_colors.pop()
    if any(a[p] == pipe for p in all_inside):
        raise ValueError("pipe and liquid roles must be distinct")
    wires = _components(outside, D8)
    # Bipartite graph: a shell transmits between its attached pipes. Mere
    # shell-to-shell contact never adds an edge.
    graph = [set() for _ in range(len(tanks) + len(wires))]
    for j, wire in enumerate(wires, len(tanks)):
        near = _neighbors(wire)
        for i, shell in enumerate(shells):
            if shell & near:
                graph[i].add(j)
                graph[j].add(i)
    parts = _components_graph(graph)
    systems = []
    for part in parts:
        ids = sorted(i for i in part if i < len(tanks))
        if not ids:
            raise ValueError("unattached pipe fragment")
        cells = set().union(*(tanks[i] for i in ids))
        wet = {p for p in cells if a[p] != bg}
        colors = {int(a[p]) for p in wet}
        if colors & {wall, pipe} or len(colors) > 1:
            raise ValueError("each hydraulic system has at most one liquid color")
        systems.append(dict(ids=ids, cells=cells, wet=wet,
                            color=next(iter(colors)) if colors else None,
                            volume=len(wet)))
    return dict(a=a, bg=bg, wall=wall, pipe=pipe, tanks=tanks,
                shells=shells, wires=wires, graph=graph, systems=systems)


def _components_graph(graph):
    remaining = set(range(len(graph)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, queue = {start}, [start]
        for node in queue:
            for other in sorted(graph[node] & remaining):
                remaining.remove(other)
                part.add(other)
                queue.append(other)
        parts.append(part)
    return parts


def _allocate(cells, volume):
    """Consume complete absolute-height rows, with no horizontal tie break."""
    result = set()
    for row in sorted({r for r, c in cells}, reverse=True):
        if not volume:
            break
        level = {p for p in cells if p[0] == row}
        if volume < len(level):
            raise ValueError("volume would require an ambiguous partial level")
        result.update(level)
        volume -= len(level)
    if volume:
        raise ValueError("volume exceeds capacity")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _solve_analyzed(_analyze(input_grid))


def _solve_analyzed(info):
    out = info["a"].copy()
    for system in info["systems"]:
        if system["color"] is None:
            continue
        for p in system["cells"]:
            out[p] = info["bg"]
        for p in _allocate(system["cells"], system["volume"]):
            out[p] = system["color"]
    return out


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _shape(rng, small=False):
    """Row-interval tanks with boxes, slopes, steps, waists and deep strips."""
    kind = _ri(rng, 0, 5)
    h = _ri(rng, 2, 4 if small else 6)
    w = _ri(rng, 2, 4 if small else 6)
    if kind == 5:
        h, w = _ri(rng, 4, 6), _ri(rng, 1, 2)
    cells, left = set(), 0
    direction = int(rng.choice([-1, 1]))
    for row in range(h):
        if kind == 1:
            left = direction * (row // 2)
        elif kind == 2 and row:
            left += _ri(rng, -1, 1)
        width = w
        if kind == 3:
            width = max(1, w - abs(2 * row - h + 1) // 2)
        elif kind == 4 and 0 < row < h - 1:
            width = max(1, w - 1)
        cells.update((row, c) for c in range(left, left + width))
    # Very thin drifting rows must remain connected.
    if len(_components(cells)) != 1:
        return _shape(rng, small)
    shell = _neighbors(cells, D4)
    r0 = min(r for r, c in shell)
    c0 = min(c for r, c in shell)
    return ({(r-r0, c-c0) for r, c in cells},
            {(r-r0, c-c0) for r, c in shell})


def _path(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    todo = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[i] for i in rng.permutation(len(D8))]
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                todo.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _level_choices(tanks, ids, require_two=True, require_dry=False):
    cells = set().union(*(tanks[i] for i in ids))
    choices = []
    for level in sorted({r for r, c in cells}):
        target = {p for p in cells if p[0] >= level}
        counts = [len(target & tanks[i]) for i in ids]
        partial = sum(0 < n < len(tanks[i]) for i, n in zip(ids, counts))
        if not (0 < len(target) < len(cells)):
            continue
        if require_two and partial < 2:
            continue
        if require_dry and not any(n == 0 for n in counts):
            continue
        choices.append((level, target))
    return choices


def _source(rng, tanks, ids, target, style):
    cells = set().union(*(tanks[i] for i in ids))
    volume = len(target)
    candidates = [i for i in ids if len(tanks[i]) >= volume]
    if style == "concentrated" and candidates:
        # An initially dry receiver proves that the line transports liquid.
        i = candidates[_ri(rng, 0, len(candidates)-1)]
        available = sorted(tanks[i])
    else:
        available = sorted(cells)
    if style == "scattered":
        indices = rng.permutation(len(available))[:volume]
        wet = {available[int(i)] for i in indices}
    else:
        # Liquid may occupy several rows and retain some settled cells.
        jitter = {p: float(rng.random()) for p in available}
        wet = set(sorted(available, key=lambda p: (p[0], jitter[p]))[:volume])
        common = sorted(set(available) & target)
        if common and not wet & target:
            wet.remove(max(wet))
            wet.add(common[_ri(rng, 0, len(common)-1)])
    return wet


def _construct(rng, bg, wall, pipe, fluids, sizes, inactive, lesson, shift, is_test):
    count = sum(sizes) + inactive
    h = _ri(rng, 21, 30) if count > 4 else _ri(rng, 17, 28)
    w = _ri(rng, 22, 30) if count > 4 else _ri(rng, 18, 29)
    canvas = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
    tanks, shells, occupied = [], [], set()
    for _ in range(count):
        inner, edge = _shape(rng, count >= 6)
        hh = max(r for r, c in edge) + 1
        ww = max(c for r, c in edge) + 1
        blocked_placement = occupied | _neighbors(occupied)
        for attempt in range(100):
            dr, dc = _ri(rng, 1, h-hh-1), _ri(rng, 1, w-ww-1)
            cells = {(r+dr, c+dc) for r, c in inner}
            shell = {(r+dr, c+dc) for r, c in edge}
            region = cells | shell
            if not region & blocked_placement:
                tanks.append(cells)
                shells.append(shell)
                occupied.update(region)
                break
        else:
            return None
    # Random assignment, then require visible evidence of shared head in each
    # of the two lessons. No semantic active-above-inactive ordering exists.
    order = [int(i) for i in rng.permutation(count)]
    groups, pos = [], 0
    for size in sizes:
        groups.append(order[pos:pos+size])
        pos += size
    inactive_ids = order[pos:]
    targets, levels = [], []
    for j, ids in enumerate(groups):
        dry = (shift == "high_dry" and (lesson == 1 or is_test) and j == 0)
        two = lesson in (0, 1) or dry
        choices = _level_choices(tanks, ids, two, dry)
        if shift == "supply" and not is_test:
            choices = [(level, target) for level, target in choices
                       if len(target) <= max(len(tanks[i]) for i in ids)]
        if lesson == 0:
            choices = [(level, target) for level, target in choices
                       if len(target) <= max(len(tanks[i]) for i in ids)
                       and len({max(r for r, c in tanks[i]) for i in ids}) > 1
                       and len({len(target & tanks[i]) for i in ids}) > 1
                       and any(max(r for r, c in tanks[k]) > max(r for r, c in target)
                               for k in inactive_ids)]
        if j and lesson == 1:
            choices = [(lev, target) for lev, target in choices if lev != levels[0]]
        if not choices:
            return None
        level, target = choices[_ri(rng, 0, len(choices)-1)]
        targets.append(target)
        levels.append(level)
    # A spanning tree of pipes, using already connected tanks as branching
    # points. Routing respects all other shells and all other systems.
    pipe_sets = []
    for ids in groups:
        connected = [ids[0]]
        tree = set()
        others = set().union(*pipe_sets) if pipe_sets else set()
        for dest in ids[1:]:
            # Allow any connected shell as a source: chains and stars coexist.
            own_shell = set().union(*(shells[i] for i in connected))
            other_shell = set().union(*(s for i, s in enumerate(shells)
                                        if i not in connected and i != dest))
            blocked = occupied | _neighbors(other_shell) | others | _neighbors(others)
            allowed = canvas - blocked
            starts = (_neighbors(own_shell) | _neighbors(tree)) & allowed
            goals = _neighbors(shells[dest]) & allowed
            route = _path(rng, starts, goals, allowed)
            if route is None:
                return None
            tree.update(route)
            connected.append(dest)
        pipe_sets.append(tree)
    a = np.full((h, w), bg, dtype=np.int8)
    b = a.copy()
    for shell in shells:
        for p in shell:
            a[p] = b[p] = wall
    for tree in pipe_sets:
        for p in tree:
            a[p] = b[p] = pipe
    wet_sets = []
    for j, ids in enumerate(groups):
        style = "scattered" if ((is_test and shift == "supply") or
                (shift != "supply" and lesson < 0 and rng.random() < .45)) else "concentrated"
        wet = _source(rng, tanks, ids, targets[j], style)
        if is_test and shift == "supply" and sum(bool(wet & tanks[i]) for i in ids) < 2:
            return None
        wet_sets.append(wet)
        for p in wet:
            a[p] = fluids[j]
        for p in targets[j]:
            b[p] = fluids[j]
    if np.array_equal(a, b):
        return None
    if lesson == 0:
        wet_tanks = sum(bool(wet_sets[0] & tanks[i]) for i in groups[0])
        final_tanks = sum(bool(targets[0] & tanks[i]) for i in groups[0])
        if wet_tanks >= final_tanks or not wet_sets[0] & targets[0]:
            return None
    if rng.random() < .5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    try:
        info = _analyze(a)
        actual = sorted(sorted(i["cells"]) for i in info["systems"] if i["volume"])
        # Reconstructed ownership, not the sampled group labels, decides
        # validity. Compare the output as well as active/inactive counts.
        if len(actual) != len(groups) or len(info["tanks"]) != count:
            return None
        if not np.array_equal(_solve_analyzed(info), b):
            return None
    except ValueError:
        return None
    return {"input": a, "output": b}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(n) or not low <= int(n) <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    bg, first, second, third = palette[:4]
    shift = ("span", "same_color", "high_dry", "supply")[_ri(rng, 0, 3)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            for attempt in range(1200):
                roles = [int(c) for c in rng.permutation(10) if c not in (bg, first, second, third)]
                wall, pipe = roles[:2]
                if lesson == 0:
                    sizes, inactive, fluids = [2], 1, [first]
                elif lesson == 1:
                    sizes = [3 if shift == "high_dry" else 2, 2]
                    inactive = _ri(rng, 0, 1)
                    fluids = [second, second if shift == "same_color" else third]
                elif split == "test" and shift == "span":
                    sizes, inactive = [4], _ri(rng, 1, 2)
                    fluids = [int(rng.choice([first, second, third]))]
                elif split == "test" and shift == "same_color":
                    sizes, inactive = [2, 2, 2], _ri(rng, 0, 1)
                    fluids = [second] * len(sizes)
                elif split == "test" and shift == "high_dry":
                    sizes, inactive, fluids = [4], _ri(rng, 1, 2), [int(rng.choice([first, second, third]))]
                else:
                    n = _ri(rng, 1, 2)
                    upper = 3 if split == "train" and shift in ("span", "high_dry") else 4
                    sizes = [_ri(rng, 2, upper) for _ in range(n)]
                    if sum(sizes) > 7:
                        sizes[-1] = 3
                    inactive = _ri(rng, 0, 2)
                    fluids = [int(rng.choice([first, second, third])) for _ in sizes]
                pair = _construct(rng, bg, wall, pipe, fluids, sizes, inactive,
                                  lesson, shift, split == "test")
                if pair is not None:
                    key = _key(pair["input"])
                    if key not in seen:
                        seen.add(key)
                        puzzle[split].append(pair)
                        break
            else:
                raise RuntimeError(f"construction exhausted: seed={seed}, split={split}, "
                                   f"pair={index}, shift={shift}, counts={num_train}/{num_test}")
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _inspect_pair(pair):
    info = _analyze(pair["input"])
    a, b = info["a"], _array(pair["output"])
    assert a.shape == b.shape and np.any(a != b)
    assert np.array_equal(_solve_analyzed(info), b)
    inside = set().union(*info["tanks"])
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if (r, c) not in inside:
                assert a[r, c] == b[r, c], "exterior, shells and pipes must stay fixed"
    levels, partial, dry, transferred = [], 0, 0, False
    for s in info["systems"]:
        if s["color"] is None:
            assert all(b[p] == info["bg"] for p in s["cells"])
            continue
        color = s["color"]
        wet = {p for p in s["cells"] if b[p] == color}
        assert len(wet) == s["volume"]
        assert all(b[p] in (info["bg"], color) for p in s["cells"])
        level = min(r for r, c in wet)
        levels.append(level)
        assert wet == {p for p in s["cells"] if p[0] >= level}, "incomplete hydraulic level"
        # The stable arrangement minimizes total height potential.
        assert sum(r for r, c in wet) >= sum(r for r, c in s["wet"])
        for i in s["ids"]:
            n = len(wet & info["tanks"][i])
            partial += 0 < n < len(info["tanks"][i])
            dry += n == 0
            transferred |= n > len(s["wet"] & info["tanks"][i])
    active = [s for s in info["systems"] if s["volume"]]
    return dict(info=info, shape=a.shape, active=len(active),
                tanks=len(info["tanks"]), counts=tuple(len(s["ids"]) for s in active),
                levels=tuple(levels), partial=partial, dry=dry, transferred=transferred,
                inactive=sum(not s["volume"] for s in info["systems"]),
                volumes=tuple(s["volume"] for s in active),
                repeated=len({s["color"] for s in active}) < len(active))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, stats = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            f = _inspect_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            if split == "train":
                stats.append(f)
    first, second = stats[:2]
    assert first["active"] == 1 and first["partial"] >= 2
    assert first["inactive"] >= 1 and first["transferred"]
    assert second["active"] == 2 and len(set(second["levels"])) == 2
    training_colors = {s["color"] for f in stats for s in f["info"]["systems"] if s["volume"]}
    assert len(training_colors) >= 2
    # Explicit ablations of both parents in the smallest demonstration.
    info, a = first["info"], first["info"]["a"]
    b = np.asarray(puzzle["train"][0]["output"])
    active = next(s for s in info["systems"] if s["volume"])
    assert len({max(r for r, c in info["tanks"][i]) for i in active["ids"]}) == 2
    assert len({sum(b[p] == active["color"] for p in info["tanks"][i])
                for i in active["ids"]}) == 2
    assert active["wet"] & {p for p in active["cells"] if b[p] == active["color"]}
    assert any(not s["volume"] and max(r for r, c in s["cells"]) >
               max(r for r, c in active["cells"]) for s in info["systems"])
    all_full = a.copy()
    for s in info["systems"]:
        if s["color"] is not None:
            for p in s["cells"]:
                all_full[p] = s["color"]
    assert not np.array_equal(all_full, b), "ignoring conserved volume must fail"
    # A no-pipe interpretation cannot put liquid in an initially dry tank.
    assert any(all(a[p] == info["bg"] for p in t)
               and any(b[p] != info["bg"] for p in t) for t in info["tanks"])
    global_fill = a.copy()
    for p in set().union(*info["tanks"]):
        global_fill[p] = info["bg"]
    for p in sorted(set().union(*info["tanks"]), key=lambda p: (-p[0], p[1]))[:active["volume"]]:
        global_fill[p] = active["color"]
    assert not np.array_equal(global_fill, b), "ignoring route membership must fail"
    return None


def self_test():
    # Hand-specified geometry, independent of procedural construction: unequal
    # floors, incidental shell contact, two disconnected equal-color supplies.
    a = np.zeros((26, 26), dtype=np.int8)
    rectangles = ((5, 8, 3, 4), (6, 9, 15, 17), (11, 12, 5, 7),
                  (18, 20, 4, 5), (18, 22, 20, 22))
    tanks = [{(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}
             for r0, r1, c0, c1 in rectangles]
    for tank in tanks:
        for p in _neighbors(tank, D4):
            a[p] = 5
    a[4, 6:15] = 8
    a[16, 6:20] = 8
    source1 = tanks[0] | {(6, c) for c in range(15, 18)} | {(7, 15), (7, 16)}
    source2 = {(r, c) for r in range(18, 21) for c in range(20, 23)} | {(18, 4), (18, 5)}
    for p in source1 | source2:
        a[p] = 2
    expected = a.copy()
    expected[expected == 2] = 0
    for p in {p for p in tanks[0] | tanks[1] if p[0] >= 7}:
        expected[p] = 2
    for p in {p for p in tanks[3] | tanks[4] if p[0] >= 20}:
        expected[p] = 2
    before = a.copy()
    assert np.array_equal(solve(a), expected)
    assert np.array_equal(a, before), "solve must not mutate its input"
    assert all(expected[p] == 0 for p in tanks[2]), "shell contact is not a pipe"
    bad_level = a.copy()
    bad_level[5, 3] = 0
    try:
        solve(bad_level)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous partial common level accepted")
    shapes, geometry, roles, sizes, volumes, puzzles = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for split in ("train", "test")
                          for p in puzzle[split] for k in ("input", "output"))
        puzzles.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect_pair(pair)
                info = f["info"]
                records.append(f)
                shapes.add(f["shape"])
                roles.add((info["bg"], info["wall"], info["pipe"]))
                sizes.add(f["tanks"])
                volumes.update(f["volumes"])
                norm = np.full(info["a"].shape, 0, dtype=np.int8)
                norm[info["a"] == info["wall"]] = 1
                norm[info["a"] == info["pipe"]] = 2
                for s in info["systems"]:
                    for p in s["wet"]:
                        norm[p] = 3
                geometry.add(_key(norm))
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for split in puzzle
                       for p, q in zip(puzzle[split], again[split]) for k in p)
    assert len(puzzles) == 42 and len(geometry) >= 200
    assert len(shapes) >= 40 and len(roles) >= 100
    assert len(sizes) >= 5 and len(volumes) >= 20
    assert any(f["repeated"] for f in records)
    assert any(f["dry"] for f in records)
    assert any(max(f["counts"]) >= 4 for f in records)
    assert sum(f["transferred"] for f in records) >= 150
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(600 + 10*a+b, a, b)
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()}
               for p in puzzle[s]] for s in puzzle}
    p = bad["test"][0]
    r, c = np.argwhere(p["input"] != p["output"])[0]
    p["output"][r, c] = p["input"][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal explicit counts, semantic ablations)")
