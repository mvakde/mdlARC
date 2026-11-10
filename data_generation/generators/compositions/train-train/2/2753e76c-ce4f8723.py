"""Network elections: transparent assembly, object voting, and color propagation.

Only standard Python and NumPy are used. No parent code or examples are loaded.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


HOST_SEEDS = (0, 10688174, 515139419, 389088350, 992900732, 19062489,
              309019162, 936099292, 313904599, 128419323, 578651132,
              845102085, 245898920, 579983918, 2391648, 179567016,
              155868108, 965977296, 634581147, 848713777, 805453831,
              232918580, 105484961, 761525809, 596612866, 424985577,
              772511596, 714518680, 397253450, 272672193, 139879549,
              551350813, 369530833, 537481919, 283557331, 36897980,
              157911106, 81578239, 608504790, 131772528, 480326857,
              544227829)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _parts(mask):
    """Four-neighbor connected components, returned as coordinate lists."""
    remaining = np.asarray(mask, dtype=bool).copy()
    h, w = remaining.shape
    result = []
    for r, c in np.argwhere(remaining):
        r, c = int(r), int(c)
        if not remaining[r, c]:
            continue
        remaining[r, c] = False
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and remaining[yy, xx]:
                    remaining[yy, xx] = False
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _read(input_grid):
    """Find the unique separator, local palette roles, and colored union."""
    inp = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        length = inp.shape[axis]
        if length < 5 or length % 2 == 0:
            continue
        middle = length // 2
        if axis == 0:
            line = inp[middle, :]
            a, b = inp[:middle, :], inp[middle+1:, :]
        else:
            line = inp[:, middle]
            a, b = inp[:, :middle], inp[:, middle+1:]
        wire = int(line[0])
        if not np.all(line == wire):
            continue
        bg = int(a[0, 0])
        if bg == wire:
            continue
        # A clear one-cell background frame is a visible local role cue.
        if not all(np.all(edge == bg) for panel in (a, b)
                   for edge in (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        both = (a != bg) & (b != bg)
        if np.any(both & (a != b)):
            continue
        assembled = np.where(a != bg, a, b)
        if not np.any(assembled == wire):
            continue
        voters = set(int(x) for x in np.unique(assembled)) - {bg, wire}
        if len(voters) < 2:
            continue
        candidates.append((axis, bg, wire, a.copy(), b.copy(), assembled))
    if len(candidates) != 1:
        raise ValueError("input must have one framed pair of aligned network layers")
    return candidates[0]


def _analyze(input_grid):
    axis, bg, wire, a, b, union = _read(input_grid)
    components = _parts(union != bg)
    labels = np.full(union.shape, -1, dtype=int)
    for i, cells in enumerate(components):
        yy, xx = zip(*cells)
        labels[yy, xx] = i
    counts = [Counter() for _ in components]
    areas = [Counter() for _ in components]
    objects = []
    colors = sorted(set(int(x) for x in np.unique(union)) - {bg, wire})
    for color in colors:
        for cells in _parts(union == color):
            y, x = cells[0]
            i = int(labels[y, x])
            counts[i][color] += 1
            areas[i][color] += len(cells)
            objects.append((i, color, cells))
    winners = []
    for tally in counts:
        if not tally:
            raise ValueError("every assembled network needs a voter")
        best = max(tally.values())
        leading = [c for c, n in tally.items() if n == best]
        if len(leading) != 1:
            raise ValueError("largest voter count must be unique within a network")
        winners.append(leading[0])
    return dict(axis=axis, bg=bg, wire=wire, a=a, b=b, union=union,
                components=components, labels=labels, counts=counts,
                areas=areas, winners=winners, objects=objects)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble both views; each network takes its most numerous voter color."""
    info = _analyze(input_grid)
    output = np.full(info["union"].shape, info["bg"], dtype=np.int8)
    for cells, color in zip(info["components"], info["winners"]):
        yy, xx = zip(*cells)
        output[yy, xx] = color
    return output


def _neighbors(node, nr, nc):
    r, c = node
    return [(rr, cc) for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < nr and 0 <= cc < nc]


def _partition(rng, nr, nc, number, minimum):
    """Random connected territories, with no visible territory borders."""
    nodes = [(r, c) for r in range(nr) for c in range(nc)]
    for _ in range(160):
        seeds = [nodes[int(rng.integers(len(nodes)))]]
        while len(seeds) < number:
            distances = [min(abs(r-y)+abs(c-x) for y, x in seeds) for r, c in nodes]
            far = max(distances)
            pool = [n for n, d in zip(nodes, distances) if d >= max(1, far-1)]
            seeds.append(pool[int(rng.integers(len(pool)))])
        owner = {n: i for i, n in enumerate(seeds)}
        groups = [{n} for n in seeds]
        while len(owner) < len(nodes):
            frontiers = [[(u, v) for u in sorted(group) for v in _neighbors(u, nr, nc)
                          if v not in owner] for group in groups]
            possible = [i for i, f in enumerate(frontiers) if f]
            smallest = min(len(groups[i]) for i in possible)
            choices = [i for i in possible if len(groups[i]) <= smallest+1]
            i = choices[int(rng.integers(len(choices)))]
            _, node = frontiers[i][int(rng.integers(len(frontiers[i])))]
            owner[node] = i
            groups[i].add(node)
        if all(len(g) >= minimum for g in groups):
            return groups
    raise RuntimeError("could not construct balanced connected networks")


def _tree_edges(rng, group, nr, nc):
    start = sorted(group)[int(rng.integers(len(group)))]
    visited, edges = {start}, []
    while len(visited) < len(group):
        options = [(u, v) for u in sorted(visited) for v in _neighbors(u, nr, nc)
                   if v in group and v not in visited]
        u, v = options[int(rng.integers(len(options)))]
        visited.add(v)
        edges.append((u, v))
    used = {frozenset(e) for e in edges}
    extra_probability = float(rng.uniform(0, .8))
    for u in sorted(group):
        for v in _neighbors(u, nr, nc):
            if v in group and u < v and frozenset((u, v)) not in used:
                if rng.random() < extra_probability:
                    edges.append((u, v))
    return edges


def _positions(rng, count, limit):
    gaps = [4] * (count-1)
    slack = limit - (4*(count-1)+5)
    for j in rng.permutation(len(gaps)):
        if slack and rng.random() < .55:
            gaps[int(j)] += 1
            slack -= 1
    pos = [2]
    for gap in gaps:
        pos.append(pos[-1]+gap)
    length = pos[-1]+3
    if length < limit and rng.random() < .5:
        length += 1
    return pos, length


def _vote_counts(rng, size, categories, maximum=4):
    options = []
    # Losing colors may tie; only the leading count must be unique.
    for winning in range(2, maximum+1):
        def extend(values):
            if len(values) == categories:
                if sum(values) <= size:
                    options.append(tuple(values))
                return
            for losing in range(1, winning):
                extend(values+[losing])
        extend([winning])
    if not options:
        raise RuntimeError("not enough nodes for the voter counts")
    return options[int(rng.integers(len(options)))]


def _make_pair(rng, kind, orientation, previous_bg=None):
    # kind 0/1 are the two evidence-rich demonstrations; 2 is more training;
    # 3 is the test regime, which extends the winning count from <=4 to 5.
    if kind == 0:
        nr, nc = (2, int(rng.integers(5, 7))) if rng.random() < .3 else (3, int(rng.integers(4, 7)))
        number, minimum, ncolors = 2, 5, 2
    elif kind == 1:
        nr, nc = 3, int(rng.integers(5, 7))
        number, minimum, ncolors = 2, 7, 3
    elif kind == 3:
        nr, nc = 3, int(rng.integers(5, 8))
        number = 3 if nc >= 6 and rng.random() < .4 else 2
        minimum = 6 if number == 3 else 7
        ncolors = int(rng.integers(2, 5))
    else:
        nr, nc = int(rng.integers(2, 4)), int(rng.integers(4, 8))
        number = 3 if nr*nc >= 15 and rng.random() < .4 else 2
        minimum = 4
        ncolors = int(rng.integers(2, 5))
    groups = _partition(rng, nr, nc, number, minimum)
    # Sorting is solely for construction; the solver never uses spatial order.
    groups.sort(key=lambda g: min(g))
    ys, h = _positions(rng, nr, 14)
    xs, w = _positions(rng, nc, 30)
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    palette = [int(c) for c in rng.choice([c for c in range(10) if c != bg],
                                         size=ncolors+1, replace=False)]
    wire, colors = palette[0], palette[1:]
    assembled = np.full((h, w), bg, dtype=np.int8)
    state = np.zeros((h, w), dtype=np.int8)
    for i, group in enumerate(groups):
        winner = colors[i % len(colors)]
        losing = [c for c in colors if c != winner]
        rng.shuffle(losing)
        if kind == 0:
            counts = (3, 2) if i == 0 else (3, 1)
        elif kind == 1 and i == 0:
            counts = (3, 2, 2)  # plurality wins even without an absolute majority
        elif kind == 3 and i == 0:
            # Keep a unique five-voter winner without requiring a new rule.
            categories = min(ncolors, len(group)-4)
            counts = (5,) + (1,)*(categories-1)
        else:
            categories = min(ncolors, max(2, len(group)-2))
            counts = _vote_counts(rng, len(group), categories)
        vote_colors = [winner]+losing[:len(counts)-1]
        edges = _tree_edges(rng, group, nr, nc)
        for u, v in edges:
            r1, c1 = ys[u[0]], xs[u[1]]
            r2, c2 = ys[v[0]], xs[v[1]]
            section = (slice(min(r1, r2), max(r1, r2)+1),
                       slice(min(c1, c2), max(c1, c2)+1))
            assembled[section] = wire
            # Whole edges, rather than random pixels, form readable fragments.
            state[section] |= int(rng.choice((1, 2, 3), p=(.42, .42, .16)))
        for node in sorted(group):
            y, x = ys[node[0]], xs[node[1]]
            assembled[y, x] = wire
            state[y, x] |= int(rng.integers(1, 4))
        nodes = sorted(group)
        rng.shuffle(nodes)
        cursor = 0
        for rank, (color, count) in enumerate(zip(vote_colors, counts)):
            for j in range(count):
                y, x = ys[nodes[cursor][0]], xs[nodes[cursor][1]]
                cursor += 1
                if kind == 0 and i == 0:
                    rh = rw = 1 if rank == 0 else 3
                    exposure = (1 if j < 2 else 2) if rank == 0 else 3
                else:
                    rh, rw = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 3))[
                        int(rng.integers(5))]
                    exposure = int(rng.choice((1, 2, 3), p=(.43, .43, .14)))
                rr, cc = y-rh//2, x-rw//2
                section = (slice(rr, rr+rh), slice(cc, cc+rw))
                assembled[section] = color
                state[section] = exposure
    a = np.where((state & 1) != 0, assembled, bg).astype(np.int8)
    b = np.where((state & 2) != 0, assembled, bg).astype(np.int8)
    inp = np.vstack((a, np.full((1, w), wire, dtype=np.int8), b))
    if orientation == 1:
        # Transpose keeps coordinates directly aligned, changing the separator.
        inp = inp.T.copy()
    return {"input": inp, "output": solve(inp)}


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return int(value)


def _key(grid):
    a = np.asarray(grid)
    return (a.shape, tuple(int(x) for x in a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    used = set()
    first_orientation = int(rng.integers(2))
    previous_bg = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = min(index, 2) if split == "train" else 3
            orientation = ((first_orientation+index) % 2 if split == "train" and index < 2
                           else int(rng.integers(2)))
            for _ in range(40):
                pair = _make_pair(rng, kind, orientation, previous_bg)
                key = _key(pair["input"])
                if key not in used and (kind != 0 or _evidence(_analyze(pair["input"]))[2]):
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
            used.add(key)
            puzzle[split].append(pair)
            previous_bg = _read(pair["input"])[1]
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info):
    """Counterfactuals measured from visible grids, not construction metadata."""
    area_reversal = False
    separate_reversal = False
    per_layer = [Counter() for _ in info["components"]]
    for panel in (info["a"], info["b"]):
        for color in set(int(x) for x in np.unique(panel)) - {info["bg"], info["wire"]}:
            for obj in _parts(panel == color):
                y, x = obj[0]
                per_layer[int(info["labels"][y, x])][color] += 1
    for i, winner in enumerate(info["winners"]):
        area_reversal |= any(area > info["areas"][i][winner]
                             for color, area in info["areas"][i].items() if color != winner)
        separate_reversal |= any(n > per_layer[i][winner]
                                 for color, n in per_layer[i].items() if color != winner)
    fragmented = (len(_parts(info["a"] != info["bg"])) > len(info["components"])
                  and len(_parts(info["b"] != info["bg"])) > len(info["components"]))
    return area_reversal, separate_reversal, fragmented


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(inp)
            analyses[split].append(info)
            assert 2 <= len(info["components"]) <= 3
            assert len(set(info["winners"])) >= 2, "local elections must differ"
            assert all(len(tally) >= 2 for tally in info["counts"])
            assert out.shape == info["union"].shape
            assert np.array_equal(out != info["bg"], info["union"] != info["bg"])
            assert np.array_equal(out, solve(inp)), "incorrect election output"
            assert all(any(info["union"][r, c] == info["wire"] for r, c in obj)
                       for obj in info["components"])
            token_labels = np.full(info["union"].shape, -1, dtype=int)
            for j, (_, color, cells) in enumerate(info["objects"]):
                yy, xx = zip(*cells)
                token_labels[yy, xx] = j
                assert len(cells) == (max(yy)-min(yy)+1)*(max(xx)-min(xx)+1), "voters must be rectangles"
                for panel in (info["a"], info["b"]):
                    visible = panel[yy, xx] == color
                    assert visible.all() or not visible.any(), "show a whole voter or none"
            # Neither districts nor distinct voter objects rely on corner-only
            # contact being interpreted one way rather than another.
            for labels in (info["labels"], token_labels):
                for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rows = labels.shape[0]-dy
                    cols = labels.shape[1]-abs(dx)
                    left = labels[:rows, max(0, -dx):max(0, -dx)+cols]
                    right = labels[dy:dy+rows, max(0, dx):max(0, dx)+cols]
                    assert not np.any((left >= 0) & (right >= 0) & (left != right)), "distinct objects need clearance"
    first = analyses["train"][0]
    states = (first["a"] != first["bg"]).astype(int) + 2*(first["b"] != first["bg"])
    assert set(np.unique(states)) == {0, 1, 2, 3}, "training must show all union states"
    assert _evidence(first)[:2] == (True, True), "missing area and duplicate-vote counterexamples"
    assert any(_evidence(i)[2] for i in analyses["train"]), "assembly needs a connectivity witness"
    assert {i["axis"] for i in analyses["train"]} == {0, 1}
    assert len({i["bg"] for i in analyses["train"]}) >= 2
    assert len(first["counts"]) == 2
    assert max(len(t) for t in analyses["train"][1]["counts"]) >= 3
    assert any(max(t.values()) < sum(t.values())-max(t.values())
               for t in analyses["train"][1]["counts"]), "show a plurality below half the votes"
    assert all(max(max(t.values()) for t in i["counts"]) <= 4 for i in analyses["train"])
    assert all(max(max(t.values()) for t in i["counts"]) == 5 for i in analyses["test"])


def self_test():
    """Bounded 42-seed family audit, count API checks, and corruption checks."""
    shapes, palettes, profiles, geometries, orientations, network_numbers = set(), set(), set(), set(), set(), set()
    backgrounds, wires, object_shapes = set(), set(), set()
    fragmented = cycles = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(tuple(pair["input"].shape))
                palettes.add((info["bg"], info["wire"], tuple(sorted(set(info["union"].flat)))))
                backgrounds.add(info["bg"])
                wires.add(info["wire"])
                orientations.add(info["axis"])
                network_numbers.add(len(info["components"]))
                profiles.add(tuple(sorted(tuple(sorted(t.values(), reverse=True)) for t in info["counts"])))
                geometries.add(_key((info["union"] != info["bg"]).astype(np.int8)))
                fragmented += _evidence(info)[2]
                # A hole in the occupied mask witnesses a closed network cycle.
                for hole in _parts(info["union"] == info["bg"]):
                    if all(0 < r < info["union"].shape[0]-1 and 0 < c < info["union"].shape[1]-1 for r, c in hole):
                        cycles += 1
                for _, _, cells in info["objects"]:
                    yy, xx = zip(*cells)
                    object_shapes.add((max(yy)-min(yy)+1, max(xx)-min(xx)+1))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(shapes) >= 20 and len(palettes) >= 100 and len(profiles) >= 20
    assert len(geometries) >= 160 and orientations == {0, 1}
    assert network_numbers == {2, 3} and len(backgrounds) == 10 and len(wires) == 10
    assert len(object_shapes) >= 5 and fragmented >= 100 and cycles >= 10
    _handworked_checks()
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(888+nt*10+ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(7, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(314159, 5, 2), generate_puzzle(314159, 5, 2)
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(p[s], q[s]) for k in ("input", "output"))
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")


def _handworked_checks():
    """Small explicit expectations, without generation or scaffold metadata."""
    whole = np.zeros((9, 19), dtype=np.int8)
    whole[2, 1:18] = 5
    whole[6, 1:18] = 5
    whole[2, [2, 4, 6]] = 2
    whole[1:3, 10:12] = 3
    whole[1:3, 14:16] = 3
    whole[6, [2, 4, 6]] = 3
    whole[6, 10] = 2
    expected = np.zeros_like(whole)
    expected[:5] = np.where(whole[:5] != 0, 2, 0)
    expected[5:] = np.where(whole[5:] != 0, 3, 0)

    def stack(a, b):
        return np.vstack((a, np.full((1, a.shape[1]), 5, dtype=np.int8), b))

    # Complete repeated evidence does not double any vote or alter its color.
    repeated = stack(whole, whole)
    assert np.array_equal(solve(repeated), expected)
    # Source exchange and transposition preserve the same geometric operation.
    first, second = whole.copy(), whole.copy()
    first[:, 8:] = 0
    second[:, :8] = 0
    assert np.array_equal(solve(stack(first, second)), expected)
    assert np.array_equal(solve(stack(second, first)), expected)
    assert np.array_equal(solve(stack(first, second).T), expected.T)
    permutation = np.array([8, 4, 7, 0, 9, 2, 1, 3, 6, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[repeated]), permutation[expected])
    # Expanding a voter changes its area but does not add a vote.
    expanded = whole.copy()
    expanded[1, 2] = 2  # a winning voter becomes a 2x1 rectangle
    expanded_expected = expected.copy()
    expanded_expected[1, 2] = 2
    assert np.array_equal(solve(stack(expanded, expanded)), expanded_expected)
    # Two new objects change the top election 3:2 -> 3:4, preserving occupancy.
    changed = whole.copy()
    changed[2, [8, 17]] = 3
    changed_expected = expected.copy()
    changed_expected[:5] = np.where(whole[:5] != 0, 3, 0)
    assert np.array_equal(solve(stack(changed, changed)), changed_expected)
    # Removing a neutral bridge changes the district, hence distant wire color.
    cut = whole.copy()
    cut[2, 7] = 0
    cut_expected = expected.copy()
    cut_expected[2, 7] = 0
    region = cut[:5, 8:]
    cut_expected[:5, 8:] = np.where(region != 0, 3, 0)
    assert np.array_equal(solve(stack(cut, cut)), cut_expected)
    # A tied leading count is outside the family, and must never pick a color.
    tied = whole.copy()
    tied[2, 8] = 3
    try:
        solve(stack(tied, tied))
    except ValueError:
        pass
    else:
        raise AssertionError("a tied leader was silently resolved")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
