"""Geodesic echo tiles: a boundary pulse clocks binary block substitution.

Only the standard library and NumPy are used. The decoder, shortest-path
calculation, and solver depend on the supplied input alone.
"""
from collections import deque
import numpy as np


SIDES = ("top", "right", "bottom", "left")
STYLES = ("maze", "snake", "walk", "percolation", "chambers")
TEACHING = ("repeat", "branch", "loop", "remote", "dormant", "multi_source",
            "no_pulse", "no_exchange", "straight", "manhattan", "odd_visible",
            "unreachable_visible", "first_root", "rotate_tile", "depth_first")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= x <= 30 for x in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _ring_box(cells):
    points = np.array(cells)
    r0, c0 = map(int, points.min(0))
    r1, c1 = map(int, points.max(0) + 1)
    if min(r1 - r0, c1 - c0) < 4:
        return None
    ring = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
            if r in (r0, r1 - 1) or c in (c0, c1 - 1)}
    return (r0, c0, r1, c1) if set(cells) == ring else None


def _bar(mask):
    rr, cc = np.nonzero(mask)
    if not len(rr):
        return None
    candidates = []
    for side, fixed, edge, lateral in (
        (0, rr, 0, cc), (1, cc, mask.shape[1] - 1, rr),
        (2, rr, mask.shape[0] - 1, cc), (3, cc, 0, rr),
    ):
        if np.all(fixed == edge):
            start = int(lateral.min())
            if np.array_equal(np.sort(lateral), np.arange(start, start + len(rr))):
                candidates.append((side, start, len(rr)))
    # A corner singleton would specify two different edges.
    return candidates[0] if len(candidates) == 1 else None


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the two frames need a background margin")
    interpretations = []
    for frame in map(int, np.unique(g)):
        if frame == bg:
            continue
        comps = _components(g == frame)
        if len(comps) != 2:
            continue
        boxes = [_ring_box(cells) for cells in comps]
        if any(b is None for b in boxes):
            continue
        for ti in (0, 1):
            tb, nb = boxes[ti], boxes[1 - ti]
            tr, tc, te, tf = tb
            nr, nc, ne, nf = nb
            tile = g[tr + 1:te - 1, tc + 1:tf - 1]
            net = g[nr + 1:ne - 1, nc + 1:nf - 1]
            colors = tuple(map(int, np.unique(tile)))
            if (tile.shape[0] != tile.shape[1] or not 2 <= len(tile) <= 4
                    or len(colors) != 2 or frame in colors):
                continue
            if min(net.shape) < 3 or max(net.shape) * len(tile) > 30:
                continue
            vals = set(map(int, np.unique(net)))
            if len(vals) != 3 or bg not in vals or frame in vals:
                continue
            inks = vals - {bg}
            if inks & set(colors):
                continue
            for marker in sorted(inks):
                bar = _bar(net == marker)
                if bar is None:
                    continue
                stencil = next(iter(inks - {marker}))
                # The complete input is exactly two separated framed objects.
                expected = np.full(g.shape, bg, dtype=np.int8)
                expected[tr:te, tc:tf] = g[tr:te, tc:tf]
                expected[nr:ne, nc:nf] = g[nr:ne, nc:nf]
                if not np.array_equal(g, expected):
                    continue
                interpretations.append(dict(background=bg, frame=frame, tile=tile.copy(),
                                            network=net.copy(), marker=marker, stencil=stencil,
                                            side=bar[0], start=bar[1], length=bar[2],
                                            tile_box=tb, network_box=nb))
    if len(interpretations) != 1:
        raise ValueError("input must have one unambiguous tile and rooted network")
    return interpretations[0]


def _distances(network, background, roots):
    """Multi-source breadth-first distances in the four-neighbor stencil graph."""
    d = np.full(network.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in roots:
        d[r, c] = 0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= rr < len(network) and 0 <= cc < network.shape[1]
                    and network[rr, cc] != background and d[rr, cc] < 0):
                d[rr, cc] = d[r, c] + 1
                queue.append((rr, cc))
    return d


def _render(info, d, behavior="normal"):
    """Block substitution. Alternative behaviors are evidence checks only."""
    net, tile, bg = info["network"], info["tile"], info["background"]
    n = len(tile)
    out = np.full((len(net) * n, net.shape[1] * n), bg, dtype=np.int8)
    active = (d >= 0) & (d % 2 == 0)
    if behavior == "no_pulse":
        active = net != bg
    elif behavior == "odd_visible":
        active = d >= 0
    elif behavior == "unreachable_visible":
        active |= (net != bg) & (d < 0)
    if behavior == "rotate_tile":
        tile = np.rot90(tile, -info["side"])
    a, b = map(int, np.unique(tile))
    exchanged = np.where(tile == a, b, a)
    for r, c in np.argwhere(active):
        exchange = (d[r, c] % 6 == 4 and behavior != "no_exchange") or behavior == "no_pulse"
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = exchanged if exchange else tile
    return out


def _depth_first_distances(network, background, roots):
    """Competing rule: commit to the first DFS route instead of earliest arrival."""
    d = np.full(network.shape, -1, dtype=np.int16)
    stack = []
    for r, c in roots:
        d[r, c] = 0
        stack.append((int(r), int(c)))
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= rr < len(network) and 0 <= cc < network.shape[1]
                    and network[rr, cc] != background and d[rr, cc] < 0):
                d[rr, cc] = d[r, c] + 1
                stack.append((rr, cc))
    return d


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the expanded pulse field using only the given input."""
    info = _decode(input_grid)
    roots = np.argwhere(info["network"] == info["marker"])
    d = _distances(info["network"], info["background"], roots)
    return _render(info, d)


def _alternative(info, d, name):
    net, bg = info["network"], info["background"]
    if name in ("no_pulse", "no_exchange", "odd_visible", "unreachable_visible", "rotate_tile"):
        return _render(info, d, name)
    roots = np.argwhere(net == info["marker"])
    rr, cc = np.indices(net.shape)
    if name == "first_root":
        wrong = _distances(net, bg, roots[:1])
    elif name == "depth_first":
        wrong = _depth_first_distances(net, bg, roots)
    elif name == "manhattan":
        wrong = np.minimum.reduce([abs(rr - r) + abs(cc - c) for r, c in roots])
        wrong[net == bg] = -1
    elif name == "straight":
        wrong = (rr if info["side"] == 0 else net.shape[1] - 1 - cc if info["side"] == 1
                 else len(net) - 1 - rr if info["side"] == 2 else cc).copy()
        wrong[net == bg] = -1
    else:
        raise ValueError(name)
    return _render(info, wrong)


def _evidence(info):
    net, bg = info["network"], info["background"]
    d = _distances(net, bg, np.argwhere(net == info["marker"]))
    reached = d >= 0
    # A genuine outward fork has two successors on the next wavefront;
    # merely touching neighboring zero-distance bar cells does not qualify.
    successors = np.zeros(net.shape, dtype=int)
    padded = np.pad(d, 1, constant_values=-100)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = padded[1 + dr:1 + dr + len(net), 1 + dc:1 + dc + net.shape[1]]
        successors += reached & (neighbor == d + 1)
    # Require a cycle among stencil cells, beyond a ladder formed only by the
    # adjacent zero-distance roots and their first neighbors.
    body = reached & (net != info["marker"])
    edges = int(np.sum(body[1:] & body[:-1]) + np.sum(body[:, 1:] & body[:, :-1]))
    cycle_rank = edges - int(body.sum()) + len(_components(body))
    lateral = np.indices(net.shape)[1 if info["side"] in (0, 2) else 0]
    outside = (lateral < info["start"]) | (lateral >= info["start"] + info["length"])
    out = _render(info, d)
    evidence = dict(repeat=int(d.max()) >= 10,
                    branch=bool(np.any(successors >= 2)),
                    loop=cycle_rank > 0,
                    remote=bool(np.any(outside & (d >= 0) & (d % 6 == 4))),
                    dormant=bool(np.any((net != bg) & ~reached)),
                    multi_source=info["length"] > 1)
    for name in TEACHING[6:]:
        evidence[name] = not np.array_equal(out, _alternative(info, d, name))
    return evidence


def _covered(evidence):
    return all(any(e[name] for e in evidence) for name in TEACHING)


def _maze(rng, h, w, loops):
    mask = np.zeros((h, w), dtype=bool)
    vertices = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
    root = vertices[int(rng.integers(len(vertices)))]
    stack, seen = [root], {root}
    mask[root] = True
    closed = set()
    while stack:
        r, c = stack[-1]
        neighbors = [(r + dr, c + dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
                     if 1 <= r + dr < h - 1 and 1 <= c + dc < w - 1]
        unvisited = [p for p in neighbors if p not in seen]
        if not unvisited:
            stack.pop()
            continue
        nr, nc = unvisited[int(rng.integers(len(unvisited)))]
        mask[(r + nr) // 2, (c + nc) // 2] = True
        mask[nr, nc] = True
        seen.add((nr, nc))
        stack.append((nr, nc))
    for r, c in vertices:
        for nr, nc in ((r + 2, c), (r, c + 2)):
            if (nr, nc) in seen and not mask[(r + nr) // 2, (c + nc) // 2]:
                closed.add(((r + nr) // 2, (c + nc) // 2))
    for p in sorted(closed):
        if rng.random() < loops:
            mask[p] = True
    return mask


def _network(rng, h, w, style, length, islands, loop_preference):
    main_w = w - 3 if islands else w
    length = min(length, main_w - 1)
    start = int(rng.integers(1 if length == 1 else 0,
                             main_w - length if length == 1 else main_w - length + 1))
    if style == "maze":
        main = _maze(rng, h, main_w, loop_preference)
    elif style == "snake":
        main = np.zeros((h, main_w), dtype=bool)
        left = int(rng.integers(0, max(1, main_w // 3)))
        right = int(rng.integers(max(left + 2, 2 * main_w // 3), main_w))
        rows = list(range(1, h - 1, 2))
        for j, r in enumerate(rows):
            main[r, left:right + 1] = True
            if j + 1 < len(rows):
                main[r + 1, right if j % 2 == 0 else left] = True
        if loop_preference:
            for r in range(2, h - 2, 2):
                if rng.random() < loop_preference:
                    main[r, int(rng.integers(left, right + 1))] = True
    elif style == "walk":
        main = np.zeros((h, main_w), dtype=bool)
        r, c = 1, start
        for _ in range(int(rng.integers(h * main_w, 4 * h * main_w))):
            main[r, c] = True
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            r, c = int(np.clip(r + dr, 1, h - 1)), int(np.clip(c + dc, 0, main_w - 1))
    elif style == "percolation":
        main = rng.random((h, main_w)) < rng.uniform(.32, .78)
    else:
        main = np.ones((h, main_w), dtype=bool)
        for _ in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(2, h)), int(rng.integers(main_w))
            if rng.random() < .5:
                main[r, :c + 1] = False
            else:
                main[r, c:] = False
        main &= rng.random((h, main_w)) > rng.uniform(.03, .22)
    main[0] = False
    # Join the bar to the network's first interior cross-section. For a maze,
    # choose the nearest existing row-1 vertex, rather than assuming a corner.
    anchors = np.flatnonzero(main[1])
    target = int(anchors[np.argmin(abs(anchors - start))]) if len(anchors) else start
    main[1, min(start, target):max(start + length, target + 1)] = True
    main[0, start:start + length] = True
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :main_w] = main
    if islands:
        extras = rng.random((h - 1, 2)) < rng.uniform(.15, .8)
        extras[int(rng.integers(h - 1)), int(rng.integers(2))] = True
        mask[1:, main_w + 1:] = extras
    return mask, start, length


def _tile(rng, n, colors, informative):
    for _ in range(100):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if not bits.any() or bits.all():
            continue
        if informative and any(np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            continue
        return np.where(bits, colors[0], colors[1]).astype(np.int8)
    bits = np.zeros((n, n), dtype=bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(np.int8)


def _frame(content, color):
    box = np.full((len(content) + 2, content.shape[1] + 2), color, dtype=np.int8)
    box[1:-1, 1:-1] = content
    return box


def _pack(rng, tile, net, bg, frame):
    boxes = [_frame(tile, frame), _frame(net, frame)]
    if rng.integers(2):
        boxes.reverse()
    vertical = bool(rng.integers(2))
    gap, margin = int(rng.integers(1, 4)), int(rng.integers(1, 3))
    axis = 0 if vertical else 1
    cross = 1 - axis
    extent = sum(b.shape[axis] for b in boxes) + gap + 2 * margin
    other = max(b.shape[cross] for b in boxes) + 2 * margin + int(rng.integers(0, 4))
    extent, other = min(30, extent), min(30, other)
    shape = (extent, other) if vertical else (other, extent)
    g = np.full(shape, bg, dtype=np.int8)
    # Reserve positive margins and a genuine gap even near the 30-cell bound.
    spare = extent - sum(b.shape[axis] for b in boxes)
    before = int(rng.integers(1, spare - 1))
    gap = int(rng.integers(1, spare - before))
    cursor = before
    for b in boxes:
        offset = int(rng.integers(1, other - b.shape[cross]))
        r, c = (cursor, offset) if vertical else (offset, cursor)
        g[r:r + len(b), c:c + b.shape[1]] = b
        cursor += b.shape[axis] + gap
    return g


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _make_pair(rng, n, side, style, informative, last, previous_evidence, seen):
    for attempt in range(240):
        fallback = attempt >= 200
        chosen_style = "maze" if fallback else style if rng.random() < .55 else str(rng.choice(STYLES))
        maximum = 30 // n
        low = 8 if informative else 4
        low = min(low, maximum)
        h, w = map(int, rng.integers(low, maximum + 1, size=2))
        if fallback:
            h = w = maximum
        islands = bool(rng.random() < .5) or fallback
        # A last demonstration fills missing set-level evidence. It still samples
        # fresh complete networks rather than inserting phase-aligned witnesses.
        if last and not any(e["dormant"] for e in previous_evidence):
            islands = True
        length = int(rng.integers(1, min(5, w - 2) + 1))
        if fallback:
            length = 2 + attempt % 2
        loops = float(rng.choice([0, .08, .2, .45]))
        if fallback:
            loops = .15
        if islands and w < 7:
            islands = False
        mask, start, length = _network(rng, h, w, chosen_style, length, islands, loops)
        palette = list(map(int, rng.permutation(10)))
        bg, frame, stencil, marker, ca, cb = palette[:6]
        if rng.random() < .35:
            ca = bg  # explicit frame keeps the whole binary square visible
        tile = _tile(rng, n, (ca, cb), informative)
        net = np.where(mask, stencil, bg).astype(np.int8)
        net[0, start:start + length] = marker
        net = np.rot90(net, -side).copy()
        inp = _pack(rng, tile, net, bg, frame)
        if _key(inp) in seen:
            continue
        info = _decode(inp)
        evidence = _evidence(info) if informative or last else None
        if informative and not evidence["repeat"]:
            continue
        if last and not _covered(previous_evidence + [evidence]):
            continue
        out = solve(inp)
        return dict(input=inp, output=out), evidence
    raise RuntimeError("bounded network sampling did not complete teaching coverage")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base, regime = int(rng.integers(4)), int(rng.integers(3))
    preferred = str(rng.choice(STYLES))
    puzzle, seen, evidence = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if split == "train":
                # At least two scales and two entry edges are demonstrated.
                if regime == 2:
                    offset = i % 2
                    side, n = (base + offset) % 4, 2 + offset
                else:
                    side = (base + i % (2 if regime == 0 else 4)) % 4
                    n = 2 + i % 2
            elif regime == 0:
                side, n = (base + 2 + i % 2) % 4, int(rng.integers(2, 5))
            elif regime == 1:
                side, n = int(rng.integers(4)), 4
            else:
                side, n = (base + 1) % 4, 2
            # Scale 4 has a smaller macro field; the rule remains unchanged.
            informative = split == "train"
            last = split == "train" and i == nt - 1
            pair, ev = _make_pair(rng, n, side, preferred, informative, last, evidence, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if ev is not None:
                evidence.append(ev)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_check(info, output):
    """Validate transparency and each printed pixel from its macro coordinates."""
    net, tile, bg = info["network"], info["tile"], info["background"]
    d = _distances(net, bg, np.argwhere(net == info["marker"]))
    n = len(tile)
    assert output.shape == (len(net) * n, net.shape[1] * n)
    rr, cc = np.indices(output.shape)
    depth = d[rr // n, cc // n]
    values = tile[rr % n, cc % n].astype(int)
    a, b = map(int, np.unique(tile))
    values = np.where(depth % 6 == 4, a + b - values, values)
    expected = np.where((depth >= 0) & (depth % 2 == 0), values, bg)
    assert np.array_equal(output, expected), "wrong tile, pulse phase, or transparency"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, evidence = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            _pixel_check(info, out)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(info["tile"]))) | {info["background"]}
            if split == "train":
                evidence.append(_evidence(info))
    assert _covered(evidence), "demonstrations do not distinguish the required interactions"
    # The sampler promises two different tile sizes before asking for transfer.
    assert len({len(_decode(p["input"])["tile"]) for p in puzzle["train"]}) >= 2


def _puzzle_key(p):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in p[split])


def _normalized_key(g):
    roles = {}
    return g.shape, tuple(roles.setdefault(int(c), len(roles)) for c in g.flat)


def self_test():
    """Bounded checks of semantics, curricula, counts, and 42-seed variation."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    stats = []
    structures = set()
    role_colors = {name: set() for name in ("background", "frame", "stencil", "marker", "tile")}
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                net = info["network"]
                d = _distances(net, info["background"], np.argwhere(net == info["marker"]))
                structures.add(_normalized_key(pair["input"]))
                for role in role_colors:
                    if role == "tile":
                        role_colors[role].update(map(int, np.unique(info[role])))
                    else:
                        role_colors[role].add(info[role])
                stats.append(dict(split=split, side=info["side"], n=len(info["tile"]),
                                  source=net.shape, input=pair["input"].shape,
                                  depth=int(d.max()), roots=info["length"],
                                  dormant=int(np.sum((net != info["background"]) & (d < 0))),
                                  components=len(_components(net != info["background"])),
                                  ground_tile=info["background"] in info["tile"],
                                  density=float(np.mean(net != info["background"]))))
    assert len(structures) >= 200
    assert {s["side"] for s in stats} == set(range(4))
    assert {s["n"] for s in stats} == {2, 3, 4}
    assert len({s["source"] for s in stats}) >= 35
    assert len({s["input"] for s in stats}) >= 60
    assert len({s["depth"] for s in stats}) >= 20
    assert {s["roots"] for s in stats} >= {1, 2, 3, 4}
    assert any(s["ground_tile"] for s in stats)
    assert any(s["dormant"] == 0 for s in stats) and any(s["dormant"] >= 5 for s in stats)
    assert any(s["density"] > .65 for s in stats) and any(s["density"] < .35 for s in stats)
    assert all(colors == set(range(10)) for colors in role_colors.values())
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(440 + 10 * nt + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A hand-specified bent corridor: (3,2) is three rows from the top
    # but is reached after four steps, so its block is exchanged.
    tile = np.array([[2, 3], [3, 3]], dtype=np.int8)
    net = np.array([[0, 5, 0, 0, 0], [0, 4, 4, 0, 0],
                    [0, 0, 4, 0, 4], [0, 4, 4, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    info = dict(network=net, tile=tile, background=0, marker=5, side=0)
    d = _distances(net, 0, np.argwhere(net == 5))
    assert d[3, 2] == 4 and d[3, 1] == 5 and d[2, 4] == -1
    out = _render(info, d)
    assert np.array_equal(out[6:8, 4:6], np.array([[3, 2], [2, 2]]))
    assert np.all(out[6:8, 2:4] == 0) and np.all(out[4:6, 8:10] == 0)
    rng = np.random.default_rng(2026)
    for side in range(4):
        oriented = np.rot90(net, -side).copy()
        inp = _pack(rng, tile, oriented, 0, 1)
        recovered = _decode(inp)
        assert recovered["side"] == side
        _pixel_check(recovered, solve(inp))
    # Smaller, truncated fields are valid even though demonstrations are long.
    # The diagonal island does not connect to the singleton origin.
    short = np.array([[0, 5, 0], [0, 0, 4], [0, 4, 4]], dtype=np.int8)
    ground_tile = np.array([[0, 2], [2, 2]], dtype=np.int8)
    corner_bar = np.array([[5, 5, 0], [4, 4, 0], [0, 0, 0]], dtype=np.int8)
    for side in range(4):
        for source in (short, corner_bar):
            inp = _pack(rng, ground_tile, np.rot90(source, -side).copy(), 0, 1)
            decoded = _decode(inp)
            _pixel_check(decoded, solve(inp))
        distances = _distances(np.rot90(short, -side), 0, np.argwhere(np.rot90(short, -side) == 5))
        assert np.sum(distances >= 0) == 1
    # Equal shortest arrivals and all roots at time zero, independent of order.
    full = np.ones((4, 5), dtype=np.int8)
    roots = np.array([[0, 1], [0, 2], [0, 3]])
    da = _distances(full, 0, roots)
    assert np.array_equal(da, _distances(full, 0, roots[::-1]))
    assert da[3, 2] == 3 and da[2, 0] == 3
    long_routes = _depth_first_distances(full, 0, roots)
    assert np.all(da <= long_routes) and np.any(da != long_routes)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
