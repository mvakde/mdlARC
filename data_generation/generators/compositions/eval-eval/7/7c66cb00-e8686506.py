"""Four-way gravity jigsaws: edge keys constrain an exact one-use assembly.

Only the standard library and NumPy are used. Solving reads only the input;
construction labels are painted from sampled placements, separately from search.
No execution traces, seed caches, files, or network are used by this module.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_EDGES = ("top", "right", "bottom", "left")
_SEEDS = (0, 641637268, 130832033, 922903831, 496156561, 387596606,
          812873817, 871771033, 2816308, 392029451, 904380679, 808881986,
          214062336, 969393082, 733596902, 994406223, 217709965, 787011959,
          64088340, 479030059, 12741811, 198619894, 837119239, 662995565,
          359576329, 481336292, 200543148, 170020579, 329013006, 501747138,
          163052664, 166969166, 726279814, 59662323, 591381415, 506250598,
          145668136, 964750033, 78409918, 240363768, 839563632, 70306383)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _components(points, grid=None, diagonal=False):
    """Monochromatic components when a grid is supplied, otherwise binary."""
    remaining, result = set(points), []
    neighbors = _EIGHT if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = r+dr, c+dc
                if p in remaining and (grid is None or grid[p] == grid[start]):
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _edge_cells(h, w, edge):
    if edge == "top":
        return [(0, c) for c in range(1, w-1)]
    if edge == "bottom":
        return [(h-1, c) for c in range(1, w-1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h-1)]
    return [(r, w-1) for r in range(1, h-1)]


def _parse(input_grid):
    """Recover the framed board, edge keys and complete exterior inventory."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(x == bg) for x in (a[0], a[-1], a[:, 0], a[:, -1])),
             "uniform outer background border required")
    interpretations = []
    for frame in map(int, np.unique(a)):
        if frame == bg:
            continue
        rr, cc = np.where(a == frame)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        board = a[r0:r1, c0:c1]
        h, w = board.shape
        if min(h, w) < 6 or any(board[p] != frame for p in
                                  ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))):
            continue
        if not np.all(np.isin(board[1:-1, 1:-1], (bg, frame))):
            continue
        keys, valid = {}, True
        for edge in _EDGES:
            cells = _edge_cells(h, w, edge)
            marks = [i for i, p in enumerate(cells) if board[p] != frame]
            if not marks:
                continue
            if len(marks) != 3 or marks != list(range(marks[0], marks[0]+3)):
                valid = False
                break
            paint, route, other = (int(board[cells[i]]) for i in marks)
            if paint != other or len({bg, frame, paint, route}) != 4 or route in keys:
                valid = False
                break
            keys[route] = (edge, paint)
        routes = set(keys)
        paints = {p for e, p in keys.values()}
        if not valid or not 2 <= len(keys) <= 4 or len(paints) != len(keys) or routes & paints:
            continue
        outside = a != bg
        outside = outside.copy()
        outside[r0:r1, c0:c1] = False
        points = [tuple(map(int, p)) for p in np.argwhere(outside)]
        components = _components(points, a)
        if not components or any(int(a[c[0]]) not in routes for c in components):
            continue
        inventory = [(int(a[c[0]]), _normalize(c)) for c in components]
        if set(r for r, s in inventory) != routes:
            continue
        if sum(len(s) for r, s in inventory) != int(np.count_nonzero(board == bg)):
            continue
        interpretations.append(dict(grid=a, background=bg, frame=frame, bbox=(r0, c0, h, w),
                                    board=board.copy(), keys=keys, inventory=inventory,
                                    components=components))
    _require(len(interpretations) == 1, "one recognizable framed board required")
    return interpretations[0]


class _SearchLimit(Exception):
    pass


def _offsets(shape, h, w, edge=None):
    """Interior placements: pin the normal extremum, search the parallel axis."""
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    rows, cols = range(1, h-sh), range(1, w-sw)
    if edge == "top":
        rows = (1,)
    elif edge == "bottom":
        rows = (h-1-sh,)
    elif edge == "left":
        cols = (1,)
    elif edge == "right":
        cols = (w-1-sw,)
    if sh > h-2 or sw > w-2:
        return
    for r in rows:
        for c in cols:
            yield r, c


def _assembly(analysis, restrict_edges=True, node_limit=20000):
    """Prove uniqueness of the colored board, quotienting identical-piece swaps.

    Search is bounded in time/state, retains at most two colored completions,
    and never interprets an exhausted search bound as uniqueness.
    """
    board, bg, keys = analysis["board"], analysis["background"], analysis["keys"]
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    index = {p: i for i, p in enumerate(holes)}
    inventory = Counter(analysis["inventory"])
    types = sorted(inventory)
    by_cell, candidates = [[] for p in holes], []
    for t, (route, shape) in enumerate(types):
        edge, paint = keys[route]
        moves = []
        for r, c in _offsets(shape, *board.shape, edge if restrict_edges else None):
            cells = tuple((r+dr, c+dc) for dr, dc in shape)
            if not all(p in index for p in cells):
                continue
            ids = [index[p] for p in cells]
            bits = sum(1 << i for i in ids)
            code = sum((paint+1) << (4*i) for i in ids)
            move = (t, bits, code)
            moves.append((cells, bits, code))
            for i in ids:
                by_cell[i].append(move)
        candidates.append(moves)
    nodes = 0

    @lru_cache(maxsize=8192)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise _SearchLimit("exact-cover search bound reached")
        if not remaining:
            return (0,) if not any(counts) else ()
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            choices = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes))-1, tuple(inventory[t] for t in types))
    answers = []
    for code in codes:
        out = board.copy()
        for p, i in index.items():
            out[p] = ((code >> (4*i)) & 15)-1
        answers.append(out)
    # Locally competing pigments are stronger evidence than piece-swap counts.
    conflict = any(len({keys[types[t][0]][1] for t, mask, paint in choices}) > 1
                   for choices in by_cell)
    return answers, dict(nodes=nodes, types=types, candidates=candidates,
                         local_competition=conflict,
                         multiple_placements=any(len(m) > inventory[t] for t, m in zip(types, candidates)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate every piece once into the unique edge-constrained mosaic."""
    analysis = _parse(input_grid)
    answers, _ = _assembly(analysis)
    _require(len(answers) == 1, "inventory must yield one colored completion")
    return answers[0].copy()


def _direct_alignment_possible(a):
    """Ablation: A's unchanged parallel coordinates, with no B packing search."""
    r0, c0, h, w = a["bbox"]
    holes = {tuple(map(int, p)) for p in np.argwhere(a["board"] == a["background"])}
    occupied = set()
    for cells in a["components"]:
        route = int(a["grid"][cells[0]])
        edge, paint = a["keys"][route]
        dr = dc = 0
        if edge == "top":
            dr = r0+1-min(r for r, c in cells)
        elif edge == "bottom":
            dr = r0+h-2-max(r for r, c in cells)
        elif edge == "left":
            dc = c0+1-min(c for r, c in cells)
        else:
            dc = c0+w-2-max(c for r, c in cells)
        target = {(r+dr-r0, c+dc-c0) for r, c in cells}
        if not target <= holes or target & occupied:
            return False
        occupied.update(target)
    return occupied == holes


def _source_contact(a):
    points = {p for cells in a["components"] for p in cells}
    return any((r+dr, c+dc) in points and a["grid"][r, c] != a["grid"][r+dr, c+dc]
               for r, c in points for dr, dc in _FOUR)


def _shape(rng, maxh, maxw, small=False):
    """Procedural connected masks, with no split-specific shape vocabulary."""
    h = int(rng.integers(1, min(maxh, 3 if small else 5)+1))
    w = int(rng.integers(1, min(maxw, 3 if small else 5)+1))
    kind = int(rng.integers(7))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} if rng.random() < .5 else {(0, c) for c in range(w)}
    elif kind == 2:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif kind == 3:
        cells = {(0, c) for c in range(w)} | {(r, w//2) for r in range(h)}
    elif kind == 4 and min(h, w) >= 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    elif kind == 5:
        cells, r, c = {(0, 0)}, 0, 0
        while r < h-1 or c < w-1:
            if r < h-1 and (c == w-1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells.add((r, c))
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        goal = int(rng.integers(1, min(h*w, 12)+1))
        while len(cells) < goal:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    # Shape orientation is sampled here, never changed by the transformation.
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    cells = _normalize(cells)
    if max(r for r, c in cells) >= maxh or max(c for r, c in cells) >= maxw:
        cells = _normalize({(c, r) for r, c in cells})
    return cells


def _layout(rng, lesson, test, palette):
    if lesson == "routing":
        edges = ["top", "bottom"] if rng.random() < .5 else ["left", "right"]
        count = int(rng.integers(2, 5))
        ih, iw = int(rng.integers(5, 9)), int(rng.integers(5, 10))
    else:
        count = int(rng.integers(6, 10) if test else rng.integers(4, 8))
        k = int(rng.integers(2, min(4, count)+1))
        edges = [str(e) for e in rng.choice(_EDGES, k, replace=False)]
        ih, iw = int(rng.integers(6, 11)), int(rng.integers(6, 11))
    h, w = ih+2, iw+2
    bg, frame = map(int, palette[:2])
    routes = list(map(int, palette[2:2+len(edges)]))
    paints = list(map(int, palette[2+len(edges):2+2*len(edges)]))
    keys = dict(zip(routes, zip(edges, paints)))
    board = np.full((h, w), frame, dtype=np.int8)
    for route, (edge, paint) in keys.items():
        cells = _edge_cells(h, w, edge)
        start = int(rng.integers(len(cells)-2))
        for p, value in zip(cells[start:start+3], (paint, route, paint)):
            board[p] = value
    output = board.copy()
    occupied, inventory = set(), []
    assignments = routes[:] + [int(rng.choice(routes)) for _ in range(count-len(routes))]
    if lesson != "routing":
        rng.shuffle(assignments)
    duplicate = _shape(rng, 3, 3, small=True)
    for i, route in enumerate(assignments):
        edge, paint = keys[route]
        placed = False
        for attempt in range(24):
            shape = duplicate if lesson == "routing" and i < 2 else _shape(rng, ih, iw)
            choices = []
            touching = []
            for r, c in _offsets(shape, h, w, edge):
                cells = {(r+dr, c+dc) for dr, dc in shape}
                if cells & occupied:
                    continue
                choices.append(cells)
                if any((r+dr, c+dc) in occupied for r, c in cells for dr, dc in _FOUR):
                    touching.append(cells)
            if not choices:
                continue
            pool = touching if touching and lesson != "routing" and rng.random() < .85 else choices
            cells = pool[int(rng.integers(len(pool)))]
            occupied.update(cells)
            inventory.append((route, shape))
            for p in cells:
                board[p], output[p] = bg, paint
            placed = True
            break
        if not placed:
            return None
    if len(occupied) > ih*iw*.76:
        return None
    analysis = dict(board=board, background=bg, keys=keys, inventory=inventory)
    try:
        answers, search = _assembly(analysis)
    except _SearchLimit:
        return None
    if len(answers) != 1 or not np.array_equal(answers[0], output):
        return None
    gap_components = _components(occupied)
    shared = any(len({int(output[p]) for p in cells}) > 1 for cells in gap_components)
    if lesson == "packing" or test:
        if not shared or not search["local_competition"] or not search["multiple_placements"]:
            return None
    if lesson == "routing":
        # The duplicate shape in two routed pigments must be a real witness.
        try:
            loose, _ = _assembly(analysis, restrict_edges=False)
        except _SearchLimit:
            return None
        if len(loose) != 2:
            return None
    return board, output, inventory, bg


def _scatter(rng, board, inventory, bg, teach_contact=False):
    """Keep every oriented piece intact; all placement retries stay in memory."""
    h, w = board.shape
    for attempt in range(8):
        ch = int(rng.integers(max(19, h+6), 31)) if attempt < 6 else 30
        cw = int(rng.integers(max(19, w+6), 31)) if attempt < 6 else 30
        br, bc = int(rng.integers(2, ch-h-1)), int(rng.integers(2, cw-w-1))
        a = np.full((ch, cw), bg, dtype=np.int8)
        a[br:br+h, bc:bc+w] = board
        forbidden = {(r, c) for r in range(br-1, br+h+1) for c in range(bc-1, bc+w+1)}
        source = set()
        order = sorted(range(len(inventory)), key=lambda i: (-len(inventory[i][1]), i))
        for i in order:
            route, shape = inventory[i]
            sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
            choices, contacts = [], []
            for r in range(1, ch-sh):
                for c in range(1, cw-sw):
                    cells = {(r+dr, c+dc) for dr, dc in shape}
                    if cells & forbidden or any(a[p] != bg for p in cells):
                        continue
                    if any(a[rr+dr, cc+dc] == route for rr, cc in cells for dr, dc in _EIGHT):
                        continue
                    choices.append(cells)
                    if teach_contact and any((rr+dr, cc+dc) in source
                                             for rr, cc in cells for dr, dc in _FOUR):
                        contacts.append(cells)
            if not choices:
                break
            pool = contacts if contacts else choices
            cells = pool[int(rng.integers(len(pool)))]
            source.update(cells)
            for p in cells:
                a[p] = route
        else:
            if not teach_contact or any((r+dr, c+dc) in source and a[r, c] != a[r+dr, c+dc]
                                        for r, c in source for dr, dc in _FOUR):
                return a
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Two guaranteed teaching witnesses, then varied assemblies and harder recombinations."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    palette = rng.permutation(10)
    exchange = bool(rng.integers(2))
    train_masks = set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            lesson = "routing" if split == "train" and i == 0 else "packing" if split == "train" and i == 1 else None
            roles = rng.permutation(10) if exchange else palette
            for attempt in range(400):
                data = _layout(rng, lesson, split == "test", roles)
                if data is None:
                    continue
                board, output, inventory, bg = data
                if split == "test" and not any(s not in train_masks for r, s in inventory):
                    continue
                inp = _scatter(rng, board, inventory, bg, teach_contact=lesson == "packing")
                if inp is None or _key(inp) in seen:
                    continue
                _require(np.array_equal(solve(inp), output), "construction and visible assembly differ")
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": output})
                if split == "train":
                    train_masks.update(s for r, s in inventory)
                break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    a = _parse(pair["input"])
    answers, search = _assembly(a)
    _require(len(answers) == 1 and np.array_equal(answers[0], pair["output"]), "incorrect assembly label")
    board, out, bg = a["board"], np.asarray(pair["output"]), a["background"]
    fixed = board != bg
    _require(np.array_equal(out[fixed], board[fixed]), "fixed cells changed")
    _require(not np.any(out == bg), "unfilled board")
    holes = {tuple(map(int, p)) for p in np.argwhere(~fixed)}
    _require(len(holes) == sum(len(s) for r, s in a["inventory"]), "inventory area mismatch")
    expected = Counter()
    for route, shape in a["inventory"]:
        expected[a["keys"][route][1]] += len(shape)
    actual = Counter(map(int, out[~fixed]))
    _require(actual == expected, "route-to-paint cell conservation failed")
    _require(len(_components({p for c in a["components"] for p in c}, a["grid"], True)) == len(a["components"]),
             "diagonal grouping ambiguity")
    shared = any(len({int(out[p]) for p in c}) > 1 for c in _components(holes))
    return dict(board_shape=board.shape, canvas_shape=np.asarray(pair["input"]).shape,
                edges=tuple(sorted(e for e, p in a["keys"].values())), background=bg,
                pieces=len(a["inventory"]), area=len(holes), shared_gap=shared,
                local_competition=search["local_competition"],
                multiple_placements=search["multiple_placements"], nodes=search["nodes"],
                masks=tuple(sorted({s for r, s in a["inventory"]})),
                repeated=any(v > 1 for v in Counter(a["inventory"]).values()),
                source_contact=_source_contact(a),
                direct_alignment_fails=not _direct_alignment_possible(a),
                fixed_components=len(_components({tuple(map(int, p)) for p in np.argwhere(fixed)})))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid example counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(inp.size > out.size, "return the board crop")
            records[split].append(_measure(pair))
    # Validate teaching content from pixels, not sampler flags or pair indices.
    _require(any(m["shared_gap"] and m["local_competition"] and m["multiple_placements"]
                 for m in records["train"]), "training lacks the packing lesson")
    _require(any(m["source_contact"] for m in records["train"]), "training lacks monochromatic splitting")
    _require(any(m["direct_alignment_fails"] for m in records["train"]), "training does not require lateral packing")
    routing_witness = False
    for pair in puzzle["train"]:
        a = _parse(pair["input"])
        shapes = {}
        for route, shape in a["inventory"]:
            shapes.setdefault(shape, set()).add(route)
        if not any(len(routes) > 1 for routes in shapes.values()):
            continue
        try:
            answers, _ = _assembly(a, restrict_edges=False)
        except _SearchLimit:
            continue
        routing_witness |= len(answers) > 1
        if routing_witness:
            break
    _require(routing_witness, "training lacks a color/edge-routing witness")
    _require(all(m["shared_gap"] and m["local_competition"] and m["multiple_placements"]
                 for m in records["test"]), "test must recombine the packing lesson")
    train_masks = {s for m in records["train"] for s in m["masks"]}
    _require(all(any(s not in train_masks for s in m["masks"]) for m in records["test"]),
             "test must include a new oriented mask")


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k], b[k])
               for a, b in zip(p[s], q[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """42 streamed seeds; retain feature sets, never the full sample corpus."""
    boards, canvases, masks, palettes, edge_sets, counts, fingerprints = (set() for _ in range(7))
    totals = Counter()
    max_nodes = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(hashlib.sha256(repr([_key(p["input"]) for p in puzzle["train"]]).encode()).hexdigest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _measure(pair)
                boards.add(m["board_shape"])
                canvases.add(m["canvas_shape"])
                masks.update(m["masks"])
                palettes.add(m["background"])
                edge_sets.add(m["edges"])
                counts.add(m["pieces"])
                max_nodes = max(max_nodes, m["nodes"])
                for name in ("shared_gap", "local_competition", "multiple_placements", "repeated",
                             "source_contact", "direct_alignment_fails"):
                    totals[name] += bool(m[name])
                totals[split] += 1
                # A geometric rotation of the whole scene must rotate the answer.
                if seed == 0:
                    for turn in (1, 2, 3):
                        _require(np.array_equal(solve(np.rot90(pair["input"], turn)),
                                                np.rot90(pair["output"], turn)), "rotation covariance")
    _require(len(fingerprints) == 42, "duplicate seed puzzles")
    _require(len(boards) >= 20 and len(canvases) >= 35 and len(masks) >= 50, "geometric variety")
    _require(palettes == set(range(10)) and len(edge_sets) >= 8 and len(counts) >= 5, "role/count variety")
    _require(totals["shared_gap"] >= 84 and totals["local_competition"] >= 84, "packing coverage")
    # Explicit small regression with known expected cells; no sampled label.
    board = np.ones((6, 6), dtype=np.int8)
    board[0, 1:4], board[5, 1:4] = (3, 2, 3), (5, 4, 5)
    board[1, 1:3], board[4, 2:4] = 0, 0
    domino = ((0, 0), (0, 1))
    example = dict(board=board, background=0, keys={2: ("top", 3), 4: ("bottom", 5)},
                   inventory=[(2, domino), (4, domino)])
    expected = board.copy()
    expected[1, 1:3], expected[4, 2:4] = 3, 5
    answers, _ = _assembly(example)
    _require(len(answers) == 1 and np.array_equal(answers[0], expected), "edge alignment regression")
    _require(len(_assembly(example, restrict_edges=False)[0]) == 2, "routing ablation must change the answer set")
    example["inventory"] = [(2, ((0, 0), (1, 0))), (4, domino)]
    _require(not _assembly(example)[0], "piece orientation was illegally changed")
    example["inventory"] = [(2, domino)]
    _require(not _assembly(example)[0], "missing inventory was silently copied")
    p = generate_puzzle(0, 2, 1)
    generate_puzzle(17, 3, 2)
    _require(_equal(p, generate_puzzle(0, 2, 1)), "determinism after intervening calls")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            q = generate_puzzle(nt*100+ne, nt, ne)
            _require(len(q["train"]) == nt and len(q["test"]) == ne, "explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Palettes are semantic roles, and solve must leave its argument untouched.
    inp = p["test"][0]["input"].copy()
    original = inp.copy()
    permutation = np.array([8, 4, 7, 2, 9, 0, 6, 1, 5, 3], dtype=np.int8)
    _require(np.array_equal(solve(permutation[inp]), permutation[p["test"][0]["output"]]),
             "literal-color dependency")
    solve(inp)
    _require(np.array_equal(inp, original), "solve mutated its input")
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt fixed pixel accepted")
    return dict(seeds=len(_SEEDS), pairs=totals["train"]+totals["test"],
                board_shapes=len(boards), canvas_shapes=len(canvases), masks=len(masks),
                backgrounds=len(palettes), edge_sets=len(edge_sets), piece_counts=sorted(counts),
                feature_pairs=dict(totals), max_search_nodes=max_nodes)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
