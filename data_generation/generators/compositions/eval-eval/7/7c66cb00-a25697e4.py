"""Pivoted sorting stencils: rigid orientation inferred through routed cavities.

The solver reads only pixels. Generation paints a known intended casting, then
checks that the visible input admits exactly one final output. No parent code is
imported. All random choices use the caller's NumPy Generator.
"""
from __future__ import annotations

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and
            1 <= b <= 4 and a > b):
        raise ValueError("counts require 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(np.all((g >= 0) & (g <= 9)), "colors must be 0..9")
    return g.astype(np.int8, copy=False)


def _parts(points):
    todo = set(points)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(frozenset(part))
    return result


def _bbox(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _norm(points):
    r0, _, c0, _ = _bbox(points)
    return tuple(sorted((r-r0, c-c0) for r, c in points))


def _turn(p, op):
    """Four rotations; then a left/right reflection followed by those rotations."""
    r, c = p
    if op >= 4:
        c = -c
    for _ in range(op % 4):
        r, c = c, -r
    return r, c


def _shape_class(points):
    return min(_norm({_turn(p, op) for p in points}) for op in range(8))


def _project(key, op, bands, policy=(0, 1), group_colors=False, route=True):
    """Return all cavity-compatible castings of one pivoted stencil.

    The ordinary rule has route=True. route=False is a bounded diagnostic that
    lets a shape choose any tray, retaining the pivot and edge constraints.
    """
    pr, pc = key["pivot"]
    regions = []
    local = key["colors"]
    for color in local:
        points = key["regions"][color]
        groups = [points] if group_colors else _parts(points)
        new_color = local[1-local.index(color)] if policy[int(op >= 4)] else color
        for part in groups:
            oriented = frozenset((pr+_turn((r-pr, c-pc), op)[0],
                                  pc+_turn((r-pr, c-pc), op)[1]) for r, c in part)
            regions.append((new_color, oriented))
    choices = []
    for color, part in regions:
        bottom = max(r for r, c in part)
        fits = []
        for bi, band in enumerate(bands):
            if route and color != band["route"]:
                continue
            dest = frozenset((r+band["bottom"]-bottom, c) for r, c in part)
            if group_colors:
                touched = [s for s in band["slots"] if s & dest]
                if touched and set().union(*touched) == set(dest):
                    fits.append((bi, dest, part, color, frozenset((bi, s) for s in touched)))
            elif dest in band["slots"]:
                fits.append((bi, dest, part, color, frozenset({(bi, dest)})))
        if not fits:
            return []
        choices.append(fits)
    castings = []

    def combine(i, used, moves):
        if i == len(choices):
            castings.append({"op": op, "moves": tuple(moves), "used": frozenset(used)})
            return
        for bi, dest, part, color, tokens in choices[i]:
            if not tokens & used:
                combine(i+1, used | tokens, moves+[(bi, dest, part, color)])

    combine(0, set(), [])
    return castings


def _parse_canonical(g, bg):
    """Read complete tray rows, then recover marked source objects above them."""
    h, w = g.shape
    info = []
    for row in g:
        non = np.flatnonzero(row != bg)
        if not len(non):
            info.append(None)
            continue
        l, r = int(non[0]), int(non[-1])
        inside = sorted({int(x) for x in row[l+1:r] if x != bg})
        if r-l >= 4 and row[l] == row[r] and len(inside) == 1 and inside[0] != row[l]:
            info.append((l, r, inside[0], int(row[l])))
        else:
            info.append(False)
    for first in range(3, h):
        if not info[first] or info[first-1] is not None:
            continue
        if any(x is False for x in info[first:]):
            continue
        bands, r = [], first
        while r < h:
            if info[r] is None:
                r += 1
                continue
            top, row_info = r, info[r]
            while r+1 < h and info[r+1] == row_info:
                r += 1
            l, right, route, rail = row_info
            if r == top:  # At least a full near row and one possible cavity row.
                break
            cavity_cells = {tuple(map(int, p)) for p in
                            np.argwhere(g[top:r+1, l+1:right] == bg)}
            cavity_cells = {(y+top, x+l+1) for y, x in cavity_cells}
            slots = _parts(cavity_cells)
            if not np.all(g[top, l+1:right] == route):
                break
            if any(max(y for y, x in s) != r for s in slots):
                break
            bands.append(dict(top=top, bottom=r, left=l, right=right,
                              route=route, rail=rail, slots=frozenset(slots)))
            r += 1
        else:
            if not 2 <= len(bands) <= 3:
                continue
            if len({b["route"] for b in bands}) != len(bands):
                continue
            if any(b["top"] < a["bottom"]+2 for a, b in zip(bands, bands[1:])):
                continue
            routes = {b["route"] for b in bands}
            stage_points = {tuple(map(int, p)) for p in np.argwhere(g[:first] != bg)}
            extras = {int(g[p]) for p in stage_points}-routes
            if len(extras) != 1:
                continue
            marker = next(iter(extras))
            if marker in {b["rail"] for b in bands}:
                continue
            keys = []
            for cells in _parts(stage_points):
                pivots = [p for p in cells if int(g[p]) == marker]
                colors = tuple(sorted({int(g[p]) for p in cells}-{marker}))
                if len(pivots) != 1 or len(colors) != 2 or not set(colors) <= routes:
                    break
                pivot = pivots[0]
                regions = {c: frozenset(p for p in cells if int(g[p]) == c) for c in colors}
                # A complete radius square lies in staging and away from rails.
                radius = max(max(abs(y-pivot[0]), abs(x-pivot[1])) for y, x in cells)
                if not (radius <= pivot[0] and pivot[0]+radius < first-1 and
                        radius < pivot[1] and pivot[1]+radius < w-1):
                    break
                keys.append(dict(cells=cells, pivot=pivot, colors=colors, regions=regions,
                                 radius=radius))
            else:
                if 1 <= len(keys) <= 3:
                    yield dict(grid=g, background=bg, marker=marker, bands=bands,
                               keys=keys, first=first)


def _interpret(input_grid):
    g = _grid(input_grid)
    backgrounds = sorted({int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])})
    interpretations = []
    for k in range(4):
        canon = np.rot90(g, k)
        for bg in backgrounds:
            for a in _parse_canonical(canon, bg):
                a["rotation"] = k
                interpretations.append(a)
    _require(len(interpretations) == 1, "one visible staging/tray interpretation required")
    return interpretations[0]


def _answers(a, policy=(0, 1), group_colors=False, route=True, ops=range(8), limit=2):
    candidates = []
    for key in a["keys"]:
        local = []
        for op in ops:
            local.extend(_project(key, op, a["bands"], policy, group_colors, route))
        candidates.append(local)
    base = a["grid"].copy()
    for key in a["keys"]:
        for p in key["cells"]:
            if p != key["pivot"]:
                base[p] = a["background"]
    answers = {}

    def search(i, used, chosen):
        if len(answers) >= limit:
            return
        if i == len(candidates):
            out = base.copy()
            for cast in chosen:
                for bi, dest, part, color in cast["moves"]:
                    for p in dest:
                        out[p] = a["bands"][bi]["rail"]
            answers.setdefault(out.tobytes(), (out, tuple(chosen)))
            return
        for cast in candidates[i]:
            if not cast["used"] & used:
                search(i+1, used | cast["used"], chosen+[cast])

    search(0, frozenset(), [])
    return list(answers.values())


def _solve_analysis(input_grid):
    a = _interpret(input_grid)
    answers = _answers(a)
    _require(len(answers) == 1, "casting must have one final colored output")
    a["output"], a["castings"] = answers[0]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer source pivots, routes, cavities, orientation parity, and casting."""
    a = _solve_analysis(input_grid)
    return np.rot90(a["output"], -a["rotation"]).copy()


def _grow(rng, allowed, start, n):
    cells = {start}
    while len(cells) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4}
                          & (set(allowed)-cells))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _stencil(rng, small=False, lesson=False, reserve=False):
    """Two compositional grammars, with irregular islands and variable contours."""
    frame = lesson or rng.random() < .5
    if frame:
        h = (4 if reserve else int(rng.integers(4, 6))) if lesson else (
            3 if reserve else int(rng.integers(3, 5 if small else 6)))
        w = (6 if reserve else int(rng.integers(6, 8))) if lesson else (
            5 if reserve else int(rng.integers(5, 7 if small else 8)))
        a = {(r, c) for r in range(h) for c in range(w)
             if r in (0, h-1) or c in (0, w-1)}
        mid = w//2
        pivot = (1, mid)
        b = set()
        windows = [list(range(1, mid)), list(range(mid+1, w-1))]
        if not lesson:
            number = int(rng.integers(1, 4 if w >= 7 else 3))
            if number == 1:
                windows = [windows[int(rng.integers(2))]]
            elif number == 3:
                windows = [[1], [mid], [w-2]]
                pivot = (0, mid)
                a.remove(pivot)
        for side, cols in enumerate(windows):
            depth = (h-2 if side == 0 else 1) if lesson else int(rng.integers(1, h-1))
            allowed = {(r, c) for r in range(1, depth+1) for c in cols}
            start = (1, cols[int(rng.integers(len(cols)))])
            count = len(allowed) if reserve else int(rng.integers(max(1, depth), len(allowed)+1))
            island = _grow(rng, allowed, start, count)
            # Reach the chosen depth so the independence lesson is reliable.
            island.update((r, start[1]) for r in range(1, depth+1))
            if lesson and side == 0:
                # A variable rectangular island with one missing far corner
                # distinguishes vertical reflection after independent settling.
                island = set(allowed)
                island.remove((depth, cols[int(rng.integers(len(cols)))]))
            b.update(island)
        # An irregular lip makes the outer region asymmetric; optional extra
        # internal struts introduce more substantial masks than plain frames.
        if not reserve and w >= 6 and rng.random() < .65:
            side = 1 if rng.random() < .5 else w-2
            a.update((r, side) for r in range(1, h-1) if (r, side) not in b | {pivot})
    else:
        h = int(rng.integers(3, 5 if small else 6))
        w = int(rng.integers(5, 7 if small else 8))
        mid = w//2
        pivot = (h-2, mid)
        a = {(h-1, c) for c in range(w)}
        b = set()
        fingers = [list(range(0, mid)), list(range(mid+1, w))]
        if rng.random() < .3:
            fingers = [fingers[int(rng.integers(2))]]
        for cols in fingers:
            allowed = {(r, c) for r in range(h-1) for c in cols}
            start = (h-2, cols[int(rng.integers(len(cols)))])
            b.update(_grow(rng, allowed, start, int(rng.integers(1, len(allowed)+1))))
        available = {(r, c) for r in range(h) for c in range(w)}-b-{pivot}
        for _ in range(int(rng.integers(w+1))):
            frontier = sorted({(r+dr, c+dc) for r, c in a for dr, dc in _D4} & (available-a))
            if frontier:
                a.add(frontier[int(rng.integers(len(frontier)))])
    _require(not a & b and pivot not in a | b, "stencil grammar overlap")
    _require(len(_parts(a | b | {pivot})) == 1, "stencil must be connected")
    return (frozenset((r-pivot[0], c-pivot[1]) for r, c in a),
            frozenset((r-pivot[0], c-pivot[1]) for r, c in b))


def _slot_clear(holes, cells):
    # Cavities remain separate four-connected masks; they may enclose islands.
    return not any((r+dr, c+dc) in holes for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _construct(rng, key_count, band_count, parity, turn, lesson, reserve=False):
    palette = list(map(int, rng.permutation(10)))
    bg, marker = palette[:2]
    routes = palette[2:2+band_count]
    external = palette[2+band_count:]
    rails = []
    for i, color in enumerate(routes):
        choices = external + [r for r in routes if r != color]
        rails.append(int(rng.choice(choices)))
    stencils = [_stencil(rng, small=key_count == 3 or band_count == 3,
                         lesson=lesson and i == 0, reserve=reserve) for i in range(key_count)]
    ops = [int(rng.integers(4)) + 4*int(parity)]
    ops += [int(rng.integers(8)) for _ in range(key_count-1)]
    incoming = [tuple(frozenset(_turn(p, _INVERSE[op]) for p in reg) for reg in shape)
                for shape, op in zip(stencils, ops)]
    radii = [max(max(abs(r), abs(c)) for reg in shape for r, c in reg) for shape in incoming]
    widths = [2*r+1 for r in radii]
    margin = 2 if reserve else int(rng.integers(1, 4))
    gap = 1 if reserve else int(rng.integers(1, 3))
    width = sum(widths)+gap*(key_count-1)+2*margin
    if width > 30:
        return None
    stage = 2*max(radii)+1 + (0 if reserve else int(rng.integers(2)))
    max_height = max(max(r for r, c in part)-min(r for r, c in part)+1
                     for shape in stencils for reg in shape for part in _parts(reg))
    # Near-side route rows remain intact; even a ring-shaped cavity is visible.
    depths = [max_height+1+(0 if reserve else int(rng.integers(2))) for _ in routes]
    gaps = [1 if reserve else int(rng.integers(1, 3)) for _ in routes]
    height = stage+sum(depths)+sum(gaps)+(0 if reserve else int(rng.integers(2)))
    if height > 30:
        return None
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    bands, cursor = [], stage
    for i in rng.permutation(band_count):
        i = int(i)
        cursor += gaps[i]
        # Insets never invade any pivot's full rotation square.
        left = 0 if reserve else int(rng.integers(margin))
        right = width-1-(0 if reserve else int(rng.integers(margin)))
        band = dict(top=cursor, bottom=cursor+depths[i]-1, left=left, right=right,
                    route=routes[i], rail=rails[i])
        bands.append(band)
        inp[cursor:cursor+depths[i], left:right+1] = routes[i]
        inp[cursor:cursor+depths[i], [left, right]] = rails[i]
        cursor += depths[i]
    out[:] = inp
    holes = [set() for _ in bands]
    selected = []
    x = margin
    for i, (shape, original, radius, op) in enumerate(zip(stencils, incoming, radii, ops)):
        pc = x+radius
        pr = radius+int(rng.integers(stage-2*radius))
        x += widths[i]+gap
        colors = tuple(map(int, rng.choice(routes, 2, replace=False)))
        inp[pr, pc] = out[pr, pc] = marker
        for reg, color in zip(original, colors):
            for r, c in reg:
                inp[pr+r, pc+c] = color
        final_colors = colors[::-1] if op >= 4 else colors
        for reg, color in zip(shape, final_colors):
            bi = next(j for j, band in enumerate(bands) if band["route"] == color)
            band = bands[bi]
            for part in _parts(reg):
                far = max(r for r, c in part)
                dest = {(band["bottom"]+r-far, pc+c) for r, c in part}
                if not _slot_clear(holes[bi], dest):
                    return None
                holes[bi].update(dest)
                selected.append((bi, frozenset(dest)))
                for p in dest:
                    inp[p] = bg
                    out[p] = band["rail"]
    # A congruent wrong-route cavity at the same lateral coordinates creates a
    # direct color-routing counterexample. The third tray is locally unused by
    # each key, though it may be active for another key elsewhere on the grid.
    twins = 0
    if band_count == 3:
        order = list(map(int, rng.permutation(len(selected))))
        for si in order:
            bi, cavity = selected[si]
            for bj in rng.permutation(band_count):
                bj = int(bj)
                if bj == bi:
                    continue
                delta = bands[bj]["bottom"]-bands[bi]["bottom"]
                twin = {(r+delta, c) for r, c in cavity}
                if _slot_clear(holes[bj], twin):
                    holes[bj].update(twin)
                    for p in twin:
                        inp[p] = out[p] = bg
                    twins += 1
                    break
            if twins >= (1 if key_count == 1 else 2):
                break
        if not twins:
            return None
    # Additional unused sockets use independently sampled masks and locations.
    # They all reach the far edge, so edge contact alone does not select them.
    extra = 0
    for _ in range(45):
        bi = int(rng.integers(band_count))
        band = bands[bi]
        mh = int(rng.integers(1, min(4, band["bottom"]-band["top"])+1))
        mw = int(rng.integers(1, min(4, band["right"]-band["left"]-1)+1))
        mask = _grow(rng, {(r, c) for r in range(mh) for c in range(mw)},
                     (mh-1, int(rng.integers(mw))), int(rng.integers(1, mh*mw+1)))
        nr, _, nc, _ = _bbox(mask)
        mask = {(r-nr, c-nc) for r, c in mask}
        size = max(c for r, c in mask)+1
        l = int(rng.integers(band["left"]+1, band["right"]-size+1))
        bottom = max(r for r, c in mask)
        cells = {(band["bottom"]+r-bottom, l+c) for r, c in mask}
        if _slot_clear(holes[bi], cells):
            holes[bi].update(cells)
            for p in cells:
                inp[p] = out[p] = bg
            extra += 1
            if extra >= 1+(key_count > 1):
                break
    if twins+extra < 1:
        return None
    pair = {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}
    try:
        a = _solve_analysis(pair["input"])
        if not np.array_equal(np.rot90(a["output"], -a["rotation"]), pair["output"]):
            return None
        if int(a["castings"][0]["op"] >= 4) != parity:
            return None
        if lesson and not _independent(a):
            return None
    except AssertionError:
        return None
    return pair


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _independent(a):
    for cast in a["castings"]:
        moves = cast["moves"]
        for i, (bi, dest, part, color) in enumerate(moves):
            for bj, other_dest, other, other_color in moves[i+1:]:
                if bi == bj and max(r for r, c in part) != max(r for r, c in other):
                    return True
    return False


def _training_hypotheses(train):
    analyses = [_solve_analysis(p["input"]) for p in train]
    survivors = []
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if all(any(np.array_equal(out, a["output"]) for out, moves in _answers(a, policy, limit=512))
               for a in analyses):
            survivors.append(policy)
    return survivors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    turn = int(rng.integers(4))
    one_key_training = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                key_count = 1 if one_key_training or i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                band_count = 3 if i == 0 else 2 if i == 1 else int(rng.integers(2, 4))
                parity = i % 2 if i < 2 else int(rng.integers(2))
            else:
                # Both schedules shift to more simultaneous castings. The
                # single-key schedule permits either two or three at test.
                key_count = int(rng.integers(2, 4)) if one_key_training else 3
                band_count = int(rng.integers(2, 4))
                parity = int(rng.integers(2))
            pair = None
            for attempt in range(100):
                candidate = _construct(rng, key_count, band_count, parity, turn,
                                       lesson=split == "train" and i < 2,
                                       reserve=attempt >= 80)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                if split == "train" and i < 2:
                    # Color exchange must be inferable from geometry, rather
                    # than becoming distinguishable only after assuming it.
                    a = _solve_analysis(candidate["input"])
                    wrong = (1, 1) if parity == 0 else (0, 0)
                    if any(np.array_equal(o, a["output"]) for o, casts in
                           _answers(a, wrong, limit=512)):
                        continue
                pair = candidate
                break
            _require(pair is not None, "bounded stencil construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, selected and unused sockets, and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            signature = _key(inp)
            _require(signature not in seen, "distinct inputs required")
            seen.add(signature)
            a = _solve_analysis(inp)
            analyses[split].append(a)
            canonical_out = np.rot90(out, a["rotation"])
            _require(np.array_equal(a["output"], canonical_out), "incorrect casting label")
            _require(not np.array_equal(inp, out), "nontrivial casting")
            used = set().union(*(set(c["used"]) for c in a["castings"]))
            all_slots = {(bi, s) for bi, band in enumerate(a["bands"]) for s in band["slots"]}
            _require(len(all_slots-used) >= 1, "an unused cavity must remain")
            for band in a["bands"]:
                occupied = set()
                for slot in sorted(band["slots"], key=min):
                    _require(_slot_clear(occupied, slot), "cavity connectivity moat")
                    occupied.update(slot)
            for key, cast in zip(a["keys"], a["castings"]):
                _require(canonical_out[key["pivot"]] == a["marker"], "pivot preservation")
                _require(all(canonical_out[p] == a["background"] for p in key["cells"]
                             if p != key["pivot"]), "source clearing")
                _require(len(cast["moves"]) >= 2, "both colors must contribute")
                for color in key["colors"]:
                    components = _parts(key["regions"][color])
                    for i, part in enumerate(components):
                        others = set().union(*components[:i], *components[i+1:])
                        _require(_slot_clear(others, part), "diagonal-only source grouping")
                for bi, dest, oriented, color in cast["moves"]:
                    band = a["bands"][bi]
                    _require(color == band["route"], "color address")
                    _require(max(r for r, c in dest) == band["bottom"], "independent far edge")
                    _require(_norm(dest) == _norm(oriented), "exact oriented mask")
                    _require({c for r, c in dest} == {c for r, c in oriented}, "lateral invariance")
            for bi, slot in all_slots-used:
                _require(all(canonical_out[p] == a["background"] for p in slot), "unused cavity changed")
    train = analyses["train"]
    _require(min(len(a["keys"]) for a in analyses["test"]) > max(len(a["keys"]) for a in train),
             "generated tests must shift to more stencils")
    _require(any(_independent(a) for a in train), "independent-component training witness")
    _require({int(c["op"] >= 4) for a in train for c in a["castings"]} == {0, 1},
             "both orientation parities must be taught")
    _require(_training_hypotheses(puzzle["train"]) == [(0, 1)], "parity rule not distinguished")
    _require(any(not _answers(a, group_colors=True) or
                 any(not np.array_equal(out, a["output"]) for out, moves in _answers(a, group_colors=True))
                 for a in train), "training does not distinguish component grouping")
    _require(any(len(_answers(a, route=False)) > 1 for a in train),
             "training needs a congruent wrong-route cavity")
    _require(any(not _answers(a, ops=(0,)) or
                 any(not np.array_equal(out, a["output"]) for out, moves in _answers(a, ops=(0,)))
                 for a in train), "training does not require rigid reorientation")


def measure_puzzle(puzzle):
    """Actual geometry measurements; useful for bounded diversity checks."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _solve_analysis(pair["input"])
            chosen = [m for cast in a["castings"] for m in cast["moves"]]
            total_slots = sum(len(b["slots"]) for b in a["bands"])
            records.append(dict(split=split, shape=tuple(np.asarray(pair["input"]).shape),
                                direction=a["rotation"], background=a["background"],
                                keys=len(a["keys"]), bands=len(a["bands"]),
                                depths=tuple(b["bottom"]-b["top"]+1 for b in a["bands"]),
                                ops=tuple(c["op"] for c in a["castings"]),
                                components=len(chosen), unused=total_slots-len(chosen),
                                masks=tuple(_shape_class(m[1]) for m in chosen),
                                source_sizes=tuple(len(k["cells"])-1 for k in a["keys"]),
                                independent=_independent(a),
                                reuse=any(b["rail"] in {x["route"] for x in a["bands"]} for b in a["bands"]),
                                inset=any(b["left"] or b["right"] != a["grid"].shape[1]-1 for b in a["bands"]),
                                route_ambiguous=len(_answers(a, route=False)) > 1))
    return records


def self_test():
    """42 seeds, all legal counts, finite ablations, and corruption checks."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        records.extend(measure_puzzle(puzzle))
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
    _require(len(fingerprints) == 42, "seed diversity")
    _require({r["direction"] for r in records} == set(range(4)), "movement directions")
    _require({op for r in records for op in r["ops"]} == set(range(8)), "rigid operations")
    _require({r["keys"] for r in records} == {1, 2, 3}, "stencil counts")
    _require({r["bands"] for r in records} == {2, 3}, "tray counts")
    _require(len({r["shape"] for r in records}) >= 30, "canvas variation")
    _require(len({m for r in records for m in r["masks"]}) >= 35, "mask variation")
    _require(len({r["depths"] for r in records}) >= 10, "tray depth variation")
    _require(len({r["components"] for r in records}) >= 6, "component count variation")
    _require(len({n for r in records for n in r["source_sizes"]}) >= 12, "stencil area variation")
    _require(all(any(r[x] for r in records) for x in ("reuse", "inset", "independent", "route_ambiguous")),
             "unexercised structural feature")
    p = generate_puzzle(991, 2, 1)
    q = generate_puzzle(991, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "reproducibility")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(1200+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise the same-rule reserve directly, without waiting for an unlucky
    # random seed to use the final twenty construction attempts.
    reserve_rng = np.random.default_rng(603)
    for n in (1, 2, 3):
        for bands in (2, 3):
            for parity in (0, 1):
                pair = None
                for attempt in range(12):
                    pair = _construct(reserve_rng, n, bands, parity, (n+bands+parity) % 4,
                                      lesson=n < 3, reserve=True)
                    if pair is not None:
                        break
                _require(pair is not None, "reserve case exhausted")
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "reserve casting")
    p["test"][0]["output"] = p["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Explicit expected pixels, separate from constructive shape generation.
    fixture = np.zeros((15, 13), dtype=np.int8)
    fixture[2, 6] = 9
    fixture[1, 5:8] = 1
    fixture[2:4, 5] = 2
    fixture[6:10, 1:-1] = 1
    fixture[6:10, [0, -1]] = 3
    fixture[11:15, 1:-1] = 2
    fixture[11:15, [0, -1]] = 4
    fixture[9, 5:8] = 0
    fixture[13:15, 5] = 0
    fixture[14, 7] = 0  # Unused singleton, not a two-cell socket.
    expected = fixture.copy()
    expected[1, 5:8] = expected[2:4, 5] = 0
    expected[9, 5:8] = 3
    expected[13:15, 5] = 4
    for turn in range(4):
        inp = np.rot90(fixture, turn).copy()
        unchanged = inp.copy()
        _require(np.array_equal(solve(inp), np.rot90(expected, turn)), "explicit casting fixture")
        _require(np.array_equal(inp, unchanged), "solve must not mutate its input")
    # Completing the second two-cell socket lets a 180-degree source rotation
    # choose the other cavity. Both operations preserve colors; outputs differ.
    fixture[13, 7] = 0
    try:
        solve(fixture)
    except AssertionError as error:
        _require("one final colored output" in str(error), "wrong ambiguity rejection")
    else:
        raise AssertionError("competing cavity selections accepted")
    return {"seeds": 42, "pairs": len(records), "grid_shapes": len({r["shape"] for r in records}),
            "mask_classes": len({m for r in records for m in r["masks"]}),
            "operations": sorted({op for r in records for op in r["ops"]}),
            "component_counts": sorted({r["components"] for r in records}), "reserve_cases": 12}
