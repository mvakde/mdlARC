"""Hinged tile routes: reconstruct a route, reroute its tail, carry bend parity.

All output-affecting information is visible. No parent generator is imported.
Annotation colors 5 (repair) and 6 (redirect) are fixed family conventions;
all other roles are decoded from the lattice and its unique binary source tile.
"""
from itertools import combinations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 512511426, 310714683, 419545072, 501021977, 628073535,
              244486231, 161051714, 372619460, 254929976, 715317511,
              610943895, 195392258, 324621425, 652719273, 236460943,
              707974499, 967364267, 495483905, 330273155, 303771804,
              610758616, 827723219, 610449926, 687942599, 490988541,
              482927037, 111079452, 291021669, 280043549, 969963874,
              813823315, 81284281, 575017097, 494796533, 785243356,
              790562053, 426377752, 5066788, 865174999, 905993250,
              17585672)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid must contain integer ARC colors")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _delta(a, b):
    return b[0] - a[0], b[1] - a[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _components(cells):
    todo, result = set(cells), []
    while todo:
        stack = [min(todo)]
        todo.remove(stack[0])
        group = set(stack)
        while stack:
            for q in _neighbors(stack.pop(), todo):
                todo.remove(q)
                group.add(q)
                stack.append(q)
        result.append(group)
    return result


def _walk(cells, root, shape):
    """One induced, open chain; the other endpoint is on the board boundary."""
    _require(len(cells) >= 4 and root in cells, "route too short or lacks source")
    degrees = {p: len(_neighbors(p, cells)) for p in cells}
    ends = [p for p, degree in degrees.items() if degree == 1]
    _require(len(ends) == 2 and root in ends and all(1 <= d <= 2 for d in degrees.values()),
             "route must be a simple chain rooted at the tile")
    terminal = next(p for p in ends if p != root)
    _require(_boundary(terminal, shape), "terminal must reach the board edge")
    route = [root]
    previous = None
    while route[-1] != terminal:
        options = [q for q in _neighbors(route[-1], cells) if q != previous]
        _require(len(options) == 1, "ambiguous route traversal")
        previous = route[-1]
        route.append(options[0])
        _require(len(route) <= len(cells), "closed route")
    _require(len(route) == len(cells), "disconnected route")
    return route


def _decode(input_grid):
    a = _grid(input_grid)
    separator = int(a[0, 0])
    _require(separator not in (5, 6), "separator is an annotation color")
    rows = np.flatnonzero(np.all(a == separator, axis=1))
    cols = np.flatnonzero(np.all(a == separator, axis=0))
    _require(len(rows) >= 3 and len(cols) >= 3 and rows[0] == cols[0] == 0
             and rows[-1] == a.shape[0] - 1 and cols[-1] == a.shape[1] - 1,
             "missing enclosing lattice")
    pitch = int(rows[1] - rows[0])
    n = pitch - 1
    _require(2 <= n <= 4 and np.all(np.diff(rows) == pitch)
             and np.all(np.diff(cols) == pitch), "lattice cells must be equal squares")
    shape = (len(rows) - 1, len(cols) - 1)
    blocks, nonuniform, counts = {}, [], {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            b = a[r * pitch + 1:(r + 1) * pitch, c * pitch + 1:(c + 1) * pitch]
            _require(not np.any(b == separator), "separator inside a cell")
            blocks[r, c] = b.copy()
            values = np.unique(b)
            if len(values) == 1:
                v = int(values[0])
                counts[v] = counts.get(v, 0) + 1
            else:
                _require(len(values) == 2 and not set(values) & {5, 6}, "invalid source tile")
                nonuniform.append((r, c))
    _require(len(nonuniform) == 1, "exactly one complete binary source tile is required")
    root = nonuniform[0]
    tile = blocks[root]
    background = max(counts, key=counts.get)
    _require(counts[background] * 2 > np.prod(shape) and background not in (5, 6),
             "empty cells must be a strict majority")
    colors = set(map(int, np.unique(tile)))
    _require(background not in colors, "source tile cannot contain background")
    guide_colors = set(counts) - {background, 5, 6}
    _require(len(guide_colors) == 1 and guide_colors <= colors, "guide must use one source-tile color")
    guide = next(iter(guide_colors))
    macro = np.zeros(shape, int)
    for p, b in blocks.items():
        if p == root:
            macro[p] = 4
        else:
            v = int(b[0, 0])
            macro[p] = {background: 0, guide: 1, 5: 2, 6: 3}[v]
    _require(np.count_nonzero(macro == 3) <= 1, "at most one redirect per route")
    return dict(macro=macro, root=root, tile=tile, background=background,
                separator=separator, guide=guide, n=n)


def _repair(macro, root):
    """Join components through isolated straight omissions, never fill a loop.

    Each omission joins two observed path-component endpoints. Enumerating
    at most three such joins checks uniqueness without consulting any label.
    """
    observed = {tuple(map(int, p)) for p in np.argwhere(np.isin(macro, [1, 2, 4]))}
    groups = _components(observed)
    _require(1 <= len(groups) <= 4, "at most three isolated omissions are supported")
    if len(groups) == 1:
        return _walk(observed, root, macro.shape), set()
    component = {p: i for i, group in enumerate(groups) for p in group}
    degree = {p: len(_neighbors(p, observed)) for p in observed}
    candidates = []
    for p0 in np.argwhere(macro == 0):
        p = tuple(map(int, p0))
        ns = _neighbors(p, observed)
        if (len(ns) == 2 and ns[0][0] + ns[1][0] == 2 * p[0]
                and ns[0][1] + ns[1][1] == 2 * p[1]
                and component[ns[0]] != component[ns[1]]
                and all(degree[q] <= 1 and (q != root or degree[q] == 0) for q in ns)):
            candidates.append(p)
    answers = []
    for selected in combinations(candidates, len(groups) - 1):
        missing = set(selected)
        if any(_neighbors(p, missing) for p in missing):
            continue
        try:
            route = _walk(observed | missing, root, macro.shape)
        except ValueError:
            continue
        answers.append((route, missing))
        if len(answers) > 1:
            break
    _require(len(answers) == 1, "gap reconstruction is not unique")
    return answers[0]


def _ray(pivot, direction, shape):
    result = []
    p = _add(pivot, direction)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, direction)
    return result


def _redirect(macro, route):
    markers = np.argwhere(macro == 3)
    if not len(markers):
        return route, None
    marker = tuple(map(int, markers[0]))
    candidates = []
    for i in range(1, len(route) - 1):
        pivot = route[i]
        direction = _delta(pivot, marker)
        if direction not in DIRS or _delta(pivot, route[i + 1]) != (-direction[0], -direction[1]):
            continue
        incoming = _delta(route[i - 1], pivot)
        if incoming[0] * direction[0] + incoming[1] * direction[1] != 0:
            continue
        final = route[:i + 1] + _ray(pivot, direction, macro.shape)
        try:
            _require(len(set(final)) == len(final), "ray revisits the retained route")
            _require(_walk(set(final), route[0], macro.shape) == final,
                     "ray touches the retained route")
        except ValueError:
            continue
        candidates.append((final, dict(index=i, pivot=pivot, direction=direction,
                                       old=route[i + 1:], new=final[i + 1:])))
    _require(len(candidates) == 1, "redirect marker must identify one corner pivot")
    return candidates[0]


def _turns(route):
    return [i for i in range(1, len(route) - 1)
            if _delta(route[i - 1], route[i]) != _delta(route[i], route[i + 1])]


def _stamp(shape, tile, background, route, behavior="normal", markers=()):
    n = len(tile)
    out = np.full((shape[0] * n, shape[1] * n), background, int)
    swapped = int(tile.min()) + int(tile.max()) - tile
    phase, rotation, marker_phase = 0, 0, 0
    turns = set(_turns(route))
    for i, (r, c) in enumerate(route):
        if i in turns:
            phase ^= 1
            incoming, outgoing = _delta(route[i - 1], route[i]), _delta(route[i], route[i + 1])
            rotation += 1 if incoming[0] * outgoing[1] - incoming[1] * outgoing[0] < 0 else -1
        state = phase
        if behavior == "no_exchange":
            state = 0
        elif behavior == "all_exchange":
            state = int(i > 0)
        elif behavior == "step_exchange":
            state = i % 2
        elif behavior == "marker_exchange":
            marker_phase ^= int((r, c) in markers)
            state = marker_phase
        patch = swapped if state else tile
        if behavior == "rotate_tile":
            patch = np.rot90(patch, rotation)
        elif behavior == "no_substitution" and i:
            patch = np.full(tile.shape, int(tile[0, 0]), int)
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = patch
    return out


def _analyze(input_grid):
    info = _decode(input_grid)
    route, missing = _repair(info["macro"], info["root"])
    final, edit = _redirect(info["macro"], route)
    info.update(route=route, missing=missing, final=final, edit=edit)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction: complete the unique route and transport its tile."""
    info = _analyze(input_grid)
    return _stamp(info["macro"].shape, info["tile"], info["background"], info["final"])


def _line(a, b):
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)]


def _polyline(vertices):
    route = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        route.extend(_line(a, b)[1:])
    return route


def _induced(route):
    if len(set(route)) != len(route):
        return False
    cells = set(route)
    return all(set(_neighbors(p, cells)) == set(route[max(0, i - 1):i] + route[i + 1:i + 2])
               for i, p in enumerate(route))


def _random_route(rng, shape):
    """Grow varying orthogonal segments inward from a random boundary cell."""
    side = int(rng.integers(4))
    h, w = shape
    start = ((0, int(rng.integers(1, w - 1))),
             (int(rng.integers(1, h - 1)), w - 1),
             (h - 1, int(rng.integers(1, w - 1))),
             (int(rng.integers(1, h - 1)), 0))[side]
    direction = DIRS[(side + 2) % 4]
    route = [start]
    for _ in range(int(rng.integers(3, 7))):
        options = []
        for distance in range(1, max(shape)):
            end = (route[-1][0] + distance * direction[0], route[-1][1] + distance * direction[1])
            if not _inside(end, shape) or _boundary(end, shape):
                continue
            candidate = route + _line(route[-1], end)[1:]
            if len(candidate) * 2 + 2 < h * w and _induced(candidate):
                options.append(candidate)
        if not options:
            break
        # Short and long legs both occur; minimum one is useful on small boards.
        route = options[int(rng.integers(len(options)))]
        direction = DIRS[(DIRS.index(direction) + int(rng.choice([1, 3]))) % 4]
    return list(reversed(route))


def _candidate_edit(route, shape, direction, old_bends, prefix_bends):
    choices = []
    occupied = set(route)
    for i in _turns(route):
        pivot = route[i]
        if _delta(pivot, route[i + 1]) != (-direction[0], -direction[1]):
            continue
        q = _add(pivot, direction)
        if not _inside(q, shape) or q in occupied:
            continue
        final = route[:i + 1] + _ray(pivot, direction, shape)
        if (_induced(final) and len(_turns(route[i:])) >= old_bends
                and len(_turns(route[:i + 1])) >= prefix_bends):
            choices.append((i, final))
    return choices


def _encode(macro, tile, background, separator, guide):
    n = len(tile)
    pitch = n + 1
    out = np.full((macro.shape[0] * pitch + 1, macro.shape[1] * pitch + 1), separator, int)
    for r, c in np.ndindex(macro.shape):
        value = int(macro[r, c])
        block = tile if value == 4 else np.full(tile.shape, (background, guide, 5, 6)[value], int)
        out[r * pitch + 1:(r + 1) * pitch, c * pitch + 1:(c + 1) * pitch] = block
    return out


def _tile(rng, n, colors):
    for _ in range(64):
        bits = rng.random((n, n)) < float(rng.uniform(.2, .8))
        if bits.any() and not bits.all() and all(
                not np.array_equal(bits, np.rot90(bits, k))
                and not np.array_equal(bits, ~np.rot90(bits, k)) for k in (1, 2, 3)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _rotate_route(route, shape, k):
    for _ in range(k):
        route = [(shape[1] - 1 - c, r) for r, c in route]
        shape = (shape[1], shape[0])
    return route, shape


def _curled_route(rng, n, direction, prefix_bend=False):
    """An old tail curls across its replacement ray; deletion must precede drawing.

    This is a topology mixture, with variable leg lengths and an optional
    retained bend, rather than a new rule or an exception in the solver.
    """
    h = int(rng.integers(7, (9 if n == 2 else 7) + 1))
    w = int(rng.integers(7, (9 if n == 2 else 7) + 1))
    col = int(rng.integers(3, w - 3))
    row = int(rng.integers(3, h - 3))
    left = int(rng.integers(1, col - 1))
    bottom = int(rng.integers(row + 2, h - 1))
    right = int(rng.integers(col + 2, w - 1))
    vertices = [(row, right), (row, col), (bottom, col), (bottom, left),
                (1, left), (1, w - 1)]
    if prefix_bend or rng.random() < .5:
        root_row = int(rng.integers(row + 2, h - 1))
        vertices.insert(0, (root_row, right))
    route = _polyline(vertices)
    k = (-DIRS.index(direction)) % 4
    return _rotate_route(route, (h, w), k)


def _make_pair(rng, n, direction=None, witness=0):
    # witness 1: three hinge transitions, a retained gap and odd ray state;
    # witness 2: bent tail deletion, an annotated pivot and an old-tail marker.
    for attempt in range(241):
        fallback = attempt == 240
        low, high = ({2: (6, 9), 3: (5, 7), 4: (5, 5)})[n]
        if witness:
            low = max(low, 6)
        shape = (int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1)))
        if witness == 2:
            # Teach phase ownership at old/new overlap before any query uses it.
            route, shape = _curled_route(rng, n, direction, prefix_bend=True)
        elif not fallback and witness == 0 and n < 4 and direction is not None and rng.random() < .18:
            route, shape = _curled_route(rng, n, direction)
        elif fallback:
            shape = (6, 7) if witness == 1 else (6, 6) if n < 4 else (5, 5)
            vertices = ([(1, 3), (1, 1), (4, 1), (4, 5), (5, 5)] if witness == 1 else
                        [(4, 1), (1, 1), (1, 4), (3, 4), (3, 5)] if n < 4 else
                        [(3, 1), (1, 1), (1, 3), (0, 3)])
            route = _polyline(vertices)
            # Canonical pivots point up for size 6, down for size 5.
            canonical = (-1, 0) if n < 4 else (1, 0)
            k = (DIRS.index(canonical) - DIRS.index(direction or canonical)) % 4
            route, shape = _rotate_route(route, shape, k)
        else:
            route = _random_route(rng, shape)
        if len(route) < 6 or len(_turns(route)) < (2 if witness else 1):
            continue
        if direction is not None:
            choices = _candidate_edit(route, shape, direction, int(witness == 2),
                                      2 if witness == 1 else int(witness == 2))
            if witness == 1:
                choices = [(i, final) for i, final in choices if len(_turns(final)) % 2 == 1]
            if not choices:
                continue
            pivot_index, final = choices[int(rng.integers(len(choices)))]
        else:
            pivot_index, final = None, route
        macro = np.zeros(shape, int)
        for p in route:
            macro[p] = 1
        macro[route[0]] = 4
        if direction is not None:
            macro[_add(route[pivot_index], direction)] = 3
        # Ordinary repair annotations can replace bends, endpoints and pivots.
        forced = []
        if witness == 1:
            straight = [route[i] for i in range(1, pivot_index)
                        if i not in _turns(route)]
            if len(straight) < 2:
                continue
            forced = [route[_turns(route)[0]], straight[-1]]
        elif witness == 2:
            forced = [route[pivot_index], route[-1]]
            old_turns = [i for i in _turns(route) if i > pivot_index]
            if old_turns:
                forced.append(route[old_turns[0]])
        others = list(route[1:])
        rng.shuffle(others)
        sites = list(dict.fromkeys(forced + others[:int(rng.integers(1, 5))]))
        for p in sites:
            macro[p] = 2
        if witness == 1:
            # Intact and repair-marked hinges have identical state semantics;
            # a repair-marked straight cell does not exchange the tile.
            macro[route[_turns(final)[1]]] = 1
        # Isolated unmarked straight gaps are accepted only if uniquely recovered.
        available = [route[i] for i in range(1, len(route) - 1)
                     if i not in _turns(route) and macro[route[i]] == 1]
        rng.shuffle(available)
        if witness:
            available.sort(key=lambda p: p not in final)
        desired = int(rng.integers(1, 4)) if witness == 1 else int(rng.integers(0, 4))
        removed = []
        for p in available:
            if len(removed) >= desired:
                break
            if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 1 for q in removed):
                continue
            macro[p] = 0
            try:
                recovered, _ = _repair(macro, route[0])
                if recovered != route:
                    raise ValueError("recovered a different route")
            except ValueError:
                macro[p] = 1
            else:
                removed.append(p)
        if witness == 1 and not set(removed) & set(final):
            if fallback:
                # Reserve a known straight cell rather than retry without evidence.
                for q in removed:
                    macro[q] = 1
                p = route[1]
                macro[p] = 0
                try:
                    _require(_repair(macro, route[0])[0] == route, "fallback gap")
                except ValueError:
                    raise RuntimeError("witness fallback could not show gap reconstruction")
            else:
                continue
        if (np.count_nonzero(macro == 0) * 2 <= np.prod(shape)
                or np.count_nonzero(macro == 1) < 2):
            continue
        palette = list(map(int, rng.permutation([0, 1, 2, 3, 4, 7, 8, 9])))
        background, separator, guide, other = palette[:4]
        tile = _tile(rng, n, (guide, other))
        inp = _encode(macro, tile, background, separator, guide)
        out = _stamp(shape, tile, background, final)
        try:
            info = _analyze(inp)
            _require(info["route"] == route and info["final"] == final, "construction not recoverable")
            _require(np.array_equal(solve(inp), out), "construction/solver disagreement")
        except ValueError:
            if fallback:
                raise RuntimeError("conservative fallback is not recoverable")
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded route construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held = int(rng.integers(4))
    train_dirs = [d for i, d in enumerate(DIRS) if i != held]
    first_n = int(rng.choice([2, 3]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            n = first_n if witness == 1 else 5 - first_n if witness == 2 else int(rng.integers(2, 5))
            direction = (DIRS[held] if split == "test" else
                         train_dirs[int(rng.integers(3))] if witness or
                         (witness == 0 and rng.random() < .65) else None)
            for _ in range(12):
                pair = _make_pair(rng, n, direction, witness)
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


def validate_generated_puzzle(puzzle):
    """Full labels plus route-specific evidence; raises AssertionError on failure."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        nt, nq = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
        seen, records = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                _require(_key(inp) not in seen, "duplicate input")
                seen.add(_key(inp))
                info = _analyze(inp)
                predicted = _stamp(info["macro"].shape, info["tile"], info["background"], info["final"])
                _require(np.array_equal(predicted, out), "output violates route or tile semantics")
                # Background is preserved at every inactive block; every active
                # block uses both source colors, without residual annotations.
                n = info["n"]
                cells = set(info["final"])
                for r, c in np.ndindex(info["macro"].shape):
                    b = out[r * n:(r + 1) * n, c * n:(c + 1) * n]
                    if (r, c) in cells:
                        _require(set(map(int, np.unique(b))) == set(map(int, np.unique(info["tile"]))),
                                 "tile substitution lost a color")
                    else:
                        _require(np.all(b == info["background"]), "inactive block was changed")
                _require(not np.any(np.isin(out, [5, 6])), "annotation survived")
                records[split].append(info)
        train = records["train"]
        _require(any(len(_turns(f["final"])) >= 2 and f["missing"] & set(f["final"])
                     and np.any(f["macro"] == 2) for f in train), "missing repair/hinge witness")
        _require(any(f["edit"] and len(_turns([f["edit"]["pivot"]] + f["edit"]["old"])) >= 1
                     and f["macro"][f["edit"]["pivot"]] == 2
                     and any(f["macro"][p] == 2 for p in f["edit"]["old"]) for f in train),
                 "missing bent-deletion/marked-pivot witness")
        for behavior in ("no_exchange", "all_exchange", "step_exchange", "rotate_tile", "no_substitution",
                         "marker_exchange"):
            _require(any(not np.array_equal(
                _stamp(f["macro"].shape, f["tile"], f["background"], f["final"], behavior,
                       {tuple(p) for p in np.argwhere(f["macro"] == 2)}),
                _stamp(f["macro"].shape, f["tile"], f["background"], f["final"])) for f in train),
                f"training does not distinguish {behavior}")
        _require(any(f["edit"] and not np.array_equal(
            _stamp(f["macro"].shape, f["tile"], f["background"], f["route"]),
            _stamp(f["macro"].shape, f["tile"], f["background"], f["final"])) for f in train),
            "training does not distinguish ignoring redirection")
        overlap_witness = False
        for f in train:
            if f["edit"] is None:
                continue
            before = _stamp(f["macro"].shape, f["tile"], f["background"], f["route"])
            after = _stamp(f["macro"].shape, f["tile"], f["background"], f["final"])
            n = f["n"]
            for r, c in set(f["edit"]["new"]) & set(f["edit"]["old"]):
                block = np.s_[r * n:(r + 1) * n, c * n:(c + 1) * n]
                overlap_witness |= not np.array_equal(before[block], after[block])
        _require(overlap_witness, "training must show the new ray replacing an old tile state")
        ray_phases = {len(_turns(f["final"])) % 2 for f in train if f["edit"]}
        _require(ray_phases == {0, 1}, "training must show both original and exchanged ray states")
        train_dirs = {f["edit"]["direction"] for f in train if f["edit"]}
        _require(all(f["edit"] and f["edit"]["direction"] not in train_dirs for f in records["test"]),
                 "test lacks the documented held-out direction shift")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise AssertionError(str(exc)) from exc


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)


def self_test():
    """Bounded 42-seed checks, structural variation, equivariance and negatives."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    signatures = {tuple(_key(p["input"]) for s in ("train", "test") for p in task[s])
                  for task in puzzles}
    assert len(signatures) == 42
    features, normalized, shapes, directions = [], set(), set(), set()
    for task in puzzles:
        validate_generated_puzzle(task)
        for split in ("train", "test"):
            for pair in task[split]:
                info = _analyze(pair["input"])
                features.append(info)
                normalized.add(_normalized(pair["input"]))
                shapes.add(pair["input"].shape)
                if info["edit"]:
                    directions.add(info["edit"]["direction"])
                # Repair annotations are waypoints, not state-change commands.
                marker_sites = np.argwhere(info["macro"] == 2)
                if len(marker_sites):
                    macro = info["macro"].copy()
                    macro[tuple(marker_sites[0])] = 1
                    unmarked = _encode(macro, info["tile"], info["background"],
                                       info["separator"], info["guide"])
                    assert np.array_equal(solve(unmarked), pair["output"])
                # Rotating the entire input rotates its complete template too.
                assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
                assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    assert {f["n"] for f in features} == {2, 3, 4}
    assert directions == set(DIRS)
    assert len(shapes) >= 15 and len(normalized) >= 150
    assert len({len(f["route"]) for f in features}) >= 12
    assert len({tuple(f["macro"].shape) for f in features}) >= 15
    assert len({(f["background"], f["separator"], f["guide"]) for f in features}) >= 80
    assert max(len(_turns(f["route"])) for f in features) >= 4
    assert {len(f["missing"]) for f in features} >= {0, 1, 2}
    assert any(f["edit"] and len(f["edit"]["new"]) == 1 for f in features)
    assert any(f["edit"] and set(f["edit"]["new"]) & set(f["edit"]["old"]) for f in features)
    # All non-annotation palette roles can change without changing the rule.
    recolor = np.arange(10)
    free = np.array([0, 1, 2, 3, 4, 7, 8, 9])
    recolor[free] = np.roll(free, 1)
    example = puzzles[0]["test"][0]
    assert np.array_equal(solve(recolor[example["input"]]), recolor[example["output"]])
    a, b = generate_puzzle(2026), generate_puzzle(2026)
    assert all(_key(x[r]) == _key(y[r]) for s in a for x, y in zip(a[s], b[s]) for r in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(17, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Pixel-level negatives include retained tiles, deleted tails, and blank cells.
    task = puzzles[0]
    query = task["test"][0]
    info = _analyze(query["input"])
    sites = [info["root"], info["edit"]["new"][0], info["edit"]["old"][-1]]
    blank = next(p for p in np.ndindex(info["macro"].shape) if p not in info["final"])
    sites.append(blank)
    for r, c in sites:
        bad = {"train": task["train"], "test": [dict(query)]}
        bad["test"][0]["output"] = query["output"].copy()
        n = info["n"]
        out = bad["test"][0]["output"]
        out[r * n, c * n] = (int(out[r * n, c * n]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": len(features), "input_shapes": len(shapes),
            "normalized_inputs": len(normalized), "tile_sizes": [2, 3, 4],
            "directions": len(directions), "checks": "generator-owned only"}


if __name__ == "__main__":
    print(self_test())
