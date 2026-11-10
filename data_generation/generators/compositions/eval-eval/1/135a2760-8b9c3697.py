"""Phase-locked docking: recover a ribbon, then dock solid cargo by its phase.

Only Python's standard library and NumPy are used. The public solver receives
one input and has no access to puzzle seeds or generation metadata.
"""

from itertools import product

import numpy as np


TRAIL = 0
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rectangular(cells):
    rows, cols = zip(*cells)
    return len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _cell(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    out, perp = fork["out"], fork["perp"]
    return k * out[0] + q * perp[0], k * out[1] + q * perp[1]


def _parse_fork(cells, grid):
    """Find a full base and three uninterrupted positive rectangular spokes."""
    parses = []
    for out in DIRS:
        perp = -out[1], out[0]
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 30)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, n in enumerate(lengths) for k in range(1, n + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        q0, q1, length = runs[1]
        if q0 + q1 != width:
            continue
        f = dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                 width=width, runs=tuple(runs), q0=q0, span=q1-q0,
                 length=length, cells=cells)
        center = {_cell(f, k, q) for k in range(1, length + 1) for q in range(q0, q1)}
        structure = cells - center
        structural_colors = {int(grid[p]) for p in structure}
        if len(structural_colors) != 1:
            continue
        colors = {int(grid[p]) for p in center}
        if len(colors) != 2 or colors & structural_colors:
            continue
        f.update(center=center, structure=structure,
                 structural_color=structural_colors.pop(), colors=tuple(sorted(colors)))
        parses.append(f)
    if len(parses) != 1:
        raise ValueError("A fork needs one base and three positive, separated spokes")
    return parses[0]


def _recover(sequence):
    """Strict-majority periodic explanations must agree on the repaired ribbon."""
    span, length = sequence.shape
    colors = np.unique(sequence)
    if len(colors) != 2:
        raise ValueError("A ribbon must contain exactly two colors")
    candidates = []
    for period in range(1, length // 4 + 1):
        unit = np.empty((span, period), dtype=np.int8)
        valid = True
        for q, phase in product(range(span), range(period)):
            group = sequence[q, phase::period]
            n = int(np.count_nonzero(group == colors[0]))
            if 2 * n == len(group):
                valid = False
                break
            unit[q, phase] = colors[0] if 2 * n > len(group) else colors[1]
        if not valid or len(np.unique(unit)) != 2:
            continue
        repaired = unit[:, np.arange(length) % period]
        if np.count_nonzero(repaired != sequence) <= max(1, sequence.size // 8):
            candidates.append((period, unit, repaired))
    if not candidates:
        raise ValueError("No supported periodic repair")
    repaired = candidates[0][2]
    if any(not np.array_equal(repaired, c[2]) for c in candidates):
        raise ValueError("Competing periodic repairs")
    return repaired, candidates[0][1], candidates[0][0]


def _code_fits(unit, k, length, color):
    # The first central-spoke slice has depth one, hence phase zero.
    return bool(np.all(unit[:, (np.arange(k, k + length) - 1) % unit.shape[1]] == color))


def _candidate(fork, token, grid, background):
    local = {(r * fork["out"][0] + c * fork["out"][1] - fork["kmin"],
              r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"])
             for r, c in token["cells"]}
    qs = {q for _, q in local}
    if qs != set(range(fork["q0"], fork["q0"] + fork["span"])):
        return None
    start, end = min(k for k, _ in local), max(k for k, _ in local)
    length = end - start + 1
    if start <= fork["length"] + 1 or token["color"] not in fork["colors"]:
        return None
    # A whole footprint must match, not just its leading slice or middle cell.
    finals = [k for k in range(fork["length"] + 1, start)
              if _code_fits(fork["unit"], k, length, token["color"])]
    if not finals:
        return None
    final_start = finals[0]
    route = {_cell(fork, k, q) for k in range(final_start, end + 1) for q in qs}
    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
           or (p not in token["cells"] and int(grid[p]) != background)
           for p in route for r, c in [p]):
        return None
    final = {_cell(fork, k, q) for k in range(final_start, final_start + length) for q in qs}
    return dict(route=route, final=final, start=start, final_start=final_start,
                token_length=length, distance=start-final_start,
                phase_gap=final_start-fork["length"]-1)


def _analyze(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if len(np.unique(border)) != 1 or int(border[0]) == TRAIL or np.any(grid == TRAIL):
        raise ValueError("The input needs a uniform nonblack margin; black is reserved for trails")
    background = int(border[0])
    forks, tokens = [], []
    for cells in _components(grid != background):
        if _rectangular(cells):
            colors = {int(grid[p]) for p in cells}
            if len(colors) != 1:
                raise ValueError("Cargo must be a solid single-color rectangle")
            tokens.append(dict(cells=cells, color=colors.pop()))
        else:
            f = _parse_fork(cells, grid)
            sequence = np.array([[grid[_cell(f, k, q)] for k in range(1, f["length"] + 1)]
                                 for q in range(f["q0"], f["q0"] + f["span"])], dtype=np.int8)
            repaired, unit, period = _recover(sequence)
            f.update(sequence=sequence, repaired=repaired, unit=unit, period=period,
                     errors=int(np.count_nonzero(sequence != repaired)))
            forks.append(f)
    if not forks or not tokens:
        raise ValueError("At least one fork and one cargo rectangle are required")
    matches, used, occupied_routes = [], set(), set()
    for fi, f in enumerate(forks):
        options = [(ti, m) for ti, t in enumerate(tokens)
                   if (m := _candidate(f, t, grid, background)) is not None]
        if len(options) > 1:
            raise ValueError("Multiple eligible cargo rectangles for a fork")
        if options:
            ti, match = options[0]
            if ti in used or not occupied_routes.isdisjoint(match["route"]):
                raise ValueError("Assignments or routes interact")
            used.add(ti)
            occupied_routes |= match["route"]
            matches.append(dict(match, fi=fi, ti=ti))
    return dict(grid=grid, background=background, forks=forks, tokens=tokens,
                matches=matches, used=used)


def _render(analysis, *, repair=True, phase=True, move=True):
    result = analysis["grid"].copy()
    if repair:
        for f in analysis["forks"]:
            for q in range(f["span"]):
                for k in range(1, f["length"] + 1):
                    result[_cell(f, k, f["q0"] + q)] = f["repaired"][q, k - 1]
    if not move:
        return result
    for t in analysis["tokens"]:
        _paint(result, t["cells"], analysis["background"])
    for m in analysis["matches"]:
        f, t = analysis["forks"][m["fi"]], analysis["tokens"][m["ti"]]
        if phase:
            route, final = m["route"], m["final"]
        else:
            # A diagnostic shortcut: ordinary contact docking, after repair.
            k0 = f["length"] + 1
            qs = range(f["q0"], f["q0"] + f["span"])
            final = {_cell(f, k, q) for k in range(k0, k0 + m["token_length"]) for q in qs}
            route = {_cell(f, k, q) for k in range(k0, m["start"] + m["token_length"]) for q in qs}
        _paint(result, route - final, TRAIL)
        _paint(result, final, t["color"])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_analyze(input_grid))


def _sample_ribbon(rng, span, period, length, cargo_length, gap_kind):
    """Sample a primitive binary vector unit with an admissible docking phase."""
    for attempt in range(160):
        bits = rng.integers(0, 2, size=(span, period), dtype=np.int8)
        # Reserve a homogeneous run for the full cargo; the other phases can
        # be arbitrary vector slices. Random cyclic rotation changes its phase.
        color_bit = int(rng.integers(2))
        bits[:, :cargo_length] = color_bit
        bits[0, cargo_length % period] = 1 - color_bit
        bits = np.roll(bits, int(rng.integers(period)), axis=1)
        if attempt >= 128:
            bits[:] = 1
            bits[:, :cargo_length] = 0
            bits = np.roll(bits, (attempt - 128) % period, axis=1)
        clean = bits[:, np.arange(length) % period]
        # Choose a feasible color by the stripe, independent of literal palette.
        options = []
        for color in (0, 1):
            k = next((k for k in range(length + 1, length + period + 1)
                      if _code_fits(bits, k, cargo_length, color)), None)
            if k is None or k + cargo_length > 27:
                continue
            gap = k - length - 1
            if gap_kind == "gap" and gap == 0 or gap_kind == "contact" and gap != 0:
                continue
            if gap_kind == "edge" and (gap == 0 or not _code_fits(bits, length + 1, 1, color)):
                continue
            options.append((color, k))
        if not options:
            continue
        chosen = options[int(rng.integers(len(options)))]
        noisy = clean.copy()
        errors = int(rng.integers(1, min(3, max(1, clean.size // 10), span * period) + 1))
        # Corrupt the latest observation of a phase needed by the final cargo.
        # Copying the end of the ribbon therefore gives a wrong docking phase.
        group0 = (chosen[1] - 1) % period
        last = list(range(group0, length, period))[-1]
        noisy[0, last] ^= 1
        groups = [g for g in range(span * period) if g != group0]
        for group in rng.choice(groups, size=errors - 1, replace=False):
            q, phase = divmod(int(group), period)
            positions = list(range(phase, length, period))
            k = int(rng.choice(positions))
            noisy[q, k] ^= 1
        try:
            repaired, unit, found = _recover(noisy)
        except ValueError:
            continue
        if found == period and np.array_equal(repaired, clean):
            return bits, noisy, chosen
    raise RuntimeError("Bounded ribbon construction exhausted")


def _module(rng, span, period, cargo_length, gap_kind, structure, colors, geometry):
    # Longer cargo near the grid boundary restricts the possible last partial cycle.
    maximum = min(4 * period + period - 1, 26 - cargo_length)
    lengths = list(range(4 * period, maximum + 1))
    rng.shuffle(lengths)
    for length in lengths:
        try:
            bits, noisy, (color_bit, final_start) = _sample_ribbon(
                rng, span, period, length, cargo_length, gap_kind)
        except RuntimeError:
            continue
        break
    else:
        raise RuntimeError("No ribbon fits the grid")
    side, clearance, tail = geometry
    width = span + 2 * (side + clearance + tail)
    q0 = tail + side + clearance
    cap = min(length + 3, 26 - cargo_length)
    mode = str(rng.choice(["standard", "unequal", "long_center", "independent"]))
    if mode == "standard" and cap > length:
        left, right = map(int, rng.integers(length + 1, cap + 1, size=2))
    elif mode == "long_center":
        left, right = map(int, rng.integers(1, length, size=2))
    else:
        left, right = map(int, rng.integers(1, cap + 1, size=2))
        if mode == "unequal" and left == right:
            right = right % cap + 1
    start_min = max(final_start + 1, left + 2, right + 2)
    start_max = 28 - cargo_length
    if start_min > start_max:
        raise RuntimeError("Cargo route cannot fit")
    start = int(rng.integers(start_min, start_max + 1))
    depth = start + cargo_length
    source = np.full((depth, width), -1, dtype=np.int8)
    target = source.copy()
    source[0, :] = target[0, :] = structure
    for qa, qb, n in ((tail, tail + side, left), (width-tail-side, width-tail, right)):
        source[1:n+1, qa:qb] = structure
        target[1:n+1, qa:qb] = structure
    source[1:length+1, q0:q0+span] = np.asarray(colors)[noisy.T]
    target[1:length+1, q0:q0+span] = np.asarray(colors)[bits[:, np.arange(length) % period].T]
    cargo_color = colors[color_bit]
    source[start:start+cargo_length, q0:q0+span] = cargo_color
    target[final_start+cargo_length:start+cargo_length, q0:q0+span] = TRAIL
    target[final_start:final_start+cargo_length, q0:q0+span] = cargo_color
    return source, target


def _pair(rng, axis, count, split, index, background, structure):
    spans = list(map(int, rng.integers(1, 3 if split == "train" else 4, size=count)))
    spans[0] = (1 if index == 0 else 2) if split == "train" and index < 2 else spans[0]
    if split == "test":
        spans[0] = 3
    band_gap = int(rng.integers(1, 3))
    # Reserve three cross-axis cells for separated distractors; no divider is drawn.
    budget = 25 - band_gap * (count - 1)
    geometries = [(1, 1, 0)] * count
    choices = list(product((1, 2), (1, 2), (0, 1)))
    for i in rng.permutation(count):
        other = sum(spans[j] + 2 * sum(geometries[j]) for j in range(count) if j != i)
        options = [g for g in choices if other + spans[i] + 2 * sum(g) <= budget]
        geometries[i] = options[int(rng.integers(len(options)))]
    available = [c for c in range(1, 10) if c not in (background, structure)]
    modules = []
    for i, (span, geometry) in enumerate(zip(spans, geometries)):
        period = (2 if index == 0 else 3) if split == "train" and index < 2 else int(
            rng.integers(2, 4) if split == "train" else rng.integers(4, 6))
        cargo_length = min(period - 1, int(rng.integers(1, 3 if split == "train" else 4)))
        if i == 0:
            cargo_length = (1 if index == 0 else 2) if split == "train" and index < 2 else cargo_length
            if split == "test":
                cargo_length = 3
        gap_kind = "any"
        if split == "train" and index < 2:
            gap_kind = ("gap" if index == 0 else "edge") if i == 0 else "contact"
            if i > 0:
                cargo_length = 1
        colors = tuple(map(int, rng.choice(available, size=2, replace=False)))
        a, b = _module(rng, span, period, cargo_length, gap_kind, structure, colors, geometry)
        if rng.integers(2):
            a, b = a[::-1], b[::-1]
        modules.append((a, b))
    depth = max(a.shape[0] for a, _ in modules) + 2
    depth = int(rng.integers(depth, 31))
    cross = sum(a.shape[1] for a, _ in modules) + band_gap * (count - 1) + 5
    cross = int(rng.integers(cross, 31))
    source = np.full((depth, cross), background, dtype=np.int8)
    target = source.copy()
    q = 1
    for a, b in modules:
        k = int(rng.integers(1, depth - a.shape[0]))
        region = np.s_[k:k+a.shape[0], q:q+a.shape[1]]
        source[region] = np.where(a == -1, background, a)
        target[region] = np.where(b == -1, background, b)
        q += a.shape[1] + band_gap
    if axis == 1:
        source, target = source.T.copy(), target.T.copy()
    if rng.integers(2):
        source, target = source[:, ::-1].copy(), target[:, ::-1].copy()
    analysis = _analyze(source)
    assert np.array_equal(target, _render(analysis))
    # Distractors must neither touch an existing component nor obstruct a route.
    blocked = np.zeros_like(source, dtype=bool)
    occupied = {p for obj in analysis["forks"] + analysis["tokens"] for p in obj["cells"]}
    occupied |= {p for m in analysis["matches"] for p in m["route"]}
    for r, c in occupied:
        for dr, dc in ((0, 0),) + DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < source.shape[0] and 0 <= cc < source.shape[1]:
                blocked[rr, cc] = True
    blocked[0] = blocked[-1] = True
    blocked[:, 0] = blocked[:, -1] = True
    for _ in range(int(rng.integers(2, 5))):
        color = int(rng.choice(available))
        sizes = [(int(rng.integers(1, 4)), int(rng.integers(1, 4))), (1, 1)]
        for h, w in sizes:
            options = []
            for r in range(1, source.shape[0] - h):
                for c in range(1, source.shape[1] - w):
                    if np.any(blocked[r:r+h, c:c+w]):
                        continue
                    cells = {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)}
                    token = dict(cells=cells, color=color)
                    if not any(_candidate(f, token, source, background) for f in analysis["forks"]):
                        options.append(cells)
            if options:
                cells = options[int(rng.integers(len(options)))]
                break
        else:
            raise RuntimeError("Reserved distractor space exhausted")
        _paint(source, cells, color)
        for r, c in cells:
            for dr, dc in ((0, 0),) + DIRS:
                blocked[r+dr, c+dc] = True
    assert np.array_equal(solve(source), target)
    return {"input": source, "output": target}


def _count(name, value, low, high):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    role_pairs = list(product(range(1, 10), repeat=2))
    role_pairs = [p for p in role_pairs if p[0] != p[1]]
    rng.shuffle(role_pairs)
    first_axis, first_count = int(rng.integers(2)), int(rng.integers(1, 3))
    puzzle = {"train": [], "test": []}
    serial = 0
    for split, n in (("train", num_train), ("test", num_test)):
        for i in range(n):
            axis = (first_axis ^ i) if split == "train" and i < 2 else int(rng.integers(2))
            count = (first_count if i == 0 else 3-first_count) if split == "train" and i < 2 else (
                int(rng.integers(1, 3)) if split == "train" else 3)
            bg, structure = role_pairs[serial]
            serial += 1
            # Local failures are spatial/template constraints, with a finite retry.
            for _ in range(40):
                try:
                    pair = _pair(rng, axis, count, split, i, bg, structure)
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("Bounded pair construction exhausted")
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_analyses = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            info = _analyze(a)
            assert np.array_equal(_render(info), b)
            assert not np.array_equal(a, b)
            assert len(info["matches"]) == len(info["forks"])
            assert len(info["tokens"]) - len(info["used"]) >= 2
            assert set(map(int, np.unique(b))) - set(map(int, np.unique(a))) == {TRAIL}
            for f in info["forks"]:
                assert min(run[2] for run in f["runs"]) >= 1
                assert f["errors"] >= 1 and f["length"] >= 4 * f["period"]
                assert all(int(b[p]) == f["structural_color"] for p in f["structure"])
                for q, phase in product(range(f["span"]), range(f["period"])):
                    assert np.count_nonzero(f["sequence"][q, phase::f["period"]]
                                            != f["repaired"][q, phase::f["period"]]) <= 1
            for m in info["matches"]:
                f, t = info["forks"][m["fi"]], info["tokens"][m["ti"]]
                assert m["distance"] >= 1
                assert _code_fits(f["unit"], m["final_start"], m["token_length"], t["color"])
                latest = np.array([[f["sequence"][q, list(range(phase, f["length"], f["period"]))[-1]]
                                    for phase in range(f["period"])] for q in range(f["span"])])
                assert not _code_fits(latest, m["final_start"], m["token_length"], t["color"]), (
                    "Copying the last repetition should misread the final phase")
                assert not any(_code_fits(f["unit"], k, m["token_length"], t["color"])
                               for k in range(f["length"] + 1, m["final_start"]))
                assert all(int(b[p]) == t["color"] for p in m["final"])
                assert all(int(b[p]) == TRAIL for p in m["route"] - m["final"])
                assert all(int(b[_cell(f, k, q)]) == info["background"]
                           for k in range(f["length"] + 1, m["final_start"])
                           for q in range(f["q0"], f["q0"] + f["span"]))
            for ti, t in enumerate(info["tokens"]):
                if ti not in info["used"]:
                    assert all(int(b[p]) == info["background"] for p in t["cells"])
            assert not np.array_equal(_render(info, move=False), b), "Omitting B has no effect"
            if split == "train":
                assert 1 <= len(info["forks"]) <= 2
                assert all(f["period"] in (2, 3) for f in info["forks"])
                train_analyses.append(info)
            else:
                assert len(info["forks"]) == 3
                assert all(f["period"] in (4, 5) for f in info["forks"])
                assert any(m["token_length"] == 3 for m in info["matches"])
    assert {len(d["forks"]) for d in train_analyses} == {1, 2}
    assert {f["period"] for d in train_analyses for f in d["forks"]} == {2, 3}
    assert {int(bool(f["out"][0])) for d in train_analyses for f in d["forks"]} == {0, 1}
    assert any(m["phase_gap"] == 0 for d in train_analyses for m in d["matches"])
    assert any(m["phase_gap"] > 0 for d in train_analyses for m in d["matches"])
    assert any(m["token_length"] > 1 and m["phase_gap"] > 0
               and _code_fits(d["forks"][m["fi"]]["unit"],
                             d["forks"][m["fi"]]["length"] + 1, 1,
                             d["tokens"][m["ti"]]["color"])
               for d in train_analyses for m in d["matches"]), "Leading-slice shortcut not exposed"
    assert any(not np.array_equal(_render(d, phase=False), _render(d)) for d in train_analyses), (
        "A must affect docking, not only ribbon recoloring")


def _puzzle_key(puzzle):
    return tuple((s, tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                           np.asarray(p["output"]).tobytes()) for p in puzzle[s]))
                 for s in ("train", "test"))


def self_test():
    shapes, structures, palettes, periods, spans, lengths, gaps, directions = (set() for _ in range(8))
    puzzles, geometries, partial = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        for pairs in puzzle.values():
            for pair in pairs:
                info = _analyze(pair["input"])
                a = info["grid"]
                shapes.add(a.shape)
                roles = {}
                structures.add(tuple(roles.setdefault(int(x), len(roles)) for x in a.ravel()))
                palettes.add(tuple(sorted(map(int, np.unique(a)))))
                for f in info["forks"]:
                    periods.add(f["period"])
                    spans.add(f["span"])
                    directions.add(f["out"])
                    partial += f["length"] % f["period"] != 0
                    geometries.add((f["width"], f["runs"], f["period"], f["length"]))
                for m in info["matches"]:
                    lengths.add(m["token_length"])
                    gaps.add(m["phase_gap"])
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 25 and len(palettes) >= 20 and len(geometries) >= 150
    assert periods == {2, 3, 4, 5} and spans == {1, 2, 3}
    assert lengths == {1, 2, 3} and len(gaps) >= 3 and directions == set(DIRS)
    assert partial >= 100
    assert _puzzle_key(generate_puzzle(781, 2, 1)) == _puzzle_key(generate_puzzle(781, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1) + 1):
            generate_puzzle(2200 + 10 * nt + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Output corruption must fail semantic validation.
    puzzle = generate_puzzle(13, 2, 1)
    pair = puzzle["train"][0]
    info = _analyze(pair["input"])
    pair["output"][min(info["matches"][0]["final"])] = TRAIL
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("A destroyed final cargo footprint was accepted")
    try:
        _recover(np.array([[1, 2, 1, 2, 2, 1, 2, 1]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("A phase-majority tie was accepted")
    # No hidden seed state: rotate and recolor a supported input; the black
    # trail remains fixed and every other semantic color can be permuted.
    puzzle = generate_puzzle(29, 2, 1)
    perm = np.array([0, 7, 4, 9, 2, 8, 1, 6, 3, 5], dtype=np.int8)
    for pair in puzzle["train"] + puzzle["test"]:
        for k in range(4):
            assert np.array_equal(solve(np.rot90(perm[pair["input"]], k)),
                                  np.rot90(perm[pair["output"]], k))


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, phase/geometry variation and all legal counts")
