"""Paired-pixel portals: noisy bilateral figures redirect diagonal L-rays.

Only NumPy and the standard library are used. No parent code or saved example
is loaded. Axis inference, portal entry, and ray tracing use the query alone.
"""
from collections import Counter, defaultdict

import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size / 2:
        raise ValueError("background must occupy a strict majority")
    return bg


def _cells(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _source(cells):
    if len(cells) != 3:
        return None
    rows, cols = zip(*cells)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    missing = next(iter({(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - cells))
    elbow = (r0 + r1 - missing[0], c0 + c1 - missing[1])
    return elbow, (elbow[0] - missing[0], elbow[1] - missing[1])


def _reflect(p, orientation, axis2):
    return (p[0], axis2 - p[1]) if orientation == "v" else (axis2 - p[0], p[1])


def _turn(direction, orientation):
    dr, dc = direction
    return (dr, -dc) if orientation == "v" else (-dr, dc)


def _best_axis(cells):
    """Count supported cells, including self-support, for both axis directions."""
    rows, cols = defaultdict(list), defaultdict(list)
    for r, c in sorted(cells):
        rows[r].append(c)
        cols[c].append(r)
    scores = Counter()
    for orientation, lines in (("v", rows), ("h", cols)):
        for values in lines.values():
            for a in values:
                for b in values:
                    scores[orientation, a + b] += 1
    maximum = max(scores.values())
    winners = [axis for axis, score in scores.items() if score == maximum]
    if len(winners) != 1:
        raise ValueError("gate has tied strongest reflection axes")
    orientation, axis2 = winners[0]
    keep = {p for p in cells if _reflect(p, orientation, axis2) in cells}
    if len(keep) < 6 or not any(_reflect(p, orientation, axis2) != p for p in keep):
        raise ValueError("gate needs at least six supported cells and an off-axis pair")
    return orientation, axis2, keep


def _parse(grid):
    bg = _background(grid)
    sources, gates = {}, {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        cells = _cells(grid, color)
        launch = _source(cells)
        if launch is not None:
            sources[color] = {"cells": cells, "elbow": launch[0], "direction": launch[1]}
        else:
            orientation, axis2, keep = _best_axis(cells)
            gates[color] = {"cells": cells, "orientation": orientation,
                            "axis2": axis2, "keep": keep, "noise": cells - keep}
    if not sources or not gates:
        raise ValueError("a portal scene requires sources and gates")
    return bg, sources, gates


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _ray_trace(shape, sources, gates):
    """Trace against supported gate cells only; noise has no optical effect."""
    source_cells = set().union(*(s["cells"] for s in sources.values()))
    ports = {p: color for color, gate in gates.items() for p in gate["keep"]}
    strokes, routes, used = {}, {}, set()
    for source_color, source in sources.items():
        direction, color = source["direction"], source_color
        anchor = source["elbow"]
        p = (anchor[0] + direction[0], anchor[1] + direction[1])
        visited, path, events = set(), [], []
        while _inside(p, shape):
            state = p, direction
            if state in visited:
                raise ValueError("ray entered a loop")
            visited.add(state)
            if p in source_cells:
                raise ValueError("ray strikes a source")
            if p in ports:
                gate_color = ports[p]
                gate = gates[gate_color]
                q = _reflect(p, gate["orientation"], gate["axis2"])
                events.append((gate_color, p, q, direction, color))
                used.add(gate_color)
                direction, color, anchor = _turn(direction, gate["orientation"]), gate_color, q
                p = (q[0] + direction[0], q[1] + direction[1])
            else:
                if p in strokes and strokes[p] != color:
                    raise ValueError("rays demand incompatible colors at a shared cell")
                strokes[p] = color
                path.append((p, color))
                p = (p[0] + direction[0], p[1] + direction[1])
        routes[source_color] = {"path": path, "events": events,
                                "tail_anchor": anchor, "tail_direction": direction}
    return strokes, routes, used


def _solve_details(grid):
    bg, sources, gates = _parse(grid)
    strokes, routes, used = _ray_trace(grid.shape, sources, gates)
    if used != set(gates):
        raise ValueError("every gate must participate in a ray route")
    output = np.full_like(grid, bg)
    for color, source in sources.items():
        for p in source["cells"]:
            output[p] = color
    for color, gate in gates.items():
        for p in gate["keep"]:
            output[p] = color
    for p, color in strokes.items():
        output[p] = color
    return output, (bg, sources, gates, strokes, routes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from the input, without training or a seed."""
    return _solve_details(_grid(input_grid))[0]


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {elbow, (r - dr, c), (r, c - dc)}


def _straight_tail(anchor, direction, shape):
    cells = []
    p = (anchor[0] + direction[0], anchor[1] + direction[1])
    while _inside(p, shape):
        cells.append(p)
        p = (p[0] + direction[0], p[1] + direction[1])
    return cells


def _core(rng, p, q, orientation, forbidden, shape):
    """Sample a silhouette in the gate's own coordinates, keeping both ports."""
    normal = 1 if orientation == "v" else 0
    along = 1 - normal
    lo, hi = sorted((p[normal], q[normal]))
    span = hi - lo
    axis2 = lo + hi
    center = p[along]
    before, after = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    start, end = max(0, center - before), min(shape[along] - 1, center + after)
    family = int(rng.integers(5))
    core = {p, q}
    for t in range(start, end + 1):
        # Contours, ladders, changing lobes, sparse islands, and thin stems.
        distances = list(range(span // 2 + 1))
        if family == 0:
            choices = distances if t in (start, end) else [0]
        elif family == 1:
            choices = [0] + ([int(rng.choice(distances))] if rng.random() < .5 else [])
        elif family == 2:
            level = int(rng.integers(len(distances)))
            choices = list(range(level, len(distances)))
        elif family == 3:
            choices = [d for d in distances if rng.random() < .35]
        else:
            choices = [distances[-1]] + ([0] if rng.random() < .3 else [])
        if rng.random() < .15 and t != center:
            choices = []
        for d in choices:
            a = (t, lo + d) if orientation == "v" else (lo + d, t)
            b = _reflect(a, orientation, axis2)
            if a not in forbidden and b not in forbidden:
                core.update((a, b))
    try:
        o, a, supported = _best_axis(core)
    except ValueError:
        return None
    return core if (o, a) == (orientation, axis2) and supported == core else None


def _noise(rng, inp, core, orientation, axis2, color, path, required):
    bg = _background(inp)
    rows, cols = zip(*core)
    local = [(r, c) for r in range(max(0, min(rows) - 2), min(inp.shape[0], max(rows) + 3))
             for c in range(max(0, min(cols) - 2), min(inp.shape[1], max(cols) + 3))]
    rng.shuffle(local)
    selected = set()

    def add(p):
        if (inp[p] != bg or p in core or p in selected
                or _reflect(p, orientation, axis2) in core | selected):
            return False
        try:
            o, a, supported = _best_axis(core | selected | {p})
        except ValueError:
            return False
        if (o, a) != (orientation, axis2) or supported != core:
            return False
        selected.add(p)
        return True

    # Teaching noise includes a transparent false contact on the incoming ray
    # and an off-ray outlier that is visibly erased.
    on_path = [p for p in path if p in local]
    rng.shuffle(on_path)
    if required:
        if not any(add(p) for p in on_path):
            return None
        if not any(add(p) for p in local if p not in path):
            return None
    wanted = int(rng.integers(2, 7)) if required else (0 if rng.random() < .18 else int(rng.integers(1, 8)))
    for p in local:
        if len(selected) >= wanted:
            break
        add(p)
    return selected


def _extend_gate(inp, rng, primary, gate_color, orientation, parity, required_noise):
    bg = _background(inp)
    # Before the first gate, the scene is deliberately just one L source.
    if set(map(int, np.unique(inp))) == {bg, primary}:
        launch = _source(_cells(inp, primary))
        anchor, direction = launch
        old_events, old_strokes = [], {}
        for p in _straight_tail(anchor, direction, inp.shape):
            old_strokes[p] = primary
    else:
        _, details = _solve_details(inp)
        old_strokes, route = details[3], details[4][primary]
        anchor, direction = route["tail_anchor"], route["tail_direction"]
        old_events = route["events"]
    tail = _straight_tail(anchor, direction, inp.shape)
    options = [(i, jump, sign) for i in range(2, len(tail))
               for jump in range(3, 9) if jump % 2 == parity
               for sign in (-1, 1)]
    rng.shuffle(options)
    occupied = _cells(inp, bg)
    occupied = {(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1])} - occupied
    normal = 1 if orientation == "v" else 0
    for i, jump, sign in options[:150]:
        p = tail[i]
        if inp[p] != bg:
            continue
        q = list(p)
        q[normal] += sign * jump * direction[normal]
        q = tuple(q)
        turned = _turn(direction, orientation)
        if not _inside(q, inp.shape) or q in occupied or q in old_strokes:
            continue
        outgoing = _straight_tail(q, turned, inp.shape)
        if len(outgoing) < 3:
            continue
        forbidden = occupied | (set(old_strokes) - {p}) | set(outgoing)
        axis2 = p[normal] + q[normal]
        for _ in range(5):
            core = _core(rng, p, q, orientation, forbidden, inp.shape)
            if core is None:
                continue
            additions = _noise(rng, inp, core, orientation, axis2, gate_color, tail[:i], required_noise)
            if additions is None:
                continue
            trial = inp.copy()
            for cell in core | additions:
                trial[cell] = gate_color
            try:
                _, details = _solve_details(trial)
            except ValueError:
                continue
            new = details[4][primary]["events"]
            if (len(new) == len(old_events) + 1 and new[:-1] == old_events
                    and new[-1][:3] == (gate_color, p, q)
                    and details[2][gate_color]["keep"] == core):
                return trial
    return None


def _add_sources(inp, rng, desired, colors):
    grid = inp.copy()
    for color in colors[:desired - 1]:
        bg = _background(grid)
        choices = [(r, c, d) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                   for d in _DIRECTIONS]
        rng.shuffle(choices)
        for r, c, d in choices[:750]:
            cells = _source_cells((r, c), d)
            if any(not _inside(p, grid.shape) or grid[p] != bg for p in cells):
                continue
            trial = grid.copy()
            for p in cells:
                trial[p] = color
            try:
                _, details = _solve_details(trial)
            except ValueError:
                continue
            if details[4][color]["events"] and len(details[4][color]["path"]) >= 3:
                grid = trial
                break
        else:
            break
    return grid


def _fallback(rng, orientation, parity, mixed, palette, direction=None):
    """Explicit clear construction after bounded retries; never a new rule."""
    bg, primary, first, second = palette[:4]
    inp = np.full((28, 28), bg, dtype=np.int8)
    for p in _source_cells((2, 12), (1, 1)):
        inp[p] = primary
    jump = 6 if parity == 0 else 5
    qcol = 17 - jump
    core1 = {(7, 17), (7, qcol), (6, qcol + 2), (6, 15), (8, qcol + 1), (8, 16)}
    if jump == 6:
        core1.add((9, 14))
    for p in core1 | {(5, 15), (10, 18)}:
        inp[p] = first
    if mixed:
        c = qcol - 6
        jump2 = int(rng.choice((5, 6)))
        qr = 13 - jump2
        core2 = {(13, c), (qr, c), (qr + 2, c + 1), (11, c + 1),
                 (qr + 1, c + 2), (12, c + 2)}
        if jump2 == 6:
            core2.add((10, c + 4))
        for p in core2 | {(11, c + 2), (14, c + 4)}:
            inp[p] = second
    if orientation == "h":
        inp = inp.T.copy()
    flip_rows = direction[0] < 0 if direction is not None else bool(rng.integers(2))
    flip_cols = direction[1] < 0 if direction is not None else bool(rng.integers(2))
    if flip_rows:
        inp = inp[::-1].copy()
    if flip_cols:
        inp = inp[:, ::-1].copy()
    _solve_details(inp)
    return inp


def _build_pair(rng, orientation, parity, mixed, target_gates, desired_sources,
                direction=None, teaching_colors=None):
    palette = list(map(int, rng.permutation(10)))
    if teaching_colors is not None:
        primary, first_gate = teaching_colors
        others = [c for c in palette if c not in teaching_colors]
        palette = [others[0], primary, first_gate, *others[1:]]
    bg, primary = palette[:2]
    gate_colors = palette[2:2 + target_gates]
    for attempt in range(18):
        shape = tuple(map(int, rng.integers(12, 31, size=2)))
        inp = np.full(shape, bg, dtype=np.int8)
        d = direction if direction is not None else _DIRECTIONS[int(rng.integers(4))]
        candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                      if all(_inside(p, shape) for p in _source_cells((r, c), d))
                      and len(_straight_tail((r, c), d, shape)) >= 8]
        elbow = candidates[int(rng.integers(len(candidates)))]
        for p in _source_cells(elbow, d):
            inp[p] = primary
        count = target_gates if attempt < 10 else (2 if mixed else 1)
        success = True
        for i in range(count):
            o = orientation if not mixed or i % 2 == 0 else ("h" if orientation == "v" else "v")
            p = parity if i == 0 else int(rng.integers(2))
            trial = _extend_gate(inp, rng, primary, gate_colors[i], o, p, required_noise=i == 0)
            if trial is None:
                success = False
                break
            inp = trial
        if success:
            break
    else:
        inp = _fallback(rng, orientation, parity, mixed, palette, direction)
        count = 2 if mixed else 1
    inp = _add_sources(inp, rng, desired_sources, palette[2 + target_gates:])
    return {"input": inp, "output": solve(inp)}


def _checked_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _checked_count(num_train, "num_train", 2, 10)
    ntest = _checked_count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    first_orientation = str(rng.choice(("v", "h")))
    first_parity = int(rng.integers(2))
    first_direction = _DIRECTIONS[int(rng.integers(4))]
    teaching_colors = list(map(int, rng.choice(10, size=3, replace=False)))
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            teacher = split == "train" and index < 2
            mixed = split == "test"
            if teacher:
                orientation = first_orientation if index == 0 else ("h" if first_orientation == "v" else "v")
                parity = first_parity if index == 0 else 1 - first_parity
                direction = first_direction if index == 0 else (first_direction[0], -first_direction[1])
                target_gates, desired_sources = 1, 1
                colors = (teaching_colors[0], teaching_colors[index + 1])
            else:
                orientation, parity, direction = str(rng.choice(("v", "h"))), int(rng.integers(2)), None
                target_gates = int(rng.choice((2, 3))) if mixed else int(rng.choice((1, 2, 3), p=(.4, .45, .15)))
                desired_sources = int(rng.choice((1, 2, 3), p=(.35, .45, .20)))
                colors = None
            for _ in range(12):
                pair = _build_pair(rng, orientation, parity, mixed, target_gates, desired_sources, direction, colors)
                evidence = _metrics(pair["input"])
                if (_key(pair["input"]) not in seen
                        and evidence["erased"] and evidence["ray_noise"]
                        and (not teacher or evidence["bbox_axis_wrong"])):
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(inp):
    out, details = _solve_details(inp)
    bg, sources, gates, strokes, routes = details
    noise = set().union(*(g["noise"] for g in gates.values()))
    return {"shape": inp.shape, "background": bg, "sources": len(sources), "gates": len(gates),
            "axes": {(g["orientation"], g["axis2"] % 2) for g in gates.values()},
            "directions": {s["direction"] for s in sources.values()},
            "counts": tuple(len(r["events"]) for r in routes.values()),
            "noise": len(noise), "ray_noise": len(noise & set(strokes)),
            "erased": sum(out[p] == bg for p in noise),
            "bbox_axis_wrong": any(min(p[1 if g["orientation"] == "v" else 0] for p in g["cells"])
                                   + max(p[1 if g["orientation"] == "v" else 0] for p in g["cells"])
                                   != g["axis2"] for g in gates.values()),
            "mixed_route": any(len({gates[e[0]]["orientation"] for e in r["events"]}) > 1 for r in routes.values()),
            "shared_gate": any(sum(any(e[0] == c for e in r["events"]) for r in routes.values()) > 1 for c in gates),
            "core_sizes": tuple(len(g["keep"]) for g in gates.values()),
            "jump_sizes": tuple(abs(e[1][0] - e[2][0]) + abs(e[1][1] - e[2][1])
                                for r in routes.values() for e in r["events"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, training_axes, directions = set(), set(), set()
    successors, biased_box = defaultdict(set), False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, details = _solve_details(inp)
            assert np.array_equal(predicted, out), "incorrect portal output"
            bg, sources, gates, strokes, routes = details
            # Check exact per-cell behavior, rather than only changes or hashes.
            for color, source in sources.items():
                assert all(out[p] == color for p in source["cells"])
            for color, gate in gates.items():
                assert all(out[p] == color and _reflect(p, gate["orientation"], gate["axis2"]) in gate["keep"]
                           for p in gate["keep"])
                assert all(out[p] == strokes.get(p, bg) for p in gate["noise"])
            assert all(out[p] == color for p, color in strokes.items())
            original_bg = (inp == bg)
            assert np.all((out[original_bg] == bg) | np.isin(out[original_bg], list(sources) + list(gates)))
            # Ignoring A leaves a deletion-only picture and necessarily differs.
            clean = inp.copy()
            for gate in gates.values():
                for p in gate["noise"]:
                    clean[p] = bg
            assert not np.array_equal(clean, out)
            # Ignoring B leaves off-ray outliers and necessarily differs.
            unfiltered = inp.copy()
            for p, color in strokes.items():
                unfiltered[p] = color
            assert not np.array_equal(unfiltered, out)
            m = _metrics(inp)
            assert m["ray_noise"] and m["erased"], "need transparent and erased outlier evidence"
            if split == "train":
                training_axes |= m["axes"]
                directions |= m["directions"]
                assert not m["mixed_route"], "training uses one portal orientation per scene"
                biased_box |= m["bbox_axis_wrong"]
                for route in routes.values():
                    for e in route["events"]:
                        successors[e[4]].add(e[0])
            else:
                assert m["mixed_route"], "test recombines both orientations on one ray"
    assert {o for o, _ in training_axes} == {"v", "h"}
    assert {p for _, p in training_axes} == {0, 1}
    assert len(directions) >= 2
    assert biased_box, "training must refute the noisy bounding-box axis"
    assert any(len(colors) > 1 for colors in successors.values()), "training must refute fixed successor colors"


def self_test():
    """Bounded semantic, mutation, count, determinism, and 42-seed coverage checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                assert np.array_equal(solve(inp.tolist()), pair["output"])
                roles = {}
                structures.add((inp.shape, bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                records.append(_metrics(inp))
    assert len(structures) >= 180
    assert len({r["shape"] for r in records}) >= 65
    assert {r["background"] for r in records} == set(range(10))
    assert {d for r in records for d in r["directions"]} == set(_DIRECTIONS)
    assert {axis for r in records for axis in r["axes"]} == {("v", 0), ("v", 1), ("h", 0), ("h", 1)}
    assert {r["gates"] for r in records} >= {1, 2, 3}
    assert {r["sources"] for r in records} >= {1, 2, 3}
    assert sum(r["shared_gate"] for r in records) >= 8
    assert len({n for r in records for n in r["core_sizes"]}) >= 15
    assert len({n for r in records for n in r["jump_sizes"]}) >= 5
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    again = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in again for a, b in zip(again[s], puzzles[0][s]) for k in a)
    # A wrong port continuation or erased supported pixel must be rejected.
    sample = puzzles[0]["test"][0]
    _, details = _solve_details(sample["input"])
    for p in (next(iter(details[3])), next(iter(next(iter(details[2].values()))["keep"]))):
        wrong = sample["output"].copy()
        wrong[p] = details[0]
        bad = {"train": puzzles[0]["train"], "test": [{"input": sample["input"], "output": wrong}]}
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect output accepted")
    # Ties receive no invented leftmost/integral/orientation preference.
    try:
        _best_axis({(r, c) for r in range(3) for c in range(3)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous axis accepted")
    for orientation in ("v", "h"):
        for parity in (0, 1):
            for mixed in (False, True):
                inp = _fallback(np.random.default_rng(9), orientation, parity, mixed, list(range(10)))
                m = _metrics(inp)
                assert m["mixed_route"] == mixed and (orientation, parity) in m["axes"]
    _rule_regressions()


def _rule_regressions():
    # Expectations below are enumerated directly, rather than painted by the
    # generator or inferred by its tracer. The gap between paired ports stays
    # blank, noise on the ray is recolored, and an axis-only island survives.
    for half in (False, True):
        inp = np.zeros((12, 14), dtype=np.int8)
        c = 5 if half else 4
        source = {(0, c), (1, c - 1), (1, c)}
        core = {(4, c + 3), (4, 5), (3, 4), (3, 8 + int(half)),
                (5, 3), (5, 9 + int(half)), (6, 6)}
        if half:
            core.add((6, 7))
        for p in source:
            inp[p] = 1
        for p in core | {(2, c + 1), (8, 8)}:
            inp[p] = 2
        expected = np.zeros_like(inp)
        for p in source:
            expected[p] = 1
        for p in core:
            expected[p] = 2
        expected[2, c + 1] = expected[3, c + 2] = 1
        for p in ((5, 4), (6, 3), (7, 2), (8, 1), (9, 0)):
            expected[p] = 2
        for transpose in (False, True):
            a, b = (inp.T, expected.T) if transpose else (inp, expected)
            for flip in (False, True):
                x, y = (a[::-1, ::-1], b[::-1, ::-1]) if flip else (a, b)
                assert np.array_equal(solve(x), y)
    # A supported pixel on the axis is a self-portal and reverses the normal
    # component in place; self-support is never mistaken for additive noise.
    inp = np.zeros((12, 14), dtype=np.int8)
    source = {(1, 2), (2, 1), (2, 2)}
    core = {(4, 7), (4, 5), (3, 4), (3, 8), (5, 3), (5, 9), (6, 6)}
    for p in source:
        inp[p] = 1
    for p in core:
        inp[p] = 2
    expected = inp.copy()
    for p in ((3, 3), (4, 4), (5, 5)):
        expected[p] = 1
    for p in ((7, 5), (8, 4), (9, 3), (10, 2), (11, 1)):
        expected[p] = 2
    assert np.array_equal(solve(inp), expected)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus legal counts and regressions)")
