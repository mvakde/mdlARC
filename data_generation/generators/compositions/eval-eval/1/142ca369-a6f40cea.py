"""Occluded mirror circuits: light selects which hidden loops are revealed.

Only NumPy and the standard library are used. Nothing is written on import,
generation, validation, or self-test. Scene descriptors are construction aids;
solve reconstructs all geometry and colors afresh from its one input.
"""

from collections import Counter
import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0,448864963,305792913,410990811,457543304,879904097,231834961,
              846593766,758414585,701088969,859028377,205237654,480823156,
              552964341,680361208,721069055,107444490,384494942,667052884,
              547039107,159554178,298016717,228562159,878004728,194077574,
              239609283,912496787,366962909,695103044,93782220,249479154,
              892848691,231806791,689474943,918849824,192347166,100118227,
              845114581,804600438,712700646,220426549,420685081)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    return int(value)


def _counts(rng, num_train, num_test):
    # Explicit counts follow the public signature. None is also supported for
    # the supplied inspection helper, which explicitly passes it for defaults.
    if num_train is not None:
        num_train = _integer(num_train, "num_train")
        _require(2 <= num_train <= 10, "num_train must be 2..10")
    if num_test is not None:
        num_test = _integer(num_test, "num_test")
        _require(1 <= num_test <= 4, "num_test must be 1..4")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    _require(num_train > num_test, "num_train must exceed num_test")
    return num_train, num_test


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} | {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} | {(y, r) for y in range(t + 1, b)})


def _in_box(p, box, strict=False):
    t, b, l, r = box
    if strict:
        return t < p[0] < b and l < p[1] < r
    return t <= p[0] <= b and l <= p[1] <= r


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _halo(points):
    return {q for y, x in points for q in ((y, x), (y-1, x), (y+1, x), (y, x-1), (y, x+1))}


def _components(points):
    todo, result = set(points), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _source(elbow, direction, color):
    y, x = elbow
    dy, dx = direction
    return {"elbow": elbow, "direction": direction, "color": int(color),
            "cells": {elbow, (y-dy, x), (y, x-dx)}}


def _contacts(box):
    t, b, l, r = box
    cy, cx = (t+b)//2, (l+r)//2
    # (contact cell, reversed component, incoming normal sign, wall cell)
    return (((t-1, cx), 0, 1, (t, cx)), ((b+1, cx), 0, -1, (b, cx)),
            ((cy, l-1), 1, 1, (cy, l)), ((cy, r+1), 1, -1, (cy, r)))


def _trace(shape, sources, loops, *, visible_only=None, reflect=True, recolor=True):
    """Trace in the complete world; the window is an observation mask only."""
    h, w = shape
    occupied = {p for loop in loops for p in loop["pixels"]
                if visible_only is None or not _in_box(p, visible_only)}
    occupied.update(p for s in sources for p in s["cells"])
    events = {}
    if reflect:
        for i, loop in enumerate(loops):
            for p, axis, sign, wall in _contacts(loop["box"]):
                if visible_only is None or not _in_box(wall, visible_only):
                    events.setdefault(p, []).append((axis, sign, i, wall))
    painted, routes, activated = {}, [], set()
    for source in sources:
        _require(not any(_in_box(source["elbow"], loop["box"]) for loop in loops),
                 "source lies inside a mirror loop")
        p, direction, shade = source["elbow"], source["direction"], source["color"]
        visited, path, hits = set(), [], []
        while True:
            p = (p[0]+direction[0], p[1]+direction[1])
            if not (0 <= p[0] < h and 0 <= p[1] < w):
                break
            state = (p, direction)
            _require(state not in visited, "ray loop")
            visited.add(state)
            _require(p not in occupied, "ray strikes a noncontact object cell")
            candidates = [v for v in events.get(p, ()) if direction[v[0]] == v[1]]
            _require(len(candidates) <= 1, "ambiguous mirror contact")
            if candidates:
                axis, sign, index, wall = candidates[0]
                incoming = shade
                if recolor:
                    shade = loops[index]["pixels"][wall]
                direction = (-direction[0], direction[1]) if axis == 0 else (direction[0], -direction[1])
                hits.append((index, axis, p, wall, incoming, shade))
                activated.add(index)
            _require(p not in painted or painted[p] == shade, "conflicting ray colors")
            painted[p] = shade
            path.append((p, shade))
        routes.append({"path": path, "hits": hits})
    return painted, routes, activated


def _periodic_completion(grid, box, visible):
    full = sorted(_perimeter(box))
    predictions, fitting = set(), []
    for sign in (1, -1):
        for period in range(1, 5):
            residues, counts = {}, Counter()
            for y, x in visible:
                phase = (y + sign*x) % period
                color = int(grid[y, x])
                if phase in residues and residues[phase] != color:
                    break
                residues[phase] = color
                counts[phase] += 1
            else:
                if len(residues) == period and min(counts.values()) >= 2:
                    predictions.add(tuple(residues[(y+sign*x) % period] for y, x in full))
                    fitting.append((period, sign))
    _require(len(predictions) == 1, "ambiguous or unwitnessed loop color pattern")
    return dict(zip(full, next(iter(predictions)))), min(p for p, _ in fitting)


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "unclear background")
    frames = []
    for color in np.unique(grid):
        if color == bg:
            continue
        for group in _components(map(tuple, np.argwhere(grid == color))):
            box = _bbox(group)
            t, b, l, r = box
            if b-t < 5 or r-l < 5 or group != _perimeter(box):
                continue
            interior = grid[t+1:b, l+1:r]
            if np.all(interior == interior[0, 0]) and int(interior[0, 0]) != color:
                frames.append(((b-t-1)*(r-l-1), box, int(interior[0, 0]), int(color)))
    frames.sort(reverse=True)
    _require(bool(frames) and (len(frames) == 1 or frames[0][0] > frames[1][0]), "no unique largest window")
    _, window, fill, frame = frames[0]
    outside = {tuple(map(int, p)) for p in np.argwhere(grid != bg) if not _in_box(p, window)}
    sources, fragments = [], []
    for group in _components(outside):
        box = _bbox(group)
        t, b, l, r = box
        colors = {int(grid[p]) for p in group}
        if len(group) == 3 and b-t == r-l == 1 and len(colors) == 1:
            missing = next(iter({(y, x) for y in (t, b) for x in (l, r)} - group))
            elbow = (t+b-missing[0], l+r-missing[1])
            sources.append(_source(elbow, (elbow[0]-missing[0], elbow[1]-missing[1]), next(iter(colors))))
        else:
            _require(len(group) >= 5, "too little exposed fragment")
            fragments.append(group)
    _require(bool(sources), "no visible L source")
    candidates = []
    for i, first in enumerate(fragments):
        for j in range(i, len(fragments)):
            visible = first | fragments[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b-t < 4 or r-l < 4 or (b-t) % 2 or (r-l) % 2:
                continue
            perimeter = _perimeter(box)
            if {p for p in perimeter if not _in_box(p, window)} != visible:
                continue
            try:
                pixels, period = _periodic_completion(grid, box, visible)
            except ValueError:
                continue
            candidates.append((frozenset((i, j)), {"box": box, "pixels": pixels, "period": period}))
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for parts, loop in candidates:
            cells = set(loop["pixels"])
            if first in parts and parts <= remaining and not (cells & _halo(occupied)):
                cover(remaining-parts, chosen+[loop], occupied | cells)

    cover(set(range(len(fragments))), [], set())
    _require(len(covers) == 1 and bool(covers[0]), "loop geometry is not uniquely reconstructible")
    loops = covers[0]
    all_loop_pixels = {p for loop in loops for p in loop["pixels"]}
    all_source_pixels = {p for s in sources for p in s["cells"]}
    _require(not (_halo(all_source_pixels) & all_loop_pixels), "objects touch")
    colors = {c for loop in loops for c in loop["pixels"].values()}
    _require(not (colors & {bg, frame, fill}), "loop and window roles share colors")
    _require(all(s["color"] not in colors | {bg, frame, fill} for s in sources), "source palette overlaps other roles")
    return grid, {"background": bg, "window": window, "fill": fill, "frame": frame,
                  "sources": sources, "loops": loops}


def _render_answer(scene, painted, active):
    t, b, l, r = scene["window"]
    output = np.full((b-t-1, r-l-1), scene["fill"], dtype=np.int8)
    for i in active:
        for p, color in scene["loops"][i]["pixels"].items():
            if _in_box(p, scene["window"], strict=True):
                output[p[0]-t-1, p[1]-l-1] = color
    for p, color in painted.items():
        if _in_box(p, scene["window"], strict=True):
            output[p[0]-t-1, p[1]-l-1] = color
    return output


def _analyze(input_grid):
    grid, scene = _read(input_grid)
    painted, routes, active = _trace(grid.shape, scene["sources"], scene["loops"])
    scene.update(painted=painted, routes=routes, active=active)
    return _render_answer(scene, painted, active), scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer latent loops, trace and activate them, then return the window crop."""
    return _analyze(input_grid)[0]


def _make_loop(box, rng, palette, period):
    word = [int(c) for c in rng.choice(palette, period, replace=False)]
    sign = int(rng.choice((-1, 1)))
    phase = int(rng.integers(period))
    return {"box": box, "period": period,
            "pixels": {p: word[(p[0]+sign*p[1]+phase) % period] for p in sorted(_perimeter(box))}}


def _fits_objects(box, shape, sources, loops):
    t, b, l, r = box
    if not (1 <= t < b < shape[0]-1 and 1 <= l < r < shape[1]-1):
        return False
    cells = _perimeter(box)
    old = {p for loop in loops for p in loop["pixels"]} | {p for s in sources for p in s["cells"]}
    return not (cells & _halo(old)) and not any(_in_box(p, box) for s in sources for p in s["cells"])


def _append_loop(shape, sources, loops, rng, palette, period, axis):
    _, routes, _ = _trace(shape, sources, loops)
    route, source = routes[0], sources[0]
    direction = source["direction"]
    for _, a, _, _, _, _ in route["hits"]:
        direction = (-direction[0], direction[1]) if a == 0 else (direction[0], -direction[1])
    anchor = route["hits"][-1][2] if route["hits"] else source["elbow"]
    options = [(d, a, hh, ww) for d in range(2, max(shape))
               for a in ((axis,) if axis is not None else (0, 1))
               for hh in (5, 7, 9, 11) for ww in (5, 7, 9, 11)]
    rng.shuffle(options)
    old_hits = [(v[0], v[1], v[2]) for v in route["hits"]]
    for distance, a, hh, ww in options[:600]:
        y, x = anchor[0]+distance*direction[0], anchor[1]+distance*direction[1]
        if a == 0:
            top = y+1 if direction[0] == 1 else y-hh
            box = (top, top+hh-1, x-ww//2, x+ww//2)
        else:
            left = x+1 if direction[1] == 1 else x-ww
            box = (y-hh//2, y+hh//2, left, left+ww-1)
        if not _fits_objects(box, shape, sources, loops):
            continue
        loop = _make_loop(box, rng, palette, period)
        try:
            _, new_routes, _ = _trace(shape, sources, loops+[loop])
        except ValueError:
            continue
        hits = [(v[0], v[1], v[2]) for v in new_routes[0]["hits"]]
        if hits == old_hits + [(len(loops), a, (y, x))] and len(new_routes[0]["path"]) >= 5:
            return loops+[loop]
    return None


def _extra_source(shape, sources, loops, rng, palette, window, hidden_target):
    original_paint, _, original_active = _trace(shape, sources, loops)
    scene = {"window": window, "fill": -1, "loops": loops}
    original_answer = _render_answer(scene, original_paint, original_active)
    proposals = []
    for loop in loops:
        for p, axis, sign, wall in _contacts(loop["box"]):
            for direction in DIRECTIONS:
                if direction[axis] == sign:
                    for distance in range(2, 11):
                        elbow = (p[0]-distance*direction[0], p[1]-distance*direction[1])
                        proposals.append((elbow, direction))
    rng.shuffle(proposals)
    occupied = {p for loop in loops for p in loop["pixels"]} | {p for s in sources for p in s["cells"]}
    for elbow, direction in proposals[:300]:
        source = _source(elbow, direction, int(rng.choice(palette)))
        if not all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in source["cells"]):
            continue
        if any(_in_box(p, window) for p in source["cells"]):
            continue
        if _halo(source["cells"]) & occupied:
            continue
        try:
            paint, routes, active = _trace(shape, sources+[source], loops)
        except ValueError:
            continue
        hidden_points = {h[2] for route in routes for h in route["hits"] if _in_box(h[2], window, True)}
        if hidden_target == 1 and len(hidden_points) != 1:
            continue
        if active != original_active:
            continue
        answer = _render_answer(scene, paint, active)
        if np.array_equal(answer, original_answer):
            continue
        # The old source must also remain visible in the combined answer.
        try:
            solo_paint, _, solo_active = _trace(shape, [source], loops)
        except ValueError:
            continue
        if np.array_equal(answer, _render_answer(scene, solo_paint, solo_active)):
            continue
        if routes[-1]["hits"]:
            return sources+[source]
    return None


def _add_dormant(shape, sources, loops, rng, palette, period):
    before, routes, used = _trace(shape, sources, loops)
    for _ in range(400):
        hh, ww = int(rng.choice((5, 7, 9))), int(rng.choice((5, 7, 9)))
        t, l = int(rng.integers(1, shape[0]-hh)), int(rng.integers(1, shape[1]-ww))
        box = (t, t+hh-1, l, l+ww-1)
        if not _fits_objects(box, shape, sources, loops):
            continue
        loop = _make_loop(box, rng, palette, period)
        try:
            after, new_routes, active = _trace(shape, sources, loops+[loop])
        except ValueError:
            continue
        if after == before and active == used:
            return loops+[loop]
    return None


def _choose_window(shape, sources, loops, rng, hidden_target, require_dormant):
    painted, routes, used = _trace(shape, sources, loops)
    hit_points = {v[2] for route in routes for v in route["hits"]}
    source_cells = {p for s in sources for p in s["cells"]}
    for _ in range(900):
        hh, ww = int(rng.integers(8, min(shape[0]-2, 21))), int(rng.integers(8, min(shape[1]-2, 21)))
        t, l = int(rng.integers(1, shape[0]-hh)), int(rng.integers(1, shape[1]-ww))
        window = (t, t+hh-1, l, l+ww-1)
        if any(_in_box(p, window) for p in source_cells):
            continue
        hidden_hits = sum(_in_box(p, window, True) for p in hit_points)
        if (hidden_target == 1 and hidden_hits != 1) or (hidden_target == 2 and hidden_hits < 2):
            continue
        if hidden_target == 2 and max(sum(_in_box(v[2], window, True) for v in route["hits"])
                                      for route in routes) < 2:
            continue
        if sum(_in_box(p, window, True) for p in painted) < 4:
            continue
        active_hidden, dormant_hidden, valid = 0, 0, True
        for i, loop in enumerate(loops):
            cells = set(loop["pixels"])
            visible = {p for p in cells if not _in_box(p, window)}
            hidden = {p for p in cells if _in_box(p, window, True)}
            if not visible or _bbox(visible) != loop["box"] or any(len(g) < 5 for g in _components(visible)):
                valid = False
                break
            if len(visible) < 2*loop["period"]:
                valid = False
                break
            if hidden:
                if i in used:
                    active_hidden += len(hidden)
                else:
                    dormant_hidden += len(hidden)
        if valid and active_hidden >= 3 and (not require_dormant or dormant_hidden >= 3):
            return window
    return None


def _pair(rng, *, count, source_count, period, axis, direction, hidden_target,
          dormant, require_dormant):
    # All searches are bounded, and no rejected scene is retained or logged.
    for attempt in range(240):
        # A second source is a sampling preference in tests. After half the
        # budget, prefer a simpler valid circuit while preserving the required
        # two-hidden-contact chain. The two-source demonstration is mandatory.
        desired_sources = 1 if hidden_target == 2 and attempt >= 120 else source_count
        shape = (int(rng.integers(22, 31)), int(rng.integers(22, 31)))
        palette = [int(c) for c in rng.permutation(10)]
        bg, frame, fill, source_color = palette[:4]
        if rng.random() < .4:
            fill = bg
        mirror_palette, source_palette = palette[4:8], [source_color, palette[8]]
        elbow = (int(rng.integers(1, shape[0]-1)), int(rng.integers(1, shape[1]-1)))
        sources, loops = [_source(elbow, direction, source_color)], []
        for i in range(count):
            loops = _append_loop(shape, sources, loops, rng, mirror_palette, period,
                                 axis if i == 0 else None)
            if loops is None:
                break
        if loops is None:
            continue
        for _ in range(dormant):
            loops = _add_dormant(shape, sources, loops, rng, mirror_palette, period)
            if loops is None:
                break
        if loops is None:
            continue
        window = _choose_window(shape, sources, loops, rng, hidden_target, require_dormant)
        if window is None:
            continue
        if desired_sources > 1:
            sources = _extra_source(shape, sources, loops, rng, source_palette, window, hidden_target)
            if sources is None:
                continue
        grid = np.full(shape, bg, dtype=np.int8)
        for loop in loops:
            for p, color in loop["pixels"].items():
                grid[p] = color
        for source in sources:
            for p in source["cells"]:
                grid[p] = source["color"]
        t, b, l, r = window
        grid[t:b+1, l:r+1] = fill
        for p in _perimeter(window):
            grid[p] = frame
        scene = {"window": window, "fill": fill, "loops": loops}
        painted, routes, active = _trace(shape, sources, loops)
        output = _render_answer(scene, painted, active)
        # Every source must make an observable contribution in this crop.
        essential = True
        if len(sources) > 1:
            for omitted in range(len(sources)):
                others = sources[:omitted]+sources[omitted+1:]
                try:
                    partial_paint, _, partial_active = _trace(shape, others, loops)
                except ValueError:
                    essential = False
                    break
                if np.array_equal(output, _render_answer(scene, partial_paint, partial_active)):
                    essential = False
                    break
        if not essential:
            continue
        try:
            prediction, decoded = _analyze(grid)
        except ValueError:
            continue
        if (np.array_equal(output, prediction) and decoded["window"] == window
                and {v["box"] for v in decoded["loops"]} == {v["box"] for v in loops}
                and len(decoded["sources"]) == len(sources)):
            return {"input": grid, "output": output}
    raise RuntimeError("bounded circuit construction exhausted")


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    ntrain, ntest = _counts(rng, num_train, num_test)
    max_period = int(rng.integers(2, 5))
    axes = [int(rng.integers(2))]
    axes.append(1-axes[0])
    directions = [DIRECTIONS[int(i)] for i in rng.permutation(4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            teaching = split == "train" and index < 2
            plan = dict(count=1 if teaching and index == 0 else 2,
                        source_count=1 if teaching and index == 0 else int(rng.integers(1, 3)),
                        period=1 if teaching and index == 0 else (max_period if teaching else int(rng.integers(1, max_period+1))),
                        axis=axes[index] if teaching else int(rng.integers(2)),
                        direction=directions[index] if teaching else DIRECTIONS[int(rng.integers(4))],
                        hidden_target=1 if split == "train" else 2,
                        dormant=1, require_dormant=True)
            # The second demonstration explicitly teaches superposition.
            if teaching and index == 1:
                plan["source_count"] = 2
            for _ in range(4):
                try:
                    pair = _pair(rng, **plan)
                except RuntimeError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct scene search exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene, shape):
    hidden = [h for route in scene["routes"] for h in route["hits"] if _in_box(h[2], scene["window"], True)]
    return {"shape": tuple(shape), "window": scene["window"],
            "sources": len(scene["sources"]), "loops": len(scene["loops"]),
            "active": len(scene["active"]), "hidden_contacts": len({h[2] for h in hidden}),
            "hidden_chain": max(sum(_in_box(h[2], scene["window"], True) for h in route["hits"])
                                for route in scene["routes"]),
            "axes": tuple(sorted({h[1] for route in scene["routes"] for h in route["hits"]})),
            "directions": tuple(sorted({s["direction"] for s in scene["sources"]})),
            "periods": tuple(sorted(loop["period"] for loop in scene["loops"])),
            "background": scene["background"], "fill_matches": scene["fill"] == scene["background"],
            "bounces": tuple(len(route["hits"]) for route in scene["routes"]),
            "loop_shapes": tuple((v["box"][1]-v["box"][0]+1, v["box"][3]-v["box"][2]+1) for v in scene["loops"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            predicted, scene = _analyze(inp)
            _require(np.array_equal(predicted, out), "incorrect activated-loop crop")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "new output color")
            active = scene["active"]
            _require(bool(active) and len(active) < len(scene["loops"]), "missing active/dormant contrast")
            active_hidden = sum(_in_box(p, scene["window"], True) for i in active for p in scene["loops"][i]["pixels"])
            dormant_hidden = sum(_in_box(p, scene["window"], True) for i, loop in enumerate(scene["loops"]) if i not in active for p in loop["pixels"])
            _require(active_hidden >= 3 and dormant_hidden >= 3, "missing hidden selection evidence")
            if len(scene["sources"]) > 1:
                for omitted in range(len(scene["sources"])):
                    sources = scene["sources"][:omitted]+scene["sources"][omitted+1:]
                    paint, _, used = _trace(inp.shape, sources, scene["loops"])
                    _require(not np.array_equal(_render_answer(scene, paint, used), out),
                             "a source has no observable effect")
            # Omitting A: reveal every reconstructed loop, with the same rays.
            without_a = _render_answer(scene, scene["painted"], set(range(len(scene["loops"]))))
            _require(not np.array_equal(without_a, out), "activation is irrelevant")
            # Omitting B: retain only visible wall contacts. A collision with
            # an inferred wall also witnesses the need for hidden reconstruction.
            try:
                paint, _, used = _trace(inp.shape, scene["sources"], scene["loops"], visible_only=scene["window"])
                without_b = _render_answer(scene, paint, used)
                _require(not np.array_equal(without_b, out), "hidden mirrors are irrelevant")
            except ValueError as error:
                _require(str(error) != "hidden mirrors are irrelevant", str(error))
            try:
                paint, _, used = _trace(inp.shape, scene["sources"], scene["loops"], recolor=False)
                unchanged_color = _render_answer(scene, paint, used)
            except ValueError:
                # Conflicting colors can appear when correct recoloring is removed.
                unchanged_color = None
            _require(unchanged_color is None or not np.array_equal(unchanged_color, out),
                     "midpoint color sampling is irrelevant")
            feature = _features(scene, inp.shape)
            (training if split == "train" else testing).append(feature)
    _require({axis for f in training for axis in f["axes"]} == {0, 1}, "both mirror axes must be taught")
    _require(len({d for f in training for d in f["directions"]}) >= 2, "launch rotations not taught")
    _require(any(f["sources"] >= 2 for f in training), "superposition not taught")
    _require(any(max(f["periods"]) >= 2 for f in training), "periodic completion not taught")
    _require(all(f["hidden_contacts"] == 1 for f in training), "training shift contract")
    _require(all(f["hidden_chain"] >= 2 for f in testing), "test must chain hidden contacts")


def self_test():
    """Stream 42 seeds; keep only bounded aggregate sets and one puzzle."""
    shapes, windows, structures, backgrounds, periods, loop_shapes = set(), set(), set(), set(), set(), set()
    fills, source_counts, active_counts, axes, directions = set(), set(), set(), set(), set()
    fingerprints = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                output, scene = _analyze(pair["input"])
                assert np.array_equal(output, pair["output"])
                f = _features(scene, pair["input"].shape)
                shapes.add(f["shape"])
                windows.add(output.shape)
                backgrounds.add(f["background"])
                periods.update(f["periods"])
                loop_shapes.update(f["loop_shapes"])
                fills.add(f["fill_matches"])
                source_counts.add(f["sources"])
                active_counts.add(f["active"])
                axes.update(f["axes"])
                directions.update(f["directions"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(pair["input"].shape)+normalized).digest())
                fingerprint.update(bytes(pair["input"].shape)+pair["input"].tobytes()+output.tobytes())
        fingerprints.add(fingerprint.digest())
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(windows) >= 45 and len(loop_shapes) >= 10
    assert backgrounds == set(range(10)) and periods == {1, 2, 3, 4}
    assert fills == {False, True} and source_counts == {1, 2} and active_counts == {1, 2}
    assert axes == {0, 1} and directions == set(DIRECTIONS)
    first, second = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s]) for k in p)
    for counts in ((10, 4), (2, 1), (None, 4), (2, None)):
        generate_puzzle(31, *counts)
    for counts in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    _rule_regressions()
    return {"seeds": 42, "pairs": 210, "structures": len(structures),
            "input_shapes": len(shapes), "output_shapes": len(windows),
            "loop_shapes": len(loop_shapes), "periods": sorted(periods)}


def _rule_regressions():
    """An explicit answer, not an expected grid obtained from the tracer."""
    inp = np.zeros((26, 28), dtype=np.int8)
    for y, x in _perimeter((8, 16, 12, 18)):
        inp[y, x] = (4, 5, 6)[(y+x) % 3]
    for p in _perimeter((18, 24, 11, 17)):
        inp[p] = 7
    inp[8, 7] = inp[7, 7] = inp[8, 6] = 2
    inp[10:21, 9:16] = 8
    for p in _perimeter((10, 20, 9, 15)):
        inp[p] = 9
    expected = np.array([[int(v) for v in row] for row in
                         ("28688", "84488", "48588", "88688", "88488",
                          "88564", "88888", "88888", "88888")], dtype=np.int8)
    # The ray reaches hidden left contact (12,11), reads color 4 from
    # wall (12,12), and turns southwest. The dormant loop stays invisible.
    # Reconstructing only the touched side would miss the bottom-side 6,4.
    for turns in range(4):
        for flip in (False, True):
            query, answer = np.rot90(inp, turns), np.rot90(expected, turns)
            if flip:
                query, answer = query[:, ::-1], answer[:, ::-1]
            assert np.array_equal(solve(query), answer)
    equal_sources = [_source((1, 1), (1, 1), 2), _source((1, 9), (1, -1), 2)]
    painted, _, _ = _trace((11, 11), equal_sources, [])
    assert painted[(5, 5)] == 2
    try:
        _trace((11, 11), [equal_sources[0], _source((1, 9), (1, -1), 3)], [])
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting shared-cell colors accepted")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
