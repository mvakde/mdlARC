"""Repair categorical terminal panels, then route their missing identities.

The solver parses frames, guide stencils and a directional wire graph from one
input. Generation independently renders answers from completed panels and paths.
No parent files, stored examples, RNG state or training outputs are used to solve.
"""
from itertools import permutations
import hashlib

import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3))
HOST_SEEDS = (0, 822348682, 173153567, 53563021, 608326248, 798862518,
              741460195, 708314102, 558713991, 486258405, 441327147,
              444146310, 849141057, 650859859, 455943705, 650264083,
              290476020, 741370798, 558989889, 644742765, 480631107,
              509955912, 938632417, 261115166, 517232784, 495663308,
              742623660, 731479233, 374212044, 159515338, 352622715,
              519874662, 534702968, 969161335, 651904525, 483675333,
              259654638, 940260590, 826754790, 985365738, 964966268,
              730920352)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "bad grid size")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((g >= 0) & (g <= 9)), "bad grid color")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.tobytes()


def _motif(pattern, color, bg):
    return np.where(pattern, color, bg).astype(np.int8)


def _hist(g, marker):
    counts = np.bincount(g.ravel(), minlength=10)
    counts[marker] = 0
    return counts


def _infer(panels, marker):
    area = panels[0].size
    observed = [_hist(p, marker) for p in panels]
    _require(all(p.size == area and np.count_nonzero(p == marker) == 1
                 for p in panels), "exactly one hole per equal-area panel required")
    target = np.maximum.reduce(observed)
    # A completion must dominate this maximum. If the maximum fills the area,
    # no additional cell can be added: it is the unique common histogram.
    _require(int(target.sum()) == area, "common histogram is not unique")
    missing = []
    for counts in observed:
        deficit = target - counts
        _require(np.all(deficit >= 0) and deficit.sum() == 1 and
                 np.count_nonzero(deficit) == 1, "inconsistent histogram")
        missing.append(int(np.flatnonzero(deficit)[0]))
    return target, missing


def _trace(g, start, end, centers, wire, crossing):
    """Trace actual four-neighbor geometry; marked intersections go straight."""
    visible = {(int(r), int(c)) for r, c in np.argwhere(g == wire)}
    if crossing is not None:
        visible.update((int(r), int(c)) for r, c in np.argwhere(g == crossing))
    graph = {p: {d: (p[0] + dr, p[1] + dc) for d, (dr, dc) in enumerate(DIRS)
                 if (p[0] + dr, p[1] + dc) in visible} for p in visible}
    endpoints = {(start, c) for c in centers} | {(end, c) for c in centers}
    crossings = 0
    for p, edges in graph.items():
        if p in endpoints:
            expected = {2} if p[0] == start else {0}
            _require(set(edges) == expected and g[p] == wire, "invalid wire terminal")
        elif len(edges) == 4:
            _require(crossing is not None and g[p] == crossing, "unmarked junction")
            crossings += 1
        else:
            _require(len(edges) == 2 and g[p] == wire, "branch, break or misplaced crossing")
    _require(endpoints <= visible, "missing terminal wire")
    pairing, routes, visited = [], [], set()
    for c in centers:
        p, incoming, states, route = (start, c), 0, set(), []
        while True:
            _require((p, incoming) not in states, "wire cycle")
            states.add((p, incoming))
            visited.add(p)
            route.append(p)
            if p in endpoints and p != (start, c):
                _require(p[0] == end, "wire returns to source bank")
                pairing.append(centers.index(p[1]))
                break
            edges = graph[p]
            if p == (start, c):
                outgoing = 2
            elif len(edges) == 4:
                _require(incoming in edges, "bad crossing entrance")
                outgoing = (incoming + 2) % 4
            else:
                _require(incoming in edges, "bad wire entrance")
                outgoing = next(d for d in edges if d != incoming)
            p = edges[outgoing]
            incoming = (outgoing + 2) % 4
        routes.append(route)
    _require(sorted(pairing) == list(range(4)), "wire routing is not bijective")
    _require(visited == visible, "unattached wire fragment")
    return pairing, routes, crossings


def _runs(indices):
    result = []
    for x in map(int, indices):
        if result and x == result[-1][-1] + 1:
            result[-1].append(x)
        else:
            result.append([x])
    return result


def _parse(input_grid):
    original = _grid(input_grid)
    bg = int(original[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (original[0], original[-1], original[:, 0], original[:, -1])),
             "a blank exterior border is required")
    candidates = []
    for rotation in range(4):
        g = np.rot90(original, rotation)
        points = np.argwhere(g != bg)
        if not len(points):
            continue
        top, last = int(points[:, 0].min()), int(points[:, 0].max())
        runs = _runs(np.flatnonzero(g[top] != bg))
        if len(runs) != 4 or len({len(run) for run in runs}) != 1:
            continue
        pw = len(runs[0]) - 2
        if pw not in (2, 3):
            continue
        marker = int(g[top, runs[0][0]])
        centers = [run[0] + (pw + 1) // 2 for run in runs]
        if len(set(np.diff(centers))) != 1 or centers[1] - centers[0] < pw + 3:
            continue
        for ph in (2, 3, 4):
            try:
                sr = top + ph + 2
                start, end = sr + 3, last - 3
                _require(end - start >= 4, "insufficient wire field")
                allowed = np.zeros(g.shape, dtype=bool)
                panels, panel_slices, guide_slices = [], [], []
                for run, c in zip(runs, centers):
                    left = run[0]
                    block = g[top:top + ph + 2, left:left + pw + 2]
                    _require(block.shape == (ph + 2, pw + 2), "truncated frame")
                    _require(all(np.all(edge == marker) for edge in
                                 (block[0], block[-1], block[:, 0], block[:, -1])),
                             "incomplete panel frame")
                    ps = (slice(top + 1, top + ph + 1), slice(left + 1, left + pw + 1))
                    panels.append(g[ps])
                    panel_slices.append(ps)
                    allowed[top:top + ph + 2, left:left + pw + 2] = True
                    for row, pattern in ((sr, DIAMOND), (end + 1, CHECKER)):
                        gs = (slice(row, row + 3), slice(c - 1, c + 2))
                        _require(np.array_equal(g[gs], _motif(pattern, marker, bg)),
                                 "missing bank guide stencil")
                        allowed[gs] = True
                        guide_slices.append(gs)
                allowed[start:end + 1] = True
                _require(np.all(g[~allowed] == bg), "unexplained cells outside panels and wires")
                target, missing = _infer(panels, marker)
                payload = set(map(int, np.flatnonzero(target)))
                _require(bg not in payload and marker not in payload, "structural payload color")
                wire = int(g[start, centers[0]])
                _require(wire not in payload | {bg, marker}, "invalid carrier role")
                _require(all(g[start, c] == wire and g[end, c] == wire for c in centers),
                         "unmatched bank terminals")
                field_colors = set(map(int, np.unique(g[start:end + 1]))) - {bg, wire}
                _require(len(field_colors) <= 1 and not (field_colors & (payload | {marker})),
                         "unrecognized routing role")
                crossing = next(iter(field_colors)) if field_colors else None
                field = np.full(g.shape, bg, dtype=np.int8)
                field[start:end + 1] = g[start:end + 1]
                pairing, routes, crossings = _trace(field, start, end, centers, wire, crossing)
                candidates.append(dict(grid=g, rotation=rotation, bg=bg, marker=marker,
                                       wire=wire, crossing=crossing, top=top, sr=sr,
                                       start=start, end=end, centers=centers, shape=(ph, pw),
                                       panels=panels, panel_slices=panel_slices,
                                       guide_slices=guide_slices, histogram=target,
                                       missing=missing, pairing=pairing, routes=routes,
                                       crossings=crossings))
            except (ValueError, IndexError, StopIteration):
                continue
    _require(len(candidates) == 1, f"expected one input interpretation, got {len(candidates)}")
    return candidates[0]


def _complete(info, *, pairing=None, labels=None, repair=True):
    g = info["grid"].copy()
    labels = info["missing"] if labels is None else labels
    pairing = info["pairing"] if pairing is None else pairing
    for i, (ps, c) in enumerate(zip(info["panel_slices"], info["centers"])):
        if repair:
            panel = g[ps]
            panel[panel == info["marker"]] = labels[i]
        g[info["sr"]:info["sr"] + 3, c - 1:c + 2] = _motif(DIAMOND, labels[i], info["bg"])
        dest = info["centers"][pairing[i]]
        g[info["end"] + 1:info["end"] + 4, dest - 1:dest + 2] = _motif(CHECKER, labels[i], info["bg"])
    return np.rot90(g, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover a unique common multiset, then carry each deficit along its wire."""
    return _complete(_parse(input_grid))


def _moves(target, rng):
    """Realize arbitrary permutations with an empty exterior routing track."""
    state, moves = list(range(4)) + [None], []
    for j in map(int, rng.permutation(4)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, 4))
        state[4], state[j] = state[j], None
        vacancy = j
        while target[vacancy] != saved:
            old = state.index(target[vacancy])
            moves.append((old, vacancy))
            state[vacancy], state[old] = state[old], None
            vacancy = old
        moves.append((4, vacancy))
        state[vacancy], state[4] = state[4], None
    _require(state[:4] == list(target), "permutation construction failed")
    return moves


def _routing(rng, ph, pw, target, detour):
    moves = _moves(target, rng)
    if detour and len(moves) <= 4:
        lane = int(rng.integers(4))
        moves = [(lane, 4), (4, lane)] + moves
    # Four framed panels, an exterior track, and a one-cell blank border.
    pitch = int(rng.integers(pw + 3, 8))
    offset = (pw + 1) // 2
    min_width = 1 + offset + 3 * pitch + 3 + 2
    _require(min_width <= 30, "width budget exhausted")
    budget = 30 - min_width
    left_extra = int(rng.integers(min(2, budget) + 1))
    budget -= left_extra
    spare_extra = int(rng.integers(min(2, budget) + 1))
    budget -= spare_extra
    right_extra = int(rng.integers(min(2, budget) + 1))
    centers = [1 + offset + left_extra + i * pitch for i in range(4)]
    columns = centers + [centers[-1] + 3 + spare_extra]
    width = columns[-1] + 2 + right_extra
    top, bottom = 1, 1
    start = top + ph + 5
    gaps = [2] * max(0, len(moves) - 1)
    lead, tail = 2, 2
    end = start + lead + sum(gaps) + tail
    height = end + 4 + bottom
    _require(height <= 30, "height budget exhausted")
    budget = 30 - height
    top_extra = int(rng.integers(min(2, budget) + 1))
    budget -= top_extra
    top += top_extra
    start += top_extra
    extra = int(rng.integers(budget + 1))
    pieces = [lead, tail] + gaps
    for _ in range(extra):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, gaps = pieces[0], pieces[1], pieces[2:]
    rows, row = [], start + lead
    for k in range(len(moves)):
        rows.append(row)
        if k < len(gaps):
            row += gaps[k]
    end = rows[-1] + tail if moves else start + lead + tail
    height = end + 4 + bottom
    paths, state, previous = [[(start, c)] for c in centers], list(range(4)) + [None], start
    for row, (old, new) in zip(rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        owner = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[owner].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    field = np.zeros((height, width), dtype=np.int8)
    owners = {}
    for i, path in enumerate(paths):
        for p in path:
            field[p] = 1
            owners.setdefault(p, set()).add(i)
    for p, visitors in owners.items():
        if len(visitors) > 1:
            _require(len(visitors) == 2, "triple crossing")
            field[p] = 2
    return field, paths, centers, top, start, end


def _common_and_missing(rng, area, palette):
    k = int(rng.integers(2, min(6, area - 1) + 1))
    colors = list(map(int, rng.choice(palette[4:], k, replace=False)))
    counts = np.ones(k, dtype=int)
    for _ in range(area - k):
        counts[int(rng.integers(k))] += 1
    if len(set(map(int, counts))) == 1:
        counts[0] += 1
        counts[1] -= 1
    # area > k ensures a repeated color. Both frequent and less frequent
    # missing identities are included, so set completion and majority fail.
    common = np.zeros(10, dtype=int)
    common[colors] = counts
    frequent = colors[int(np.argmax(counts))]
    minority = int(rng.choice([c for c in colors if common[c] < common[frequent]]))
    missing = [frequent, minority]
    if k >= 4 and rng.random() < .5:
        missing += list(map(int, rng.choice([c for c in colors if c not in missing], 2, replace=False)))
    else:
        missing += list(map(int, rng.choice(colors, 2)))
    rng.shuffle(missing)
    return common, missing


def _target(rng, missing, lesson, split, shift):
    options = list(permutations(range(4)))
    if lesson == 0:
        # One swapped pair, with distinguishable labels and visible crossings.
        options = [p for p in options if sum(i != p[i] for i in range(4)) == 2
                   and any(missing[p[i]] != missing[i] for i in range(4))]
    elif lesson == 1:
        options = [p for p in options if sum(i != p[i] for i in range(4)) == 3
                   and any(missing[p[i]] != missing[i] for i in range(4))]
    elif shift == "routing" and split == "test":
        options = [p for p in options if all(i != p[i] for i in range(4))
                   and any(missing[p[i]] != missing[i] for i in range(4))]
    elif shift == "routing":
        options = [p for p in options if sum(i != p[i] for i in range(4)) <= 3]
    elif rng.random() < .2:
        options = [tuple(range(4))]
    return options[int(rng.integers(len(options)))]


def _make_pair(rng, shape, palette, rotation, lesson, split, shift):
    ph, pw = shape
    bg, marker, wire, crossing = map(int, palette[:4])
    common, missing = _common_and_missing(rng, ph * pw, palette)
    target = _target(rng, missing, lesson, split, shift)
    field, paths, centers, top, start, end = _routing(rng, ph, pw, target, rng.random() < .3)
    lut = np.array([bg, wire, crossing], dtype=np.int8)
    inp, out = lut[field], lut[field].copy()
    patterns, holes = [], []
    for i, c in enumerate(centers):
        left = c - (pw + 1) // 2
        for g in (inp, out):
            g[top:top + ph + 2, left:left + pw + 2] = marker
        # At most 32 proposals per panel; no retained failed payloads or logs.
        for _ in range(32):
            values = np.repeat(np.arange(10, dtype=np.int8), common)
            rng.shuffle(values)
            pos = int(rng.choice(np.flatnonzero(values == missing[i])))
            if not patterns or (values.tobytes() not in patterns and (i != 3 or len(set(holes + [pos])) > 1)):
                break
        patterns.append(values.tobytes())
        holes.append(pos)
        panel = values.reshape(ph, pw)
        ps = (slice(top + 1, top + ph + 1), slice(left + 1, left + pw + 1))
        inp[ps], out[ps] = panel, panel
        inp[top + 1 + pos // pw, left + 1 + pos % pw] = marker
        sr = start - 3
        inp[sr:sr + 3, c - 1:c + 2] = _motif(DIAMOND, marker, bg)
        out[sr:sr + 3, c - 1:c + 2] = _motif(DIAMOND, missing[i], bg)
        dest = paths[i][-1][1]
        inp[end + 1:end + 4, dest - 1:dest + 2] = _motif(CHECKER, marker, bg)
        out[end + 1:end + 4, dest - 1:dest + 2] = _motif(CHECKER, missing[i], bg)
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             f"{name} must be an integer")
    _require(low <= value <= high, f"{name} must be in {low}..{high}")
    return int(value)


def _distinct_fallback(pair, seen, payload_pool):
    """At least 6P2=30 relabelings exceed the thirteen prior input bound."""
    info = _parse(pair["input"])
    used = list(map(int, np.flatnonzero(info["histogram"])))
    for values in permutations(map(int, payload_pool), len(used)):
        lut = np.arange(10, dtype=np.int8)
        lut[used] = values
        candidate = {role: lut[grid] for role, grid in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("distinct-label finite bound violated")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    vary_palette = bool(rng.integers(2))
    shift = "panel_area" if rng.integers(2) else "routing"
    base_rotation = int(rng.integers(4))
    mixed_orientation = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else None
            if lesson == 0:
                shape = (2, 2)
            elif lesson == 1:
                shape = SHAPES[1 + int(rng.integers(2))]
            elif shift == "panel_area":
                choices = SHAPES[:3] if split == "train" else SHAPES[3:]
                shape = choices[int(rng.integers(len(choices)))]
            else:
                shape = SHAPES[int(rng.integers(len(SHAPES)))]
            rotation = int(rng.integers(4)) if mixed_orientation else base_rotation
            local_palette = rng.permutation(10) if vary_palette else palette
            for _ in range(24):
                pair = _make_pair(rng, shape, local_palette, rotation, lesson, split, shift)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                pair = _distinct_fallback(pair, seen, local_palette[4:])
                key = _key(pair["input"])
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check input-derived uniqueness, causal necessity and exact preservation."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape change or duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(np.array_equal(_complete(info), out), "input-derived solution mismatch")
            canonical_out = np.rot90(out, info["rotation"])
            mutable = np.zeros(inp.shape if not info["rotation"] % 2 else inp.T.shape, dtype=bool)
            for ps in info["panel_slices"]:
                mutable[ps] = info["grid"][ps] == info["marker"]
                _require(np.array_equal(_hist(canonical_out[ps], info["marker"]), info["histogram"]),
                         "completed panel histograms differ")
            for gs in info["guide_slices"]:
                mutable[gs] = info["grid"][gs] == info["marker"]
            _require(np.array_equal(info["grid"][~mutable], canonical_out[~mutable]),
                     "noneditable cells changed")
            _require(np.count_nonzero(inp != out) == 40, "expected four repairs and 36 stencil cells")
            counts = info["histogram"]
            positive = counts[counts > 0]
            _require(len(set(map(int, positive))) > 1, "multiplicities must be unequal")
            _require(len(set(info["missing"])) >= 2, "uninformative repair identities")
            _require(any(counts[c] >= 2 for c in info["missing"]), "set-only shortcut not distinguished")
            _require(any(counts[c] < counts.max() for c in info["missing"]), "majority shortcut not distinguished")
            if split == "train":
                training.append(info)
    _require(any(i["crossings"] and not np.array_equal(
        _complete(i, pairing=list(range(4))), _complete(i)) for i in training),
        "training does not establish answer-affecting cable pairing")
    _require(any(i["shape"] == (2, 2) for i in training) and
             any(np.prod(i["shape"]) >= 6 for i in training), "missing panel-size demonstrations")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed semantic checks; retains only hashes and feature sets."""
    shapes, panel_shapes, rotations, pairings, kinds, roles, hashes = (set() for _ in range(7))
    identities = four_different = repeated = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        _require(digest == _digest(generate_puzzle(seed)), "nondeterministic generation")
        hashes.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, expected = pair["input"], pair["output"]
                before = inp.copy()
                predicted = solve(inp)
                _require(np.array_equal(predicted, expected) and np.array_equal(inp, before),
                         "solver mismatch or input mutation")
                info = _parse(inp)
                shapes.add(inp.shape)
                panel_shapes.add(info["shape"])
                rotations.add(info["rotation"])
                pairings.add(tuple(info["pairing"]))
                kinds.add(tuple(sorted(int(c) for c in info["histogram"] if c)))
                roles.add((info["bg"], info["marker"], info["wire"]))
                identities += info["pairing"] == list(range(4))
                four_different += len(set(info["missing"])) == 4
                repeated += len(set(info["missing"])) < 4
                # Palette relabeling must commute with solving; numeric order
                # and interval size have no meaning in categorical histograms.
                lut = np.arange(9, -1, -1, dtype=np.int8)
                _require(np.array_equal(solve(lut[inp]), lut[expected]), "numeric-color shortcut")
                # Histogram ablation changes even identity-route examples.
                # Pairing ablation is checked on the training set by validation.
                no_histogram = _complete(info, labels=[info["marker"]] * 4, repair=False)
                _require(not np.array_equal(no_histogram, expected), "histogram concept unnecessary")
    _require(len(hashes) == 42 and len(shapes) >= 25, "insufficient puzzle/shape variation")
    _require(panel_shapes == set(SHAPES) and rotations == set(range(4)), "geometry coverage")
    _require(len(pairings) >= 18 and identities > 0, "routing coverage")
    _require(len(kinds) >= 12 and len(roles) >= 25 and four_different and repeated,
             "categorical multiplicity or palette variation missing")
    # Exhaust all 24 permutations, with true geometry traced independently of
    # the path metadata. No whole-puzzle generation is chained here.
    rng = np.random.default_rng(20261023)
    for target in permutations(range(4)):
        field, paths, centers, _, start, end = _routing(rng, 3, 3, target, False)
        pairing, _, _ = _trace(field, start, end, centers, 1, 2 if np.any(field == 2) else None)
        _require(pairing == [target.index(i) for i in range(4)], "permutation trace mismatch")
        _require(all(path[-1][1] == centers[pairing[i]] for i, path in enumerate(paths)),
                 "trace differs from constructed path")
    # A unique histogram need not have four different deficits. An input whose
    # four observations all omit the same identity must be rejected.
    ambiguous = [np.array([[1, 1], [6, 5]], dtype=np.int8) for _ in range(4)]
    try:
        _infer(ambiguous, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous common histogram accepted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9183, nt, nq)
            validate_generated_puzzle(p)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    old = generate_puzzle(0, 2, 1)["train"][0]
    info = _parse(old["input"])
    available = sorted(set(range(10)) - {info[k] for k in ("bg", "marker", "wire", "crossing")})[:6]
    different = _distinct_fallback(old, {_key(old["input"])}, available)
    _require(_key(different["input"]) != _key(old["input"]) and
             np.array_equal(solve(different["input"]), different["output"]),
             "distinct-input fallback failed")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        return
    raise AssertionError("nonterminal corruption accepted")
