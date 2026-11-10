"""Reciprocal cross keys: input-visible, peg-centered rigid registration.

0 is empty; 5 is scaffold material. Each eight-connected object has a body
and a lone reciprocal-color peg. All embedded plus centers, rather than outer
contours, must register. No parent module or stored pair enters prediction.
"""
from __future__ import annotations

import numpy as np

GRAY = 5
LABELS = (1, 2, 3, 4, 6, 7, 8, 9)
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30")
    return a.astype(np.int8, copy=True)


def _centers(body):
    """Complete five-cell motifs on the original coordinate set."""
    return {p for p in body if all((p[0] + dr, p[1] + dc) in body for dr, dc in N4)}


def _union(centers):
    return {p for r, c in centers for p in ((r, c), (r - 1, c),
                                          (r + 1, c), (r, c - 1), (r, c + 1))}


def _rot(points, turns):
    """Clockwise quarter turns in row/column coordinates, about (0, 0)."""
    answer = set(points)
    for _ in range(turns % 4):
        answer = {(c, -r) for r, c in answer}
    return answer


def _shift(points, origin):
    return {(r + origin[0], c + origin[1]) for r, c in points}


def _relative(points, peg):
    return {(r - peg[0], c - peg[1]) for r, c in points}


def _components(a):
    remaining = set(map(tuple, np.argwhere(a != 0)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N8:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        components.append(cells)
    return components


def _parse(a):
    if np.count_nonzero(a == 0) <= a.size // 2:
        raise ValueError("empty cells must be a strict majority")
    stations, donors = {}, []
    for cells in _components(a):
        colors = {int(a[p]) for p in cells}
        if len(colors) != 2 or GRAY not in colors:
            raise ValueError("each object must have gray and one label color")
        color = next(iter(colors - {GRAY}))
        gray = {p for p in cells if a[p] == GRAY}
        colored = cells - gray
        if len(colored) == 1 and len(gray) >= 5:
            kind, body, peg = "station", gray, next(iter(colored))
        elif len(gray) == 1 and len(colored) >= 4:
            kind, body, peg = "donor", colored, next(iter(gray))
        else:
            raise ValueError("ambiguous reciprocal body/peg roles")
        # The body itself, not just the body-plus-peg, must be connected.
        visited = {min(body)}
        stack = list(visited)
        while stack:
            r, c = stack.pop()
            for dr, dc in N8:
                q = (r + dr, c + dc)
                if q in body and q not in visited:
                    visited.add(q)
                    stack.append(q)
        if visited != body or not any((peg[0] + dr, peg[1] + dc) in body for dr, dc in N8):
            raise ValueError("body and attached peg must be eight-connected")
        obj = dict(kind=kind, color=color, body=body, peg=peg,
                   centers=_centers(body), cells=cells)
        if kind == "station":
            if color in stations or not obj["centers"]:
                raise ValueError("stations need distinct labels and complete pluses")
            stations[color] = obj
        else:
            donors.append(obj)
    if not stations:
        raise ValueError("at least one station is required")
    return stations, donors


def _registrations(station, donor, whole_body=False):
    target = station["body"] if whole_body else station["centers"]
    source = donor["body"] if whole_body else donor["centers"]
    placements = {}
    for turn in range(4):
        registered = _shift(_rot(_relative(source, donor["peg"]), turn), station["peg"])
        if registered == target:
            body = frozenset(_shift(_rot(_relative(donor["body"], donor["peg"]), turn),
                                    station["peg"]))
            placements.setdefault(body, set()).add(turn)
    return placements


def _assemble(a):
    stations, donors = _parse(a)
    out = np.zeros_like(a)
    stationary_cells = set()
    for station in stations.values():
        for p in station["body"]:
            out[p] = GRAY
        out[station["peg"]] = station["color"]
        stationary_cells.update(station["cells"])
    matched, used, targets = [], set(), set()
    for donor in donors:
        color = donor["color"]
        if color not in stations:
            continue
        if color in used:
            raise ValueError("at most one donor per station label")
        used.add(color)
        station = stations[color]
        if not donor["centers"]:
            raise ValueError("a served donor needs complete pluses")
        placements = _registrations(station, donor)
        if len(placements) != 1:
            raise ValueError("incompatible or ambiguous constellation registration")
        body, turns = next(iter(placements.items()))
        if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in body):
            raise ValueError("registered body would be clipped")
        foreign = stationary_cells - station["cells"]
        if body & foreign or body & targets:
            raise ValueError("registered bodies must not overwrite other assemblies")
        # Require the full output footprints to remain separate, including
        # diagonal contacts. Within a joint, occlusion is intentional.
        expanded = {(r + dr, c + dc) for r, c in body for dr, dc in ((0, 0),) + N8}
        if expanded & foreign or expanded & targets:
            raise ValueError("distinct output assemblies must be separated")
        targets.update(body)
        for p in body:
            out[p] = color
        out[station["peg"]] = color
        matched.append(dict(station=station, donor=donor, target=set(body), turns=turns))
    return out, dict(stations=stations, donors=donors, matched=matched)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive an answer solely from the input's colors, pegs, and motifs."""
    return _assemble(_grid(input_grid))[0]


def _core(mode, rng):
    body = set()

    def plus(r, c):
        body.update(_union({(r, c)}))

    def link(a, b):
        r, c = a
        while r != b[0]:
            body.add((r, c))
            r += 1 if b[0] > r else -1
        while c != b[1]:
            body.add((r, c))
            c += 1 if b[1] > c else -1
        body.add(b)

    if mode == "single":
        plus(4, 4)
    elif mode == "adjacent":
        body = {(r, c) for r in range(3, 6) for c in range(3, 7)}
    elif mode == "shared":
        plus(4, 3)
        plus(4, 5)
    elif mode == "elbow":
        for p in ((3, 3), (3, 6), (6, 3)):
            plus(*p)
        link((3, 3), (3, 6))
        link((3, 3), (6, 3))
    elif mode == "chain":
        n = int(rng.integers(2, 5))
        for c in range(1, 2 * n, 2):
            plus(4, c)
    elif mode == "branch":
        plus(4, 4)
        choices = ((2, 4), (4, 2), (4, 6), (6, 4))
        for i in rng.permutation(4)[:int(rng.integers(2, 5))]:
            plus(*choices[int(i)])
    elif mode == "loop":
        body = {(r, c) for r in range(2, 7) for c in range(2, 7)
                if r in (2, 6) or c in (2, 6)}
        for p in ((2, 4), (4, 2), (4, 6), (6, 4)):
            plus(*p)
    elif mode == "walk":
        choices = ((2, 2), (2, 5), (5, 2), (5, 5), (3, 3), (3, 6), (6, 3))
        points = [choices[int(i)] for i in rng.permutation(len(choices))[:int(rng.integers(3, 6))]]
        for p in points:
            plus(*p)
            link(points[0], p)
    else:
        raise ValueError("unknown core mode")
    return body


def _grow(core, peg, rng, budget):
    body, centers = set(core), _centers(core)
    for _ in range(budget):
        candidates = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N8
                             if 0 <= r + dr < 9 and 0 <= c + dc < 9}
                            - body - {peg})
        for i in rng.permutation(len(candidates)):
            q = candidates[int(i)]
            proposed = body | {q}
            if _centers(proposed) == centers:
                body = proposed
                break
        else:
            break
    return body


def _joint(mode, rng, budget):
    core = _core(mode, rng)
    rows, cols = zip(*core)
    options = sorted({(r + dr, c + dc) for r, c in core for dr, dc in N8
                      if 0 <= r + dr < 9 and 0 <= c + dc < 9
                      and (r + dr < min(rows) or r + dr > max(rows)
                           or c + dc < min(cols) or c + dc > max(cols))} - core)
    # An internal peg between shared-arm crosses gives a false one-anchor
    # quarter-turn match. Registering the complete constellation resolves it.
    peg = (3, 4) if mode == "shared" else options[int(rng.integers(len(options)))]
    for _ in range(24):
        scaffold = _grow(core, peg, rng, budget + int(rng.integers(1, 5)))
        donor = _grow(core, peg, rng, budget + int(rng.integers(1, 5)))
        # Training and generated tests expose contours which disagree in both
        # directions, rather than hiding a full-silhouette matching shortcut.
        if scaffold - donor and donor - scaffold:
            return _relative(scaffold, peg), _relative(donor, peg)
    # There is no shape-replacing fallback: use a new instance of this mode.
    raise RuntimeError("could not grow two distinct motif-compatible bodies")


def _candidate_positions(blocked, footprint):
    coords = np.array(sorted(footprint), dtype=int)
    low, high = coords.min(axis=0), coords.max(axis=0)
    h, w = high - low + 1
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return None
    rr, cc = np.indices((blocked.shape[0] - h + 1, blocked.shape[1] - w + 1))
    origins = np.column_stack((rr.ravel(), cc.ravel())) - low
    rows = origins[:, :1] + coords[:, 0]
    cols = origins[:, 1:] + coords[:, 1]
    good = np.flatnonzero(~blocked[rows, cols].any(axis=1))
    return origins[good]


def _pack(rng, shape, objects):
    blocked = np.zeros(shape, dtype=bool)
    grid = np.zeros(shape, dtype=np.int8)
    # Sorting only controls packing. No parser relies on this spatial order.
    order = list(map(int, rng.permutation(len(objects))))
    order.sort(key=lambda i: -len(objects[i]["footprint"]))
    for i in order:
        obj = objects[i]
        positions = _candidate_positions(blocked, obj["footprint"])
        if positions is None or not len(positions):
            return None
        origin = tuple(map(int, positions[int(rng.integers(len(positions)))]))
        for p in _shift(obj["body"], origin):
            grid[p] = obj["body_color"]
        grid[origin] = obj["peg_color"]
        for r, c in _shift(obj["footprint"], origin):
            blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return grid


def _make_pair(rng, labels, modes, deltas, missing, distractors, budget, dockable=False):
    for construction in range(60):
        objects = []
        try:
            for i, (color, mode) in enumerate(zip(labels, modes)):
                scaffold, donor = _joint(mode, rng, budget)
                station_turn = int(rng.integers(4))
                station_body = _rot(scaffold, station_turn)
                served = i not in missing
                target_body = _rot(donor, station_turn) if served else set()
                objects.append(dict(body=station_body, body_color=GRAY, peg_color=color,
                                    footprint=station_body | target_body | {(0, 0)}))
                if served:
                    source = _rot(donor, (station_turn - deltas[i]) % 4)
                    objects.append(dict(body=source, body_color=color, peg_color=GRAY,
                                        footprint=source | {(0, 0)}))
            for color in distractors:
                if not dockable and rng.random() < .2:
                    # No-plus near miss, still a legitimate reciprocal object.
                    core = {(4, 3), (4, 4), (4, 5), (5, 4), (6, 4)}
                    peg = (3, 3)
                    body = _relative(core, peg)
                else:
                    _, body = _joint(str(rng.choice(("single", "shared", "chain"))), rng, budget)
                body = _rot(body, int(rng.integers(4)))
                objects.append(dict(body=body, body_color=color, peg_color=GRAY,
                                    footprint=body | {(0, 0)}))
        except RuntimeError:
            continue
        if rng.random() < .22:
            shape = (int(rng.integers(16, 21)), int(rng.integers(26, 31)))
            if rng.random() < .5:
                shape = shape[::-1]
        elif rng.random() < .22:
            side = int(rng.integers(19, 31))
            shape = (side, side)
        else:
            shape = tuple(map(int, rng.integers(18, 31, size=2)))
        for attempt in range(8):
            grid = _pack(rng, shape, objects)
            if grid is not None:
                try:
                    output = solve(grid)
                except ValueError:
                    # Geometry is accepted from the finished input. In
                    # particular, competing registrations are never resolved
                    # by the orientations secretly selected above.
                    pass
                else:
                    return dict(input=grid, output=output)
            shape = tuple(min(30, d + 2) for d in shape)
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(LABELS)))
    budget = int(rng.integers(2, 8))
    regime = str(rng.choice(("constellation", "partial", "familiar")))
    rotations = list(map(int, rng.permutation(4)))
    basics = ["single", "adjacent", "shared", "elbow"]
    # Witness shapes have no fixed demonstration positions after shuffling.
    configs = []
    for i in range(2):
        labs = palette[:2] if i == 0 else palette[2:4]
        modes = basics[2 * i:2 * i + 2]
        missing = set()
        if regime == "partial":
            labs = labs + [palette[4 + i]]
            modes = modes + [str(rng.choice(basics))]
            missing = {2}
        configs.append((labs, modes, rotations[2 * i:2 * i + 2] + [0], missing,
                        [palette[2] if i == 0 else palette[0]], budget, True))
    for index in range(2, ntrain + ntest):
        test = index >= ntrain
        n = int(rng.integers(1, 4))
        if test and regime == "partial":
            n = int(rng.integers(2, 4))
        labs = list(map(int, rng.choice(LABELS, n, replace=False)))
        vocabulary = basics + ["chain", "branch", "loop", "walk"]
        if test and regime == "constellation":
            vocabulary = ["branch", "loop", "walk", "chain"]
        elif test and regime == "familiar":
            vocabulary = basics
        modes = list(map(str, rng.choice(vocabulary, n)))
        missing = set()
        if regime == "partial" and n > 1 and (test or rng.random() < .6):
            missing = {int(rng.integers(n))}
        d = int(rng.choice(3, p=(.3, .55, .15)))
        absent = [c for c in LABELS if c not in labs]
        distractors = list(map(int, rng.choice(absent, d, replace=True)))
        configs.append((labs, modes, list(map(int, rng.integers(4, size=n))),
                        missing, distractors, budget + (2 if test else 0), False))
    puzzle, seen = dict(train=[], test=[]), set()
    for index, config in enumerate(configs):
        for _ in range(12):
            pair = _make_pair(rng, *config)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if index < ntrain else "test"].append(pair)
                break
        else:
            raise RuntimeError("bounded uniqueness retries exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(ntrain)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(a, analysis, no_A):
    """Two concrete, plausible parent-omission rules for training evidence."""
    out = np.zeros_like(a)
    for station in analysis["stations"].values():
        for p in station["body"]:
            out[p] = GRAY
        out[station["peg"]] = station["color"]
    for joint in analysis["matched"]:
        station, donor = joint["station"], joint["donor"]
        if no_A:
            placements = _registrations(station, donor, whole_body=True)
            if len(placements) != 1:
                continue
            cells = next(iter(placements))
        else:
            cells = _union(station["centers"])
        for p in cells:
            out[p] = station["color"]
    return out


def _signature(points):
    signatures = []
    for k in range(4):
        rotated = _rot(points, k)
        low = min(r for r, c in rotated), min(c for r, c in rotated)
        signatures.append(tuple(sorted(_relative(rotated, low))))
    return min(signatures)


def _features(a, info):
    matches = info["matched"]
    single_anchor_ambiguous = False
    for j in matches:
        s, d = j["station"], j["donor"]
        for k in range(4):
            registered = _shift(_rot(_relative(d["centers"], d["peg"]), k), s["peg"])
            if registered & s["centers"] and registered != s["centers"]:
                single_anchor_ambiguous = True
    return dict(shape=a.shape, stations=len(info["stations"]), donors=len(info["donors"]),
                matched=len(matches), missing=len(info["stations"]) - len(matches),
                distractors=sum(d["color"] not in info["stations"] for d in info["donors"]),
                plus_counts=[len(s["centers"]) for s in info["stations"].values()],
                overlap=any(len(_union(s["centers"])) < 5 * len(s["centers"])
                            for s in info["stations"].values()),
                turns=set().union(*(j["turns"] for j in matches)) if matches else set(),
                overhang=any(j["target"] - j["station"]["body"] for j in matches),
                remaining_gray=any(j["station"]["body"] - j["target"] for j in matches),
                silhouettes={_signature(d["body"]) for d in info["donors"]},
                constellations={_signature(s["centers"]) for s in info["stations"].values()},
                match_colors={j["donor"]["color"] for j in matches},
                absent_colors={d["color"] for d in info["donors"] if d["color"] not in info["stations"]},
                edge=any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                         for j in matches for r, c in j["target"]),
                no_plus_distractor=any(not d["centers"] and d["color"] not in info["stations"]
                                       for d in info["donors"]),
                dockable_distractor=any(d["centers"] and d["color"] not in info["stations"]
                                        for d in info["donors"]),
                single_anchor_ambiguous=single_anchor_ambiguous)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    no_A_witness = no_B_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _assemble(inp)
            assert np.array_equal(out, expected), "incorrect registered assembly"
            assert not np.array_equal(inp, out), "generated scenes must demonstrate motion or deletion"
            f = _features(inp, info)
            features[split].append(f)
            for j in info["matched"]:
                # Exact body preservation, peg consumption, and color accounting.
                color, station, target = j["donor"]["color"], j["station"], j["target"]
                assert len(target) == len(j["donor"]["body"])
                assert _centers(target) == station["centers"]
                assert all(out[p] == color for p in target)
                assert out[station["peg"]] == color
                assert np.count_nonzero(out == color) == len(target) + 1
                assert all(out[p] == GRAY for p in station["body"] - target)
            for d in info["donors"]:
                if d["color"] not in info["stations"]:
                    assert not np.any(out == d["color"]), "unmatched label survives"
            if split == "train":
                no_A_witness |= not np.array_equal(out, _ablation(inp, info, True))
                no_B_witness |= not np.array_equal(out, _ablation(inp, info, False))
    train = features["train"]
    assert set().union(*(f["turns"] for f in train)) == set(range(4)), "show every quarter-turn"
    assert any(f["overlap"] for f in train), "show overlapping local joints"
    assert any(f["overhang"] and f["remaining_gray"] for f in train)
    assert any(f["distractors"] for f in train), "show color-based deletion"
    assert any(f["dockable_distractor"] for f in train), "show deletion of valid-looking keys"
    assert any(f["single_anchor_ambiguous"] for f in train), "one joint must not determine registration"
    matched_colors = set().union(*(f["match_colors"] for f in train))
    absent_colors = set().union(*(f["absent_colors"] for f in train))
    assert matched_colors & absent_colors, "show a literal color changing roles"
    assert no_A_witness and no_B_witness, "both parent concepts need counterexamples"
    if any(f["missing"] for f in features["test"]):
        assert any(f["missing"] for f in train), "partial service must be demonstrated"


def self_test() -> dict:
    """Bounded checks on motifs, registration, legal counts, and 42 seeds."""
    # Exhaust the local motif truth table, including diagonal clutter.
    for bits in range(512):
        body = {(i // 3, i % 3) for i in range(9) if (bits >> i) & 1}
        expected = {(1, 1)} if all((bits >> i) & 1 for i in (1, 3, 4, 5, 7)) else set()
        assert _centers(body) == expected
    block = {(r, c) for r in range(3) for c in range(4)}
    assert _centers(block) == {(1, 1), (1, 2)} and len(_union(_centers(block))) == 8
    assert not _centers({(0, 0), (0, 1), (0, 2), (1, 1)})
    # Hand-constructed peg origin and constellation, with unequal contours.
    centers = {(2, 2), (2, 4)}
    base = _union(centers)
    donor_body = base | {(2, 5), (3, 5), (4, 5)}
    station = dict(body=base | {(0, 1)}, centers=centers, peg=(1, 1))
    donor = dict(body=_rot(_relative(donor_body, (1, 1)), 1),
                 centers=_rot(_relative(centers, (1, 1)), 1), peg=(0, 0))
    placements = _registrations(station, donor)
    assert placements == {frozenset(donor_body): {3}}
    # A centered peg with a fourfold-symmetric constellation is legal when
    # every rotation gives identical body cells; asymmetric additions are a tie.
    symmetric = _union({(-2, 0), (2, 0), (0, -2), (0, 2)})
    s = dict(body=symmetric, centers=_centers(symmetric), peg=(0, 0))
    assert list(_registrations(s, s).values()) == [{0, 1, 2, 3}]
    asym = dict(body=symmetric | {(-3, 1)}, centers=_centers(symmetric), peg=(0, 0))
    assert len(_registrations(s, asym)) == 4
    # Complete known-answer fixture: two shared-arm crosses, an internal peg,
    # donor overhang, retained scaffold cells, a distractor, and an unserved
    # station. The expected body coordinates are written explicitly.
    fixture = np.zeros((20, 24), dtype=np.int8)
    joint_cells = {(4, 4), (3, 4), (5, 4), (4, 3), (4, 5),
                   (4, 6), (3, 6), (5, 6), (4, 7)}
    scaffold_cells = joint_cells | {(3, 2), (4, 2), (5, 2)}
    donor_target = joint_cells | {(5, 7), (6, 7), (6, 8)}
    for p in scaffold_cells:
        fixture[p] = GRAY
    fixture[3, 5] = 3
    for r, c in donor_target:
        fixture[13 + c - 5, 16 - r + 3] = 3
    fixture[13, 16] = GRAY
    for p in ((2, 19), (1, 19), (3, 19), (2, 18), (2, 20)):
        fixture[p] = 2
    fixture[2, 21] = GRAY
    unserved = {(14, 4), (13, 4), (15, 4), (14, 3), (14, 5)}
    for p in unserved:
        fixture[p] = GRAY
    fixture[16, 4] = 8
    expected = np.zeros_like(fixture)
    for p in scaffold_cells | unserved:
        expected[p] = GRAY
    expected[3, 5], expected[16, 4] = 3, 8
    for p in donor_target:
        expected[p] = 3
    original = fixture.copy()
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(fixture, original), "solve mutated its input"
    # Symmetry-equivalent registration and its inequivalent asymmetric tie.
    equivalent = np.zeros((25, 25), dtype=np.int8)
    for r, c in symmetric:
        equivalent[6 + r, 6 + c] = GRAY
        equivalent[18 + r, 18 + c] = 2
    equivalent[6, 6], equivalent[18, 18] = 2, GRAY
    expected = np.zeros_like(equivalent)
    for r, c in symmetric | {(0, 0)}:
        expected[6 + r, 6 + c] = 2
    assert np.array_equal(solve(equivalent), expected)
    equivalent[15, 19] = 2
    try:
        solve(equivalent)
    except ValueError:
        pass
    else:
        raise AssertionError("solve accepted an inequivalent registration tie")
    # A complete plus may have two arms on the canvas boundary; no wrapping.
    boundary = np.zeros((12, 12), dtype=np.int8)
    for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
        boundary[p] = GRAY
    boundary[2, 2] = 7
    for p in ((8, 8), (7, 8), (9, 8), (8, 7), (8, 9)):
        boundary[p] = 7
    boundary[9, 9] = GRAY
    expected = np.zeros_like(boundary)
    for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2), (2, 2)):
        expected[p] = 7
    assert np.array_equal(solve(boundary), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, features = set(), []
    for puzzle in puzzles:
        signatures.add(tuple(_key(p["input"]) for split in puzzle for p in puzzle[split]))
        for split in puzzle:
            for p in puzzle[split]:
                a = _grid(p["input"])
                features.append(_features(a, _assemble(a)[1]))
    assert len(signatures) == 42
    repeated = generate_puzzle(17)
    assert tuple(_key(p["input"]) for split in puzzles[17] for p in puzzles[17][split]) == tuple(
        _key(p["input"]) for split in repeated for p in repeated[split])
    shapes = {f["shape"] for f in features}
    silhouettes = set().union(*(f["silhouettes"] for f in features))
    constellations = set().union(*(f["constellations"] for f in features))
    counts = {n for f in features for n in f["plus_counts"]}
    assert len(shapes) >= 35 and any(30 in shape for shape in shapes)
    assert len(silhouettes) >= 100 and len(constellations) >= 12
    assert {1, 2, 3, 4, 5} <= counts
    assert {1, 2, 3} <= {f["stations"] for f in features}
    assert {0, 1, 2} <= {f["distractors"] for f in features}
    assert any(f["missing"] for f in features) and any(f["edge"] for f in features)
    assert any(f["no_plus_distractor"] for f in features)
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(8123 + ntrain * 10 + ntest, ntrain, ntest)
            assert len(puzzle["train"]) == ntrain and len(puzzle["test"]) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {split: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][split]]
           for split in puzzles[0]}
    bad["test"][0]["output"][0, 0] = 9 if bad["test"][0]["output"][0, 0] != 9 else 0
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")
    return dict(seeds=42, pairs=len(features), shapes=len(shapes),
                donor_silhouettes=len(silhouettes), constellations=len(constellations),
                plus_counts=sorted(counts),
                semantic_verification="generator-owned invariants and explicit counterexamples")


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
