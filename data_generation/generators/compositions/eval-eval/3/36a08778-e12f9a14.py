"""Watershed detours: colored ports feed competing fronts around transverse bars.

Only NumPy and the standard library are used. Nothing is read or written by
generation, validation, or self_test. All interpretation comes from the input.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _cores(a, background):
    """Recognize full 2x2 cores with 10/11 equal ring cells and side ports."""
    h, w = a.shape
    found = []
    if min(h, w) < 4:
        return found
    square = ((a[:-1, :-1] == a[1:, :-1]) & (a[:-1, :-1] == a[:-1, 1:])
              & (a[:-1, :-1] == a[1:, 1:]) & (a[:-1, :-1] != background))
    offsets = [(r, c) for r in range(-1, 3) for c in range(-1, 3)
               if r in (-1, 2) or c in (-1, 2)]
    sides = {(1, 0): {(2, 0), (2, 1)}, (-1, 0): {(-1, 0), (-1, 1)},
             (0, 1): {(0, 2), (1, 2)}, (0, -1): {(0, -1), (1, -1)}}
    for rr, cc in np.argwhere(square):
        r, c = int(rr), int(cc)
        if r < 1 or c < 1 or r + 2 >= h or c + 2 >= w:
            continue
        ring = [(dr, dc, int(a[r + dr, c + dc])) for dr, dc in offsets]
        holes = {(dr, dc) for dr, dc, color in ring if color == background}
        solid = {color for _, _, color in ring if color != background}
        if len(holes) not in (1, 2) or len(solid) != 1:
            continue
        structure = next(iter(solid))
        if structure == int(a[r, c]):
            continue
        directions = [d for d, cells in sides.items() if holes <= cells]
        if len(directions) == 1:
            found.append({"r": r, "c": c, "color": int(a[r, c]),
                          "structure": structure, "direction": directions[0],
                          "ports": sorted((r + dr, c + dc) for dr, dc in holes)})
    return found


def _runs(row):
    result = []
    c = 0
    while c < len(row):
        if not row[c]:
            c += 1
            continue
        lo = c
        while c + 1 < len(row) and row[c + 1]:
            c += 1
        result.append((lo, c))
        c += 1
    return result


def _analyze(value):
    original = _grid(value)
    colors, counts = np.unique(original, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    background = int(colors[counts.argmax()])
    launchers = _cores(original, background)
    if not 2 <= len(launchers) <= 4:
        raise ValueError("expected two to four full launchers")
    directions = {x["direction"] for x in launchers}
    structures = {x["structure"] for x in launchers}
    if len(directions) != 1 or len(structures) != 1:
        raise ValueError("all launchers must share their heading and structural role")
    turn = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}[directions.pop()]
    a = np.ascontiguousarray(np.rot90(original, turn))
    launchers = sorted(_cores(a, background), key=lambda x: x["c"])
    structure = structures.pop()
    footprint = np.zeros(a.shape, dtype=bool)
    injections = {}
    for obj in launchers:
        r, c = obj["r"], obj["c"]
        if footprint[r - 1:r + 3, c - 1:c + 3].any():
            raise ValueError("launcher footprints overlap")
        footprint[r - 1:r + 3, c - 1:c + 3] = True
        for row, col in obj["ports"]:
            injections.setdefault(row, {})[col] = obj["color"]
    if len({o["color"] for o in launchers}) != len(launchers):
        raise ValueError("payload colors must distinguish the launchers")
    outside = ~footprint & (a != background)
    if not np.all(a[outside] == structure):
        raise ValueError("all non-launcher objects must be structural bars")
    barrier = outside & (a == structure)
    rows = np.flatnonzero(barrier.any(axis=1))
    if len(rows) == 0 or np.any(np.diff(rows) < 3):
        raise ValueError("bar layers must be separated by at least two empty layers")
    if rows[0] < max(injections) + 1:
        raise ValueError("bars must be beyond every launcher")
    bars = {int(r): _runs(barrier[r]) for r in rows}
    for intervals in bars.values():
        if any(b[0] - a[1] < 4 for a, b in zip(intervals, intervals[1:])):
            raise ValueError("separate bars need at least three background cells between them")
    return {"a": a, "background": background, "structure": structure,
            "launchers": launchers, "footprint": footprint,
            "injections": injections, "bars": bars, "turn": turn}


def _simulate(info, variant="correct"):
    """The bar resets lateral distance; distance from a launcher is irrelevant.

    Variants are explicit wrong rules used only to audit demonstration strength.
    No execution trace is retained. The small statistics describe the final input.
    """
    a, bg = info["a"], info["background"]
    h, w = a.shape
    out = a.copy()
    active = {}
    stats = Counter()
    topology, lengths = [], []
    payload = info["launchers"][0]["color"]
    injections = {r: dict(cols) for r, cols in info["injections"].items()}
    if variant == "ignore_ports":
        for obj in info["launchers"]:
            # Incorrectly add an emission under the closed half of the side.
            r, c = obj["r"] + 3, obj["c"]
            open_columns = {col for _, col in obj["ports"]}
            injections.setdefault(r, {}).update({col: obj["color"] for col in (c, c + 1)
                                                if col not in open_columns})

    def paint(r, c, color):
        if a[r, c] != bg:
            raise ValueError("a path would overwrite input foreground")
        if out[r, c] not in (bg, color):
            raise ValueError("unresolved crossing outside a bar junction")
        out[r, c] = color

    for r in range(h):
        new = injections.get(r, {})
        if set(active) & set(new):
            raise ValueError("an emission overlaps an existing strand")
        active.update(new)
        intervals = info["bars"].get(r, [])
        hit_columns = {c for lo, hi in intervals for c in active if lo <= c <= hi}
        for c, color in active.items():
            if c not in hit_columns:
                paint(r, c, color)
        following = {c: color for c, color in active.items() if c not in hit_columns}
        layer = []
        for lo, hi in intervals:
            contacts = sorted((c, color) for c, color in active.items() if lo <= c <= hi)
            if not contacts:
                stats["inert_bars"] += 1
                if variant != "all_bars":
                    continue
                contacts = [((lo + hi) // 2, payload)]
            if variant == "ignore_barriers":
                following.update(contacts)
                continue
            # Family domain excludes an unaffected ray running along a face.
            if any(lo - 1 <= c <= hi + 1 for c in active if not lo <= c <= hi):
                raise ValueError("a crosspiece would intersect an unaffected strand")
            stats["reached_bars"] += 1
            lengths.append(hi - lo + 1)
            contact_colors = {color for _, color in contacts}
            stats["multicolor_junctions"] += len(contact_colors) >= 2
            stats["single_color_splits"] += len(contact_colors) == 1 and lo > 0 and hi < w - 1
            stats["same_color_multihits"] += len(contacts) > len(contact_colors)
            stats["clipped_junctions"] += lo == 0 or hi == w - 1
            stats["pruned_colors"] += len(contact_colors - {contacts[0][1], contacts[-1][1]})
            face = []
            for c in range(max(0, lo - 1), min(w, hi + 2)):
                distance = min(abs(c - x) for x, _ in contacts)
                nearest = [color for x, color in contacts if abs(c - x) == distance]
                if variant == "bar_midpoint":
                    nearest = ([contacts[0][1]] if 2 * c < lo + hi else
                               [contacts[-1][1]] if 2 * c > lo + hi else
                               [contacts[0][1], contacts[-1][1]])
                color = nearest[0] if len(set(nearest)) == 1 or variant == "fill_ties" else bg
                if color == bg:
                    stats["tie_cells"] += 1
                    _require(out[r - 1, c] == bg, "tie must lie on unused background")
                else:
                    paint(r - 1, c, color)
                face.append(color)
            stats["adjacent_color_meetings"] += sum(x != y and x != bg and y != bg
                                                    for x, y in zip(face, face[1:]))
            for c, color in ((lo - 1, contacts[0][1]), (hi + 1, contacts[-1][1])):
                if 0 <= c < w:
                    _require(c not in following, "two outgoing strands overlap")
                    following[c] = color
                    paint(r, c, color)
            if variant == "keep_interior":
                following.update(contacts)
            layer.append((len(contacts), len(contact_colors), int(lo == 0) + int(hi == w - 1)))
        if layer:
            stats["reached_layers"] += 1
            stats["parallel_junction_layers"] += len(layer) > 1
            topology.append(tuple(layer))
        active = following
    if variant == "collapse_payloads":
        out[out != a] = payload
    stats.update(launchers=len(info["launchers"]), ports=sum(map(len, info["injections"].values())),
                 double_ports=sum(len(x["ports"]) == 2 for x in info["launchers"]),
                 staggered=int(len({x["r"] for x in info["launchers"]}) > 1),
                 changed_cells=int(np.count_nonzero(out != a)))
    return np.ascontiguousarray(np.rot90(out, -info["turn"])), dict(stats), tuple(topology), tuple(lengths)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, emission and junction exclusively from input_grid."""
    return _simulate(_analyze(input_grid))[0]


def _advance(active, intervals, width):
    following = {c: color for c, color in active.items()
                 if not any(lo <= c <= hi for lo, hi in intervals)}
    for lo, hi in intervals:
        contacts = sorted((c, color) for c, color in active.items() if lo <= c <= hi)
        if not contacts:
            continue
        if lo:
            following[lo - 1] = contacts[0][1]
        if hi + 1 < width:
            following[hi + 1] = contacts[-1][1]
    return following


def _stage(rng, active, width, mode, clip=False):
    columns = sorted(active)
    cuts = [i for i in range(1, len(columns)) if columns[i] - columns[i - 1] >= 4]
    if mode == "gather":
        chosen = []
    elif mode == "separate":
        chosen = cuts
    else:
        chosen = [i for i in cuts if rng.random() < .6]
    if columns[0] == 0 and columns[-1] == width - 1 and not chosen:
        _require(bool(cuts), "no room to keep a continuing route")
        chosen = [max(cuts, key=lambda i: columns[i] - columns[i - 1])]
    boundaries = [0] + chosen + [len(columns)]
    result = []
    for start, stop in zip(boundaries, boundaries[1:]):
        left, right = columns[start], columns[stop - 1]
        lc = left if start == 0 else (left - columns[start - 1] - 3) // 2
        rc = width - right - 1 if stop == len(columns) else (columns[stop] - right - 3) // 2
        # Non-clipping layers retain both endpoints inside the canvas.
        if start == 0 and left > 0:
            lc = min(lc, left - 1)
        if stop == len(columns) and right < width - 1:
            rc = min(rc, width - right - 2)
        lo = left - int(rng.integers(max(0, lc) + 1))
        hi = right + int(rng.integers(max(0, rc) + 1))
        result.append((lo, hi))
    if clip:
        options = []
        if result[0][1] < width - 1:
            options.append("left")
        if result[-1][0] > 0:
            options.append("right")
        if options:
            if str(rng.choice(options)) == "left":
                result[0] = (0, result[0][1])
            else:
                result[-1] = (result[-1][0], width - 1)
    return result


def _make_input(rng, roles, witness, testing):
    bg, structure, pool = roles
    n = 2 if witness == 0 else 3 if witness == 1 else int(rng.integers(2, 5))
    width = int(rng.integers(max(16, 5 * n + 1), 31))
    stages = int(rng.integers(5, 7)) if testing else 3 if witness is not None else int(rng.integers(2, 5))
    first = int(rng.integers(9, min(12, 30 - 3 * (stages - 1) - 1)))
    minimum = first + 3 * (stages - 1) + 2
    extra_depth = int(rng.integers(min(6, 30 - minimum) + 1))
    spacing = 3 + rng.multinomial(extra_depth, np.full(stages - 1, 1 / (stages - 1)))
    event_rows = [first]
    for gap in spacing:
        event_rows.append(event_rows[-1] + int(gap))
    minimum += extra_depth
    height = int(rng.integers(minimum, min(30, minimum + 6) + 1))
    a = np.full((height, width), bg, dtype=np.int8)
    extra = width - (4 * n + (n - 1) + 2)
    gaps = rng.multinomial(extra, np.full(n + 1, 1 / (n + 1)))
    starts = [1 + int(gaps[0])]
    for i in range(1, n):
        starts.append(starts[-1] + 5 + int(gaps[i]))
    colors = list(map(int, rng.choice(pool, n, replace=False)))
    ports = []
    choices = []
    for i, left in enumerate(starts):
        double = witness != 0 and i >= (2 if witness == 1 else 0) and rng.random() < .55
        choices.append([1, 2] if double else [int(rng.integers(1, 3))])
    if witness in (0, 1):
        # Even contact spacing teaches a blank bisector; odd spacing teaches
        # adjacent different fronts. Both facts survive rotations/reflections.
        desired = 0 if witness == 0 else 1
        first_col = starts[0] + choices[0][0]
        choices[1] = [next(d for d in (1, 2) if (starts[1] + d - first_col) % 2 == desired)]
    tops = list(map(int, rng.integers(0, 3, n)))
    if witness == 0:
        tops[:2] = list(map(int, rng.choice(3, 2, replace=False)))
    for i, left in enumerate(starts):
        top = tops[i]
        a[top:top + 4, left:left + 4] = structure
        a[top + 1:top + 3, left + 1:left + 3] = colors[i]
        for offset in choices[i]:
            a[top + 3, left + offset] = bg
            ports.append((left + offset, colors[i]))
    active = dict(ports)
    if witness == 0:
        # Same-length reached/unreached singletons refute a size-based rule.
        free = [c for c in range(width) if c not in active and all(abs(c - x) > 1 for x in active)]
        a[first - 3, int(rng.choice(free))] = structure
    elif rng.random() < .6:
        free = [c for c in range(width) if c not in active]
        a[first - 3, int(rng.choice(free))] = structure
    for step, row in enumerate(event_rows):
        if witness == 0 and step == 0:
            intervals = [(c, c) for c in sorted(active)]
        else:
            mode = ("gather" if (witness == 0 and step == 1) or (witness != 0 and step == 0)
                    else "separate" if witness == 1 and step == 1
                    else str(rng.choice(["gather", "separate", "mixed"])))
            intervals = _stage(rng, active, width, mode,
                               clip=(step == stages - 1 and (witness is not None or rng.random() < .6)))
        for lo, hi in intervals:
            a[row, lo:hi + 1] = structure
        active = _advance(active, intervals, width)
        _require(bool(active), "construction unexpectedly lost all strands")
    return a


_SHORTCUTS = ("ignore_barriers", "ignore_ports", "collapse_payloads", "bar_midpoint",
              "fill_ties", "keep_interior", "all_bars")


def _different(info, output, variant):
    try:
        return not np.array_equal(_simulate(info, variant)[0], output)
    except (ValueError, AssertionError):
        # An invalid crossing under a shortcut is also evidence against it.
        return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    roles = (palette[0], palette[1], palette[2:])
    fixed_turn = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            for attempt in range(80):
                try:
                    a = _make_input(rng, roles, witness, split == "test")
                    if rng.integers(2):
                        a = np.fliplr(a)
                    turn = int(rng.integers(4)) if mixed else fixed_turn
                    a = np.ascontiguousarray(np.rot90(a, turn))
                    key = (a.shape, a.tobytes())
                    if key in seen:
                        continue
                    info = _analyze(a)
                    out, stats, _, _ = _simulate(info)
                    if witness == 0 and not (stats.get("tie_cells") and stats.get("single_color_splits")
                                             and stats.get("inert_bars") and stats.get("same_color_multihits")):
                        continue
                    if witness == 1 and not (stats.get("pruned_colors") and stats.get("adjacent_color_meetings")):
                        continue
                except (ValueError, AssertionError):
                    continue
                puzzle[split].append({"input": a, "output": out})
                seen.add(key)
                break
            else:
                raise RuntimeError("bounded geometry construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, palettes = set(), set()
    taught = Counter()
    refuted = set()
    staggered_tie = False
    train_depths, test_depths = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape and not np.array_equal(a, b), "invalid output shape or identity")
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            expected, stats, _, _ = _simulate(info)
            _require(np.array_equal(b, expected), "incorrect watershed detours")
            bg = info["background"]
            _require(np.array_equal(a[a != bg], b[a != bg]), "input foreground changed")
            payloads = {x["color"] for x in info["launchers"]}
            _require(set(map(int, b[a != b])) <= payloads, "new cells must retain a payload identity")
            palettes.add((bg, info["structure"]))
            _require(stats.get("multicolor_junctions", 0) > 0, "launchers never interact")
            if split == "train":
                taught.update(stats)
                staggered_tie |= bool(stats.get("staggered") and stats.get("tie_cells"))
                train_depths.append(stats["reached_layers"])
                refuted.update(v for v in _SHORTCUTS if _different(info, b, v))
            else:
                test_depths.append(stats["reached_layers"])
    _require(len(palettes) == 1, "background/structure conventions must be shared within a puzzle")
    for name in ("tie_cells", "adjacent_color_meetings", "pruned_colors", "single_color_splits",
                 "same_color_multihits", "clipped_junctions", "inert_bars", "parallel_junction_layers"):
        _require(taught[name] > 0, f"training lacks a visible {name} witness")
    _require(set(_SHORTCUTS) <= refuted, "training admits an intended counterfactual shortcut")
    _require(staggered_tie, "training lacks an unequal-launch-depth tie witness")
    _require(max(train_depths) >= 3 and min(test_depths) > max(train_depths),
             "training must teach iteration and tests must recombine more layers")


def self_test():
    """Bounded 42-seed structural/semantic checks; returns small aggregate data."""
    # Hand-specified pixel fixture: unequal launch distances, a blank bisector,
    # an asymmetric bar and then a clipped detour. Expected pixels are explicit.
    fixture = np.zeros((16, 16), dtype=np.int8)
    fixture[0:4, 1:5] = 1
    fixture[1:3, 2:4] = 2
    fixture[3, 2] = 0
    fixture[1:5, 9:13] = 1
    fixture[2:4, 10:12] = 4
    fixture[4, 10] = 0
    fixture[8, 1:12] = 1
    fixture[11, :5] = 1
    expected = fixture.copy()
    expected[3:7, 2] = 2
    expected[4:7, 10] = 4
    expected[7, :6] = 2
    expected[7, 7:13] = 4
    expected[8:11, 0] = 2
    expected[8:, 12] = 4
    expected[10, :6] = 2
    expected[11:, 5] = 2
    _require(expected[7, 6] == 0, "fixture seam was filled")
    for reflected in (False, True):
        a = np.fliplr(fixture) if reflected else fixture
        b = np.fliplr(expected) if reflected else expected
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn)),
                     "hand-specified detour fixture failed")
    aggregate = Counter()
    launcher_counts, double_port_counts, layer_counts, gap_sizes = set(), set(), set(), set()
    shapes, layouts, topologies, palettes, headings, lengths = set(), set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                out, stats, topology, spans = _simulate(info)
                _require(np.array_equal(out, pair["output"]), "solver mismatch")
                aggregate.update(stats)
                aggregate["pairs"] += 1
                launcher_counts.add(stats["launchers"])
                double_port_counts.add(stats["double_ports"])
                layer_counts.add(stats["reached_layers"])
                gap_sizes.update(map(int, np.diff(sorted(info["bars"]))))
                shapes.add(a.shape)
                headings.add(info["turn"])
                lengths.update(spans)
                topologies.add(topology)
                palettes.add((info["background"], info["structure"]))
                canonical = info["a"]
                normalized = np.zeros(canonical.shape, dtype=np.int8)
                normalized[canonical == info["structure"]] = 1
                for label, obj in enumerate(info["launchers"], 2):
                    normalized[canonical == obj["color"]] = label
                occupied = np.argwhere(normalized != 0)
                r0, c0 = occupied.min(axis=0)
                r1, c1 = occupied.max(axis=0) + 1
                crop = normalized[r0:r1, c0:c1]
                layouts.add((crop.shape, crop.tobytes()))
                signature.update(a.tobytes())
                # Solve is independent of NumPy array layout and caller storage.
                snapshot = a.copy()
                _require(np.array_equal(solve(a.tolist()), out), "nested-list solve mismatch")
                _require(np.array_equal(a, snapshot), "solve mutated its input")
                # Full nearest-color ownership is equivariant under rotation.
                if seed < 3:
                    for turn in range(4):
                        _require(np.array_equal(solve(np.rot90(a, turn)), np.rot90(out, turn)),
                                 "rotational equivariance failed")
        signatures.add(signature.digest())
    _require(len(signatures) == 42, "repeated puzzles across seeds")
    _require(len(shapes) >= 60 and len(layouts) >= 180, "geometry variation collapsed")
    _require(len(topologies) >= 50, "junction topology variation collapsed")
    _require(headings == set(range(4)) and len(palettes) >= 25, "role/orientation variation collapsed")
    _require(1 in lengths and max(lengths) >= 20 and len(lengths) >= 20, "bar lengths collapsed")
    _require(launcher_counts == {2, 3, 4} and max(double_port_counts) >= 2,
             "launcher/port counts collapsed")
    _require({2, 3, 4, 5, 6} <= layer_counts and len(gap_sizes) >= 4,
             "junction depths/spacings collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(983, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(983, 10, 4), generate_puzzle(983, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "whole-puzzle nondeterminism")
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")
    return {"seeds": 42, "pairs": aggregate["pairs"], "shapes": len(shapes),
            "color_normalized_cropped_layouts": len(layouts), "junction_topologies": len(topologies),
            "role_palettes": len(palettes), "headings": sorted(headings),
            "launcher_counts": sorted(launcher_counts), "double_port_counts": sorted(double_port_counts),
            "reached_layer_counts": sorted(layer_counts), "bar_layer_gaps": sorted(gap_sizes),
            "reached_bar_lengths": sorted(lengths), "feature_totals": dict(aggregate),
            "semantic_check": "generator-owned invariants and counterfactuals; no independent solver"}
