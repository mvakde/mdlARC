"""Printing plates on switchable rails: topology, sampled ink, and tile transport.

All solving conventions are visible in one input. No parent generator is imported.
"""
from itertools import combinations
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
             "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be in 0..9")
    return a.astype(np.int8, copy=True)


def _runs(row):
    start = 0
    for end in range(1, len(row) + 1):
        if end == len(row) or row[end] != row[start]:
            yield start, end, int(row[start])
            start = end


def _paths(mask):
    """Trace actual pixel adjacency, including contacts between couplers."""
    cells = set(map(tuple, np.argwhere(mask)))
    adjacency = {p: [q for dr, dc in _NEIGHBORS
                     if (q := (p[0] + dr, p[1] + dc)) in cells] for p in cells}
    _require(all(len(v) in (1, 2) for v in adjacency.values()), "branched or isolated rail")
    unseen, result = set(cells), []
    while unseen:
        start = min(unseen)
        component, todo = set(), [start]
        while todo:
            p = todo.pop()
            if p in component:
                continue
            component.add(p)
            todo.extend(adjacency[p])
        ends = sorted(p for p in component if len(adjacency[p]) == 1)
        _require(len(ends) == 2, "a rail must be an open path, not a cycle")
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [p for p in adjacency[current] if p != previous]
            _require(len(choices) == 1, "ambiguous path")
            previous, current = current, choices[0]
            path.append(current)
        _require(len(path) == len(component), "path does not cover its component")
        vertical_steps = {int(b[0] - a[0]) for a, b in zip(path, path[1:])
                          if b[0] != a[0]}
        _require(len(vertical_steps) <= 1, "internal gravity reversal")
        unseen.difference_update(component)
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _rewire(core, centers, k, background, track, gate):
    """Simultaneous thin U couplers, rather than solid, branching rectangles."""
    structural, writes, overlap = core.copy(), {}, 0
    for r, c in centers:
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                value = gate if dx == 0 else track if abs(dx) == k else background
                p = (r + dy, c + dx)
                if p in writes:
                    _require(writes[p] == value, "coupler writes conflict")
                    overlap += 1
                writes[p] = value
    for p, value in writes.items():
        structural[p] = value
    return structural, _paths(structural == track), overlap


def _slot(info, endpoint):
    r, c = endpoint
    rr = info["top"] + r - 1
    cc = info["left"] - 3 if c == 0 else info["left"] + info["width"]
    return (slice(rr, rr + 3), slice(cc, cc + 3))


def _analyze(input_grid):
    """Recover background, carrier, spacing, gates, plates and routes from pixels."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "a uniform one-cell background border is required")
    long_runs = [(r, lo, hi, color) for r, row in enumerate(a)
                 for lo, hi, color in _runs(row) if color != bg and hi - lo >= 9]
    _require(len(long_runs) >= 2, "missing parallel rails")
    lanes = [r for r, _, _, _ in long_runs]
    top, left, right, track = long_runs[0]
    _require(all((lo, hi, color) == (left, right, track)
                 for _, lo, hi, color in long_runs), "rail extents/carrier are ambiguous")
    _require(lanes == sorted(set(lanes)), "more than one long run on a lane")
    spacing = lanes[1] - lanes[0]
    _require(spacing in (4, 6) and all(b - x == spacing for x, b in zip(lanes, lanes[1:])),
             "lane spacing must be uniformly four or six")
    k, width, height = spacing // 2, right - left, lanes[-1] - top + 1
    _require(width >= 4 * k + 1, "rails are too short for a staggered pair of gates")
    _require(top >= 2 and lanes[-1] + 2 < a.shape[0] and left >= 4
             and right + 3 < a.shape[1], "plate slots must fit inside the border")
    core = a[top:top + height, left:right].copy()
    values = set(map(int, np.unique(core))) - {bg, track}
    _require(len(values) == 1, "exactly one visible gate color is required")
    gate = next(iter(values))
    expected_core = np.full(core.shape, bg, dtype=np.int8)
    expected_core[::spacing] = track
    centers, separator_counts = [], []
    for r in range(k, height, spacing):
        row_centers = []
        for lo, hi, color in _runs(core[r]):
            if color == bg:
                continue
            _require(color == gate and hi - lo == 2 * k + 1, "malformed divider bar")
            c = lo + k
            centers.append((r, c))
            row_centers.append(c)
            expected_core[r, lo:hi] = gate
        separator_counts.append(len(row_centers))
    _require(centers and np.array_equal(core, expected_core), "noise or malformed input rails")
    structural, paths, overlap = _rewire(core, centers, k, bg, track, gate)
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    expected_ends = {(r, c) for r in range(0, height, spacing) for c in (0, width - 1)}
    _require(endpoints == expected_ends and len(paths) == len(lanes),
             "every route must end at two outer plate slots")
    info = dict(input=a, background=bg, track=track, gate=gate, k=k,
                top=top, left=left, width=width, height=height, lanes=len(lanes),
                centers=centers, separator_counts=separator_counts, structural=structural,
                paths=paths, overlap=overlap)
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[top:top + height, left:right] = core
    plates = {}
    for p in sorted(endpoints):
        tile = a[_slot(info, p)].copy()
        if np.all(tile == bg):
            continue
        _require(bool(np.all(tile != bg)) and not np.any(tile == gate), "malformed plate")
        _require(np.count_nonzero(tile == track) >= 5, "plate carrier must be a majority")
        inward = (1, 2) if p[1] == 0 else (1, 0)
        ink = int(tile[inward])
        _require(ink not in (bg, track, gate), "inward-facing pixel must be a detail color")
        plates[p] = dict(tile=tile, ink=ink, side=int(p[1] != 0), inward=inward)
        expected[_slot(info, p)] = tile
    _require(plates and np.array_equal(a, expected), "noise outside the rail/plate system")
    transfers = []
    for path in paths:
        occupied = [p for p in (path[0], path[-1]) if p in plates]
        _require(len(occupied) <= 1, "a path may carry at most one printing plate")
        if not occupied:
            continue
        source = occupied[0]
        target = source if path[0][0] == path[-1][0] else max((path[0], path[-1]), key=lambda p: p[0])
        transfers.append(dict(source=source, target=target, path=path, **plates[source]))
    info.update(plates=plates, transfers=transfers)
    return info


def _dihedral(tile):
    return tuple(np.rot90(tile, j) for j in range(4)) + tuple(
        np.rot90(tile[:, ::-1], j) for j in range(4))


def _render(info, sample="inward", turn="half", always_turn=False, paint=True):
    out = info["input"].copy()
    r, c = info["top"], info["left"]
    out[r:r + info["height"], c:c + info["width"]] = info["structural"]
    for p in info["plates"]:
        out[_slot(info, p)] = info["background"]
    for t in info["transfers"]:
        ink = t["ink"] if sample == "inward" else int(t["tile"][1, 1])
        if paint:
            for y, x in t["path"]:
                out[r + y, c + x] = ink
        tile = t["tile"]
        if t["target"] != t["source"] or always_turn:
            if turn == "half":
                tile = tile[::-1, ::-1]
            elif turn == "reflection":
                tile = tile[:, ::-1]
        out[_slot(info, t["target"])] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction; the fixed movement direction is screen downward."""
    return _render(_analyze(input_grid))


def _layout(rng, lanes, width, k, mode):
    """Build interlacing couplers in connected groups; no internal breaks."""
    if mode == "cross":
        a = int(rng.integers(k, width - 3 * k))
        b = int(rng.integers(a + 2 * k, width - k))
        if rng.random() < .5:
            a, b = b, a
        return [(k, a), (3 * k, b)]
    if mode == "lesson":
        sep = int(rng.integers(lanes - 1))
        return [((2 * sep + 1) * k, int(rng.integers(k, width - k)))]
    if mode == "gaps":
        active = int(rng.integers(lanes - 1))
        return [((2 * active + 1) * k, int(rng.integers(k, width - k)))]
    blank = set()
    if mode == "mixed" and lanes > 2 and rng.random() < .55:
        blank.add(int(rng.integers(lanes - 1)))
    centers, previous = [], []
    for sep in range(lanes - 1):
        if sep in blank:
            previous = []
            continue
        # The end of this connected group bounds the number of bars that can
        # subsequently shrink without a U turn on its last lane.
        end = min([j for j in blank if j > sep] + [lanes - 1])
        remaining = end - sep
        if not previous:
            chosen = [int(rng.integers(k, width - k))]
        else:
            chosen = [int(rng.integers(a + 2 * k, b - 2 * k + 1))
                      for a, b in zip(previous, previous[1:])]
            exteriors = [(k, previous[0] - 2 * k),
                         (previous[-1] + 2 * k, width - k - 1)]
            for j in rng.permutation(2):
                lo, hi = exteriors[int(j)]
                if lo <= hi and len(chosen) < remaining and rng.random() < .72:
                    chosen.append(int(rng.integers(lo, hi + 1)))
            if not chosen:
                # A one-bar group may terminate early, leaving a level lane.
                previous = []
                continue
        previous = sorted(chosen)
        centers.extend(((2 * sep + 1) * k, c) for c in previous)
    if not centers:
        return _layout(rng, lanes, width, k, "gaps")
    return centers


def _tile(rng, track, details, side, witness=False, repeat=False, sample=None):
    tile = np.full((3, 3), track, dtype=np.int8)
    inward = (1, 0) if side else (1, 2)
    if sample is None:
        sample = int(rng.choice(details))
    auxiliary = int(rng.choice([x for x in details if x != sample]))
    tile[inward] = sample
    positions = [p for p in range(9) if p != 3 * inward[0] + inward[1]]
    if witness:
        # Three equal auxiliary pixels outnumber the sample, yet position alone
        # chooses ink. Select a mask with all eight distinct D4 images.
        masks = list(combinations(positions, 3))
        rng.shuffle(masks)
        for mask in masks:
            trial = tile.copy()
            trial.flat[list(mask)] = auxiliary
            if len({view.tobytes() for view in _dihedral(trial)}) == 8:
                return trial
        raise AssertionError("no asymmetric tile witness")
    rng.shuffle(positions)
    extra = int(rng.integers(4))
    if repeat:
        tile.flat[positions[0]] = sample
        positions = positions[1:]
        extra = min(extra, 2)
    for p in positions[:extra]:
        tile.flat[p] = int(rng.choice(details if rng.random() < .5 else [sample, auxiliary]))
    if repeat and np.array_equal(tile, tile[::-1, ::-1]):
        # Keep at least two sample pixels and at least five carrier pixels while
        # ensuring a retained plate distinguishes staying from turning.
        repaired = None
        for flat in range(9):
            if flat == 3 * inward[0] + inward[1]:
                continue
            for color in details:
                trial = tile.copy()
                trial.flat[flat] = color
                if (np.count_nonzero(trial == track) >= 5
                        and np.count_nonzero(trial == sample) >= 2
                        and not np.array_equal(trial, trial[::-1, ::-1])):
                    repaired = trial
                    break
            if repaired is not None:
                break
        assert repaired is not None
        tile = repaired
    return tile


def _make_pair(rng, k, palette, shift, split, lesson=None):
    bg, track, gate = palette
    details = [x for x in range(10) if x not in palette]
    if lesson:
        lanes, mode = (3 if lesson == "cross" else 4), lesson
    elif split == "test" and shift == "depth":
        lanes, mode = (int(rng.integers(5, 8)) if k == 2 else 5), "woven"
    elif split == "test" and shift == "density":
        lanes, mode = int(rng.integers(4, 8 if k == 2 else 6)), "mixed"
    elif split == "test":
        lanes, mode = int(rng.integers(5, 8 if k == 2 else 6)), "gaps"
    else:
        upper = (5 if k == 2 else 4) if shift == "depth" else 4 if shift == "gaps" else (8 if k == 2 else 6)
        lanes, mode = int(rng.integers(2, upper)), "mixed"
    width = int(rng.integers(4 * k + 1, 23))
    height = 2 * k * (lanes - 1) + 1
    centers = _layout(rng, lanes, width, k, mode)
    core = np.full((height, width), bg, dtype=np.int8)
    core[::2 * k] = track
    for r, c in centers:
        core[r, c - k:c + k + 1] = gate
    structural, paths, _ = _rewire(core, centers, k, bg, track, gate)
    assert len(paths) == lanes
    # At least one border cell, plus independently distributed spare margins.
    room_h, room_w = 30 - (height + 4), 30 - (width + 8)
    total_h, total_w = int(rng.integers(room_h + 1)), int(rng.integers(room_w + 1))
    before_h, before_w = int(rng.integers(total_h + 1)), int(rng.integers(total_w + 1))
    a = np.full((height + 4 + total_h, width + 8 + total_w), bg, dtype=np.int8)
    top, left = 2 + before_h, 4 + before_w
    a[top:top + height, left:left + width] = core
    geometry = dict(top=top, left=left, width=width)
    chosen = {}
    if lesson == "cross":
        for path in paths:
            if path[0][0] == 0:
                chosen[path[0]] = "witness"
    elif lesson == "lesson":
        active = [p for p in paths if p[0][0] != p[-1][0]]
        level = [p for p in paths if p[0][0] == p[-1][0]]
        chosen[active[0][-1]] = "repeat"
        rng.shuffle(level)
        chosen[level[0][0]] = "repeat"
        chosen[level[1][-1]] = "repeat"
    else:
        probability = 1.0 if split == "test" and shift == "density" else float(rng.uniform(.3, .95))
        indices = rng.permutation(len(paths))
        cap = 3 if split == "train" and shift == "density" else lanes
        for index in indices:
            if len(chosen) >= cap:
                break
            path = paths[int(index)]
            if rng.random() < probability:
                chosen[path[0] if rng.random() < .6 else path[-1]] = "normal"
        if not chosen:
            chosen[paths[0][0]] = "normal"
    witness_inks = list(map(int, rng.choice(details, 2, replace=False)))
    for p, kind in chosen.items():
        a[_slot(geometry, p)] = _tile(rng, track, details, int(p[1] != 0),
                                     witness=kind == "witness", repeat=kind == "repeat",
                                     sample=witness_inks.pop() if kind == "witness" else None)
    return {"input": a, "output": solve(a)}


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    n, m = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(n > m, "num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    k = int(rng.choice([2, 3]))
    shift = str(rng.choice(["depth", "density", "gaps"]))
    fixed_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.random() < .65)
    puzzle, seen = {"train": [], "test": []}, set()
    lessons = ["cross", "lesson"] + [None] * (n - 2)
    rng.shuffle(lessons)
    for split, count in (("train", n), ("test", m)):
        for index in range(count):
            for attempt in range(40):
                palette = tuple(map(int, rng.choice(10, 3, replace=False))) if vary_palette else fixed_palette
                pair = _make_pair(rng, k, palette, shift, split,
                                  lessons[index] if split == "train" else None)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    transfers = info["transfers"]
    moving = [t for t in transfers if t["source"] != t["target"]]
    return dict(k=info["k"], lanes=info["lanes"], width=info["width"],
                shape=info["input"].shape, gates=len(info["centers"]),
                separator_counts=tuple(info["separator_counts"]), overlap=info["overlap"],
                plates=len(transfers), moving=len(moving),
                cross=sum(t["source"][1] != t["target"][1] for t in moving),
                settled=sum(t["source"] == t["target"] and t["path"][0][0] != t["path"][-1][0]
                            for t in transfers),
                level=sum(t["path"][0][0] == t["path"][-1][0] for t in transfers),
                palette=(info["background"], info["track"], info["gate"]),
                repeated=sum(np.count_nonzero(t["tile"] == t["ink"]) > 1 for t in transfers))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen, training, queries, spacings = set(), [], [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info, b = _analyze(pair["input"]), _grid(pair["output"])
            a = info["input"]
            _require(a.shape == b.shape, "output shape changed")
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            spacings.add(info["k"])
            # Scalar checks of every route, card slot, and turned gate; they do
            # not just compare the saved output with another solve() invocation.
            expected = np.full(a.shape, info["background"], dtype=np.int8)
            expected[info["top"]:info["top"] + info["height"],
                     info["left"]:info["left"] + info["width"]] = info["structural"]
            active = {tuple(t["path"]): t for t in info["transfers"]}
            for path in info["paths"]:
                t = active.get(tuple(path))
                ink = info["track"] if t is None else t["ink"]
                for r, c in path:
                    expected[info["top"] + r, info["left"] + c] = ink
                if t is None:
                    continue
                slot = _slot(info, t["target"])
                rr, cc = slot[0].start, slot[1].start
                moving = t["source"] != t["target"]
                for y in range(3):
                    for x in range(3):
                        sy, sx = (2 - y, 2 - x) if moving else (y, x)
                        expected[rr + y, cc + x] = t["tile"][sy, sx]
            _require(np.array_equal(b, expected), "incorrect path ink, tile transport, gate, or background")
            _require(not np.array_equal(a, b), "identity pair")
            _require(set(map(int, np.unique(a))) == set(map(int, np.unique(b))), "palette not conserved")
            present = sum(not np.all(b[_slot(info, p)] == info["background"])
                          for path in info["paths"] for p in (path[0], path[-1]))
            _require(present == len(info["plates"]), "plate count not conserved")
            if split == "train":
                training.append(info)
            else:
                queries.append(info)
    _require(len(spacings) == 1, "generated puzzles keep one lane spacing")
    features = [_features(info) for info in training]
    _require(any(f["cross"] for f in features), "training must show routing across edges")
    _require(any(f["settled"] for f in features) and any(f["level"] for f in features),
             "training must show already settled and level plates")
    level_sides = {t["side"] for info in training for t in info["transfers"]
                   if t["path"][0][0] == t["path"][-1][0]}
    _require(level_sides == {0, 1}, "training must distinguish both fixed-side level tiebreaks")
    _require(any(f["repeated"] for f in features), "training must sample a repeated detail color")
    # Demonstrations identify the sampled coordinate and the full-tile transform
    # independently for plates entering from the left and from the right.
    positions, operations = [set(range(9)), set(range(9))], [set(range(8)), set(range(8))]
    moving_sides, inks = set(), set()
    for info in training:
        for t in info["transfers"]:
            side = t["side"]
            positions[side].intersection_update(i for i, v in enumerate(t["tile"].flat)
                                                if v == t["ink"])
            inks.add(t["ink"])
            if t["source"] != t["target"]:
                moving_sides.add(side)
                target = t["tile"][::-1, ::-1]
                operations[side].intersection_update(i for i, view in enumerate(_dihedral(t["tile"]))
                                                     if np.array_equal(view, target))
    _require(moving_sides == {0, 1} and positions == [{5}, {3}] and operations == [{2}, {2}],
             "training leaves sample position or half-turn ambiguous")
    _require(len(inks) >= 2, "training ink colors must vary")
    _require(any(not np.array_equal(_render(info), _render(info, turn="none")) for info in training),
             "training does not expose tile orientation")
    _require(any(not np.array_equal(_render(info), _render(info, sample="center")) for info in training),
             "sampling has no demonstrated effect")
    _require(any(not np.array_equal(_render(info), _render(info, always_turn=True)) for info in training),
             "always rotating settled plates is not distinguished")
    q_features = [_features(info) for info in queries]
    shifts = (
        all(q["lanes"] > max(f["lanes"] for f in features) for q in q_features),
        all(q["plates"] > max(f["plates"] for f in features) for q in q_features),
        all(q["separator_counts"].count(0) > max(f["separator_counts"].count(0) for f in features)
            for q in q_features),
    )
    _require(any(shifts), "missing demonstrated structural train/test shift")


def self_test():
    """Bounded 42-seed coverage, legal counts, corruption and input-only checks."""
    puzzles, features, normalized, signatures = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                features.append(_features(info))
                before = pair["input"].copy()
                predicted = solve(before)
                assert np.array_equal(before, pair["input"])
                assert np.array_equal(predicted, pair["output"])
                roles, norm = {}, []
                for value in before.flat:
                    norm.append(roles.setdefault(int(value), len(roles)))
                normalized.add((before.shape, tuple(norm)))
                signature.append((before.shape, before.tobytes(), predicted.tobytes()))
        signatures.add(tuple(signature))
    assert len(signatures) == 42 and len(normalized) >= 150
    assert len({f["shape"] for f in features}) >= 60
    assert {f["k"] for f in features} == {2, 3}
    assert {f["lanes"] for f in features} >= set(range(2, 8))
    assert {n for f in features for n in f["separator_counts"]} >= {0, 1, 2}
    assert any(f["gates"] >= 6 for f in features)
    assert any(f["overlap"] for f in features)
    assert any(f["plates"] >= 6 for f in features)
    assert all(any(f[name] for f in features) for name in ("cross", "settled", "level", "repeated"))
    assert all({f["palette"][i] for f in features} == set(range(10)) for i in range(3))
    assert any(30 in f["shape"] for f in features)
    a, b = generate_puzzle(314159, 6, 3), generate_puzzle(314159, 6, 3)
    assert all(np.array_equal(x[key], y[key]) for s in a for x, y in zip(a[s], b[s])
               for key in ("input", "output"))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(700 + n * 10 + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (3, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{key: value.copy() for key, value in pair.items()} for pair in a[s]] for s in a}
    info = _analyze(corrupt["test"][0]["input"])
    t = info["transfers"][0]
    r, c = t["path"][0]
    corrupt["test"][0]["output"][info["top"] + r, info["left"] + c] = info["track"]
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted path ink accepted")
    # Invalid occupancy: both terminal cards on one route must be rejected.
    bad = a["train"][0]["input"].copy()
    info = _analyze(bad)
    t = info["transfers"][0]
    other = t["path"][-1] if t["source"] == t["path"][0] else t["path"][0]
    tile = np.full((3, 3), info["track"], dtype=np.int8)
    tile[1, 2 if other[1] == 0 else 0] = t["ink"]
    bad[_slot(info, other)] = tile
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("two-plate collision accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
