"""Hidden twins: symmetry transports the handedness of colored beam turners.

Standard library and NumPy only. Nothing is read from disk by this module.
All tracing records are bounded, transient data; no execution traces are saved.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
import hashlib

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
KINDS = ("H", "V", "P")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    """Eight-connected components, with a deterministic traversal order."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        pending, cells = [first], []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for dr, dc in product((-1, 0, 1), repeat=2):
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _kind(marker):
    r0, r1, c0, c1 = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r0 == r1 and len(marker) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(marker) == r1 - r0 + 1:
        return "V"
    raise AssertionError("guide must be a straight segment or a singleton")


def _image(cells, marker, kind):
    ar, ac = marker[0]
    if kind == "H":
        return [(2 * ar - r, c) for r, c in cells]
    if kind == "V":
        return [(r, 2 * ac - c) for r, c in cells]
    return [(2 * ar - r, 2 * ac - c) for r, c in cells]


def _eligible(payload, marker, kind):
    if min(max(abs(r - a), abs(c - b)) for r, c in payload for a, b in marker) != 1:
        return False
    r0, r1, c0, c1 = _bbox(payload)
    ar, ac = marker[0]
    if kind == "H":
        return (r1 < ar or r0 > ar) and not (c1 < min(c for r, c in marker) or c0 > max(c for r, c in marker))
    if kind == "V":
        return (c1 < ac or c0 > ac) and not (r1 < min(r for r, c in marker) or r0 > max(r for r, c in marker))
    # Closed quadrants allow payload cells aligned with the center's row/column.
    return (r1 <= ar or r0 >= ar) and (c1 <= ac or c0 >= ac)


def _header(grid):
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "invalid key band")
    rows = grid[1:bottom]
    _require(np.all(rows == rows[0]), "key rows must repeat")
    row = rows[0]
    _require(row[0] == row[-1] == frame, "missing panel walls")
    runs, c = [], 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _require(2 <= len(runs) <= 4 and all(b - a >= 3 for a, b in runs), "ambiguous panels")
    background = int(row[runs[0][0] + 1])
    _require(background != frame, "frame aliases background")
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == background), "inconsistent panel interiors")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == background) != (right == background), "ambiguous side bar")
        color, turn = (right, 1) if left == background else (left, -1)
        _require(color not in turns and color not in (background, frame), "conflicting key")
        turns[color] = turn
    return bottom + 1, frame, background, turns


def _decode(input_grid):
    grid = _grid(input_grid)
    crop, frame, background, turns = _header(grid)
    field = grid[crop:].copy()
    other = set(map(int, np.unique(field))) - {frame, background} - set(turns)
    _require(len(other) == 1, "one unkeyed emitter color is required")
    source = next(iter(other))
    points = np.argwhere(field == source)
    _require(len(points) == 1, "the emitter is a single cell")
    origin = tuple(map(int, points[0]))
    markers = _components(field == frame)
    _require(bool(markers), "missing symmetry guides")
    payloads = []
    for color in sorted(turns):
        payloads.extend((color, cells) for cells in _components(field == color))
    _require(len(payloads) == len(markers), "one payload per guide is required")
    used, occupied, groups = set(), set(), []
    h, w = field.shape
    for marker in markers:
        kind = _kind(marker)
        # Ownership is determined before checking whether a destination is empty.
        candidates = [i for i, (_, cells) in enumerate(payloads) if _eligible(cells, marker, kind)]
        _require(len(candidates) == 1, "ambiguous or missing payload ownership")
        index = candidates[0]
        _require(index not in used, "payload has two owners")
        used.add(index)
        color, cells = payloads[index]
        twin = _image(cells, marker, kind)
        _require(all(0 <= r < h and 0 <= c < w and field[r, c] == background for r, c in twin), "twin is not clear and in bounds")
        _require(not occupied.intersection(twin), "twins overlap")
        occupied.update(twin)
        groups.append(dict(kind=kind, marker=marker, payload=cells, twin=twin, color=color))
    _require(len(used) == len(payloads), "unowned payload")
    return dict(field=field, crop=crop, frame=frame, background=background,
                source=source, origin=origin, turns=turns, groups=groups)


def _materialize(info, include_twins=True):
    field = info["field"].copy()
    owners = np.full(field.shape, -1, dtype=np.int16)
    for i, g in enumerate(info["groups"]):
        for r, c in g["payload"]:
            owners[r, c] = 2 * i
        if include_twins:
            for r, c in g["twin"]:
                owners[r, c] = 2 * i + 1
                field[r, c] = g["color"]
    return field, owners


def _trace(info, *, parity=(-1, -1, 1), glyphs=(-1, 1), include_twins=True):
    """Trace one-cell beams against immutable geometry, never against new paint.

    Directions are N/E/S/W; -1 is relative left and +1 relative right.
    Guides are transparent annotations and retain drawing priority over paths.
    State count bounds every branch by 4*height*width without a retry loop.
    """
    field, owners = _materialize(info, include_twins)
    out = field.copy()
    h, w = field.shape
    paths = np.zeros((4, h, w), dtype=bool)
    events = []
    for branch in range(4):
        r, c = info["origin"]
        d, seen = branch, set()
        while True:
            state = r, c, d
            _require(state not in seen, "cyclic beam")
            seen.add(state)
            dr, dc = DIRECTIONS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            owner = int(owners[nr, nc])
            if owner >= 0:
                g = info["groups"][owner // 2]
                base = info["turns"][g["color"]]
                side = glyphs[0 if base == -1 else 1]
                if owner % 2:
                    side *= parity[KINDS.index(g["kind"])]
                nd = (d + side) % 4
                events.append((branch, owner // 2, bool(owner % 2), d, nd, r, c))
                d = nd
            else:
                r, c = nr, nc
                paths[branch, r, c] = True
                if field[r, c] == info["background"]:
                    out[r, c] = info["source"]
    return out, paths, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; the key frame also identifies the guide color."""
    return _trace(_decode(input_grid))[0]


def _orientation_witness(group):
    cells, twin = group["payload"], group["twin"]
    r0, _, c0, _ = _bbox(cells)
    a0, _, b0, _ = _bbox(twin)
    translation = {(r + a0 - r0, c + b0 - c0) for r, c in cells}
    return translation != set(twin)


def _evidence(info):
    out, paths, events = _trace(info)
    hits = {info["groups"][i]["kind"] for _, i, twin, *_ in events if twin}
    lengths = Counter(e[0] for e in events)
    chains = []
    for b in range(4):
        sequence = [e for e in events if e[0] == b]
        chains.append({info["groups"][e[1]]["kind"] for e in sequence if e[2]})
    return out, dict(twin_hits=hits, max_turns=max(lengths.values(), default=0),
                    mixed_chain=any(len(kinds) >= 2 for kinds in chains),
                    events=events, paths=paths)


def _hypotheses_fit(infos, outputs):
    fits = []
    for glyphs in product((-1, 1), repeat=2):
        for parity in product((-1, 1), repeat=3):
            try:
                if all(np.array_equal(_trace(info, parity=parity, glyphs=glyphs)[0], out)
                       for info, out in zip(infos, outputs)):
                    fits.append((glyphs, parity))
            except AssertionError:
                pass
    return fits


def _teaching_headings(infos):
    headings = {-1: set(), 1: set()}
    originals = set()
    for info in infos:
        for branch, i, twin, d, nd, r, c in _trace(info)[2]:
            side = -1 if (nd - d) % 4 == 3 else 1
            headings[side].add(d)
            if not twin:
                originals.add(info["turns"][info["groups"][i]["color"]])
    return originals == {-1, 1} and all(len(value) >= 2 for value in headings.values())


def _silhouette(rng, witness):
    for _ in range(24):
        h, w = map(int, rng.integers(2, 5, size=2))
        style = int(rng.integers(6))
        mask = np.zeros((h, w), dtype=bool)
        if style == 0:
            mask[:, 0] = True
            mask[-1, :] = True
        elif style == 1:
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            if h >= 3 and w >= 3:
                mask[0, 1] = False
        elif style == 2:
            col = int(rng.integers(w))
            for r in range(h):
                mask[r, col] = True
                col = min(w - 1, max(0, col + int(rng.choice((-1, 1)))))
        elif style == 3:
            mask[:] = True
            mask[0:int(rng.integers(1, h)), 0:int(rng.integers(1, w))] = False
        elif style == 4:
            mask[:] = True
        else:
            cells = {(int(rng.integers(h)), int(rng.integers(w)))}
            for step in range(int(rng.integers(3, h * w + 1))):
                frontier = sorted({(r + dr, c + dc) for r, c in cells
                                   for dr, dc in product((-1, 0, 1), repeat=2)
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(len(frontier)))])
            for q in cells:
                mask[q] = True
        r, c = np.nonzero(mask)
        mask = mask[r.min():r.max() + 1, c.min():c.max() + 1]
        if rng.random() < .5:
            mask = mask[::-1]
        if rng.random() < .5:
            mask = mask[:, ::-1]
        if not witness or (not np.array_equal(mask, mask[::-1]) and
                           not np.array_equal(mask, mask[:, ::-1]) and
                           not np.array_equal(mask, mask[::-1, ::-1])):
            return mask
    return np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)


def _prototype(rng, kind, witness=True):
    mask = _silhouette(rng, witness)
    payload = [(int(r) + 1, int(c)) for r, c in np.argwhere(mask)]
    if kind == "P":
        # A diagonal or offset corner; never an arbitrary center inside a shape.
        offset = int(rng.integers(0, 2))
        payload = [(r, c + offset) for r, c in payload]
        marker = [(0, 0)]
    else:
        contact = min(c for r, c in payload if r == 1)
        start = int(rng.integers(-1, contact + 1))
        end = int(rng.integers(max(start + 1, contact), max(start + 2, mask.shape[1] + 2)))
        marker = [(0, c) for c in range(start, end + 1)]
    sr, sc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
    payload = [(sr * r, sc * c) for r, c in payload]
    marker = [(sr * r, sc * c) for r, c in marker]
    if kind == "V":
        payload = [(c, r) for r, c in payload]
        marker = [(c, r) for r, c in marker]
    twin = _image(payload, marker, kind)
    r0, _, c0, _ = _bbox(marker + payload + twin)
    return dict(kind=kind, **{key: [(r - r0, c - c0) for r, c in cells]
                             for key, cells in (("marker", marker), ("payload", payload), ("twin", twin))})


def _scene(rng, h, w, kinds, turns):
    prototypes = [_prototype(rng, k, witness=k not in kinds[:i] or rng.random() < .5)
                  for i, k in enumerate(kinds)]
    blocked = np.zeros((h, w), dtype=bool)
    groups = []
    for p in sorted(prototypes, key=lambda p: -len(p["marker"] + p["payload"] + p["twin"])):
        footprint = p["marker"] + p["payload"] + p["twin"]
        _, r1, _, c1 = _bbox(footprint)
        locations = [(r, c) for r in range(h - r1) for c in range(w - c1)]
        rng.shuffle(locations)
        if rng.random() < .25:
            locations.sort(key=lambda q: min(q[0], q[1], h - r1 - 1 - q[0], w - c1 - 1 - q[1]))
        accepted = None
        for dr, dc in locations:
            if all(not blocked[r + dr, c + dc] for r, c in footprint):
                accepted = dr, dc
                break
        if accepted is None:
            return None
        dr, dc = accepted
        g = {key: [(r + dr, c + dc) for r, c in p[key]] for key in ("marker", "payload", "twin")}
        g.update(kind=p["kind"], color=int(rng.choice(sorted(turns))))
        groups.append(g)
        for r, c in footprint:
            r, c = r + dr, c + dc
            blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    field = np.zeros((h, w), dtype=np.int8)
    reserve = np.zeros((h, w), dtype=bool)
    for g in groups:
        for q in g["marker"]:
            field[q] = 8
        for q in g["payload"]:
            field[q] = g["color"]
        for q in g["marker"] + g["payload"] + g["twin"]:
            reserve[q] = True
    return dict(field=field, frame=8, background=0, source=1, turns=turns, groups=groups), reserve


def _source_candidates(rng, info, reserve, required):
    h, w = reserve.shape
    cells_by_kind = {k: [q for g in info["groups"] if g["kind"] == k for q in g["twin"]] for k in required}
    # Intersections of rows/columns of desired twins make two simultaneous
    # branches possible, without fixing a slot, source axis, or route template.
    candidates = set()
    if len(required) > 1:
        a, b = sorted(required)[:2]
        for r, c in cells_by_kind[a]:
            for x, y in cells_by_kind[b]:
                candidates.update(((r, y), (x, c)))
    for kind in sorted(required):
        for r, c in cells_by_kind[kind]:
            candidates.update((r, x) for x in (0, w - 1, int(rng.integers(w))))
            candidates.update((x, c) for x in (0, h - 1, int(rng.integers(h))))
    candidates = sorted(q for q in candidates if not reserve[q])
    rng.shuffle(candidates)
    return candidates[:90]


def _make_header(rng, width, turns):
    count = len(turns)
    widths = np.full(count, 5, dtype=int)
    widths += rng.multinomial(width - 5 * count, np.full(count, 1 / count))
    header = np.zeros((int(rng.integers(3, 6)), width), dtype=np.int8)
    header[[0, -1]] = 8
    c = 0
    for color, size in zip(rng.permutation(sorted(turns)), widths):
        color, size = int(color), int(size)
        header[:, c] = header[:, c + size - 1] = 8
        header[1:-1, c + 1 if turns[color] == -1 else c + size - 2] = color
        c += size
    return header


def _make_pair(rng, required, *, test=False, original_side=None):
    for attempt in range(100):
        nkeys = int(rng.integers(2, 5))
        w = int(rng.integers(max(18, 5 * nkeys), 31))
        h = int(rng.integers(18 if test else 15, 26))
        turns = {c: int(rng.choice((-1, 1))) for c in range(2, 2 + nkeys)}
        if original_side is not None:
            turns[int(rng.choice(sorted(turns)))] = original_side
        kinds = list(required)
        extra = int(rng.integers(1, 3 if not test else 4))
        kinds += [str(rng.choice(KINDS)) for _ in range(extra)]
        rng.shuffle(kinds)
        scene = _scene(rng, h, w, kinds, turns)
        if scene is None:
            continue
        info, reserve = scene
        for origin in _source_candidates(rng, info, reserve, required):
            info["origin"] = origin
            info["field"][origin] = 1
            try:
                out, evidence = _evidence(info)
                needed = set(required) <= evidence["twin_hits"]
                enough = evidence["max_turns"] >= (3 if test else 1)
                original_witness = original_side is None or any(
                    not twin and turns[info["groups"][i]["color"]] == original_side
                    for b, i, twin, d, nd, r, c in evidence["events"])
                if needed and enough and original_witness and (not test or evidence["mixed_chain"]):
                    header = _make_header(rng, w, turns)
                    palette = rng.permutation(10)
                    pair = {"input": palette[np.vstack((header, info["field"]))].astype(np.int8),
                            "output": palette[out].astype(np.int8)}
                    # All associations are re-derived from the resulting pixels.
                    decoded = _decode(pair["input"])
                    if np.array_equal(solve(pair["input"]), pair["output"]):
                        return pair, decoded
            except AssertionError:
                pass
            info["field"][origin] = 0
    raise AssertionError("bounded scene search exhausted")


def _fingerprint(a):
    return a.shape, a.tobytes()


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _checked_count(num_train, "num_train", 2, 10)
    nq = _checked_count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # The two lessons cover all marker types even at the minimum count. Their
    # locations, glyph tables, colors, shapes, and demonstration order vary.
    for curriculum_attempt in range(16):
        puzzle = {"train": [], "test": []}
        infos, seen = [], set()
        plans = [("H", "P"), ("V",)]
        plans += [tuple(rng.choice(KINDS, size=int(rng.integers(1, 3)), replace=False)) for _ in range(nt - 2)]
        for lesson, required in enumerate(plans):
            for _ in range(8):
                pair, info = _make_pair(rng, required, original_side=(-1, 1)[lesson] if lesson < 2 else None)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle["train"].append(pair)
                    infos.append(info)
                    break
            else:
                raise AssertionError("unique training search exhausted")
        outputs = [p["output"] for p in puzzle["train"]]
        if _hypotheses_fit(infos, outputs) != [((-1, 1), (-1, -1, 1))]:
            continue
        # Teach continuing after a collision before asking for longer routing.
        if max(_evidence(q)[1]["max_turns"] for q in infos) < 2:
            continue
        if not _teaching_headings(infos):
            continue
        for _ in range(nq):
            for _ in range(8):
                pair, info = _make_pair(rng, ("H", "V", "P"), test=True)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle["test"].append(pair)
                    break
            else:
                raise AssertionError("unique test search exhausted")
        rng.shuffle(puzzle["train"])
        validate_generated_puzzle(puzzle)
        return puzzle
    raise AssertionError("could not obtain distinguishing demonstrations")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, training, labels = set(), [], []
    witnessed, hit_kinds = set(), set()
    no_a_witness = no_b_witness = recursion = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _fingerprint(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(inp)
            expected, ev = _evidence(info)
            _require(np.array_equal(expected, out), "incorrect output")
            _require(out.shape == info["field"].shape, "wrong field crop")
            original = info["field"]
            _require(np.array_equal(out[original != info["background"]], original[original != info["background"]]), "original cells changed")
            completed, owners = _materialize(info)
            _require(np.array_equal(out[owners >= 0], completed[owners >= 0]), "payload colors changed")
            if split == "train":
                training.append(info)
                labels.append(out)
                witnessed.update(g["kind"] for g in info["groups"] if _orientation_witness(g))
                hit_kinds.update(ev["twin_hits"])
                recursion |= ev["max_turns"] >= 2
                # Counterfactual paths are compared only on original background,
                # excluding would-be twin cells, so mere missing copies cannot pass.
                try:
                    _, bare_paths, _ = _trace(info, include_twins=False)
                    clear = (original == info["background"]) & (owners < 0)
                    no_a_witness |= bool(np.any((bare_paths.any(axis=0) != ev["paths"].any(axis=0)) & clear))
                except AssertionError:
                    no_a_witness = True
                no_b_witness |= not np.array_equal(completed, out)
            else:
                _require(set(KINDS) <= ev["twin_hits"], "test must use every symmetry kind")
                _require(ev["max_turns"] >= 3 and ev["mixed_chain"], "test lacks route recombination")
    _require(set(KINDS) <= witnessed and set(KINDS) <= hit_kinds, "training hides a symmetry mechanism")
    _require(recursion and no_a_witness and no_b_witness, "training does not expose the interaction")
    _require(_teaching_headings(training), "training hides original glyphs or relative headings")
    _require(_hypotheses_fit(training, labels) == [((-1, 1), (-1, -1, 1))], "turn and parity conventions are not identifiable")


def self_test():
    """Bounded checks over 42 seeds; only feature sets, never a puzzle corpus."""
    shapes, layouts, silhouettes, counts, palettes, turns_seen, structural = set(), set(), set(), set(), set(), set(), set()
    features = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                before = inp.copy()
                predicted = solve(inp)
                _require(np.array_equal(predicted, pair["output"]), "solver mismatch")
                _require(np.array_equal(before, inp), "solve mutated input")
                info = _decode(inp)
                out, ev = _evidence(info)
                shapes.add(info["field"].shape)
                counts.add(len(info["groups"]))
                palettes.add((info["background"], info["frame"], info["source"]))
                layouts.add((info["field"].shape, info["origin"], tuple(tuple(g["marker"]) for g in info["groups"])))
                normalized, roles = [], {}
                for c in inp.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structural.add(hashlib.sha256(bytes(normalized)).digest())
                for g in info["groups"]:
                    r0, r1, c0, c1 = _bbox(g["payload"])
                    cells = tuple(sorted((r - r0, c - c0) for r, c in g["payload"]))
                    silhouettes.add(cells)
                    features[g["kind"]] += 1
                    features["diagonal"] += len(_four_components(g["payload"])) > 1
                    features["nonrectangle"] += len(cells) < (r1 - r0 + 1) * (c1 - c0 + 1)
                for b, i, twin, d, nd, r, c in ev["events"]:
                    turns_seen.add((d, (nd - d) % 4))
                    features["twin_hit"] += twin
                features["long_chain"] += ev["max_turns"] >= 4
                features["unhit_group"] += len({e[1] for e in ev["events"]}) < len(info["groups"])
                features["crossing"] += bool(np.any(ev["paths"].sum(axis=0) >= 2))
        # Also vary legal request sizes over the same fixed 42 seeds, including
        # minimum curricula and the maximum count boundaries, without retention.
        nt = 2 + seed % 9
        nq = 1 + (seed // 3) % min(4, nt - 1)
        validate_generated_puzzle(generate_puzzle(seed, nt, nq))
    _require(len(shapes) >= 30 and len(layouts) >= 190, "insufficient geometry variation")
    _require(len(silhouettes) >= 50 and len(structural) == 210, "insufficient shape variation")
    _require(len(counts) >= 4 and len(palettes) >= 40, "insufficient count/palette variation")
    _require(len(turns_seen) == 8, "missing relative turn headings")
    _require(all(features[k] > 0 for k in ("H", "V", "P", "diagonal", "nonrectangle", "twin_hit", "long_chain", "unhit_group", "crossing")), "missing family feature")
    # Reproducibility, array/list boundary, and a corrupted-label rejection.
    first, again = generate_puzzle(748339091, 2, 1), generate_puzzle(748339091, 2, 1)
    _require(all(np.array_equal(a[n], b[n]) for s in first for a, b in zip(first[s], again[s]) for n in a), "nondeterminism")
    listed = {s: [{n: a.tolist() for n, a in pair.items()} for pair in first[s]] for s in first}
    validate_generated_puzzle(listed)
    recolor = np.array([7, 0, 9, 1, 8, 2, 6, 3, 4, 5], dtype=np.int8)
    for pair in first["train"] + first["test"]:
        _require(np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]]), "palette roles depend on literal colors")
        # Reflecting the entire input also reflects the side bars in its key.
        # Both original and twin turn handedness therefore transform together.
        _require(np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"])), "spatial/key handedness is inconsistent")
    bad = {s: [{n: a.copy() for n, a in pair.items()} for pair in first[s]] for s in first}
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label was accepted")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Small geometric regression fixtures exercise finite axes, an aligned
    # point center, disconnected four-components, and actual parity at impact.
    _require(_image([(2, 3), (3, 4)], [(4, 3), (4, 4)], "H") == [(6, 3), (5, 4)], "finite axis mapping")
    _require(_image([(4, 6), (5, 5)], [(4, 5)], "P") == [(4, 4), (3, 5)], "offset center mapping")
    _require(len(_components(np.eye(4, dtype=bool))) == 1, "diagonal payload split")
    for kind in ("H", "P"):
        marker = [(4, 6), (4, 7)] if kind == "H" else [(4, 5)]
        payload = [(3, 6), (2, 6), (2, 7)] if kind == "H" else [(3, 4), (2, 4), (2, 3)]
        field = np.zeros((9, 12), dtype=np.int8)
        field[5, 1] = 1
        for q in marker:
            field[q] = 8
        for q in payload:
            field[q] = 2
        group = dict(kind=kind, marker=marker, payload=payload,
                     twin=_image(payload, marker, kind), color=2)
        info = dict(field=field, source=1, frame=8, background=0,
                    origin=(5, 1), turns={2: -1}, groups=[group])
        out, paths, events = _trace(info)
        event = next(e for e in events if e[0] == 1)
        _require(event[2] and event[3:5] == (1, 2 if kind == "H" else 0), "incorrect twin turn")
        _require(out[8 if kind == "H" else 0, 5] == 1, "missing routed endpoint")
        _require(all(out[q] == 8 for q in marker), "guide was painted")
    # A cyclic routing geometry must fail instead of hanging or silently stopping.
    field = np.zeros((10, 10), dtype=np.int8)
    field[4, 4] = 1
    cells = [(4, 7), (7, 6), (6, 3), (3, 4)]
    for q in cells:
        field[q] = 2
    info = dict(field=field, source=1, frame=8, background=0, origin=(4, 4),
                turns={2: 1}, groups=[dict(kind="P", marker=[(0, 0)], payload=cells, twin=[], color=2)])
    try:
        _trace(info, include_twins=False)
    except AssertionError as error:
        _require(str(error) == "cyclic beam", "wrong cycle rejection")
    else:
        raise AssertionError("cyclic beam accepted")
    # Pixel-derived ownership and copy collisions have separate negative checks.
    for markers, payloads, expected_error in (
            ([(4, 4), (4, 6)], [(5, 5), (3, 5)], "ambiguous or missing payload ownership"),
            ([(4, 4), (2, 2)], [(5, 5), (1, 1)], "twins overlap")):
        field = np.zeros((9, 18), dtype=np.int8)
        field[0, 17] = 1
        for q in markers:
            field[q] = 8
        for color, q in zip((2, 3), payloads):
            field[q] = color
        header = _make_header(np.random.default_rng(1), 18, {2: -1, 3: 1})
        try:
            _decode(np.vstack((header, field)))
        except AssertionError as error:
            _require(str(error) == expected_error, "wrong geometry rejection")
        else:
            raise AssertionError("ambiguous geometry accepted")
    return {"seeds": 42, "pairs": 210, "field_shapes": len(shapes), "layouts": len(layouts),
            "silhouettes": len(silhouettes), "structures": len(structural),
            "group_counts": sorted(counts), "role_triples": len(palettes), "features": dict(features)}


def _four_components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        pending, component = [root], []
        while pending:
            r, c = pending.pop()
            component.append((r, c))
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
        result.append(component)
    return result
