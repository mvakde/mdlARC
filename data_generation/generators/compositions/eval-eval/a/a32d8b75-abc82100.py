"""Calibrated mosaic stamps. All output conventions are visible in each input.

Two framed reference cards supply a binary tile and a registered macro mask.
Remote copies of the mask's colored handle calibrate its rotation and pitch.
Only macro coordinates rotate; exchanged microtiles always remain upright.
"""
from __future__ import annotations

import numpy as np


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer ARC colors")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _rot(point, turns):
    r, c = map(int, point)
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _rings(a, color):
    rings = []
    for component in _components(a == color):
        rr, cc = zip(*component)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if min(r1 - r0, c1 - c0) < 3:
            continue
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                    if r in (r0, r1) or c in (c0, c1)}
        if component == expected:
            rings.append((r0, r1, c0, c1))
    return rings


def _interior(a, box):
    r0, r1, c0, c1 = box
    return a[r0 + 1:r1, c0 + 1:c1]


def _decode(input_grid):
    """Read background, frames, label order, exact mask and remote handles."""
    a = _array(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background mode is tied")
    bg = int(counts.argmax())
    candidates = []
    for frame in np.unique(a):
        frame = int(frame)
        if frame == bg:
            continue
        rings = _rings(a, frame)
        if len(rings) != 2:
            continue
        for tile_box, macro_box in (rings, rings[::-1]):
            tile, field = _interior(a, tile_box), _interior(a, macro_box)
            colors = list(map(int, np.unique(tile)))
            if tile.shape[0] != tile.shape[1] or not 2 <= len(tile) <= 4:
                continue
            if len(colors) != 2 or bg in colors or frame in colors:
                continue
            if min(field.shape) < 3:
                continue
            labels = np.argwhere(np.isin(field, colors))
            neutral = set(map(int, np.unique(field))) - {bg, frame, *colors}
            if len(labels) != 2 or len(neutral) != 1 or np.any(field == frame):
                continue
            if any(np.count_nonzero(field == c) != 1 for c in colors):
                continue
            readings = []
            for s, t in ((labels[0], labels[1]), (labels[1], labels[0])):
                delta = tuple(map(int, t - s))
                if delta not in _DIRS:
                    continue
                q, before = s + 2 * np.array(delta), s - np.array(delta)
                inside = lambda p: 0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1]
                if inside(q) and not inside(before) and int(field[tuple(q)]) in {bg, *neutral}:
                    readings.append((tuple(map(int, q)), _DIRS.index(delta),
                                     int(field[tuple(s)]), int(field[tuple(t)])))
            if len(readings) != 1:
                continue
            q, direction, source, target = readings[0]
            erase = np.zeros(a.shape, bool)
            for r0, r1, c0, c1 in rings:
                erase[r0:r1 + 1, c0:c1 + 1] = True
            ink = next(iter(neutral))
            if np.any(((a == frame) | (a == ink)) & ~erase):
                continue
            candidates.append(dict(grid=a, bg=bg, frame=frame, neutral=ink,
                                   tile=tile.copy(), mask=field == ink, q=q,
                                   direction=direction, source=source, target=target,
                                   erase=erase, boxes=(tile_box, macro_box)))
    _require(len(candidates) == 1, "reference cards are missing or ambiguous")
    model = candidates[0]
    source, target, erase = model["source"], model["target"], model["erase"]
    tails = list(map(tuple, np.argwhere((a == source) & ~erase)))
    heads = list(map(tuple, np.argwhere((a == target) & ~erase)))
    _require(tails and len(tails) == len(heads), "remote handle colors are unbalanced")
    n = len(model["tile"])
    handles, used = [], set()
    for s in tails:
        choices = []
        for t in heads:
            dr, dc = t[0] - s[0], t[1] - s[1]
            if (dr == 0) == (dc == 0):
                continue
            pitch = abs(dr) + abs(dc)
            if pitch < n:
                continue
            unit = (dr // pitch, dc // pitch)
            choices.append((t, pitch, _DIRS.index(unit)))
        _require(len(choices) == 1, "remote tail does not have one aligned head")
        t, pitch, direction = choices[0]
        _require(t not in used, "remote handles share a head")
        used.add(t)
        handles.append(dict(s=s, t=t, pitch=pitch,
                            turn=(direction - model["direction"]) % 4, direction=direction))
    # Degree one on both sides is stronger than merely one perfect matching.
    for t in heads:
        aligned = [s for s in tails if ((s[0] == t[0]) != (s[1] == t[1]))
                   and abs(s[0] - t[0]) + abs(s[1] - t[1]) >= n]
        _require(len(aligned) == 1, "remote head has competing tails")
    model["handles"] = handles
    return model


def _render(model, variant="normal"):
    """Paint from the original handles. Variants audit concrete shortcuts."""
    a, n, mask = model["grid"], len(model["tile"]), model["mask"]
    tile = model["tile"]
    colors = np.unique(tile)
    exchanged = colors.sum() - tile
    out = a.copy()
    erased = model["erase"].copy()
    for handle in model["handles"]:
        erased[handle["s"]] = erased[handle["t"]] = True
    out[erased] = model["bg"]
    paints, visits, clipped, complete_blocks = {}, {}, 0, 0
    for index, handle in enumerate(model["handles"]):
        s, t = np.array(handle["s"]), np.array(handle["t"])
        p, k = handle["pitch"], handle["turn"]
        q_ext = 2 * t - s
        q = model["q"]
        if variant == "at_source": q_ext = s
        if variant == "at_target": q_ext = t
        if variant == "center_registration": q = tuple((d - 1) // 2 for d in mask.shape)
        if variant == "constant_pitch":
            p = n
            q_ext = s + 2 * p * np.array(_DIRS[handle["direction"]])
        if variant == "no_macro_rotation": k = 0
        active = np.ones(mask.shape, bool) if variant == "fill_rectangle" else mask
        block = tile if variant == "no_exchange" else exchanged
        if variant == "rotate_tile": block = np.rot90(exchanged, -handle["turn"])
        if variant == "one_color": block = np.full(tile.shape, model["target"], int)
        for r, c in np.argwhere(active):
            dr, dc = _rot((r - q[0], c - q[1]), k)
            top, left = map(int, q_ext + p * np.array((dr, dc)))
            is_complete = 0 <= top and 0 <= left and top + n <= a.shape[0] and left + n <= a.shape[1]
            complete_blocks += int(is_complete)
            for u in range(n):
                for v in range(n):
                    point = top + u, left + v
                    if not (0 <= point[0] < a.shape[0] and 0 <= point[1] < a.shape[1]):
                        clipped += 1
                        continue
                    color = int(block[u, v])
                    if variant == "normal":
                        _require(point not in paints or paints[point] == color,
                                 "overlapping stamps disagree")
                    paints[point] = color
                    visits.setdefault(point, set()).add(index)
    for point, color in paints.items():
        out[point] = color
    stats = dict(paints=paints, erased=erased, clipped=clipped,
                 complete_blocks=complete_blocks,
                 visible_handles=len(set().union(*visits.values())) if visits else 0,
                 overlap=sum(len(v) > 1 for v in visits.values()),
                 covered_pins=sum(p in paints for h in model["handles"] for p in (h["s"], h["t"])))
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a valid family input with no seed or training examples."""
    return _render(_decode(input_grid))[0]


def _tile(rng, n, colors, witness):
    for _ in range(80):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if not bits.any() or bits.all():
            continue
        if witness and any(np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            continue
        return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _macro(rng, shape, bg, neutral, source, target, witness):
    h, w = shape
    qr = 1 if witness else int(rng.integers(1, h - 1))
    q = qr, 2
    rr, cc = np.indices(shape)
    style = int(rng.integers(6))
    if style == 0:
        mask = rng.random(shape) < rng.uniform(.2, .7)
    elif style == 1:
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        mask &= rng.random(shape) > .12
    elif style == 2:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(2, w)))
    elif style == 3:
        mask = ((rr + cc) % int(rng.integers(2, 4)) == 0)
    elif style == 4:
        mask = np.abs(rr - qr) + np.abs(cc - 2) == int(rng.integers(1, max(h, w)))
    else:
        mask = np.zeros(shape, bool)
        c = int(rng.integers(w))
        for r in range(h):
            nc = int(np.clip(c + rng.integers(-1, 2), 0, w - 1))
            mask[r, min(c, nc):max(c, nc) + 1] = True
            c = nc
    mask[qr, :2] = False
    if witness == 1:
        mask[q] = True
        mask[0, 0], mask[0, 1], mask[2, 3] = True, False, True
    elif witness == 2:
        # A missing registration, compatible overlap, and another stamp's pins
        # beneath the first stamp are all visible in this teaching input.
        mask[q] = False
        mask[0, 1], mask[1, 3], mask[2, 2], mask[2, 3] = True, True, True, True
        mask[0, 0] = False
    if mask.sum() < 2:
        mask[0, -1] = mask[-1, -1] = True
    field = np.where(mask, neutral, bg).astype(int)
    field[qr, 0], field[qr, 1] = source, target
    return field


def _frame(interior, color):
    return np.pad(interior, 1, constant_values=color)


def _place_cards(rng, a, cards, bg):
    occupied = np.zeros(a.shape, bool)
    for card in cards:
        h, w = card.shape
        choices = [(r, c) for r in range(a.shape[0] - h + 1)
                   for c in range(a.shape[1] - w + 1)]
        rng.shuffle(choices)
        for r, c in choices:
            # One empty cell between frames keeps their border components separate.
            halo = occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1]
            if halo.any():
                continue
            a[r:r + h, c:c + w] = card
            occupied[r:r + h, c:c + w] = True
            break
        else:
            raise ValueError("cards do not fit")
    return occupied


def _geometry(model):
    # Express geometry relative to an east-pointing handle, independent of color.
    m = np.rot90(model["mask"], model["direction"])
    return _key(m)


def _try_pair(rng, index, test, forbidden):
    witness = 0 if test or index >= 2 else index + 1
    n = (2 if witness == 1 else 3 if witness == 2 else int(rng.integers(2, 5)))
    shape = (3, 4) if witness else (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
    bg, frame, neutral, source, target = map(int, rng.permutation(10)[:5])
    tile = _tile(rng, n, (source, target), bool(witness))
    field = _macro(rng, shape, bg, neutral, source, target, witness)
    reference_turn = int(rng.integers(4))
    field = np.rot90(field, -reference_turn)
    h = int(rng.integers(24 if witness else 17, 31))
    w = int(rng.integers(24 if witness else 17, 31))
    a = np.full((h, w), bg, int)
    reserved = _place_cards(rng, a, [_frame(tile, frame), _frame(field, frame)], bg)
    # The reference direction is east turned reference_turn times clockwise.
    count = 2 if witness else int(rng.integers(1, 5))
    canonical_specs = ([(n, 0), (n + 1, 1)] if witness == 1 else
                       [(n, 0), (n, 0)] if witness == 2 else
                       [(n + int(rng.integers(2, 4)) if test and i == 0 else
                         n + int(rng.integers(2)), int(rng.integers(4))) for i in range(count)])
    positions = [(r, c) for r in range(h) for c in range(w)]
    previous = None
    final_model = None
    for i, (pitch, direction) in enumerate(canonical_specs):
        dr, dc = _DIRS[direction]
        rng.shuffle(positions)
        choices = positions
        if witness == 2 and i == 1:
            shift = _rot((n, 2 * n), direction)
            choices = [(previous[0] + shift[0], previous[1] + shift[1])]
        accepted = False
        for s in choices:
            t = s[0] + pitch * dr, s[1] + pitch * dc
            if not (0 <= s[0] < h and 0 <= s[1] < w and 0 <= t[0] < h and 0 <= t[1] < w):
                continue
            if reserved[s] or reserved[t] or a[s] != bg or a[t] != bg:
                continue
            a[s], a[t] = source, target
            try:
                model = _decode(a)
                out, stats = _render(model)
                if witness:
                    # Require each teaching stamp to be fully in bounds.
                    _require(stats["clipped"] == 0, "teaching stamp is clipped")
                _require(stats["complete_blocks"] > 0 or not witness, "no complete teaching block")
                _require(stats["paints"], "all stamps are outside the grid")
                _require(stats["visible_handles"] == i + 1, "a handle has no visible stamp")
                if witness == 2 and i == 1:
                    _require(stats["overlap"] > 0 and stats["covered_pins"] >= 2,
                             "overlap witness was lost")
            except ValueError:
                a[s] = a[t] = bg
                continue
            previous, final_model, accepted = s, model, True
            break
        if not accepted:
            raise ValueError("remote handles do not fit")
    _require(_geometry(final_model) not in forbidden, "test repeats a training macro mask")
    out, stats = _render(final_model)
    # Inert content includes small groups as well as isolated pixels. Put it
    # outside paint so preservation has an observable witness.
    pool = [c for c in range(10) if c not in (bg, frame, neutral, source, target)]
    rng.shuffle(pool)
    for color in pool[:int(rng.integers(3))]:
        rng.shuffle(positions)
        used = 0
        desired = int(rng.integers(1, 7))
        for p in positions:
            if used >= desired:
                break
            if reserved[p] or p in stats["paints"] or a[p] != bg:
                continue
            a[p] = color
            used += 1
    model = _decode(a)
    out, stats = _render(model)
    if witness:
        for variant in ("no_exchange", "rotate_tile", "no_macro_rotation", "at_source",
                        "at_target", "center_registration", "fill_rectangle", "one_color"):
            _require(not np.array_equal(out, _render(model, variant)[0]),
                     "uninformative teaching pair: " + variant)
        if witness == 1:
            _require(not np.array_equal(out, _render(model, "constant_pitch")[0]),
                     "uninformative pitch witness")
    return {"input": a, "output": out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    _require(low <= int(value) <= high, name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    _require(nt > nq, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen, train_geometry = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(120):
                try:
                    pair = _try_pair(rng, index, split == "test",
                                     train_geometry if split == "test" else set())
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_geometry.add(_geometry(_decode(pair["input"])))
                break
            else:
                raise RuntimeError("bounded generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, training, test_models = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _array(pair["input"]), _array(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            model = _decode(a)
            predicted, stats = _render(model)
            _require(np.array_equal(out, predicted), "output is not the calibrated mosaic transform")
            _require(a.shape == out.shape and stats["paints"], "empty or resized output")
            _require(stats["visible_handles"] == len(model["handles"]), "a stamp is wholly clipped")
            uncovered = ~stats["erased"]
            for p in stats["paints"]:
                uncovered[p] = False
            _require(np.array_equal(a[uncovered], out[uncovered]), "uncovered content changed")
            (training if split == "train" else test_models).append((model, out, stats))
    for variant in ("no_exchange", "rotate_tile", "no_macro_rotation", "constant_pitch",
                    "at_source", "at_target", "center_registration", "fill_rectangle", "one_color"):
        _require(any(not np.array_equal(out, _render(model, variant)[0])
                     for model, out, _ in training), "training permits shortcut: " + variant)
    _require(any(not m["mask"][m["q"]] for m, _, _ in training), "training lacks a registration hole")
    _require(any(s["overlap"] and s["covered_pins"] >= 2 for _, _, s in training),
             "training lacks simultaneous overlapping stamps")
    geometry = {_geometry(m) for m, _, _ in training}
    for m, _, _ in test_models:
        _require(_geometry(m) not in geometry, "test geometry was rehearsed")
        _require(any(h["pitch"] > len(m["tile"]) + 1 for h in m["handles"]),
                 "test lacks the spacing extrapolation")


def self_test():
    """Bounded 42-seed variation audit plus semantic and interface regressions."""
    # Hand-calculated fixture: hole registration, pitch 3 with a 2x2 tile,
    # a quarter-turn, clipping and an inert pixel. Expected cells are explicit;
    # no rendering routine is used to construct these regression answers.
    for quarter in (False, True):
        a = np.zeros((15, 17), int)
        a[0:4, 0:4] = _frame(np.array([[1, 1], [2, 1]]), 9)
        a[0:5, 6:12] = _frame(np.array([[8, 0, 0, 8], [1, 2, 0, 8], [0, 0, 8, 8]]), 9)
        a[8, 8] = 1
        a[11, 8] = 2 if quarter else 0
        a[8, 11] = 0 if quarter else 2
        a[14, 16] = 3
        expected = np.zeros_like(a)
        expected[14, 16] = 3
        if quarter:
            expected[8:10, 11:13] = [[2, 2], [1, 2]]
            expected[14, 5:7] = [2, 2]
        else:
            expected[5:7, 8:10] = [[2, 2], [1, 2]]
            expected[11:13, 14:16] = [[2, 2], [1, 2]]
        _require(np.array_equal(solve(a), expected), "hand-calculated registration regression failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    puzzle_keys = {tuple((_key(p["input"]), _key(p["output"]))
                         for s in ("train", "test") for p in puzzle[s]) for puzzle in puzzles}
    _require(len(puzzle_keys) == 42, "seed diversity failed")
    features = []
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                model = _decode(pair["input"])
                out, stats = _render(model)
                _require(np.array_equal(solve(pair["input"].copy()), out), "input-only solve failed")
                features.append((split, model, stats))
    sizes = {len(m["tile"]) for _, m, _ in features}
    masks = {_geometry(m) for _, m, _ in features}
    dimensions = {m["grid"].shape for _, m, _ in features}
    reference_directions = {m["direction"] for _, m, _ in features}
    directions = {h["direction"] for _, m, _ in features for h in m["handles"]}
    handle_counts = {len(m["handles"]) for _, m, _ in features}
    pitches = {h["pitch"] for _, m, _ in features for h in m["handles"]}
    _require(sizes == {2, 3, 4} and directions == reference_directions == set(range(4)), "geometric variation failed")
    _require(handle_counts == {1, 2, 3, 4} and len(pitches) >= 5, "handle variation failed")
    _require(len(masks) >= 65 and len(dimensions) >= 50, "structural variation failed")
    _require(any(s["clipped"] for _, _, s in features), "clipping never occurs")
    for split in ("train", "test"):
        fs = [(m, s) for sp, m, s in features if sp == split]
        _require({m["bg"] for m, _ in fs} == set(range(10)), "background palette variation failed")
        _require({len(m["tile"]) for m, _ in fs} == {2, 3, 4}, "tile sizes missing in split")
        _require(any(not m["mask"][m["q"]] for m, _ in fs), "hole registration missing in split")
    a, b = generate_puzzle(867, 2, 1), generate_puzzle(867, 2, 1)
    _require(all(np.array_equal(x[r], y[r]) for s in a for x, y in zip(a[s], b[s])
                 for r in ("input", "output")), "determinism failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(31, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "count handling failed")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{r: a.copy() for r, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong answer was accepted")
    return dict(seeds=42, pairs=len(features), macro_geometries=len(masks), dimensions=len(dimensions),
                tile_sizes=sorted(sizes), handle_counts=sorted(handle_counts), pitches=sorted(pitches),
                clipped_pairs=sum(bool(s["clipped"]) for _, _, s in features),
                overlapping_pairs=sum(bool(s["overlap"]) for _, _, s in features),
                pins_covered_pairs=sum(bool(s["covered_pins"]) for _, _, s in features))
