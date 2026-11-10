"""Mirror loom: dock containers and periodically reflect eligible chamber tiles.

Only Python's standard library and NumPy are used. The public solver reads one
input, not a puzzle, seed, generator record, or learned color convention.
"""
from __future__ import annotations

import itertools
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_REGIMES = ("combine", "more_rooms", "longer_word", "short_band")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    pending = set(zip(*np.nonzero(mask)))
    groups = []
    while pending:
        point = min(pending)
        pending.remove(point)
        group, stack = {point}, [point]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(min(cc)), int(max(rr) + 1), int(max(cc) + 1)


def _holes(mask):
    h, w = mask.shape
    return [g for g in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in g)]


def _period(tile):
    """Primitive whole-column word; a prefix period must divide the room width."""
    w = tile.shape[1]
    return next(p for p in range(1, w + 1)
                if w % p == 0 and np.array_equal(tile, tile[:, np.arange(w) % p]))


def _objects(canonical, ink, bg, rail):
    work = canonical[:, :rail]
    objects, claimed = [], np.zeros(work.shape, dtype=bool)
    previous_end = -1
    for cells in _components(work == ink):
        r0, c0, r1, c1 = _box(cells)
        _require(r0 > previous_end, "containers need a blank separating row")
        _require(r0 > 0 and r1 < len(work), "containers need exterior margins")
        previous_end = r1
        patch = work[r0:r1, c0:c1].copy()
        mask = patch == ink
        _require(int(mask.sum()) == len(cells), "overlapping container boxes")
        claimed[r0:r1, c0:c1] = True
        chambers = []
        for hole in _holes(mask):
            a, b, z, d = _box(hole)
            _require(len(hole) == (z - a) * (d - b), "closed chambers must be rectangular")
            chambers.append(dict(row=a, col=b, tile=patch[a:z, b:d].copy(),
                                 active=d == mask.shape[1] - 1))
        objects.append(dict(row=r0, col=c0, patch=patch, mask=mask,
                            shift=rail - c1, chambers=chambers))
    _require(bool(objects), "no containers")
    _require(bool(np.all(claimed | (work == bg))), "payload has no owning container")
    return objects


def _decode(input_grid):
    grid = _grid(input_grid)
    choices = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        h, w = g.shape
        for rail in range(1, w - 1):
            ink, bg = int(g[0, rail]), int(g[0, -1])
            if ink == bg or not np.all(g[:, rail] == ink):
                continue
            if not np.all(g[:, rail + 1:] == bg):
                continue
            try:
                objs = _objects(g, ink, bg, rail)
                choices.append(dict(grid=g, turn=turn, ink=ink, bg=bg,
                                    rail=rail, objects=objs))
            except AssertionError:
                continue
    _require(len(choices) == 1, "rail/work direction must have one geometric interpretation")
    return choices[0]


def _render(scene, variant="rule"):
    """Input-derived rendering, including explicit wrong-rule ablations."""
    g, rail, bg = scene["grid"], scene["rail"], scene["bg"]
    out = g.copy()
    out[:, :rail] = bg
    length = g.shape[1] - rail - 1
    for obj in scene["objects"]:
        patch, row = obj["patch"], obj["row"]
        start = obj["col"] if variant == "no_motion" else rail - patch.shape[1]
        out[row:row + len(patch), start:start + patch.shape[1]] = patch
        if variant == "local_column":
            # Intentionally wrong: inspect behind the extreme wall without
            # determining whether the non-ink run reaches the exterior.
            m = obj["mask"]
            for r in range(len(m)):
                if m.shape[1] < 3 or not m[r, -1] or m[r, -2]:
                    continue
                left = m.shape[1] - 2
                while left > 0 and not m[r, left - 1]:
                    left -= 1
                word = patch[r, left:-1][::-1]
                out[row + r, rail + 1:] = word[np.arange(length) % len(word)]
            continue
        for room in obj["chambers"]:
            if not room["active"] and variant != "all_closed":
                continue
            if variant == "moving_only" and obj["shift"] == 0:
                continue
            tile = room["tile"]
            word = tile if variant == "unreflected" else tile[:, ::-1]
            if variant == "flat":
                word = word[:, :1]
            elif variant == "prefix":
                p = next(p for p in range(1, word.shape[1] + 1)
                         if np.array_equal(word, word[:, np.arange(word.shape[1]) % p]))
                word = word[:, :p]
            phase = obj["shift"] if variant == "travel_phase" else 0
            values = word[:, (np.arange(length) + phase) % word.shape[1]]
            r = row + room["row"]
            out[r:r + len(tile), rail + 1:] = values
    return np.ascontiguousarray(np.rot90(out, -scene["turn"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output solely from this input's visible geometry."""
    return _render(_decode(input_grid))


def _integer(rng, low, high):
    return int(rng.integers(low, high + 1))


def _tile(rng, h, w, palette, bg, style=None):
    a, b = palette[:2]
    c = palette[2 % len(palette)]
    if style == "aba":
        return np.array([[a, b, a]], dtype=np.int8)
    if style == "vector":
        t = np.full((h, 1), a, dtype=np.int8)
        t[-1, 0] = b
        return t
    if style == "asymmetric" and (h, w) == (2, 2):
        return np.array([[a, b], [b, c]], dtype=np.int8)
    colors = palette[:_integer(rng, 2, len(palette))]
    t = rng.choice(colors, size=(h, w)).astype(np.int8)
    mode = _integer(rng, 0, 5)
    if mode == 0:
        t[:] = a
    elif mode == 1:
        t[:] = rng.choice(colors, size=(h, 1))
    elif mode == 2:
        t[:] = rng.choice(colors, size=(1, w))
    elif mode == 3 and h * w > 1:
        t[rng.random((h, w)) < .35] = bg
    elif mode == 4 and w >= 3:
        t[:, -1] = t[:, 0]
    if style in ("asymmetric", "primitive"):
        if w > 1:
            t[0, 0], t[0, -1] = a, b
        elif h > 1:
            t[0, 0], t[-1, 0] = a, b
    if style == "primitive" and _period(t) != w:
        t[:] = a
        t[0, -1] = b
    return t


def _roughen(rng, mask):
    """Erode an outline without changing its enclosed pixels or bounding box."""
    holes = set().union(*_holes(mask)) if _holes(mask) else set()
    h, w = mask.shape
    for _ in range(_integer(rng, 0, 10)):
        r, c = (_integer(rng, 0, h - 1), int(rng.choice([0, w - 1])))
        if rng.random() < .5:
            r, c = int(rng.choice([0, h - 1])), _integer(rng, 0, w - 1)
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        cells = _components(trial)
        if len(cells) != 1 or _box(cells[0]) != (0, 0, h, w):
            continue
        actual = set().union(*_holes(trial)) if _holes(trial) else set()
        if actual == holes:
            mask = trial
    return mask


def _container(rng, room_specs, ink, bg, *, tail=0, simple=False, shift=None):
    """Builder records are used only to construct labels, never by solve()."""
    top = bottom = 1
    if not simple:
        top, bottom = _integer(rng, 1, 2), _integer(rng, 1, 2)
    separators = [1 if simple else _integer(rng, 1, 2)
                  for _ in range(len(room_specs) - 1)]
    fronts = [1 if s["kind"] != "decoy" else _integer(rng, 2, 4) for s in room_specs]
    backs = [1 if simple else _integer(rng, 1, 3) for _ in room_specs]
    w = max(s["tile"].shape[1] + f + b
            for s, f, b in zip(room_specs, fronts, backs))
    h = top + bottom + sum(len(s["tile"]) for s in room_specs) + sum(separators)
    mask = np.ones((h, w), dtype=bool)
    rooms, row = [], top
    for i, (spec, front) in enumerate(zip(room_specs, fronts)):
        tile = spec["tile"]
        th, tw = tile.shape
        right = w - front
        left = right - tw
        mask[row:row + th, left:right] = False
        opened = spec["kind"] == "open"
        if opened:
            sides = ["back", "top", "bottom"] if simple else (
                ["back", "front", "top", "bottom"] if len(room_specs) == 1 else ["back", "front"])
            side = str(rng.choice(sides))
            rr = _integer(rng, row, row + th - 1)
            cc = _integer(rng, left, right - 1)
            if side == "back":
                mask[rr, :left] = False
            elif side == "front":
                mask[rr, right:] = False
            elif side == "top":
                mask[:row, cc] = False
            else:
                mask[row + th:, cc] = False
        rooms.append(dict(row=row, col=left, tile=tile.copy(),
                          active=not opened and front == 1 and tail == 0))
        row += th + (separators[i] if i < len(separators) else 0)
    if tail:
        extended = np.zeros((h, w + tail), dtype=bool)
        extended[:, :w] = mask
        extended[int(rng.choice([0, h - 1])), w:] = True
        mask = extended
    if not simple:
        mask = _roughen(rng, mask)
    _require(len(_components(mask)) == 1, "builder disconnected a container")
    patch = np.full(mask.shape, bg, dtype=np.int8)
    patch[mask] = ink
    for room in rooms:
        r, c, t = room["row"], room["col"], room["tile"]
        patch[r:r + len(t), c:c + t.shape[1]] = t
    return dict(patch=patch, rooms=rooms, shift_mode=shift)


def _solid(rng, ink, bg):
    shape = (_integer(rng, 2, 5), _integer(rng, 2, 7))
    mask = _roughen(rng, np.ones(shape, dtype=bool))
    return dict(patch=np.where(mask, ink, bg).astype(np.int8), rooms=[], shift_mode=None)


def _random_container(rng, palette, ink, bg, regime, training):
    if rng.random() < .17:
        return _solid(rng, ink, bg)
    n = _integer(rng, 1, 2 if training or rng.random() < .6 else 3)
    specs = []
    for _ in range(n):
        h, w = _integer(rng, 1, 3), _integer(rng, 1, 3 if training else 4)
        if regime == "combine" and training:
            h, w = (1, w) if rng.random() < .5 else (h, 1)
        kind = str(rng.choice(["active", "active", "decoy", "open"]))
        specs.append(dict(kind=kind, tile=_tile(rng, h, w, palette, bg)))
    tail = _integer(rng, 1, 3) if rng.random() < .15 else 0
    return _container(rng, specs, ink, bg, tail=tail)


def _add_extras(rng, containers, palette, ink, bg, regime, training, limit=4):
    target = _integer(rng, len(containers), max(len(containers), limit))
    for _ in range(target - len(containers)):
        obj = _random_container(rng, palette, ink, bg, regime, training)
        needed = sum(len(x["patch"]) for x in containers) + len(obj["patch"]) + len(containers)
        if needed <= 28:
            containers.append(obj)


def _layout(rng, containers, ink, bg, bands, turn):
    rng.shuffle(containers)
    widest = max(c["patch"].shape[1] for c in containers)
    work_max = 29 - bands[0]
    _require(widest + 3 <= work_max, "builder exceeded horizontal budget")
    rail = _integer(rng, widest + 3, min(work_max, widest + 9))
    band = _integer(rng, bands[0], min(bands[1], 29 - rail))
    base = sum(len(c["patch"]) for c in containers) + len(containers) - 1
    h = _integer(rng, base + 2, min(30, base + 8))
    extras = rng.multinomial(h - base - 2, np.full(len(containers) + 1, 1 / (len(containers) + 1)))
    inp = np.full((h, rail + band + 1), bg, dtype=np.int8)
    inp[:, rail] = ink
    out = inp.copy()
    row = 1 + int(extras[0])
    for i, obj in enumerate(containers):
        patch = obj["patch"]
        ph, pw = patch.shape
        maximum = rail - pw
        if obj["shift_mode"] == "aligned":
            travel = 0
        elif obj["shift_mode"] == "phase":
            travel = int(rng.choice([s for s in range(1, maximum + 1) if s % 3]))
        else:
            travel = 0 if rng.random() < .25 else _integer(rng, 1, maximum)
        left = rail - pw - travel
        inp[row:row + ph, left:left + pw] = patch
        out[row:row + ph, rail - pw:rail] = patch
        # Labels are built from sampled room rectangles, not from the solver.
        for room in obj["rooms"]:
            if not room["active"]:
                continue
            tile = room["tile"]
            for d in range(band):
                values = tile[:, tile.shape[1] - 1 - d % tile.shape[1]]
                r = row + room["row"]
                out[r:r + len(tile), rail + 1 + d] = values
        row += ph + 1 + int(extras[i + 1])
    return {k: np.ascontiguousarray(np.rot90(a, turn)) for k, a in (("input", inp), ("output", out))}


def _palette(rng, preferred=None):
    colors = list(map(int, rng.permutation(10))) if preferred is None else list(preferred)
    return colors[0], colors[1], colors[2:2 + _integer(rng, 2, 6)]


def _make_pair(rng, kind, regime, preferred, turn):
    bg, ink, payload = _palette(rng, preferred)
    containers = []
    if kind == "contrast":
        tile = _tile(rng, 1, 3, payload, bg, "aba")
        for role in ("active", "open", "tail"):
            containers.append(_container(rng, [dict(kind="open" if role == "open" else "active", tile=tile)],
                                         ink, bg, tail=_integer(rng, 1, 3) if role == "tail" else 0,
                                         simple=True, shift="phase" if role == "active" else None))
    elif kind == "second":
        if regime == "combine":
            specs = [dict(kind="active", tile=_tile(rng, _integer(rng, 2, 3), 1, payload, bg, "vector")),
                     dict(kind="active", tile=_tile(rng, 1, 2, payload, bg, "asymmetric"))]
        else:
            specs = [dict(kind="active", tile=_tile(rng, 2, 2, payload, bg, "asymmetric")),
                     dict(kind=str(rng.choice(["active", "decoy"])),
                          tile=_tile(rng, _integer(rng, 1, 2), _integer(rng, 1, 3), payload, bg))]
        containers.append(_container(rng, specs, ink, bg, shift="aligned"))
        _add_extras(rng, containers, payload, ink, bg, regime, True, 3)
    elif kind == "extra":
        containers.append(_random_container(rng, payload, ink, bg, regime, True))
        _add_extras(rng, containers, payload, ink, bg, regime, True)
    else:
        if regime == "more_rooms":
            specs = [dict(kind="active" if j < 2 else str(rng.choice(["active", "decoy"])),
                          tile=_tile(rng, _integer(rng, 2 if j == 0 else 1, 3),
                                     _integer(rng, 2 if j == 0 else 1, 4), payload, bg, "asymmetric"))
                     for j in range(_integer(rng, 3, 4))]
        else:
            w = _integer(rng, 4, 6) if regime == "longer_word" else _integer(rng, 2, 4)
            specs = [dict(kind="active", tile=_tile(rng, _integer(rng, 2, 3), w, payload, bg,
                                                    "primitive" if regime == "longer_word" else "asymmetric"))]
            if rng.random() < .55:
                specs.append(dict(kind=str(rng.choice(["active", "decoy"])),
                                  tile=_tile(rng, _integer(rng, 1, 2), _integer(rng, 1, 3), payload, bg)))
        containers.append(_container(rng, specs, ink, bg))
        _add_extras(rng, containers, payload, ink, bg, regime, False, 4)
    bands = (1, 5) if kind == "query" and regime == "short_band" else (7, 13)
    return _layout(rng, containers, ink, bg, bands, turn)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    # At least two visible roles give >=90 distinct injective recolorings.
    # At most thirteen earlier inputs are blocked; iteration returns by #14.
    colors = list(map(int, np.unique(pair["input"])))
    for values in itertools.permutations(range(10), len(colors)):
        lut = np.arange(10, dtype=np.int8)
        lut[colors] = values
        candidate = {k: lut[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("finite uniqueness orbit exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(_REGIMES))
    preferred = list(map(int, rng.permutation(10))) if rng.random() < .35 else None
    base_turn = _integer(rng, 0, 3)
    mixed = rng.random() < .55
    turns = [base_turn] * (num_train + num_test)
    if mixed:
        turns = list(map(int, rng.integers(0, 4, size=len(turns))))
        turns[1] = (turns[0] + _integer(rng, 1, 3)) % 4
    puzzle, used = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        split = "train" if i < num_train else "test"
        kind = "contrast" if i == 0 else "second" if i == 1 else "extra" if split == "train" else "query"
        pair = _unique(_make_pair(rng, kind, regime, preferred, turns[i]), used)
        used.add(_key(pair["input"]))
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(pairs, scenes):
    variants = ("all_closed", "local_column", "flat", "unreflected", "prefix",
                "travel_phase", "moving_only", "no_motion")
    for variant in variants:
        _require(any(not np.array_equal(_render(s, variant), p["output"])
                     for p, s in zip(pairs, scenes)), "training fails to distinguish " + variant)
    rooms = [c for s in scenes for o in s["objects"] for c in o["chambers"]]
    _require(any(c["active"] and len(c["tile"]) > 1 for c in rooms), "no vector-slice witness")
    _require(any(c["active"] and _period(c["tile"]) > 1 for c in rooms), "no cyclic-word witness")
    _require(any(not c["active"] for c in rooms), "no enclosed recessed contrast")
    _require(any(o["shift"] == 0 and any(c["active"] for c in o["chambers"])
                 for s in scenes for o in s["objects"]), "no already-docked printing witness")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _decode(inp)
            _require(np.array_equal(_render(scene), out), "wrong docking or printed tile")
            ink = scene["ink"]
            _require(int(np.count_nonzero(inp == ink)) == int(np.count_nonzero(out == ink)),
                     "structural ink not conserved")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented color")
            if split == "train":
                training.append(scene)
    _training_evidence(puzzle["train"], training)


def _features(pair):
    s = _decode(pair["input"])
    rooms = [c for o in s["objects"] for c in o["chambers"]]
    return dict(shape=pair["input"].shape, direction=s["turn"], bg=s["bg"], ink=s["ink"],
                count=len(s["objects"]), band=s["grid"].shape[1] - s["rail"] - 1,
                boxes=tuple(o["patch"].shape for o in s["objects"]),
                shifts=tuple(o["shift"] for o in s["objects"]),
                room_counts=tuple(len(o["chambers"]) for o in s["objects"]),
                active=sum(c["active"] for c in rooms), decoy=sum(not c["active"] for c in rooms),
                tiles=tuple(c["tile"].shape for c in rooms), periods=tuple(_period(c["tile"]) for c in rooms),
                background_tiles=sum(bool(np.any(c["tile"] == s["bg"])) for c in rooms),
                outlines=tuple((o["mask"].shape, o["mask"].tobytes()) for o in s["objects"]))


def _signature(p):
    return tuple(tuple((_key(q["input"]), _key(q["output"])) for q in p[s]) for s in ("train", "test"))


def _observed_shifts(puzzle):
    """Measure actual held-out features; do not consult the sampled regime."""
    train = [_decode(p["input"]) for p in puzzle["train"]]
    test = [_decode(p["input"]) for p in puzzle["test"]]
    tr = [c for s in train for o in s["objects"] for c in o["chambers"]]
    te = [c for s in test for o in s["objects"] for c in o["chambers"]]
    observed = set()
    if (not any(c["active"] and min(c["tile"].shape) > 1 for c in tr)
            and any(c["active"] and min(c["tile"].shape) > 1 for c in te)):
        observed.add("combine")
    if max(len(o["chambers"]) for s in test for o in s["objects"]) > max(
            len(o["chambers"]) for s in train for o in s["objects"]):
        observed.add("more_rooms")
    if max(c["tile"].shape[1] for c in te) > max(c["tile"].shape[1] for c in tr):
        observed.add("longer_word")
    if max(s["grid"].shape[1] - s["rail"] - 1 for s in test) < min(
            s["grid"].shape[1] - s["rail"] - 1 for s in train):
        observed.add("short_band")
    return observed


def _literal_regressions():
    # A hand-specified 2x3 chamber, moved two cells. Printed rows are literal
    # expected strings, intentionally not computed by the repetition code.
    inp = np.zeros((6, 17), dtype=np.int8)
    inp[:, 8] = 8
    inp[1:5, 1:6] = np.array([[8, 8, 8, 8, 8], [8, 2, 3, 4, 8],
                              [8, 5, 6, 2, 8], [8, 8, 8, 8, 8]], dtype=np.int8)
    expected = np.zeros_like(inp)
    expected[:, 8] = 8
    expected[1:5, 3:8] = inp[1:5, 1:6]
    expected[2, 9:] = list(map(int, "43243243"))
    expected[3, 9:] = list(map(int, "26526526"))
    fixtures = [(inp, expected)]
    # Opening a top door leaves the same contents but suppresses both rows.
    opened = inp.copy()
    opened[1, 3] = 0
    silent = np.zeros_like(inp)
    silent[:, 8] = 8
    silent[1:5, 3:8] = opened[1:5, 1:6]
    fixtures.append((opened, silent))
    # A protruding tail globally recesses the unchanged closed chamber.
    tailed = inp.copy()
    tailed[1, 6] = 8
    recessed = np.zeros_like(inp)
    recessed[:, 8] = 8
    recessed[1:5, 2:8] = tailed[1:5, 1:7]
    fixtures.append((tailed, recessed))
    # A diagonal leak does not open a four-connected chamber.
    diagonal, diagonal_out = inp.copy(), expected.copy()
    diagonal[1, 1], diagonal_out[1, 3] = 0, 0
    fixtures.append((diagonal, diagonal_out))
    docked = expected.copy()
    docked[:, 9:] = 0
    fixtures.append((docked, expected))
    fixtures.append((inp[:, :10], expected[:, :10]))
    # The same frame with entirely blank contents is still a legal chamber.
    blank, blank_out = inp.copy(), expected.copy()
    blank[2:4, 2:5] = 0
    blank_out[2:4, 4:7] = 0
    blank_out[:, 9:] = 0
    fixtures.append((blank, blank_out))
    for a, b in fixtures:
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn)), "literal rule regression")
    return fixtures[0]


def self_test():
    literal_in, literal_out = _literal_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(q) for p in puzzles for s in ("train", "test") for q in p[s]]
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _require(len({f["shape"] for f in records}) >= 50, "canvas diversity collapsed")
    _require(len({m for f in records for m in f["outlines"]}) >= 100, "wall geometry collapsed")
    _require({f["direction"] for f in records} == set(range(4)), "missing directions")
    _require({f["bg"] for f in records} == set(range(10)), "background palette collapsed")
    _require({f["ink"] for f in records} == set(range(10)), "ink palette collapsed")
    _require({1, 2, 3, 4} <= {f["count"] for f in records}, "container counts collapsed")
    _require({1, 2, 3} <= {n for f in records for n in f["room_counts"]}, "room counts collapsed")
    _require({1, 2, 3, 4, 5, 6} <= {w for f in records for h, w in f["tiles"]}, "word sizes collapsed")
    _require(any(f["band"] == 1 for f in records), "no first-copy clipping")
    _require(any(f["background_tiles"] for f in records), "no background tile cells")
    _require(any(0 in f["shifts"] and max(f["shifts"]) > 0 for f in records), "no mixed travel")
    _require(any(f["active"] == 0 for f in records), "no silent scene")
    shifts = [_observed_shifts(p) for p in puzzles]
    _require(all(shifts), "a puzzle has no measured train/test shift")
    _require(all(sum(name in s for s in shifts) >= 3 for name in _REGIMES), "shift diversity collapsed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9182 + a * 10 + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (None, 1),
                 (4, None), (True, 1), (4.0, 1), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _require(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "seed not reproducible")
    rng = np.random.default_rng(771)
    for _ in range(8):
        lut = rng.permutation(10).astype(np.int8)
        a, b = lut[literal_in], lut[literal_out]
        _require(np.array_equal(solve(a), b), "color equivariance failed")
        _require(np.array_equal(solve(a[::-1]), b[::-1]), "reflection equivariance failed")
    # Exercise finite duplicate repair even for a scene with only bg and ink.
    blank = literal_in.copy()
    blank[2:4, 2:5] = 0
    pair = {"input": blank, "output": solve(blank)}
    used = set()
    for _ in range(14):
        fresh = _unique(pair, used)
        used.add(_key(fresh["input"]))
        _require(np.array_equal(solve(fresh["input"]), fresh["output"]), "uniqueness repair changed rule")
    _require(len(used) == 14, "finite uniqueness fallback failed")
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len({_signature(p) for p in puzzles}),
                grid_shapes=len({f["shape"] for f in records}),
                wall_masks=len({m for f in records for m in f["outlines"]}),
                container_counts=sorted({f["count"] for f in records}),
                tile_shapes=sorted({t for f in records for t in f["tiles"]}),
                directions=sorted({f["direction"] for f in records}),
                band_lengths=sorted({f["band"] for f in records}),
                observed_shift_puzzles={name: sum(name in s for s in shifts) for name in _REGIMES},
                semantic_checks="generator-owned, plus literal regressions; not independent approval")
