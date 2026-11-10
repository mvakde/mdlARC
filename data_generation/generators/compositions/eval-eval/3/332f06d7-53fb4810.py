"""Phase-sensitive aperture looms. Standard library and NumPy only.

Every inference, collision test, and write in solve depends on input pixels.
Construction uses sampled words directly; validation reparses the returned grids.
"""
from __future__ import annotations

import numpy as np

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GAP = -1  # construction-only sentinel, converted to the visible floor color


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a


def _key(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    left = set(map(tuple, np.argwhere(mask)))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        cells, todo = {start}, [start]
        for r, c in todo:
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _box(cells):
    r, c = zip(*cells)
    return int(min(r)), int(min(c)), int(max(r)-min(r)+1), int(max(c)-min(c)+1)


def _hub_mask(h, w, kind, arm=None):
    mask = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind == "cross":
        ah, aw = arm
        mask[:] = False
        mask[(h-ah)//2:(h+ah)//2, :] = True
        mask[:, (w-aw)//2:(w+aw)//2] = True
    return mask


def _hub_kind(mask):
    h, w = mask.shape
    if not (3 <= h <= 9 and 3 <= w <= 9):
        return None
    for kind in ("rectangle", "clipped"):
        if np.array_equal(mask, _hub_mask(h, w, kind)):
            return kind
    for ah in range(1, h-1):
        for aw in range(1, w-1):
            if (h-ah) % 2 == 0 and (w-aw) % 2 == 0:
                if np.array_equal(mask, _hub_mask(h, w, "cross", (ah, aw))):
                    return "cross"
    return None


def _maximum(shape, box, direction):
    t, l, h, w = box
    return (t, shape[0]-t-h, l, shape[1]-l-w)[direction]


def _coords(box, direction, width, distance):
    """Transverse order: top-to-bottom or left-to-right in the current grid."""
    t, l, h, w = box
    if direction < 2:
        r = t-distance if direction == 0 else t+h-1+distance
        return [(r, l+(w-width)//2+j) for j in range(width)]
    c = l-distance if direction == 2 else l+w-1+distance
    return [(t+(h-width)//2+j, c) for j in range(width)]


def _parse_with_roles(g, wall, hub, floor, groups):
    claimed, envelopes, looms = set(), [], []
    for cells in groups:
        box = _box(cells)
        t, l, h, w = box
        kind = _hub_kind(g[t:t+h, l:l+w] == hub)
        _require(kind is not None, "malformed source hub")
        faces = []
        for direction in range(4):
            maximum = _maximum(g.shape, box, direction)
            if maximum < 2:
                continue
            transverse = w if direction < 2 else h
            face = _coords(box, direction, transverse, 1)
            indices = [j for j, p in enumerate(face) if int(g[p]) not in (wall, hub)]
            width = len(indices)
            offset = (transverse-width)//2
            if not (1 <= width <= 5 and transverse-width >= 2
                    and (transverse-width) % 2 == 0
                    and indices == list(range(offset, offset+width))):
                continue
            dr, dc = DIRECTIONS[direction]
            if all((r-dr, c-dc) in cells for r, c in _coords(box, direction, width, 1)):
                faces.append((direction, width, maximum))
        _require(len(faces) == 1, "source must have exactly one centered channel")
        direction, width, maximum = faces[0]
        word, band = [], set()
        ended = False
        for d in range(1, maximum+1):
            ps = _coords(box, direction, width, d)
            vals = tuple(int(g[p]) for p in ps)
            _require(hub not in vals, "channel hits source")
            payload = any(v not in (wall, floor, hub) for v in vals)
            if payload:
                _require(not ended and wall not in vals, "payload outside unobstructed seed")
                word.append(vals)
            else:
                ended = True
                _require(all(v in (wall, floor) for v in vals) and floor in vals,
                         "each aperture must retain an open lane")
            band.update(ps)
        _require(1 <= len(word) < maximum, "visible word must be nonempty and unfinished")
        # The entire rectangle denotes a complete word, not an arbitrary prefix.
        _require(len(word) <= 10, "visible seed longer than family bound")
        envelope = band | {(r, c) for r in range(t, t+h) for c in range(l, l+w)}
        for other in envelopes:
            _require(not envelope.intersection(other), "channel envelopes overlap")
        envelopes.append(envelope)
        _require(not claimed.intersection(cells | band), "objects overlap")
        claimed.update(cells | {p for p in band if int(g[p]) != wall})
        looms.append(dict(box=box, kind=kind, direction=direction, width=width,
                          maximum=maximum, word=word, band=band, hub_cells=cells))
    _require(claimed == set(map(tuple, np.argwhere(g != wall))), "unattached nonwall cells")
    return looms


def _parse(input_grid):
    g = _grid(input_grid)
    corners = [int(g[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _require(len(set(corners)) == 1, "four wall corners required")
    wall = corners[0]
    colors = [int(c) for c in np.unique(g) if c != wall]
    candidates = []
    for hub in colors:
        groups = _components(g == hub)
        if not (1 <= len(groups) <= 3):
            continue
        if any(_hub_kind(g[t:t+h, l:l+w] == hub) is None
               for t, l, h, w in map(_box, groups)):
            continue
        for floor in colors:
            if floor == hub:
                continue
            try:
                looms = _parse_with_roles(g, wall, hub, floor, groups)
                candidates.append((wall, hub, floor, looms))
            except AssertionError:
                pass
    _require(len(candidates) == 1, "missing or ambiguous geometric color roles")
    return candidates[0]


def _simulate(g, roles, variant=None):
    wall, hub, floor, looms = roles
    out, records = g.copy(), []
    for loom in looms:
        word = loom["word"]
        if variant == "last_slice":
            word = [word[-1]]
        elif variant == "prefix_period":
            p = next(p for p in range(1, len(word)+1)
                     if all(v == word[i % p] for i, v in enumerate(word)))
            word = word[:p]
        stop, admitted, transparent_passes, partial_collision = None, 0, 0, False
        for d in range(len(loom["word"])+1, loom["maximum"]+1):
            vals = word[(d-1) % len(word)]
            ps = _coords(loom["box"], loom["direction"], loom["width"], d)
            active = [(p, v) for p, v in zip(ps, vals) if v != floor]
            obstructed = [(p, v) for p, v in active if int(g[p]) != floor]
            envelope_block = any(int(g[p]) == wall for p in ps)
            blocked = bool(obstructed) or (variant == "solid_envelope" and envelope_block)
            if blocked and variant not in ("ignore_collision", "clip_lanes"):
                stop = d
                partial_collision = bool(obstructed and len(obstructed) < len(active))
                break
            if not obstructed and envelope_block:
                transparent_passes += 1
            admitted += 1
            for p, v in active:
                if variant != "clip_lanes" or int(g[p]) == floor:
                    out[p] = v
        records.append(dict(loom, stop=stop, admitted=admitted,
                            transparent_passes=transparent_passes,
                            partial_collision=partial_collision,
                            success=stop is None))
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    return _simulate(g, _parse(g))[0]


def _dilate(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _tile(rng, width, period, palette):
    word = rng.choice(palette, size=(period, width)).astype(int)
    style = int(rng.integers(4))
    for i in range(period):
        if width > 1 and rng.random() < .75:
            gaps = rng.choice(width, size=int(rng.integers(1, width)), replace=False)
            word[i, gaps] = GAP
        if style == 1:
            word[i, word[i] != GAP] = int(rng.choice(palette))
    if style == 2 and period >= 3:
        word[-1] = word[0]
    if style == 3 and period > 1:
        word[1:] = word[0]
        j = int(np.flatnonzero(word[-1] != GAP)[0])
        word[-1, j] = next(c for c in palette if c != word[-1, j])
    return word


def _period(word):
    return next(p for p in range(1, len(word)+1)
                if len(word) % p == 0
                and all(tuple(row) == tuple(word[i % p]) for i, row in enumerate(word)))


def _anchor_word(rng, width, palette):
    a = rng.choice(palette, size=width).astype(int)
    gaps = rng.choice(width, size=int(rng.integers(1, width)), replace=False)
    a[gaps] = GAP
    b = rng.choice(palette, size=width).astype(int)
    b[0], b[-1] = palette[:2]
    return np.array([a, b, a], dtype=int)


def _apertures(rng, word, length, copies, outcome, anchor=False):
    """Construction works with sampled support, not the input parser."""
    width, visible = word.shape[1], len(word)*copies
    wall = np.zeros((length, width), dtype=bool)
    if width == 1:
        return wall, None
    if anchor:
        holes = np.flatnonzero(word[0] == GAP)
        wall[3, holes] = True  # distance 4: A fits, even though its envelope does not
        if outcome == "blocked":
            stop = 8
            wall[stop-1, holes] = True  # distance 8: B has teeth in every lane
        else:
            stop = None
            wall[5, holes] = True  # distance 6: another compatible A
    else:
        stop = (int(rng.integers(visible+1, length+1)) if outcome == "blocked" else None)
    for d in range(visible+1, length+1):
        if anchor and d in (4, 6, 8):
            continue
        phase = word[(d-1) % len(word)]
        holes = np.flatnonzero(phase == GAP)
        if rng.random() < .45:
            choices = (np.arange(width) if stop is not None and d > stop else holes)
            if len(choices):
                n = int(rng.integers(1, min(width-1, len(choices))+1))
                wall[d-1, rng.choice(choices, n, replace=False)] = True
    if stop is not None and not anchor:
        active = np.flatnonzero(word[(stop-1) % len(word)] != GAP)
        lane = int(rng.choice(active))
        wall[stop-1, lane] = True
        if wall[stop-1].all():
            wall[stop-1, (lane+1) % width] = False
    return wall, stop


def _make_spec(rng, shape, width, word, outcome, copies=1, anchor=False,
               placement=None, compact=False):
    direction = int(rng.integers(4)) if placement is None else placement[1]
    transverse = width+2 if compact else width+2*int(rng.integers(1, 3))
    longitudinal = 4 if compact else int(rng.integers(3, 7))
    h, w = (longitudinal, transverse) if direction < 2 else (transverse, longitudinal)
    if placement is None:
        if h > shape[0]-2 or w > shape[1]-2:
            return None
        t = int(rng.integers(1, shape[0]-h))
        l = int(rng.integers(1, shape[1]-w))
    else:
        t, l = placement[0]
    box = t, l, h, w
    if t < 0 or l < 0 or t+h > shape[0] or l+w > shape[1]:
        return None
    length = _maximum(shape, box, direction)
    visible = len(word)*copies
    if length < (11 if anchor else visible+2):
        return None
    kind = str(rng.choice(("rectangle", "clipped", "cross")))
    ah = [a for a in range(1, h-1) if (h-a) % 2 == 0 and (direction < 2 or a >= width)]
    aw = [a for a in range(1, w-1) if (w-a) % 2 == 0 and (direction >= 2 or a >= width)]
    mask = _hub_mask(h, w, kind, (int(rng.choice(ah)), int(rng.choice(aw))))
    cells = {(t+int(r), l+int(c)) for r, c in np.argwhere(mask)}
    band = {p for d in range(1, length+1) for p in _coords(box, direction, width, d)}
    envelope = band | {(r, c) for r in range(t, t+h) for c in range(l, l+w)}
    if any(p in envelope for p in ((0, 0), (0, shape[1]-1), (shape[0]-1, 0),
                                  (shape[0]-1, shape[1]-1))):
        return None
    apertures, stop = _apertures(rng, word, length, copies, outcome, anchor)
    return dict(box=box, direction=direction, width=width, word=word.copy(), copies=copies,
                length=length, mask=mask, cells=cells, band=band, envelope=envelope,
                apertures=apertures, stop=stop)


def _paint(shape, specs, roles):
    wall, hub, floor = roles
    inp = np.full(shape, wall, dtype=np.int8)
    for s in specs:
        for p in s["cells"]:
            inp[p] = hub
        for d in range(1, s["length"]+1):
            for j, p in enumerate(_coords(s["box"], s["direction"], s["width"], d)):
                inp[p] = wall if s["apertures"][d-1, j] else floor
        for d in range(1, len(s["word"])*s["copies"]+1):
            vals = s["word"][(d-1) % len(s["word"])]
            for p, v in zip(_coords(s["box"], s["direction"], s["width"], d), vals):
                inp[p] = floor if v == GAP else int(v)
    out = inp.copy()
    for s in specs:
        end = s["length"]+1 if s["stop"] is None else s["stop"]
        for d in range(len(s["word"])*s["copies"]+1, end):
            vals = s["word"][(d-1) % len(s["word"])]
            for p, v in zip(_coords(s["box"], s["direction"], s["width"], d), vals):
                if v != GAP:
                    out[p] = int(v)
    return dict(input=inp, output=out)


def _pair(rng, roles, palette, width_limit, period_limit, count=1,
          anchor_word=None, outcome=None, primary=None, copies=1, force_fallback=False):
    words = []
    for i in range(count):
        if i == 0 and anchor_word is not None:
            word = anchor_word.copy()
        else:
            width, period = primary if i == 0 and primary else (
                int(rng.integers(1, width_limit+1)), int(rng.integers(1, period_limit+1)))
            word = _tile(rng, width, period, palette)
            if i == 0 and primary and _period(word) != period:
                j = int(np.flatnonzero(word[-1] != GAP)[0])
                word[-1, j] = next(c for c in palette if c != word[-1, j])
                _require(_period(word) == period, "primary period repair")
        words.append(word)
    for scene in range(0 if force_fallback else 16):
        shape = (int(rng.integers(11, 31)), int(rng.integers(15, 31))) if count == 1 else (30, 30)
        specs = []
        for i, word in enumerate(words):
            result = outcome if i == 0 and outcome else str(rng.choice(("blocked", "success")))
            if word.shape[1] == 1:
                result = "success"
            for _ in range(80):
                s = _make_spec(rng, shape, word.shape[1], word, result, copies if i == 0 else 1,
                               anchor_word is not None and i == 0)
                if s is not None and not any(_dilate(s["envelope"]) & x["envelope"] for x in specs):
                    specs.append(s)
                    break
            else:
                break
        if len(specs) == count:
            pair = _paint(shape, specs, roles)
            try:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction mismatch")
                return pair
            except AssertionError:
                continue
    # Guaranteed separated bands. Maximal width has a 9-cell transverse source.
    shape, specs = (30, 30), []
    for i, word in enumerate(words):
        result = outcome if i == 0 and outcome else str(rng.choice(("blocked", "success")))
        if word.shape[1] == 1:
            result = "success"
        # Compact sources and a fixed 25-slice ray guarantee three bands fit.
        s = _make_spec(rng, shape, word.shape[1], word, result, copies if i == 0 else 1,
                       anchor_word is not None and i == 0, ((1+10*i, 1), 3), compact=True)
        _require(s is not None, "fallback geometry")
        specs.append(s)
    pair = _paint(shape, specs, roles)
    _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback mismatch")
    return pair


def _counts(num_train, num_test):
    for name, n, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= int(n) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(3))  # wider, longer word, more simultaneous looms
    train_width = 3 if regime == 0 else 4
    colors = list(map(int, rng.permutation(10)))
    roles, palette = tuple(colors[:3]), colors[3:]
    width = int(rng.integers(2, train_width+1))
    anchor = _anchor_word(rng, width, palette)
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", n), ("test", m)):
        for i in range(number):
            anchor_i = split == "train" and i < 2
            # The paired anchors share the word but exchange structural colors.
            local_roles = roles if anchor_i and i == 0 else roles[1:]+roles[:1] if anchor_i else None
            for attempt in range(48):
                if local_roles is None:
                    colors = list(map(int, rng.permutation(10)))
                    pair_roles, pair_palette = tuple(colors[:3]), colors[3:3+int(rng.integers(2, 8))]
                else:
                    pair_roles, pair_palette = local_roles, palette
                primary, count, copies = None, 1, 1
                if split == "test":
                    if regime == 0:
                        primary = (int(rng.integers(4, 6)), int(rng.integers(1, 4)))
                    elif regime == 1:
                        primary = (int(rng.integers(2, 5)), int(rng.integers(4, 6)))
                    else:
                        count = int(rng.integers(2, 4))
                    copies = int(rng.integers(1, 3))
                elif not anchor_i:
                    copies = int(rng.integers(1, 3))
                pair = _pair(rng, pair_roles, pair_palette, train_width, 3, count,
                             anchor if anchor_i else None,
                             ("blocked" if i == 0 else "success") if anchor_i else None,
                             primary, copies)
                if _key(pair["input"]) not in used:
                    used.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded unique-input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            roles = _parse(inp)
            expected, records = _simulate(inp, roles)
            _require(np.array_equal(expected, out), "wrong synchronized collision or periodic continuation")
            wall, hub, floor, looms = roles
            _require(np.array_equal(out[inp == wall], inp[inp == wall]), "wall overwritten")
            _require(np.array_equal(out[inp == hub], inp[inp == hub]), "source overwritten")
            for s in records:
                for d in range(1, len(s["word"])+1):
                    ps = _coords(s["box"], s["direction"], s["width"], d)
                    _require(all(out[p] == inp[p] for p in ps), "seed modified")
                if s["stop"] is not None:
                    ps = [p for d in range(s["stop"], s["maximum"]+1)
                          for p in _coords(s["box"], s["direction"], s["width"], d)]
                    _require(all(out[p] == inp[p] for p in ps), "continued after first collision")
            if split == "train":
                training.append((inp, roles, records, out))
    _require(any(s["success"] and s["admitted"] > 0 for _, _, rs, _ in training for s in rs),
             "training must demonstrate reaching edge")
    _require(any(s["partial_collision"] and s["admitted"] > 0 for _, _, rs, _ in training for s in rs),
             "training must demonstrate atomic stop despite open teeth")
    _require(any(s["transparent_passes"] > 0 for _, _, rs, _ in training for s in rs),
             "training must demonstrate holes passing wall")
    for variant in ("ignore_collision", "clip_lanes", "last_slice", "solid_envelope", "prefix_period"):
        _require(any(not np.array_equal(_simulate(inp, roles, variant)[0], out)
                     for inp, roles, _, out in training), "shortcut not distinguished: "+variant)


def review_features(pair):
    """Observable features, not sampled generator labels."""
    inp = _grid(pair["input"])
    roles = _parse(inp)
    _, records = _simulate(inp, roles)
    return dict(shape=inp.shape, roles=roles[:3], count=len(records),
                widths=tuple(s["width"] for s in records),
                lengths=tuple(len(s["word"]) for s in records),
                periods=tuple(_period(s["word"]) for s in records),
                directions=tuple(s["direction"] for s in records),
                kinds=tuple(s["kind"] for s in records),
                stops=tuple(s["stop"] for s in records),
                admitted=tuple(s["admitted"] for s in records),
                success=tuple(s["success"] for s in records),
                transparent=sum(s["transparent_passes"] for s in records),
                gates=sum(sum(inp[p] == roles[0] for p in s["band"]) for s in records),
                support_words=tuple(tuple(tuple(v != roles[2] for v in row) for row in s["word"])
                                    for s in records))


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    features = [review_features(x) for p in puzzles for split in ("train", "test") for x in p[split]]
    _require(len({f["shape"] for f in features}) >= 30, "canvas diversity")
    _require({v for f in features for v in f["widths"]} == {1, 2, 3, 4, 5}, "width diversity")
    _require({v for f in features for v in f["directions"]} == {0, 1, 2, 3}, "direction diversity")
    _require({f["count"] for f in features} == {1, 2, 3}, "loom-count diversity")
    _require({v for f in features for v in f["kinds"]} == {"rectangle", "clipped", "cross"}, "hub diversity")
    _require(len({f["support_words"] for f in features}) >= 80, "physical support diversity")
    _require({v for f in features for v in f["success"]} == {False, True}, "outcome diversity")
    for role in range(3):
        _require({f["roles"][role] for f in features} == set(range(10)), "role palette diversity")
    _require({1, 2, 3, 4, 5}.issubset({v for f in features for v in f["lengths"]}), "word-length diversity")
    shifts = set()
    for p in puzzles:
        train = [review_features(x) for x in p["train"]]
        test = [review_features(x) for x in p["test"]]
        if max(v for f in test for v in f["widths"]) > max(v for f in train for v in f["widths"]):
            shifts.add("width")
        if max(v for f in test for v in f["periods"]) > max(v for f in train for v in f["periods"]):
            shifts.add("word")
        if max(f["count"] for f in test) > max(f["count"] for f in train):
            shifts.add("count")
        for x in p["train"]+p["test"]:
            # All eight dihedral transformations preserve interpretation and labels.
            for rot in range(4):
                for mirror in (False, True):
                    a, b = np.rot90(x["input"], rot), np.rot90(x["output"], rot)
                    if mirror:
                        a, b = np.fliplr(a), np.fliplr(b)
                    _require(np.array_equal(solve(a), b), "rotation/reflection equivariance")
    _require(shifts == {"width", "word", "count"}, "curriculum diversity")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "reproducibility")
    for width in (1, 2, 5):
        for count in (1, 2, 3):
            pair = _pair(np.random.default_rng(100+width+count), (0, 2, 8), [1, 3, 4],
                         5, 5, count=count, primary=(width, 5), copies=2, force_fallback=True)
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback regression")
    # Direct hand-specified comb contrast: same slot, different phase.
    word = np.array([[1, GAP], [1, 3], [1, GAP]])
    for blocked, end in ((True, 8), (False, 14)):
        inp = np.zeros((6, 18), dtype=np.int8)
        inp[1:5, 1:5] = 2
        inp[2:4, 5:18] = 8
        inp[2:4, 5:8] = np.where(word.T == GAP, 8, word.T)
        inp[3, 8] = 0  # distance 4 is A: transparent gap over wall
        inp[3, 12 if blocked else 10] = 0  # distance 8 is B; distance 6 is A
        expected = inp.copy()
        for d in range(4, end):
            vals = word[(d-1) % 3]
            for j, v in enumerate(vals):
                if v != GAP:
                    expected[2+j, 5+d-1] = v
        _require(np.array_equal(solve(inp), expected), "hand-specified phase/stop contrast")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(92731, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("output mutation accepted")
