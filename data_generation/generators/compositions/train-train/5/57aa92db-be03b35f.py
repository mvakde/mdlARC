"""Growth-turn stencil completion.

Composition of ARC 57aa92db (registration-anchored stencil completion at
integer scales) and be03b35f (a sequence of successive quarter-turns whose
next member is extrapolated).

Rule.  Each input holds one complete unit-scale reference stencil: a single
registration-coloured cell plus cells of one reference colour.  Every other
object is an incomplete copy anchored by a solid k-by-k registration block and
showing some k-by-k blocks in its own payload colour.  The copies are stages of
one growth sequence: the stage-k copy (magnified k times) is the reference
turned k-1 quarter-turns about its registration cell, every stage turning in
the same direction.  The direction is not fixed by the family; each input shows
it through copies whose own blocks fix their orientation.  The output paints
every stencil block of every copy in that copy's payload colour, at its scale
and stage orientation, and leaves everything else unchanged.
"""
from __future__ import annotations

import numpy as np

CW, CCW = 1, -1

_SOURCE_FORMS = (
    ("XXX", "XMX", "X.X"),
    ("...X", "MXXX", "...X"),
    ("..X", "XXM", "..X"),
    (".X.", "XXM", ".X."),
    (".XX", "XMX", "XX."),
)
_SIZES = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (2, 4), (4, 2), (4, 4),
          (3, 3), (3, 4), (4, 3))


# --------------------------------------------------------------------------
# small helpers

def _key(grid):
    grid = np.asarray(grid)
    return (grid.shape, grid.astype(np.int8).tobytes())


def _rot(code, quarter_turns_cw):
    """Rotate a stencil code array clockwise by the given number of turns."""
    return np.rot90(code, -(int(quarter_turns_cw) % 4))


def _orientation_count(code):
    return len({_key(np.rot90(code, k)) for k in range(4)})


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r0, c0 in zip(*np.nonzero(mask)):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack, cells = [(r0, c0)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < height and 0 <= cc < width and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _connected(mask):
    return len(_components(mask)) == 1


def _dilate(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# --------------------------------------------------------------------------
# turn laws.  The family law is _law_growth; the others are the shortcuts that
# the demonstrations must refute.

def _law_growth(scale, scales):
    return scale - 1


def _law_none(scale, scales):          # parent A alone: copies never turn
    return 0


def _law_one(scale, scales):           # every enlarged copy turns once
    return 1 if scale > 1 else 0


def _law_k(scale, scales):             # off-by-one: k turns at scale k
    return scale


def _law_rank(scale, scales):          # turn count = rank among sizes present
    return sum(1 for s in set(scales) | {1} if s < scale)


def _mirror_half(code, quarter_turns_cw):
    # Half-turn stage replaced by a top/bottom mirror.
    return np.flipud(code) if quarter_turns_cw % 4 == 2 else _rot(code, quarter_turns_cw)


def _mirror_half_lr(code, quarter_turns_cw):
    # Half-turn stage replaced by a left/right mirror.
    return np.fliplr(code) if quarter_turns_cw % 4 == 2 else _rot(code, quarter_turns_cw)


def _mirror_quarter(code, quarter_turns_cw):
    # Quarter-turn stages replaced by the two diagonal mirrors.
    turns = quarter_turns_cw % 4
    if turns == 1:
        return code.T
    if turns == 3:
        return np.rot90(code, 2).T
    return _rot(code, turns)


# name -> (turn law, stage transform).  Every alternative is evaluated in both
# directions, so "mirror_quarter" lets either diagonal mirror stand for the
# scale-2 stage (and the other one for scale 4).
_ALTERNATIVES = {"no_turn": (_law_none, _rot), "one_turn": (_law_one, _rot),
                 "k_turns": (_law_k, _rot), "rank": (_law_rank, _rot),
                 "mirror_half": (_law_growth, _mirror_half),
                 "mirror_half_lr": (_law_growth, _mirror_half_lr),
                 "mirror_quarter": (_law_growth, _mirror_quarter)}


# --------------------------------------------------------------------------
# input-only interpretation (used by solve and by the validator)

def _square_components(grid, color):
    found = []
    for cells in _components(grid == color):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        top, left = min(rows), min(cols)
        side = max(rows) - top + 1
        if side != max(cols) - left + 1 or side > 4 or len(cells) != side * side:
            return None
        found.append((top, left, side))
    return sorted(found)


def _reference(grid, background, marker, row, col):
    cells = None
    for comp in _components(grid != background):
        if (row, col) in comp:
            cells = comp
            break
    if cells is None:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    top, left = min(rows), min(cols)
    height, width = max(rows) - top + 1, max(cols) - left + 1
    if not (2 <= height <= 4 and 2 <= width <= 4):
        return None
    code = np.zeros((height, width), dtype=np.int8)
    colors = set()
    for r, c in cells:
        value = int(grid[r, c])
        if value == marker:
            code[r - top, c - left] = 2
        else:
            code[r - top, c - left] = 1
            colors.add(value)
    if int((code == 2).sum()) != 1 or len(colors) != 1 or int((code == 1).sum()) < 3:
        return None
    return {"code": code, "top": top, "left": left, "color": colors.pop()}


def _candidates(grid):
    """All (background, marker, marker squares, reference) readings of an input."""
    grid = np.asarray(grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] * 2 <= grid.size:
        return []
    found = []
    for marker in sorted(int(v) for v in np.unique(grid)):
        if marker == background:
            continue
        squares = _square_components(grid, marker)
        if squares is None or len(squares) < 2:
            continue
        for index, (top, left, side) in enumerate(squares):
            if side != 1:
                continue
            ref = _reference(grid, background, marker, top, left)
            if ref is not None:
                found.append((background, marker, squares, index, ref))
    return found


def _complete(grid, candidate, direction, law, transform=_rot):
    """Complete every copy under one reading, one direction and one turn law."""
    background, marker, squares, ref_index, ref = candidate
    code, ref_color = ref["code"], ref["color"]
    height, width = grid.shape
    expected_in = np.full(grid.shape, background, dtype=np.int8)
    ref_box = (ref["top"], ref["left"], ref["top"] + code.shape[0],
               ref["left"] + code.shape[1])
    for (a, b) in np.argwhere(code != 0):
        expected_in[ref["top"] + a, ref["left"] + b] = marker if code[a, b] == 2 else ref_color
    expected_out = expected_in.copy()
    occupied = expected_in != background
    scales = [side for i, (_, _, side) in enumerate(squares) if i != ref_index]
    copies, payloads = [], set()
    for index, (mrow, mcol, scale) in enumerate(squares):
        if index == ref_index:
            continue
        turns = (direction * law(scale, scales)) % 4
        rotated = transform(code, turns)
        prow, pcol = (int(v) for v in np.argwhere(rotated == 2)[0])
        top, left = mrow - prow * scale, mcol - pcol * scale
        fh, fw = rotated.shape[0] * scale, rotated.shape[1] * scale
        if top < 0 or left < 0 or top + fh > height or left + fw > width:
            return None
        footprint = np.kron(rotated != 0, np.ones((scale, scale), dtype=bool))
        if (occupied[top:top + fh, left:left + fw] & footprint).any():
            return None
        occupied[top:top + fh, left:left + fw] |= footprint
        supplied, missing, colors = [], 0, set()
        for (a, b) in np.argwhere(rotated == 1):
            block = grid[top + a * scale:top + (a + 1) * scale,
                         left + b * scale:left + (b + 1) * scale]
            value = int(block.flat[0])
            if (block != value).any():
                return None
            if value == background:
                missing += 1
            elif value in (marker, ref_color):
                return None
            else:
                supplied.append((int(a), int(b)))
                colors.add(value)
        if not supplied or len(colors) != 1:
            return None
        payload = colors.pop()
        if payload in payloads or (scale == 1 and missing == 0):
            return None
        payloads.add(payload)
        for target, everything in ((expected_in, False), (expected_out, True)):
            for (a, b) in np.argwhere(rotated != 0):
                a, b = int(a), int(b)
                if rotated[a, b] == 2:
                    color = marker
                elif everything or (a, b) in supplied:
                    color = payload
                else:
                    continue
                target[top + a * scale:top + (a + 1) * scale,
                       left + b * scale:left + (b + 1) * scale] = color
        copies.append({"scale": scale, "turns": turns, "payload": payload,
                       "marker_at": (mrow, mcol), "supplied": len(supplied),
                       "missing": missing, "box": (top, left, top + fh, left + fw)})
    if not any(c["missing"] for c in copies):
        return None
    if not np.array_equal(expected_in, grid):
        return None
    return {"output": expected_out, "background": background, "marker": marker,
            "reference_color": ref_color, "code": code, "reference_box": ref_box,
            "direction": direction, "copies": copies}


def _interpret(grid, law=_law_growth, candidates=None, transform=_rot):
    grid = np.asarray(grid)
    if candidates is None:
        candidates = _candidates(grid)
    models = []
    for candidate in candidates:
        for direction in (CW, CCW):
            model = _complete(grid, candidate, direction, law, transform)
            if model is not None:
                models.append(model)
    return models


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid).astype(np.int8)
    models = _interpret(grid)
    if not models:
        raise ValueError("no consistent growth-turn stencil reading")
    return models[0]["output"].astype(int)


# --------------------------------------------------------------------------
# evidence analysis

def _own_options(grid, background, code, marker_at, scale, payload):
    return set(_own_turns(grid, background, code, marker_at, scale, payload).values())


def _own_turns(grid, background, code, marker_at, scale, payload):
    """Turns (and their completions) a solver would accept from one copy's own blocks.

    Every quarter-turn about the registration block is tried.  A turn is
    accepted when its footprint fits, every payload cell lies in one of its
    foreground blocks, and each of those blocks is wholly payload or wholly
    background.  The scale-to-turn law is deliberately ignored here.
    """
    mrow, mcol = marker_at
    height, width = grid.shape
    payload_mask = grid == payload
    options = {}
    for turns in range(4):
        rotated = _rot(code, turns)
        prow, pcol = (int(v) for v in np.argwhere(rotated == 2)[0])
        top, left = mrow - prow * scale, mcol - pcol * scale
        if (top < 0 or left < 0 or top + rotated.shape[0] * scale > height
                or left + rotated.shape[1] * scale > width):
            continue
        cover = np.zeros(grid.shape, dtype=bool)
        blocks, ok = [], True
        for (a, b) in np.argwhere(rotated == 1):
            region = (slice(top + a * scale, top + (a + 1) * scale),
                      slice(left + b * scale, left + (b + 1) * scale))
            values = np.unique(grid[region])
            if len(values) != 1 or int(values[0]) not in (background, payload):
                ok = False
                break
            cover[region] = True
            blocks.append((int(a) - prow, int(b) - pcol))
        if ok and not (payload_mask & ~cover).any():
            options[turns] = frozenset(blocks)
    return options


def _analyze_pair(source, answer):
    source = np.asarray(source)
    answer = np.asarray(answer)
    candidates = _candidates(source)
    models = _interpret(source, _law_growth, candidates)
    assert models, "input has no growth-turn reading"
    readings = {(m["marker"], m["reference_box"]) for m in models}
    assert len(readings) == 1, "input has more than one stencil reading"
    assert all(np.array_equal(m["output"], models[0]["output"]) for m in models), \
        "turn direction is not determined where it matters"
    assert np.array_equal(models[0]["output"], answer), "output is not the completion"
    model = models[0]
    code = model["code"]
    assert _orientation_count(code) >= 2, "stencil is quarter-turn symmetric"
    copies = []
    for copy in model["copies"]:
        turns = _own_turns(source, model["background"], code,
                           copy["marker_at"], copy["scale"], copy["payload"])
        lawful = {turns[t] for t in {(copy["scale"] - 1) % 4, (1 - copy["scale"]) % 4}
                  if t in turns}
        # own: completions allowed by this copy's blocks alone; lawful: those
        # also allowed by the growth law when the direction is still unknown.
        copies.append(dict(copy, own=len(set(turns.values())), lawful=len(lawful)))
    assert 1 <= len(copies) <= 3, "one to three copies per input"
    assert all(1 <= c["scale"] <= 4 for c in copies)
    assert any(c["scale"] >= 2 for c in copies), "no enlarged copy"
    refuted = {}
    for name, (law, transform) in _ALTERNATIVES.items():
        alt = _interpret(source, law, candidates, transform)
        refuted[name] = not any(np.array_equal(m["output"], answer) for m in alt)
    return {"directions": tuple(sorted({m["direction"] for m in models})),
            "copies": copies, "refuted": refuted, "code": code,
            "background": model["background"], "marker": model["marker"],
            "reference_color": model["reference_color"],
            "orientations": _orientation_count(code), "shape": source.shape}


def _visible_scales(info):
    return {c["scale"] for c in info["copies"] if c["scale"] >= 2 and c["own"] == 1}


def _has_ambiguous(info):
    return any(c["own"] >= 2 for c in info["copies"])


def _determined(info):
    return info["directions"][0] if len(info["directions"]) == 1 else None


# --------------------------------------------------------------------------
# generation

def _sample_half_turn(rng):
    """A 3x3 stencil centred on its registration cell with only half-turn symmetry."""
    pairs = (((0, 0), (2, 2)), ((0, 1), (2, 1)), ((0, 2), (2, 0)), ((1, 0), (1, 2)))
    while True:
        chosen = [pairs[i] for i in range(4) if rng.random() < 0.55]
        code = np.zeros((3, 3), dtype=np.int8)
        for cell_a, cell_b in chosen:
            code[cell_a] = code[cell_b] = 1
        code[1, 1] = 2
        if (int((code == 1).sum()) >= 4 and int((code == 0).sum()) > 0
                and _connected(code != 0) and _orientation_count(code) == 2):
            return code


def _sample_stencil(rng, half_turn=False):
    if half_turn:
        return _sample_half_turn(rng)
    for _ in range(500):
        if rng.random() < 0.2:
            form = _SOURCE_FORMS[int(rng.integers(len(_SOURCE_FORMS)))]
            code = np.array([[{"X": 1, "M": 2, ".": 0}[ch] for ch in row]
                             for row in form], dtype=np.int8)
            code = np.rot90(code, int(rng.integers(4)))
            if rng.integers(2):
                code = np.fliplr(code)
            code = code.copy()
        else:
            height, width = _SIZES[int(rng.integers(len(_SIZES)))]
            occupied = rng.random((height, width)) < rng.uniform(0.4, 0.85)
            if not (occupied[0].any() and occupied[-1].any()
                    and occupied[:, 0].any() and occupied[:, -1].any()):
                continue
            count = int(occupied.sum())
            if count < 4 or count == occupied.size or not _connected(occupied):
                continue
            cells = np.argwhere(occupied)
            code = occupied.astype(np.int8)
            code[tuple(cells[int(rng.integers(len(cells)))])] = 2
        if _orientation_count(code) == 4:
            return code
    return np.array([[1, 0, 0], [2, 1, 1]], dtype=np.int8)


def _layout(rng, code, copies, wide):
    """Place copies (scale, turns) and then the reference; return geometry."""
    footprints, markers = [], []
    for scale, turns in copies:
        rotated = _rot(code, turns)
        footprints.append(np.kron(rotated != 0, np.ones((scale, scale), dtype=bool)))
        prow, pcol = (int(v) for v in np.argwhere(rotated == 2)[0])
        markers.append((prow * scale, pcol * scale, scale))
    need_h = max(f.shape[0] for f in footprints)
    need_w = max(f.shape[1] for f in footprints)
    colored = sum(int(f.sum()) for f in footprints) + int((code != 0).sum())
    if need_h > 28 or need_w > 28:
        return None
    for _ in range(6):
        low = 12 if wide else 8
        height = int(rng.integers(max(low, need_h + 2), 31))
        width = int(rng.integers(max(low, need_w + 2), 31))
        if colored > 0.40 * height * width:
            continue
        occupied = np.zeros((height, width), dtype=bool)
        halo = np.zeros((height, width), dtype=bool)
        boxes, places = [], [None] * len(copies)
        order = sorted(range(len(copies)), key=lambda i: -int(footprints[i].sum()))
        failed = False
        for i in order:
            fp = footprints[i]
            fh, fw = fp.shape
            mr, mc, scale = markers[i]
            near = _dilate(occupied)
            free, interlock, adjacent = [], [], []
            for top in range(height - fh + 1):
                for left in range(width - fw + 1):
                    if (occupied[top:top + fh, left:left + fw] & fp).any():
                        continue
                    if halo[top + mr:top + mr + scale, left + mc:left + mc + scale].any():
                        continue
                    free.append((top, left))
                    box = (top, left, top + fh, left + fw)
                    if any(_boxes_overlap(box, other) for other in boxes):
                        interlock.append((top, left))
                    if (near[top:top + fh, left:left + fw] & fp).any():
                        adjacent.append((top, left))
            if not free:
                failed = True
                break
            mode = int(rng.choice(3, p=[0.5, 0.28, 0.22]))
            pool = interlock if mode == 1 else adjacent if mode == 2 else free
            if not pool:
                pool = free
            top, left = pool[int(rng.integers(len(pool)))]
            places[i] = (top, left)
            occupied[top:top + fh, left:left + fw] |= fp
            block = np.zeros((height, width), dtype=bool)
            block[top + mr:top + mr + scale, left + mc:left + mc + scale] = True
            halo |= _dilate(block)
            boxes.append((top, left, top + fh, left + fw))
        if failed:
            continue
        th, tw = code.shape
        spots = [(t, l) for t in range(height - th + 1) for l in range(width - tw + 1)
                 if not any(_boxes_overlap((t - 1, l - 1, t + th + 1, l + tw + 1), b)
                            for b in boxes)]
        if not spots:
            continue
        return (height, width), spots[int(rng.integers(len(spots)))], places
    return None


def _paint_block(grid, top, left, a, b, scale, color):
    grid[top + a * scale:top + (a + 1) * scale,
         left + b * scale:left + (b + 1) * scale] = color


def _choose_evidence(rng, grid, background, code, turns, place, scale, payload, kind):
    """Paint a supplied subset of one copy's blocks that has the requested own-evidence kind."""
    rotated = _rot(code, turns)
    prow, pcol = (int(v) for v in np.argwhere(rotated == 2)[0])
    marker_at = (place[0] + prow * scale, place[1] + pcol * scale)
    blocks = [tuple(int(v) for v in x) for x in np.argwhere(rotated == 1)]
    n = len(blocks)

    def painted(chosen):
        trial = grid.copy()
        for a, b in chosen:
            _paint_block(trial, place[0], place[1], a, b, scale, payload)
        return trial

    def options(trial):
        return len(_own_options(trial, background, code, marker_at, scale, payload))

    def relay(trial):
        # Both law-permitted turns (k-1 either way) fit and complete differently,
        # so only the direction shown by another copy decides this one.
        found = _own_turns(trial, background, code, marker_at, scale, payload)
        first, second = (scale - 1) % 4, (1 - scale) % 4
        return first in found and second in found and found[first] != found[second]

    if kind in ("ambiguous", "relay"):
        # Sparse evidence: try every single block, then random pairs and triples.
        trials = [[blocks[i]] for i in rng.permutation(n)]
        for size in (2, 3):
            if size < n:
                trials += [[blocks[i] for i in rng.choice(n, size=size, replace=False)]
                           for _ in range(12)]
        for chosen in trials:
            trial = painted(chosen)
            if relay(trial) if kind == "relay" else options(trial) >= 2:
                return trial
        return None
    for attempt in range(16):
        if kind == "visible" and scale >= 2 and attempt == 0 and rng.random() < 0.08:
            size = n
        elif kind == "visible":
            size = int(rng.integers(max(1, (n + 1) // 2), n)) if attempt < 12 else n - 1
        else:
            size = int(rng.integers(1, n))
        trial = painted([blocks[i] for i in rng.choice(n, size=size, replace=False)])
        if kind == "any" or options(trial) == 1:
            return trial
    return None


def _make_pair(rng, spec, palette, used):
    for _ in range(12):
        scales = spec["scales"](rng)
        kinds = spec["kinds"](rng, scales)
        direction = spec["direction"]
        copies = [(k, (direction * (k - 1)) % 4) for k in scales]
        for _ in range(4):
            made = _build_pair(rng, spec, palette, used, copies, kinds)
            if made is not None:
                return made
    return None


def _build_pair(rng, spec, palette, used, copies, kinds):
    half_turn = spec.get("half", False) and rng.random() < 0.1
    code = _sample_stencil(rng, half_turn)
    layout = _layout(rng, code, copies, spec["wide"])
    if layout is None:
        return None
    (height, width), (rtop, rleft), places = layout
    background, marker, ref_color, payloads = palette(rng, len(copies))
    source = np.full((height, width), background, dtype=np.int8)
    for (a, b) in np.argwhere(code != 0):
        source[rtop + a, rleft + b] = marker if code[a, b] == 2 else ref_color
    answer = source.copy()
    for (scale, turns), (top, left), payload in zip(copies, places, payloads):
        rotated = _rot(code, turns)
        for (a, b) in np.argwhere(rotated != 0):
            color = marker if rotated[a, b] == 2 else payload
            _paint_block(answer, top, left, int(a), int(b), scale, color)
            if rotated[a, b] == 2:
                _paint_block(source, top, left, int(a), int(b), scale, marker)
    # Sparse (ambiguous) copies receive their evidence last: other copies'
    # blocks can only remove candidate turns, never add them.
    order = sorted(rng.permutation(len(copies)).tolist(),
                   key=lambda i: kinds[i] in ("ambiguous", "relay"))
    for i in order:
        scale, turns = copies[i]
        source = _choose_evidence(rng, source, background, code, turns, places[i],
                                  scale, payloads[i], kinds[i])
        if source is None:
            return None
    if np.array_equal(source, answer) or _key(source) in used:
        return None
    try:
        info = _analyze_pair(source, answer)
    except AssertionError:
        return None
    if not spec["accept"](info):
        return None
    return {"input": source.astype(int), "output": answer.astype(int)}, info


def _pick(rng, options, weights):
    weights = np.asarray(weights, dtype=float)
    return options[int(rng.choice(len(options), p=weights / weights.sum()))]


def _random_scales(rng, count_weights, scale_weights):
    count = _pick(rng, (1, 2, 3), count_weights)
    while True:
        scales = [_pick(rng, (1, 2, 3, 4), scale_weights) for _ in range(count)]
        if any(s >= 2 for s in scales):
            return sorted(scales)


def _kinds_visible(rng, scales):
    return ["visible" if s >= 2 else "any" for s in scales]


def _kinds_mixed(probability_ambiguous):
    def kinds(rng, scales):
        result = []
        for s in scales:
            u = rng.random()
            if u < probability_ambiguous:
                result.append("ambiguous")
            elif u < probability_ambiguous + 0.55 * (1 - probability_ambiguous):
                result.append("visible")
            else:
                result.append("any")
        return result
    return kinds


def _kinds_challenge(rng, scales):
    kinds = [_pick(rng, ("visible", "any"), (0.6, 0.4)) for _ in scales]
    big = [i for i, s in enumerate(scales) if s >= 2]
    kinds[big[int(rng.integers(len(big)))]] = "ambiguous"
    return kinds


def _kinds_relay(rng, scales):
    """One odd-stage copy shows its turn, another odd-stage copy is sparse."""
    kinds = [_pick(rng, ("visible", "any"), (0.5, 0.5)) for _ in scales]
    odd = [i for i, s in enumerate(scales) if s in (2, 4)]
    witness, sparse = (odd[int(i)] for i in rng.permutation(len(odd))[:2])
    kinds[witness], kinds[sparse] = "visible", "relay"
    return kinds


def _is_relay(info):
    return any(c["own"] >= 2 and c["lawful"] >= 2 for c in info["copies"])


_RELAY = ([2, 4], [2, 2], [1, 2, 4], [2, 3, 4], [2, 2, 3], [2, 4], [2, 2, 4])


_LADDER = ([2, 3], [2, 3], [1, 2, 3], [2, 4], [3, 4], [2, 3], [1, 2, 4], [2, 3, 4])
_GAP_ANY = ([3], [1, 3], [3], [3, 4], [2, 4], [1, 2, 4], [3, 3], [4, 3])
_GAP_TURN = ([2, 4], [3, 4], [1, 2, 4], [2, 4], [1, 3, 4])

# Puzzle-level train/test shift modes.
#   sparse:      same configuration ranges; tests hold sparsely evidenced copies.
#   extrapolate: training stops at scale 3 (mostly one or two copies); the first
#                test adds a scale-4 stage (one more quarter-turn) among two or
#                three copies.
#   familiar:    tests use the training configuration distribution.
_MODES = ("sparse", "extrapolate", "familiar")
_MODE_WEIGHTS = (0.42, 0.33, 0.25)


def _choice_of(sets, cap):
    allowed = [list(s) for s in sets if max(s) <= cap]
    return lambda r: list(allowed[int(r.integers(len(allowed)))])


def _with_stage_four(scales_fn):
    def scales(r):
        result = scales_fn(r)
        if 4 not in result:
            result[int(np.argmax(result))] = 4
        return sorted(result)
    return scales


def _spec(rng, role, direction, mixed, mode):
    cap = 3 if mode == "extrapolate" else 4
    if cap == 3:
        train_counts, train_scales = (0.55, 0.45, 0.0), (0.15, 0.45, 0.4, 0.0)
    else:
        train_counts, train_scales = (0.5, 0.38, 0.12), (0.14, 0.36, 0.32, 0.18)
    test_counts, test_scales = (0.1, 0.4, 0.5), (0.04, 0.26, 0.3, 0.4)
    if mode == "familiar":
        test_counts, test_scales = train_counts, train_scales
    elif mode == "extrapolate":
        test_counts = (0.0, 0.5, 0.5)
    spec = {"direction": direction, "wide": False}
    if role == "ladder":
        spec["scales"] = _choice_of(_LADDER, cap)
        spec["kinds"] = _kinds_visible
        spec["accept"] = lambda info: (len(_visible_scales(info)) >= 2
                                       and _determined(info) == direction)
    elif role == "gap":
        turn_sets = mixed and cap == 4
        spec["scales"] = _choice_of(_GAP_TURN if turn_sets else _GAP_ANY, cap)
        spec["kinds"] = _kinds_visible
        spec["accept"] = lambda info: (info["refuted"]["rank"]
                                       and (not turn_sets or _determined(info) == direction))
    elif role == "free" and rng.random() < 0.12:
        spec["scales"] = _choice_of(_RELAY, cap)
        spec["kinds"] = _kinds_relay
        spec["accept"] = _is_relay
    elif role == "free":
        spec["scales"] = lambda r: _random_scales(r, train_counts, train_scales)
        spec["kinds"] = _kinds_mixed(0.15)
        spec["half"] = True
        spec["accept"] = lambda info: True
    elif role == "challenge" and rng.random() < 0.35:
        spec["wide"] = mode != "familiar"
        spec["scales"] = _choice_of([s for s in _RELAY if mode != "extrapolate" or 4 in s], 4)
        spec["kinds"] = _kinds_relay
        spec["accept"] = lambda info: (_is_relay(info) and info["refuted"]["no_turn"]
                                       and _determined(info) == direction)
    elif role == "challenge":
        spec["wide"] = mode != "familiar"
        spec["scales"] = lambda r: _random_scales(r, test_counts, test_scales)
        if mode == "extrapolate":
            spec["scales"] = _with_stage_four(spec["scales"])
        spec["kinds"] = _kinds_challenge
        spec["accept"] = lambda info: (_has_ambiguous(info) and info["refuted"]["no_turn"]
                                       and _determined(info) in (None, direction))
    else:  # additional test pair
        spec["wide"] = mode != "familiar" and rng.random() < 0.5
        spec["scales"] = lambda r: _random_scales(r, test_counts, test_scales)
        spec["kinds"] = _kinds_mixed(0.4)
        spec["half"] = True
        spec["accept"] = lambda info: _determined(info) in (None, direction)
    return spec


def _palette_factory(rng):
    mode = _pick(rng, ("stable", "background", "fresh"), (0.3, 0.3, 0.4))
    base = [int(v) for v in rng.permutation(10)[:3]]

    def palette(r, count):
        if mode == "stable":
            background, marker, ref_color = base
        else:
            background = base[0] if (mode == "background" or r.random() < 0.4) else int(r.integers(10))
            rest = [int(v) for v in r.permutation(10) if v != background]
            marker, ref_color = rest[0], rest[1]
        rest = [int(v) for v in r.permutation(10) if v not in (background, marker, ref_color)]
        return background, marker, ref_color, rest[:count]
    return palette


def _attempt(rng, num_train, num_test):
    mixed = rng.random() < 0.6
    first = CW if rng.random() < 0.5 else CCW
    mode = _pick(rng, _MODES, _MODE_WEIGHTS)
    palette = _palette_factory(rng)
    roles = ["ladder", "gap"] + ["free"] * (num_train - 2)
    used, train, infos = set(), [], []
    for index, role in enumerate(roles):
        # Mixed puzzles alternate the turn direction between training pairs.
        direction = -first if mixed and index % 2 else first
        made = _make_pair(rng, _spec(rng, role, direction, mixed, mode), palette, used)
        if made is None:
            return None
        pair, info = made
        used.add(_key(pair["input"]))
        train.append(pair)
        infos.append(info)
    if not _training_ok(infos):
        return None
    trained = sorted({_determined(i) for i in infos} - {None})
    tests, test_infos = [], []
    for index in range(num_test):
        direction = trained[int(rng.integers(len(trained)))]
        role = "challenge" if index == 0 else "extra"
        made = _make_pair(rng, _spec(rng, role, direction, mixed, mode), palette, used)
        if made is None:
            return None
        pair, info = made
        used.add(_key(pair["input"]))
        tests.append(pair)
        test_infos.append(info)
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    order = rng.permutation(num_test)
    tests = [tests[int(i)] for i in order]
    return {"train": train, "test": tests}


def _training_ok(infos):
    visible = set()
    for info in infos:
        visible |= _visible_scales(info)
    if len(visible) < 2:
        return False
    if not all(any(info["refuted"][name] for info in infos) for name in _ALTERNATIVES):
        return False
    return any(_determined(info) is not None for info in infos)


def _counts(num_train, num_test):
    def as_int(value, name):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)
    train, test = as_int(num_train, "num_train"), as_int(num_test, "num_test")
    if test is None:
        test = 1
    if train is None:
        train = max(4, test + 1)
    if not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    for _ in range(80):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not build a valid growth-turn puzzle")


# --------------------------------------------------------------------------
# validation

def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "bad grid size"
    assert np.issubdtype(grid.dtype, np.integer), "grid must be integer"
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9, "bad colour"
    return grid.astype(np.int8)


def _puzzle_infos(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "splits must be lists"
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "counts"
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            source, answer = _grid(pair["input"]), _grid(pair["output"])
            assert source.shape == answer.shape, "output keeps the canvas"
            assert _key(source) not in seen, "duplicate input"
            seen.add(_key(source))
            assert not np.array_equal(source, answer), "identity pair"
            info = _analyze_pair(source, answer)
            assert np.all(source[source != answer] == info["background"]), \
                "only background cells change"
            infos[split].append(info)
    return infos


def validate_generated_puzzle(puzzle: dict) -> None:
    infos = _puzzle_infos(puzzle)
    train, test = infos["train"], infos["test"]
    assert _training_ok(train), "training does not demonstrate the growth-turn law"
    trained = {_determined(i) for i in train} - {None}
    for info in test:
        direction = _determined(info)
        assert direction is None or direction in trained, "test direction never demonstrated"
    assert any(_has_ambiguous(i) for i in test), "no test copy needs the sequence law"
    assert any(i["refuted"]["no_turn"] for i in test), "test solvable without turning"


# --------------------------------------------------------------------------
# self test

def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _must_reject(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError("a damaged puzzle was accepted")


def _copy(puzzle):
    return {s: [{k: np.array(v) for k, v in p.items()} for p in puzzle[s]]
            for s in ("train", "test")}


def self_test() -> None:
    one = generate_puzzle(11)
    assert _same(one, generate_puzzle(11)), "nondeterministic"
    assert not _same(one, generate_puzzle(12)), "seed ignored"
    for train, test in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(500 + train, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (2, 2), (3, 4), (6, 5), (True, 1)):
        try:
            generate_puzzle(3, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Damage: wrong completion, missing evidence, unturned completion.
    damaged = _copy(one)
    pair = damaged["test"][0]
    cells = np.argwhere(pair["input"] != pair["output"])
    pair["output"][tuple(cells[0])] = pair["input"][tuple(cells[0])]
    _must_reject(damaged)
    damaged = _copy(one)
    info = _analyze_pair(one["train"][0]["input"], one["train"][0]["output"])
    grid = damaged["train"][0]["input"]
    grid[grid == info["copies"][0]["payload"]] = info["background"]
    _must_reject(damaged)
    # A per-copy solver that ignores the growth law may pick another turn for a
    # sparsely evidenced copy; that completion must be rejected.
    damaged = _copy(one)
    replaced = False
    for p in damaged["test"]:
        info = _analyze_pair(p["input"], p["output"])
        for copy in info["copies"]:
            if copy["own"] < 2:
                continue
            k, (mrow, mcol), colour = copy["scale"], copy["marker_at"], copy["payload"]
            options = _own_options(p["input"], info["background"], info["code"],
                                   copy["marker_at"], k, colour)
            wrong = p["output"].copy()
            wrong[(wrong == colour) & (p["input"] != colour)] = info["background"]
            for option in options:
                trial = wrong.copy()
                for dr, dc in option:
                    trial[mrow + dr * k:mrow + (dr + 1) * k, mcol + dc * k:mcol + (dc + 1) * k] = colour
                if not np.array_equal(trial, p["output"]):
                    p["output"] = trial
                    replaced = True
                    break
            if replaced:
                break
        if replaced:
            break
    assert replaced, "test lacks a sparsely evidenced copy"
    _must_reject(damaged)

    stats = {"stencils": set(), "shapes": set(), "train_scales": set(), "test_scales": set(),
             "directions": set(), "mixed": 0, "ambiguous_train": 0, "complete": 0,
             "interlock": 0, "copy_counts": set(), "backgrounds": set(), "markers": set(),
             "half": 0, "unit_copies": 0, "rank_test": 0, "scale4_test": 0, "pairs": 0,
             "relay_test": 0, "relay_train": 0, "new_stage_test": 0, "palette_mixed": 0}
    signatures = set()
    for seed in range(42):
        counts = ((4, 1), (3, 1), (5, 2), (2, 1), (6, 3), (4, 1), (7, 4))[seed % 7]
        puzzle = generate_puzzle(seed, *counts)
        signatures.add(repr([[p["input"].tolist() for p in puzzle[s]] for s in ("train", "test")]))
        infos = _puzzle_infos(puzzle)
        trained = {_determined(i) for i in infos["train"]} - {None}
        stats["mixed"] += len(trained) == 2
        top = {split: max(c["scale"] for i in infos[split] for c in i["copies"])
               for split in ("train", "test")}
        stats["new_stage_test"] += top["test"] > top["train"]
        stats["palette_mixed"] += len({(i["background"], i["marker"], i["reference_color"])
                                       for s in ("train", "test") for i in infos[s]}) > 1
        for split in ("train", "test"):
            for info in infos[split]:
                stats["pairs"] += 1
                stats["stencils"].add(_key(info["code"]))
                stats["shapes"].add(info["shape"])
                stats["backgrounds"].add(info["background"])
                stats["markers"].add(info["marker"])
                stats["directions"].update(info["directions"])
                stats["copy_counts"].add(len(info["copies"]))
                stats["half"] += info["orientations"] == 2
                scales = [c["scale"] for c in info["copies"]]
                stats[split + "_scales"].update(scales)
                stats["unit_copies"] += scales.count(1)
                stats["complete"] += sum(c["missing"] == 0 for c in info["copies"])
                boxes = [c["box"] for c in info["copies"]]
                stats["interlock"] += any(_boxes_overlap(a, b) for i, a in enumerate(boxes)
                                          for b in boxes[i + 1:])
                stats["relay_" + split] += _is_relay(info)
                if split == "train":
                    stats["ambiguous_train"] += _has_ambiguous(info)
                else:
                    stats["rank_test"] += info["refuted"]["rank"]
                    stats["scale4_test"] += 4 in scales
    assert len(signatures) == 42, "seeds collide"
    assert len(stats["stencils"]) >= 60 and len(stats["shapes"]) >= 40, "too little geometry variety"
    assert stats["train_scales"] == {1, 2, 3, 4} and stats["test_scales"] == {1, 2, 3, 4}
    assert stats["directions"] == {CW, CCW} and stats["mixed"] >= 8, "direction variety"
    assert stats["copy_counts"] == {1, 2, 3}
    assert len(stats["backgrounds"]) >= 8 and len(stats["markers"]) >= 8, "palette variety"
    assert stats["palette_mixed"] >= 15, "palette roles never change inside puzzles"
    assert min(stats["ambiguous_train"], stats["complete"], stats["interlock"], stats["half"],
               stats["unit_copies"], stats["rank_test"], stats["scale4_test"],
               stats["relay_test"], stats["relay_train"], stats["new_stage_test"]) > 0, stats


if __name__ == "__main__":
    self_test()
    print("self_test ok")
