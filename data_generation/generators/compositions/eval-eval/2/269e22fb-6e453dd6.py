"""Texture portals: crop registration controls an enclosure-gated projection.

Only NumPy and the standard library are used. No files, saved templates, parent
generators, labels, seeds, or generation metadata are consulted by solve().
"""
from collections import deque
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    queue.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = np.asarray(cells).T
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _holes(mask):
    h, w = mask.shape
    return [set(c) for c in _components(~mask)
            if all(0 < r < h - 1 and 0 < col < w - 1 for r, col in c)]


def _matches(reference, patch):
    """Include even matches that would have an empty outgoing continuation."""
    result = []
    ph, pw = patch.shape
    for t in range(8):
        full = _d4(reference, t)
        if ph > full.shape[0] or pw > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, patch.shape)
        for swap in (0, 1):
            rr, cc = np.nonzero(np.all(windows == (patch ^ swap), axis=(-2, -1)))
            result.extend((t, swap, int(r), int(c)) for r, c in zip(rr, cc))
    return result


def _scene(grid):
    """Infer all six color roles, the rail direction, and the reference card."""
    choices = []
    for turn in range(4):
        a = np.rot90(grid, turn).copy()
        h, w = a.shape
        bg = int(a[0, -1])
        for col in range(2, w - 1):
            rail = int(a[0, col])
            if (rail == bg or not np.all(a[:, col] == rail)
                    or np.count_nonzero(a == rail) != h
                    or not np.all(a[:, col + 1:] == bg)):
                continue
            colors = set(map(int, np.unique(a)))
            if len(colors) != 6:
                continue
            for frame in sorted(colors - {bg, rail}):
                cells = list(zip(*np.nonzero(a == frame)))
                r0, r1, c0, c1 = _bbox(cells)
                fh, fw = r1 - r0, c1 - c0
                if min(fh, fw) < 6 or c1 > col:
                    continue
                border = np.ones((fh, fw), dtype=bool)
                border[2:-2, 2:-2] = False
                if not np.array_equal(a[r0:r1, c0:c1] == frame, border):
                    continue
                tex = a[r0 + 2:r1 - 2, c0 + 2:c1 - 2]
                palette = tuple(map(int, np.unique(tex)))
                if len(palette) != 2 or set(palette) & {bg, rail, frame}:
                    continue
                other = colors - {bg, rail, frame, *palette}
                if len(other) != 1:
                    continue
                wall = other.pop()
                choices.append(dict(a=a, turn=turn, bg=bg, rail=rail, col=col,
                                    frame=frame, card=(r0, r1, c0, c1), palette=palette,
                                    reference=(tex == palette[1]).astype(np.int8), wall=wall))
    _require(len(choices) == 1, "rail direction and double-framed reference must be unique")
    return choices[0]


def _objects(scene):
    a, wall, bg, col = (scene[k] for k in ("a", "wall", "bg", "col"))
    p0, p1 = scene["palette"]
    objects, used = [], np.zeros(a.shape, dtype=bool)
    cr0, cr1, cc0, cc1 = scene["card"]
    used[cr0:cr1, cc0:cc1] = True
    previous_end = -1
    for cells in _components(a[:, :col] == wall):
        r0, r1, c0, c1 = _bbox(cells)
        _require(r0 > previous_end, "carrier row bands must have a blank separating row")
        previous_end = r1
        _require(r1 <= cr0 or r0 >= cr1 + 1, "reference and carriers need a blank separating row")
        local = a[r0:r1, c0:c1]
        mask = local == wall
        _require(len(cells) == int(mask.sum()), "carrier bounding boxes overlap")
        _require(not np.any(used[r0:r1, c0:c1]), "overlapping payloads")
        _require(set(map(int, np.unique(local))) <= {bg, wall, p0, p1}, "unknown payload color")
        used[r0:r1, c0:c1] = True
        holes = _holes(mask)
        samples = []
        texture = (local == p0) | (local == p1)
        for pixels in _components(texture):
            q0, q1, x0, x1 = _bbox(pixels)
            _require(len(pixels) == (q1 - q0) * (x1 - x0), "texture samples must be rectangles")
            patch = local[q0:q1, x0:x1]
            _require(len(np.unique(patch)) == 2, "both texture roles must be visible")
            enclosed = set(pixels) in holes
            active = enclosed and x1 == mask.shape[1] - 1
            samples.append(dict(row=r0 + q0, col=c0 + x0, patch=patch.copy(),
                                enclosed=enclosed, active=active))
        _require(all(any(hole == set(c) for c in _components(texture)) for hole in holes),
                 "every enclosed cavity must be filled by exactly one texture sample")
        _require(c1 <= col, "carrier crosses the rail")
        objects.append(dict(box=(r0, r1, c0, c1), local=local.copy(),
                            shift=col - c1, samples=samples, mask=mask))
    _require(bool(objects), "no carriers")
    used[:, col:] = True
    _require(not np.any((a != bg) & ~used), "unassigned cells outside card/carriers")
    return objects


def _registration(scene, sample):
    patch = (sample["patch"] == scene["palette"][1]).astype(np.int8)
    matches = _matches(scene["reference"], patch)
    _require(len(matches) == 1, "sample must have exactly one orientation, role map and offset")
    t, swap, u, v = matches[0]
    full = _d4(scene["reference"], t) ^ swap
    edge = v + patch.shape[1]
    _require(edge < full.shape[1], "sample must leave an outgoing suffix")
    suffix = full[:, edge:]
    _require(0 < int(suffix.sum()) < suffix.size, "continuation must contain both texture roles")
    return dict(t=t, swap=swap, u=u, v=v, full=full, suffix=suffix,
                row=sample["row"] - u, col=scene["col"] + 1)


def _render(scene, objects, gates=True, register=True):
    """Render from pixels; flags implement explicit concept ablations."""
    a, bg = scene["a"], scene["bg"]
    result = a.copy()
    occupied = np.zeros(a.shape, dtype=bool)
    registrations = []
    for obj in objects:
        r0, r1, c0, c1 = obj["box"]
        result[r0:r1, c0:c1] = bg
    for obj in objects:
        r0, r1, c0, c1 = obj["box"]
        result[r0:r1, c0 + obj["shift"]:c1 + obj["shift"]] = obj["local"]
        for sample in obj["samples"]:
            reg = _registration(scene, sample)
            if gates and not sample["active"]:
                continue
            r, c = reg["row"], reg["col"]
            tile = reg["suffix"]
            if not register:
                # A plausible local continuation: repeat the visible window,
                # using the same bounding footprint to isolate pixel recognition.
                patch = (sample["patch"] == scene["palette"][1]).astype(np.int8)
                tile = np.tile(patch, (1 + tile.shape[0] // patch.shape[0],
                                      1 + tile.shape[1] // patch.shape[1]))[:tile.shape[0], :tile.shape[1]]
            hh, ww = tile.shape
            _require(0 <= r and r + hh <= a.shape[0] and c + ww <= a.shape[1],
                     "outgoing continuation must fit the canvas")
            _require(not np.any(occupied[r:r + hh, c:c + ww]), "projected rectangles overlap")
            _require(np.all(a[r:r + hh, c:c + ww] == bg), "projection side is not blank")
            occupied[r:r + hh, c:c + ww] = True
            p0, p1 = scene["palette"]
            result[r:r + hh, c:c + ww] = np.where(tile, p1, p0)
            registrations.append(reg)
    return np.rot90(result, -scene["turn"]).copy(), registrations


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _scene(_grid(input_grid))
    return _render(scene, _objects(scene))[0]


def _crop_bank(reference):
    """Index small windows under all D4/color candidates, including edge matches."""
    index = {}
    for t in range(8):
        full = _d4(reference, t)
        h, w = full.shape
        for ph in range(2, h + 1):
            for pw in range(2, w + 1):
                for r in range(h - ph + 1):
                    for c in range(w - pw + 1):
                        patch = full[r:r + ph, c:c + pw]
                        for swap in (0, 1):
                            key = _key(patch ^ swap)
                            value = (t, swap, r, c, ph, pw)
                            index[key] = value if key not in index else None
    bank = []
    for item in index.values():
        if item is None:
            continue
        t, swap, u, v, ph, pw = item
        full = _d4(reference, t) ^ swap
        if ph >= full.shape[0] or v + pw >= full.shape[1]:
            continue
        patch, suffix = full[u:u + ph, v:v + pw], full[:, v + pw:]
        if len(np.unique(patch)) != 2 or len(np.unique(suffix)) != 2:
            continue
        # Reject exact local-repeat shortcuts at construction, not just in tests.
        repeat = np.tile(patch, (1 + suffix.shape[0] // ph, 1 + suffix.shape[1] // pw))[:suffix.shape[0], :suffix.shape[1]]
        if np.array_equal(repeat, suffix):
            continue
        bank.append(item)
    return bank


def _reference(rng):
    h, w = int(rng.integers(4, 7)), int(rng.integers(5, 9))
    for attempt in range(80):
        a = np.zeros((h, w), dtype=np.int8)
        for _ in range(int(rng.integers(4, 9))):
            r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
            c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
            kind = int(rng.integers(4))
            if kind == 0:  # bend
                a[r0:r1 + 1, c0] = 1
                a[r1, c0:c1 + 1] = 1
            elif kind == 1:  # loop
                a[r0:r1 + 1, c0] = a[r0:r1 + 1, c1] = 1
                a[r0, c0:c1 + 1] = a[r1, c0:c1 + 1] = 1
            elif kind == 2:  # separated horizontal or vertical bars
                a[r0, c0:c1 + 1] = 1
                if rng.random() < .5:
                    a[r1, c0:c1 + 1] = 1
            else:  # a stepped path
                for r in range(r0, r1 + 1):
                    c = c0 + (r - r0) * (c1 - c0) // max(1, r1 - r0)
                    a[r, c0:c + 1] = 1
        if not .2 <= float(a.mean()) <= .8:
            continue
        if len({_key(_d4(a, t) ^ s) for t in range(8) for s in (0, 1)}) != 16:
            continue
        bank = _crop_bank(a)
        if all(sum(x[0] == t and x[1] == 0 for x in bank) >= 3 for t in range(8)):
            return a, bank
    # A fixed structured emergency reference, never a parent image. Random
    # references are overwhelmingly typical; this path is explicitly bounded.
    a = np.array([[int(c) for c in row] for row in
                  ("1100010", "0101110", "0111011", "0010010", "1110100")], dtype=np.int8)
    bank = _crop_bank(a)
    _require(all(any(x[0] == t and x[1] == 0 for x in bank) for t in range(8)), "fallback crop bank")
    return a, bank


def _record(rng, bank, t, swap):
    choices = [x for x in bank if x[0] == t and x[1] == swap]
    return choices[int(rng.integers(len(choices)))]


def _carrier(rng, reference, records, kinds, starts):
    """Construct one body with one or several independently addressed windows."""
    widths = [x[5] for x in records]
    back = int(rng.integers(1, 4))
    body_width = max(widths) + back + 1 + (1 if "inset" in kinds else 0)
    holes, top, bottom = [], 30, 0
    for rec, kind, slot in zip(records, kinds, starts):
        t, swap, u, v, ph, pw = rec
        row = slot + 1 + u
        right = body_width - 1 - (1 if kind == "inset" else 0)
        holes.append((row, right - pw, _d4(reference, t)[u:u + ph, v:v + pw] ^ swap, kind))
        top, bottom = min(top, row - 1), max(bottom, row + ph + 1)
    body = np.ones((bottom - top, body_width), dtype=np.int8)
    # -1 is empty space, 0/1 are texture; 2 is carrier wall.
    body[:] = 2
    for row, left, patch, kind in holes:
        r, ph, pw = row - top, *patch.shape
        body[r:r + ph, left:left + pw] = patch
        if kind == "open":
            side = int(rng.integers(2 if len(records) > 1 else 4))
            if side == 0:
                body[r, :left] = -1
            elif side == 1:
                body[r, left + pw:] = -1
            elif side == 2:
                body[:r, left + pw - 1] = -1
            else:
                body[r + ph:, left + pw - 1] = -1
    if len(records) == 1 and kinds[0] == "tail":
        extension = int(rng.integers(1, 4))
        extra = np.full((body.shape[0], extension), -1, dtype=np.int8)
        extra[int(rng.choice([0, body.shape[0] - 1])), :] = 2
        body = np.concatenate((body, extra), axis=1)
    # Some multi-window carriers are lobes joined along the front spine.
    if len(records) > 1 and rng.random() < .6:
        for first, second in zip(holes, holes[1:]):
            gap_start = first[0] + first[2].shape[0] + 1
            gap_end = second[0] - 1
            if gap_start < gap_end:
                body[gap_start - top:gap_end - top, :-1] = -1
    _require(len(_components(body == 2)) == 1, "carrier wall disconnected")
    return top, body


def _make_pair(rng, reference, bank, split, index, nt, palette, turn):
    rh, rw = reference.shape
    card_h, card_w = rh + 4, rw + 4
    first = split == "train" and index == 0
    second = split == "train" and index == 1
    group = list(range(8 if nt >= 3 else 4))
    if split == "train":
        anchor = 0 if first else 1 if second else 4 if index == 2 else int(rng.choice(group))
        plan = [(anchor, index % 2, "active")]
        if first or second or rng.random() < .55:
            plan.append((int(rng.choice([0, 2])), int(rng.integers(2)), "open" if first else "tail"))
        elif rng.random() < .3:
            plan.append((int(rng.choice(group)), int(rng.integers(2)), "active"))
        combined = False
    else:
        plan = [(2, int(rng.integers(2)), "active"),
                (3, int(rng.integers(2)), "active")]
        if index > 0:
            plan[0] = (int(rng.choice(group)), int(rng.integers(2)), "active")
            plan[1] = (int(rng.choice(group)), int(rng.integers(2)),
                       str(rng.choice(["active", "active", "inset", "open"])))
        combined = True
    # A third independent window/carrier is possible on shorter references.
    def plan_height(p):
        return sum(_d4(reference, x[0]).shape[0] + 2 for x in p) + len(p) - 1
    base_top = card_h + 1
    extra = (int(rng.choice([0, 2])), int(rng.integers(2)), "active" if combined else "open")
    if plan_height(plan + [extra]) + base_top <= 30 and rng.random() < .35:
        plan.append(extra)
    # Later tests may choose two tall orientations; retain a random valid
    # orientation of the same group instead of clipping any reconstruction.
    while plan_height(plan) + base_top > 30:
        k = max(range(len(plan)), key=lambda j: _d4(reference, plan[j][0]).shape[0])
        t, swap, kind = plan[k]
        plan[k] = (int(rng.choice([0, 2])), swap, kind)
    rng.shuffle(plan)
    records = [_record(rng, bank, t, swap) for t, swap, _ in plan]
    kinds = [p[2] for p in plan]
    slots, cursor = [], base_top
    for j, rec in enumerate(records):
        slots.append(cursor)
        cursor += _d4(reference, rec[0]).shape[0] + 2 + int(j + 1 < len(records))
    slack = 30 - cursor
    # Vary spacing as well as shapes. Every footprint is still fully visible.
    additions = rng.multinomial(int(rng.integers(slack + 1)), np.full(len(slots) + 1, 1 / (len(slots) + 1)))
    for j in range(len(slots)):
        slots[j] += int(sum(additions[:j + 1]))
    height = cursor + int(additions.sum())
    if combined:
        bodies = [_carrier(rng, reference, records, kinds, slots)]
    else:
        bodies = [_carrier(rng, reference, [rec], [kind], [slot])
                  for rec, kind, slot in zip(records, kinds, slots)]
    widest = max(card_w, *(body.shape[1] for _, body in bodies))
    rail_col = int(rng.integers(widest + 1, min(20, 29 - max(rh, rw)) + 1))
    band_min = max(_d4(reference, rec[0]).shape[1] - rec[3] - rec[5] for rec in records)
    width = int(rng.integers(rail_col + 1 + band_min, 31))
    bg, rail, wall, frame, p0, p1 = palette
    a = np.full((height, width), bg, dtype=np.int8)
    a[:, rail_col] = rail
    card_left = int(rng.integers(rail_col - card_w + 1))
    a[:card_h, card_left:card_left + card_w] = frame
    a[2:card_h - 2, card_left + 2:card_left + card_w - 2] = np.where(reference, p1, p0)
    for j, (row, body) in enumerate(bodies):
        limit = rail_col - body.shape[1]
        # First two demonstrations expose positive travel and zero travel.
        shift = (int(rng.integers(1, limit + 1)) if first and j == 0
                 else 0 if second and j == 0 else int(rng.integers(limit + 1)))
        left = rail_col - body.shape[1] - shift
        painted = np.where(body == 2, wall, np.where(body == 1, p1, np.where(body == 0, p0, bg)))
        a[row:row + body.shape[0], left:left + body.shape[1]] = painted
    a = np.rot90(a, turn).copy()
    return {"input": a, "output": solve(a)}


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    reference, bank = _reference(rng)
    stable = bool(rng.integers(2))
    shared_palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    base_turn, mixed = int(rng.integers(4)), bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for retry in range(40):
                palette = shared_palette if stable else tuple(map(int, rng.choice(10, 6, replace=False)))
                turn = (base_turn + (i if split == "train" else i + 2)) % 4 if mixed else base_turn
                pair = _make_pair(rng, reference, bank, split, i, nt, palette, turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, reference_classes, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _scene(inp)
            objects = _objects(scene)
            reference_classes.add(min(_key(_d4(scene["reference"], t) ^ s) for t in range(8) for s in (0, 1)))
            expected, regs = _render(scene, objects)
            _require(np.array_equal(out, expected), "incorrect output")
            _require(np.array_equal(solve(inp.copy()), out), "input-only solve failed")
            _require(set(map(int, np.unique(inp))) == set(map(int, np.unique(out))), "color roles lost or introduced")
            _require(bool(regs), "generated example must demonstrate a portal")
            if split == "train":
                training.append((scene, objects, out))
    _require(len(reference_classes) == 1, "puzzle does not share a reference")
    samples = [s for _, objects, _ in training for o in objects for s in o["samples"]]
    _require(any(s["active"] for s in samples), "no active training window")
    _require(any(not s["enclosed"] for s in samples), "no open-window contrast")
    _require(any(s["enclosed"] and not s["active"] for s in samples), "no recessed-window contrast")
    _require(any(o["shift"] > 0 for _, objects, _ in training for o in objects), "motion not demonstrated")
    _require(any(o["shift"] == 0 for _, objects, _ in training for o in objects), "already-aligned case missing")
    _require(any(not np.array_equal(_render(sc, obs, gates=False)[0], out) for sc, obs, out in training),
             "omitting enclosure/leading-edge gating does not change any training answer")
    _require(all(not np.array_equal(_render(sc, obs, register=False)[0], out) for sc, obs, out in training),
             "local texture repetition shortcut survived")


def self_test():
    """Bounded, streaming 42-seed suite; no puzzle corpus or trace is saved."""
    references, shapes, masks, crop_shapes, offsets, suffix_shapes = set(), set(), set(), set(), set(), set()
    directions, colors, carrier_counts, window_counts, shifts = set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(pair["output"].tobytes())
                sc = _scene(inp)
                obs = _objects(sc)
                references.add(min(_key(_d4(sc["reference"], t) ^ s) for t in range(8) for s in (0, 1)))
                shapes.add(inp.shape)
                directions.add(sc["turn"])
                colors.update(map(int, np.unique(inp)))
                carrier_counts.add(len(obs))
                for obj in obs:
                    masks.add(_key(obj["mask"]))
                    shifts.add(obj["shift"])
                    window_counts.add(len(obj["samples"]))
                    for sample in obj["samples"]:
                        reg = _registration(sc, sample)
                        crop_shapes.add(sample["patch"].shape)
                        offsets.add((reg["u"], reg["v"]))
                        suffix_shapes.add(reg["suffix"].shape)
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(references) >= 38, "reference/puzzle diversity collapsed")
    _require(len(shapes) >= 35 and len(masks) >= 100, "geometry diversity collapsed")
    _require(len(crop_shapes) >= 10 and len(offsets) >= 12 and len(suffix_shapes) >= 12, "registration diversity collapsed")
    _require(directions == set(range(4)) and colors == set(range(10)), "orientation/palette diversity collapsed")
    _require({1, 2, 3} <= carrier_counts and {1, 2, 3} <= window_counts, "count diversity collapsed")
    _require(0 in shifts and len(shifts) >= 7, "travel distances collapsed")
    a, b = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(408152485, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (2, 2), (4, 0), (6, 5), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "reference_classes": len(references), "grid_shapes": len(shapes),
            "carrier_masks": len(masks), "crop_shapes": len(crop_shapes), "offsets": len(offsets),
            "suffix_shapes": len(suffix_shapes), "carrier_counts": sorted(carrier_counts),
            "windows_per_carrier": sorted(window_counts), "travel_distances": sorted(shifts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
