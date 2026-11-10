"""Interference looms: unary reservoirs and an input-local crossing analogy.

Only Python/NumPy are used. Construction paints logical beams directly;
solve independently reads rectangles, pointed components and their cavities.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_DIR = {"top": (1, 0), "right": (0, -1),
        "bottom": (-1, 0), "left": (0, 1)}
_REACTIONS = ("erase", "horizontal", "vertical", "accent")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _perimeter(top, left, h, w):
    return ({(top, c) for c in range(left, left + w)}
            | {(top + h - 1, c) for c in range(left, left + w)}
            | {(r, left) for r in range(top, top + h)}
            | {(r, left + w - 1) for r in range(top, top + h)})


def _tip(h, w, side, position):
    # Coordinates relative to the interior. A one-cell rim transmits beams.
    return {"top": (-2, position), "bottom": (h + 1, position),
            "left": (position, -2), "right": (position, w + 1)}[side]


def _profile(a, b, tail, stem=0):
    """Thin, 8-connected fork; unequal reaches make off-center tips."""
    frame, cavity = {(y, 0) for y in range(stem + 1)}, set()
    for step in range(1, max(a, b) + tail + 1):
        y = stem + step
        lo, hi = -min(a, step), min(b, step)
        frame.update(((y, lo), (y, hi)))
        cavity.update((y, x) for x in range(lo + 1, hi))
    return frame, cavity


def _reservoir_cells(h, w, emitter):
    side, pos, profile, quantity, arrangement = emitter
    frame, cavity = _profile(*profile)
    tr, tc = _tip(h, w, side, pos)
    dr, dc = _DIR[side]
    # x is the transverse coordinate; y runs backwards from the tip.
    def world(p):
        y, x = p
        return tr - y * dr - x * dc, tc - y * dc + x * dr
    return {world(p) for p in frame}, {world(p) for p in cavity}


def _beam_masks(h, w, emitters, constant=None):
    horizontal, vertical = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for side, pos, _, quantity, _ in emitters:
        q = quantity if constant is None else constant
        if side == "left":
            horizontal[pos, :min(q, w)] = True
        elif side == "right":
            horizontal[pos, max(0, w - q):] = True
        elif side == "top":
            vertical[:min(q, h), pos] = True
        else:
            vertical[max(0, h - q):, pos] = True
    return horizontal, vertical


def _combine(horizontal, vertical, neutral, hcolor, vcolor, reaction, accent):
    out = np.full(horizontal.shape, neutral, dtype=np.int8)
    out[horizontal & ~vertical] = hcolor
    out[vertical & ~horizontal] = vcolor
    collision = {"erase": neutral, "horizontal": hcolor,
                 "vertical": vcolor, "accent": accent}[reaction]
    out[horizontal & vertical] = collision
    return out


def _payload_subset(rng, cavity, q, mode):
    cells = sorted(cavity)
    if mode == "random":
        return {cells[int(i)] for i in rng.choice(len(cells), q, replace=False)}
    selected = {cells[int(rng.integers(len(cells)))]}
    while len(selected) < q:
        remaining = sorted(cavity - selected)
        distances = [min(abs(p[0] - t[0]) + abs(p[1] - t[1])
                         for t in selected) for p in remaining]
        best = min(distances) if mode == "compact" else max(distances)
        choices = [p for p, d in zip(remaining, distances) if d == best]
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _render_board(rng, h, w, emitters, colors, reaction, filled):
    neutral, rim, frame_color, hcolor, vcolor, accent = colors
    all_frame, all_payload = {}, {}
    guards = set()
    for emitter in emitters:
        frame, cavity = _reservoir_cells(h, w, emitter)
        if not frame.isdisjoint(guards) or not cavity.isdisjoint(guards):
            raise ValueError("reservoirs touch")
        guards.update((r + dr, c + dc) for r, c in frame | cavity
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        color = hcolor if emitter[0] in ("left", "right") else vcolor
        all_frame.update({p: frame_color for p in frame})
        all_payload.update({p: color for p in _payload_subset(
            rng, cavity, emitter[3], emitter[4])})
    border = _perimeter(-1, -1, h + 2, w + 2)
    assert border.isdisjoint(all_frame) and border.isdisjoint(all_payload)
    points = border | set(all_frame) | set(all_payload)
    minr, minc = min(r for r, c in points), min(c for r, c in points)
    maxr, maxc = max(r for r, c in points), max(c for r, c in points)
    block = np.full((maxr - minr + 1, maxc - minc + 1), neutral, np.int8)
    for p in border:
        block[p[0] - minr, p[1] - minc] = rim
    for mapping in (all_frame, all_payload):
        for (r, c), color in mapping.items():
            block[r - minr, c - minc] = color
    hm, vm = _beam_masks(h, w, emitters)
    answer = _combine(hm, vm, neutral, hcolor, vcolor, reaction, accent)
    if filled:
        block[-minr:h - minr, -minc:w - minc] = answer
    return block, answer, (-minr, -minc, h, w)


def _pack_boards(rng, reference, target, neutral):
    rh, rw = reference.shape
    th, tw = target.shape
    gap = int(rng.integers(1, 3))
    options = []
    for axis in (0, 1):
        need = ((rh + th + gap, max(rw, tw)) if axis == 0
                else (max(rh, th), rw + tw + gap))
        if max(need) <= 30:
            options.append((axis, need))
    if not options and gap == 2:
        gap = 1
        for axis in (0, 1):
            need = ((rh + th + gap, max(rw, tw)) if axis == 0
                    else (max(rh, th), rw + tw + gap))
            if max(need) <= 30:
                options.append((axis, need))
    if not options:
        raise ValueError("two looms do not fit")
    axis, need = options[int(rng.integers(len(options)))]
    shape = tuple(int(rng.integers(n, 31)) for n in need)
    canvas = np.full(shape, neutral, np.int8)
    blocks = [reference, target]
    if rng.integers(2):
        blocks.reverse()
    along = int(rng.integers(shape[axis] - need[axis] + 1))
    positions = []
    for block in blocks:
        cross = int(rng.integers(shape[1 - axis] - block.shape[1 - axis] + 1))
        r, c = (along, cross) if axis == 0 else (cross, along)
        canvas[r:r + block.shape[0], c:c + block.shape[1]] = block
        positions.append((r, c))
        along += block.shape[axis] + gap
    # Extremes of a block are reservoir centers, never its bounding corners.
    if any(canvas[p] != neutral for p in
           ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0),
            (shape[0] - 1, shape[1] - 1))):
        raise ValueError("canvas corners must show neutral")
    return canvas


def _palette(rng):
    c = [int(x) for x in rng.permutation(10)]
    neutral, rim, frame, rh, rv, th, tv, accent = c[:8]
    mapping = int(rng.integers(3))
    if mapping == 0:
        th, tv = rh, rv
    elif mapping == 1:
        th = rh
    return ((neutral, rim, frame, rh, rv, accent),
            (neutral, rim, frame, th, tv, accent))


def _reference_plan(rng, tiny=False):
    h = 4 if tiny else int(rng.integers(4, 7))
    w = 4 if tiny else int(rng.integers(4, 7))
    qh = 2 if tiny else int(rng.integers(2, min(4, w - 1) + 1))
    qv = 2 if tiny else int(rng.integers(2, min(4, h - 1) + 1))
    ph = (1, 1, max(1, qh - 1), int(rng.integers(2)) if not tiny else 0)
    pv = (1, 1, max(1, qv - 1), int(rng.integers(2)) if not tiny else 0)
    emitters = [("left", 1, ph, qh, "compact"),
                ("top", 1, pv, qv, "scattered")]
    # Reverse either direction while retaining a guaranteed crossing/solo cells.
    if not tiny and rng.integers(2):
        emitters = [("right", 1, ph, qh, "compact"),
                    ("top", w - 2, pv, qv, "scattered")]
    if not tiny and rng.integers(2):
        emitters = [(s, h - 2 if s in ("left", "right") else pos, p, q, a)
                    if s in ("left", "right") else ("bottom", pos, p, q, a)
                    for s, pos, p, q, a in emitters]
    return h, w, emitters


def _witness_plan(rng):
    h, w = int(rng.integers(7, 10)), int(rng.integers(7, 10))
    profile = (1, 1, 3, int(rng.integers(2)))  # capacity four
    emitters = [("left", 1, profile, 4, "compact"),
                ("left", h - 2, profile, 2, "scattered"),
                ("top", 2, (2, 2, 2, 0), 4, "random")]
    return h, w, emitters


def _random_plan(rng, split, fallback=False):
    if fallback:
        h = w = 6
    else:
        h, w = int(rng.integers(6, 10)), int(rng.integers(6, 10))
    if split == "test":
        sides = list(_SIDES)
        clip_side = str(rng.choice(_SIDES))
        duplicate_side = None
        if not fallback and rng.random() < .35:
            duplicate_side = str(rng.choice([s for s in _SIDES if s != clip_side]))
            sides.append(duplicate_side)
            if duplicate_side in ("left", "right"):
                h = max(h, 7)
            else:
                w = max(w, 7)
    else:
        sides = list(rng.choice(_SIDES, int(rng.integers(2, 5)), replace=False))
        clip_side = None
        duplicate_side = None
        if not (set(sides) & {"left", "right"} and set(sides) & {"top", "bottom"}):
            raise ValueError("both beam axes are needed")
    emitters, occupied = [], set()
    for side in sides:
        size = h if side in ("left", "right") else w
        limit = w if side in ("left", "right") else h
        if side == clip_side:
            profile = (2, 2, 3, 0)
            q = limit + 1  # capacity thirteen; one deliberate clipped export
        else:
            profile = ((1, 1, 2, 0) if fallback or side == duplicate_side else
                       (int(rng.integers(1, 3)), int(rng.integers(1, 3)),
                        int(rng.integers(1, 5)), int(rng.integers(2))))
            capacity = len(_profile(*profile)[1])
            q = int(rng.integers(1, min(capacity, limit - 1) + 1))
        frame, cavity = _reservoir_cells(h, w, (side, 0, profile, q, "random"))
        axis = 0 if side in ("left", "right") else 1
        lo, hi = min(p[axis] for p in frame), max(p[axis] for p in frame)
        candidates = list(range(-lo, size - hi))
        rng.shuffle(candidates)
        if side == duplicate_side:
            candidates = [size - 2 if any(e[0] == side for e in emitters) else 1]
        for pos in candidates:
            e = (side, pos, profile, q, str(rng.choice(["compact", "scattered", "random"])))
            f, c = _reservoir_cells(h, w, e)
            guard = {(r + dr, col + dc) for r, col in f | c
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if occupied.isdisjoint(guard):
                occupied.update(f | c)
                emitters.append(e)
                break
        else:
            raise ValueError("cannot fit a reservoir")
    hm, vm = _beam_masks(h, w, emitters)
    if not (np.any(hm & vm) and np.any(hm & ~vm) and np.any(vm & ~hm)):
        raise ValueError("need crossing and both solo roles")
    return h, w, emitters


def _read_pointed_component(cells, direction):
    """Read cross sections; no profile enumeration or generator metadata."""
    dr, dc = direction
    front = max(r * dr + c * dc for r, c in cells)
    tips = [p for p in cells if p[0] * dr + p[1] * dc == front]
    if len(tips) != 1:
        return None
    tip = tips[0]
    rows = {}
    for r, c in cells:
        y = front - r * dr - c * dc
        x = -(r - tip[0]) * dc + (c - tip[1]) * dr
        rows.setdefault(y, []).append(x)
    if set(rows) != set(range(max(rows) + 1)) or rows[0] != [0]:
        return None
    split, left, right, cavity = False, 0, 0, set()
    for y in range(1, max(rows) + 1):
        xs = sorted(rows[y])
        if xs == [0] and not split:
            continue
        if len(xs) != 2:
            return None
        lo, hi = xs
        if not (lo <= 0 <= hi and lo < hi and 0 <= left - lo <= 1
                and 0 <= hi - right <= 1):
            return None
        split, left, right = True, lo, hi
        cavity.update((tip[0] - y * dr - x * dc,
                       tip[1] - y * dc + x * dr) for x in range(lo + 1, hi))
    if not cavity or not left < 0 < right:
        return None
    return tip, cavity


def _read_scene(input_grid):
    grid = _grid(input_grid)
    neutral = int(grid[0, 0])
    assert all(grid[p] == neutral for p in ((0, -1), (-1, 0), (-1, -1)))
    candidates = []
    for color in np.unique(grid):
        if color == neutral:
            continue
        comps = _components(map(tuple, np.argwhere(grid == color)))
        if len(comps) != 2:
            continue
        boxes = []
        for comp in comps:
            rs, cs = zip(*comp)
            top, left = min(rs), min(cs)
            h, w = max(rs) - top + 1, max(cs) - left + 1
            if min(h, w) < 6 or comp != _perimeter(top, left, h, w):
                break
            boxes.append((top, left, h, w))
        if len(boxes) == 2:
            candidates.append((int(color), boxes))
    assert len(candidates) == 1, "need two unambiguous same-color chamber rims"
    rim, boxes = candidates[0]
    boards = []
    covered = set()
    for top, left, height, width in boxes:
        region = {(r, c) for r in range(top, top + height)
                  for c in range(left, left + width)}
        assert covered.isdisjoint(region), "overlapping chambers"
        covered.update(region)
        boards.append(dict(box=(top, left, height, width), h=height - 2, w=width - 2,
                           interior=grid[top + 1:top + height - 1,
                                         left + 1:left + width - 1].copy(), emitters=[]))
    used = set()
    for color in np.unique(grid):
        if color in (neutral, rim):
            continue
        for comp in _components(map(tuple, np.argwhere(grid == color)), True):
            if len(comp) < 5 or not comp.isdisjoint(covered):
                continue
            matches = []
            for side in _SIDES:
                geometry = _read_pointed_component(comp, _DIR[side])
                if geometry is None:
                    continue
                tip, cavity = geometry
                for index, board in enumerate(boards):
                    r, c, bh, bw = board["box"]
                    if side in ("top", "bottom"):
                        pos = tip[1] - c - 1
                        expected = (r - 1 if side == "top" else r + bh, tip[1])
                        lateral = all(c < x < c + bw - 1 for y, x in comp | cavity)
                        span = board["w"]
                    else:
                        pos = tip[0] - r - 1
                        expected = (tip[0], c - 1 if side == "left" else c + bw)
                        lateral = all(r < y < r + bh - 1 for y, x in comp | cavity)
                        span = board["h"]
                    if tip != expected or not 0 <= pos < span or not lateral:
                        continue
                    assert cavity.isdisjoint(covered)
                    assert all(0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]
                               for y, x in cavity)
                    values = {int(grid[p]) for p in cavity} - {neutral}
                    if len(values) != 1 or int(color) in values or rim in values:
                        continue
                    payload_color = next(iter(values))
                    payload = {p for p in cavity if grid[p] == payload_color}
                    matches.append((index, dict(side=side, pos=pos, quantity=len(payload),
                                               payload_color=payload_color, frame_color=int(color),
                                               frame=comp, cavity=cavity, payload=payload)))
            assert len(matches) <= 1, "ambiguous reservoir association"
            if matches:
                index, obj = matches[0]
                assert used.isdisjoint(comp | obj["cavity"]), "overlapping reservoirs"
                used.update(comp | obj["cavity"])
                boards[index]["emitters"].append(obj)
    nonneutral = set(map(tuple, np.argwhere(grid != neutral)))
    assert nonneutral <= covered | used, "unassigned exterior cells"
    frame_colors = set()
    for board in boards:
        objects = board["emitters"]
        frame_colors.update(o["frame_color"] for o in objects)
        hc = {o["payload_color"] for o in objects if o["side"] in ("left", "right")}
        vc = {o["payload_color"] for o in objects if o["side"] in ("top", "bottom")}
        assert len(hc) == len(vc) == 1 and hc != vc, "two readable local beam palettes needed"
        board["hc"], board["vc"] = next(iter(hc)), next(iter(vc))
        tuples = [(o["side"], o["pos"], None, o["quantity"], None) for o in objects]
        board["hm"], board["vm"] = _beam_masks(board["h"], board["w"], tuples)
        hm, vm = board["hm"], board["vm"]
        assert np.any(hm & vm) and np.any(hm & ~vm) and np.any(vm & ~hm)
    assert len(frame_colors) == 1
    assert not frame_colors & {neutral, rim, *(b["hc"] for b in boards), *(b["vc"] for b in boards)}
    blank = [i for i, b in enumerate(boards) if np.all(b["interior"] == neutral)]
    assert len(blank) == 1, "exactly one chamber is blank"
    target, reference = boards[blank[0]], boards[1 - blank[0]]
    colors = np.unique(reference["interior"][reference["hm"] & reference["vm"]])
    assert len(colors) == 1, "reference crossings must share one reaction"
    mixed = int(colors[0])
    reaction = ("erase" if mixed == neutral else "horizontal" if mixed == reference["hc"]
                else "vertical" if mixed == reference["vc"] else "accent")
    assert mixed not in frame_colors | {rim}
    if reaction == "accent":
        assert mixed not in {target["hc"], target["vc"]}
    expected = _combine(reference["hm"], reference["vm"], neutral,
                        reference["hc"], reference["vc"], reaction, mixed)
    assert np.array_equal(reference["interior"], expected), "reference disagrees with counted beams"
    return dict(neutral=neutral, rim=rim, reference=reference, target=target,
                reaction=reaction, accent=mixed, boards=boards)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover only the missing chamber, with no generation/training state."""
    scene = _read_scene(input_grid)
    target = scene["target"]
    return _combine(target["hm"], target["vm"], scene["neutral"],
                    target["hc"], target["vc"], scene["reaction"], scene["accent"])


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                  (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    # Diagnostic twins: target pixels and all sources are identical. Only the
    # demonstrated crossing color changes. Their labels are built from masks.
    for attempt in range(80):
        rc, tc = _palette(rng)
        rh, rw, re = _reference_plan(rng)
        th, tw, te = _witness_plan(rng)
        reactions = [str(x) for x in rng.choice(_REACTIONS, 2, replace=False)]
        try:
            rb, _, _ = _render_board(rng, rh, rw, re, rc, reactions[0], True)
            tb, first_answer, _ = _render_board(rng, th, tw, te, tc, reactions[0], False)
            first = _pack_boards(rng, rb, tb, rc[0])
            scene = _read_scene(first)
            second = first.copy()
            reference = scene["reference"]
            r, c, bh, bw = reference["box"]
            second[r + 1:r + bh - 1, c + 1:c + bw - 1] = _combine(
                reference["hm"], reference["vm"], rc[0], rc[3], rc[4], reactions[1], rc[5])
            hm, vm = _beam_masks(th, tw, te)
            second_answer = _combine(hm, vm, tc[0], tc[3], tc[4], reactions[1], tc[5])
            assert np.array_equal(solve(first), first_answer)
            assert np.array_equal(solve(second), second_answer)
            assert not np.array_equal(first_answer, second_answer)
            break
        except (ValueError, AssertionError):
            continue
    else:
        raise RuntimeError("could not build informative training twins")
    twins = [{"input": first, "output": first_answer},
             {"input": second, "output": second_answer}]
    if rng.integers(2):
        twins.reverse()
    puzzle["train"].extend(twins)
    seen.update(_key(p["input"]) for p in twins)
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for index in range(count):
            for attempt in range(160):
                rc, tc = _palette(rng)
                reaction = str(rng.choice(_REACTIONS))
                rh, rw, re = _reference_plan(rng, tiny=split == "test")
                try:
                    th, tw, te = _random_plan(rng, split, fallback=attempt >= 120)
                    rb, _, _ = _render_board(rng, rh, rw, re, rc, reaction, True)
                    tb, answer, _ = _render_board(rng, th, tw, te, tc, reaction, False)
                    inp = _pack_boards(rng, rb, tb, rc[0])
                    if _key(inp) in seen:
                        continue
                    assert np.array_equal(solve(inp), answer)
                    observed = _read_scene(inp)
                    assert not np.array_equal(_constant_answer(observed, 30), answer)
                    break
                except (ValueError, AssertionError):
                    continue
            else:
                raise RuntimeError("could not build a distinct readable loom")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": answer})
    return puzzle


def _constant_answer(scene, quantity):
    target = scene["target"]
    emitters = [(o["side"], o["pos"], None, o["quantity"], None)
                for o in target["emitters"]]
    hm, vm = _beam_masks(target["h"], target["w"], emitters, constant=quantity)
    return _combine(hm, vm, scene["neutral"], target["hc"], target["vc"],
                    scene["reaction"], scene["accent"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            scene = _read_scene(inp)
            scenes[split].append(scene)
            t = scene["target"]
            assert out.shape == (t["h"], t["w"])
            assert np.array_equal(solve(inp), out), "incorrect missing chamber"
            assert np.any(out != scene["neutral"])
            assert any(o["quantity"] > 1 for o in t["emitters"])
            assert not np.array_equal(_constant_answer(scene, 1), out), "counting is invisible"
            assert not np.array_equal(_constant_answer(scene, 30), out), "edge extension bypasses counting"
    # The first two demonstrations are reference-dependence counterfactuals.
    a, b = scenes["train"][:2]
    ta, tb = a["target"], b["target"]
    assert ta["box"] == tb["box"]
    assert a["reaction"] != b["reaction"]
    assert np.array_equal(ta["hm"], tb["hm"]) and np.array_equal(ta["vm"], tb["vm"])
    ia, ib = _grid(puzzle["train"][0]["input"]), _grid(puzzle["train"][1]["input"])
    ra, ca, ha, wa = a["reference"]["box"]
    allowed = np.zeros(ia.shape, bool)
    allowed[ra + 1:ra + ha - 1, ca + 1:ca + wa - 1] = a["reference"]["hm"] & a["reference"]["vm"]
    assert ia.shape == ib.shape and np.all((ia == ib) | allowed)
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    # Two different full counts on the same geometry/direction defeat fixed
    # length, frame-size, and component-count substitutions, even at counts 2/1.
    witnesses = {}
    for o in ta["emitters"]:
        if o["side"] not in ("left", "right"):
            continue
        if o["quantity"] >= ta["w"]:
            continue
        rs, cs = zip(*o["frame"])
        profile = tuple(sorted((y - min(rs), x - min(cs)) for y, x in o["frame"]))
        witnesses.setdefault((o["side"], profile, o["payload_color"]), set()).add(o["quantity"])
    assert any(len(v) >= 2 for v in witnesses.values())
    assert any(len(_components(o["payload"])) < o["quantity"] for o in ta["emitters"])
    assert all(o["quantity"] < (ta["w"] if o["side"] in ("left", "right") else ta["h"])
               for o in ta["emitters"])
    for scene in scenes["test"]:
        target = scene["target"]
        assert set(o["side"] for o in target["emitters"]) == set(_SIDES)
        assert any(o["quantity"] > (target["w"] if o["side"] in ("left", "right") else target["h"])
                   for o in target["emitters"])


def _diagnostic_test():
    """Hand-written expected interiors, independent of beam-mask construction."""
    rc, tc = (0, 8, 9, 1, 2, 3), (0, 8, 9, 5, 6, 3)
    reference_plan = [("left", 1, (1, 1, 1, 0), 2, "compact"),
                      ("top", 1, (1, 1, 1, 0), 2, "compact")]
    witness = [("left", 1, (1, 1, 3, 0), 4, "compact"),
               ("left", 5, (1, 1, 3, 0), 2, "scattered"),
               ("top", 2, (2, 2, 2, 0), 4, "random")]
    tb, _, _ = _render_board(np.random.default_rng(71), 7, 7, witness, tc, "erase", False)
    answers, inputs = [], []
    for reaction, crossing in zip(_REACTIONS, (0, 5, 6, 3)):
        rb, _, _ = _render_board(np.random.default_rng(72), 4, 4, reference_plan,
                                 rc, reaction, True)
        inp = _pack_boards(np.random.default_rng(73), rb, tb, 0)
        expected = np.zeros((7, 7), np.int8)
        expected[1, :4] = 5
        expected[5, :2] = 5
        expected[:4, 2] = 6
        expected[1, 2] = crossing
        before = _key(inp)
        assert np.array_equal(solve(inp), expected)
        assert _key(inp) == before, "solve mutated its input"
        answers.append(_key(expected))
        inputs.append(inp)
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
        # Arbitrary bijective relabeling includes a nonzero neutral.
        permutation = np.array([7, 4, 1, 0, 8, 2, 3, 6, 5, 9], np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[expected])
    assert len(set(answers)) == 4
    # Same target pixels under all four reference treatments.
    a, b = _read_scene(inputs[0]), _read_scene(inputs[3])
    assert np.array_equal(a["target"]["hm"], b["target"]["hm"])
    assert np.array_equal(a["target"]["vm"], b["target"]["vm"])
    for clipped in (False, True):
        # Two opposing horizontal beams unite into a full row; their overlap
        # alone must stay H-colored. One vertical beam triggers chemistry.
        plan = [("left", 2, (2, 2, 3, 0) if clipped else (1, 1, 3, 0),
                 9 if clipped else 4, "random"),
                ("right", 2, (1, 1, 3, 0), 4, "compact"),
                ("top", 2, (2, 2, 2, 0), 4, "scattered")]
        target, _, _ = _render_board(np.random.default_rng(74), 7, 7, plan, tc, "accent", False)
        rb, _, _ = _render_board(np.random.default_rng(75), 4, 4, reference_plan, rc, "accent", True)
        inp = _pack_boards(np.random.default_rng(76), rb, target, 0)
        expected = np.zeros((7, 7), np.int8)
        expected[2, :] = 5
        expected[:4, 2] = 6
        expected[2, 2] = 3
        assert np.array_equal(solve(inp), expected)


def self_test():
    """Bounded checks plus conceptual diversity measurements over 42 seeds."""
    sizes, output_sizes, profiles, reactions, palettes, fingerprints = (set() for _ in range(6))
    quantities, counts, sides, modes = (set() for _ in range(4))
    same_axis_overlap = clipped = asymmetric = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], repeated[s]) for k in ("input", "output"))
        for split in puzzle:
            for pair in puzzle[split]:
                scene = _read_scene(pair["input"])
                target = scene["target"]
                sizes.add(np.asarray(pair["input"]).shape)
                output_sizes.add(np.asarray(pair["output"]).shape)
                reactions.add(scene["reaction"])
                palettes.add((scene["neutral"], scene["rim"], target["hc"], target["vc"]))
                counts.add(len(target["emitters"]))
                ref = scene["reference"]
                modes.add("same" if (ref["hc"], ref["vc"]) == (target["hc"], target["vc"])
                          else "shared" if ref["hc"] == target["hc"] else "remapped")
                summed = 0
                for o in target["emitters"]:
                    quantities.add(o["quantity"])
                    sides.add(o["side"])
                    limit = target["w"] if o["side"] in ("left", "right") else target["h"]
                    summed += min(o["quantity"], limit)
                    clipped += o["quantity"] > limit
                    rs, cs = zip(*o["frame"])
                    profiles.add(tuple(sorted((r - min(rs), c - min(cs)) for r, c in o["frame"])))
                    dr, dc = _DIR[o["side"]]
                    transverse = [-r * dc + c * dr for r, c in o["frame"]]
                    tip, _ = _read_pointed_component(o["frame"], (dr, dc))
                    asymmetric += 2 * (-tip[0] * dc + tip[1] * dr) != min(transverse) + max(transverse)
                same_axis_overlap += summed > int(target["hm"].sum() + target["vm"].sum())
                for rotation in (1, 2, 3):
                    if seed == 0:
                        assert np.array_equal(solve(np.rot90(pair["input"], rotation)),
                                              np.rot90(pair["output"], rotation))
    assert len(fingerprints) == 42 and len(sizes) >= 20 and len(output_sizes) >= 10
    assert len(profiles) >= 30 and len(palettes) >= 30
    assert reactions == set(_REACTIONS) and sides == set(_SIDES)
    assert modes == {"same", "shared", "remapped"}
    assert {2, 3, 4, 5} <= counts and max(quantities) >= 8
    assert same_axis_overlap and clipped >= 42 and asymmetric
    # Exercise every legal explicit count, and reject bad types as well as bounds.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _diagnostic_test()
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(sizes),
            "output_shapes": len(output_sizes), "frame_profiles": len(profiles),
            "target_quantities": sorted(quantities), "target_counts": sorted(counts),
            "reactions": sorted(reactions), "palette_modes": sorted(modes),
            "clipped_emitters": int(clipped), "same_axis_overlap_pairs": int(same_axis_overlap),
            "asymmetric_emitters": int(asymmetric)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
