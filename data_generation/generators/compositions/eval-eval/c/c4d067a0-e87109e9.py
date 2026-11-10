"""A clocked, keyed, patterned bead ray. Standard library and NumPy only."""
from __future__ import annotations

import json
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid cells")
    return a.astype(np.int8, copy=False)


def _components(field, colors):
    seen, boxes = set(), []
    h, w = field.shape
    for rr, cc in np.argwhere(np.isin(field, list(colors))):
        r, c = int(rr), int(cc)
        if (r, c) in seen:
            continue
        color = int(field[r, c])
        todo, cells = [(r, c)], []
        seen.add((r, c))
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in DIRS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen and field[yy, xx] == color:
                    seen.add((yy, xx))
                    todo.append((yy, xx))
        a, b = np.min(cells, axis=0)
        e, f = np.max(cells, axis=0) + 1
        _require(len(cells) == (e - a) * (f - b), "nonrectangular component")
        boxes.append((int(a), int(b), int(e - a), int(f - b), color))
    return boxes


def _schedule(plan, order="snake"):
    if order == "raster":
        return tuple(map(int, plan.flat))
    values = tuple(int(v) for r, row in enumerate(plan) for v in (row if r % 2 == 0 else row[::-1]))
    return values[::-1] if order == "reverse" else values


def _matches(sequence, clue):
    n = len(sequence)
    return [p for p in range(n) if all(sequence[(p + i) % n] == v for i, v in enumerate(clue))]


def _parse(input_grid):
    inp = _grid(input_grid)
    frame = int(inp[0, 0])
    _require(np.all(inp[0] == frame), "missing band frame")
    rows = np.flatnonzero(np.all(inp[1:] == frame, axis=1)) + 1
    _require(len(rows) > 0, "missing band bottom")
    bottom = int(rows[0])
    _require(bottom in (4, 6, 8) and bottom < len(inp) - 2, "invalid band height")
    band = inp[1:bottom]
    walls = np.all(band == frame, axis=0)
    _require(walls[0] and walls[-1], "missing band walls")
    panels, start = [], None
    for c in range(inp.shape[1] + 1):
        open_cell = c < inp.shape[1] and not walls[c]
        if open_cell and start is None:
            start = c
        if not open_cell and start is not None:
            panels.append((start, c, band[:, start:c]))
            start = None
    keys, minis, backgrounds = {}, [], set()
    for a, b, panel in panels:
        _require(panel.shape[1] >= 3 and not np.any(panel == frame), "bad panel interior")
        if np.all(panel == panel[0]):
            row = panel[0]
            bg = int(row[1])
            _require(np.all(row[1:-1] == bg), "bad key interior")
            left, right = int(row[0]), int(row[-1])
            _require((left == bg) != (right == bg), "ambiguous key bar")
            color, side = (right, 1) if left == bg else (left, -1)
            _require(color not in keys and color not in (bg, frame), "conflicting key")
            keys[color] = side
            backgrounds.add(bg)
        else:
            minis.append((a, b, panel))
    _require(len(backgrounds) == 1 and 2 <= len(keys) <= 3 and len(minis) == 1, "ambiguous band")
    bg = backgrounds.pop()
    a, b, mini = minis[0]
    _require(mini.shape[0] % 2 == mini.shape[1] % 2 == 1, "bad miniature dimensions")
    mask = np.zeros(mini.shape, dtype=bool)
    mask[::2, ::2] = True
    _require(np.all(mini[~mask] == bg), "bad miniature separators")
    plan = mini[::2, ::2].copy()
    payload = set(map(int, plan.flat)) - {bg}
    _require(len(payload) >= 2 and not payload & (set(keys) | {frame}), "conflicting payload roles")
    _require(2 <= plan.shape[0] <= 4 and 3 <= plan.shape[1] <= 6, "bad plan size")
    field = inp[bottom + 1:].copy()
    _require(set(map(int, field.flat)) <= {bg, frame} | payload | set(keys), "unknown field color")
    tag = _components(field, [frame])
    _require(len(tag) == 1, "cap must be unique")
    tr, tc, th, tw, _ = tag[0]
    _require((th == 1) != (tw == 1) and 2 <= max(th, tw) <= 4, "cap must be a transverse line")
    k = max(th, tw)
    beads = _components(field, payload)
    _require(2 <= len(beads) <= 4 and all(h == w == k for r, c, h, w, v in beads), "bad seed beads")
    candidates = []
    for r, c, h, w, v in beads:
        for d, (dr, dc) in enumerate(DIRS):
            if d == 0:
                match = th == 1 and (tr, tc) == (r + k, c)
            elif d == 1:
                match = tw == 1 and (tr, tc) == (r, c - 1)
            elif d == 2:
                match = th == 1 and (tr, tc) == (r - 1, c)
            else:
                match = tw == 1 and (tr, tc) == (r, c + k)
            if match:
                candidates.append((r, c, d))
    _require(len(candidates) == 1, "cap must identify one forward seed")
    r, c, d = candidates[0]
    dr, dc = DIRS[d]
    beads.sort(key=lambda q: (q[0] - r) * dr + (q[1] - c) * dc)
    distances = [(y - r) * dr + (x - c) * dc for y, x, _, _, _ in beads]
    stride = distances[1]
    _require(k < stride <= k + 3, "bad measured stride")
    for i, (y, x, _, _, v) in enumerate(beads):
        _require((y, x) == (r + i * stride * dr, c + i * stride * dc), "seed must be a consecutive straight run")
    clue = tuple(q[4] for q in beads)
    seq = _schedule(plan)
    matches = _matches(seq, clue)
    _require(len(matches) == 1, "seed fragment has ambiguous cycle phase")
    blockers = _components(field, keys)
    return dict(field=field, background=bg, frame=frame, plan=plan, sequence=seq,
                turns=keys, source=(r, c, d), k=k, stride=stride, phase=matches[0],
                clue=clue, blockers=blockers, crop=bottom + 1, mini_column=a)


def _trace(field, bg, frame, source, k, stride, turns):
    """Move the square head cell by cell; only complete clock ticks are beads."""
    h, w = field.shape
    r, c, d = source
    step = 0
    sites, events, seen = [(r, c, d)], [], set()
    after_turn = False
    corridor = np.zeros(field.shape, dtype=bool)
    corridor[r:r + k, c:c + k] = True
    for _ in range(4 * h * w + 1):
        state = (r, c, d)
        _require(state not in seen, "cyclic route")
        seen.add(state)
        if (r == 0, c + k == w, r + k == h, c == 0)[d]:
            break
        if d == 0:
            rr, cc = np.full(k, r - 1), np.arange(c, c + k)
        elif d == 1:
            rr, cc = np.arange(r, r + k), np.full(k, c + k)
        elif d == 2:
            rr, cc = np.full(k, r + k), np.arange(c, c + k)
        else:
            rr, cc = np.arange(r, r + k), np.full(k, c - 1)
        values = field[rr, cc]
        keyed = np.isin(values, list(turns))
        if np.any(keyed):
            _require(np.all(keyed) and np.all(values == values[0]), "partial or conflicting blocker face")
            _require(step % stride == 0 and not after_turn, "turn is off clock lattice or has no clear exit")
            color = int(values[0])
            events.append((step // stride, d, turns[color], color, r, c))
            d = (d + turns[color]) % 4
            after_turn = True
        else:
            _require(not np.any(values == frame), "ray meets its cap")
            dr, dc = DIRS[d]
            r, c = r + dr, c + dc
            step += 1
            after_turn = False
            corridor[r:r + k, c:c + k] = True
            if step % stride == 0:
                sites.append((r, c, d))
    else:
        raise ValueError("unterminated route")
    mask = np.zeros(field.shape, dtype=bool)
    for y, x, _ in sites:
        region = field[y:y + k, x:x + k]
        _require(not np.any(np.isin(region, list(turns) + [frame])), "blocked bead footprint")
        _require(not np.any(mask[y:y + k, x:x + k]), "overlapping bead footprints")
        mask[y:y + k, x:x + k] = True
    return sites, events, corridor


def _paint(info, sites, sequence=None, phase=None, reset=False, skip_blanks=False, advance_turn=False):
    seq = info["sequence"] if sequence is None else sequence
    phase = info["phase"] if phase is None else phase
    out = info["field"].copy()
    k, bg = info["k"], info["background"]
    events = info.get("events", [])
    pointer = phase
    event_ticks = {e[0] for e in events}
    for i, (r, c, _) in enumerate(sites):
        if i and reset and i - 1 in event_ticks:
            pointer = phase
        if advance_turn and i in event_ticks:
            pointer += 1
        if skip_blanks:
            while seq[pointer % len(seq)] == bg:
                pointer += 1
        color = seq[pointer % len(seq)]
        out[r:r + k, c:c + k] = color
        pointer += 1
    return out


def _decode(input_grid):
    info = _parse(input_grid)
    sites, events, corridor = _trace(info["field"], info["background"], info["frame"],
                                    info["source"], info["k"], info["stride"], info["turns"])
    info.update(sites=sites, events=events, corridor=corridor)
    out = _paint(info, sites)
    occupied = info["field"] != info["background"]
    _require(np.array_equal(out[occupied], info["field"][occupied]), "completion would change input objects")
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, phase, scale, stride and route from this input alone."""
    return _decode(input_grid)[0]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_header(rng, plan, width, turns):
    h = 2 * plan.shape[0] + 1
    mini_width = 2 * plan.shape[1] + 1
    n = len(turns)
    widths = [mini_width] + [5] * n
    extra = width - sum(widths)
    _require(extra >= 0, "panels do not fit")
    # Extra width belongs to key panels; miniature pitch and extent stay explicit.
    for q in rng.integers(1, n + 1, size=extra):
        widths[int(q)] += 1
    entries = [(None, widths[0])] + list(zip(turns, widths[1:]))
    rng.shuffle(entries)
    header = np.zeros((h, width), dtype=np.int8)
    header[[0, -1]] = 1
    c = 0
    for color, size in entries:
        header[:, c] = header[:, c + size - 1] = 1
        if color is None:
            header[1:-1:2, c + 1:c + size - 1:2] = plan
        else:
            header[1:-1, c + 1 if turns[color] == -1 else c + size - 2] = color
        c += size
    return header


def _tag(field, source, k):
    r, c, d = source
    if d == 0:
        field[r + k, c:c + k] = 1
    elif d == 1:
        field[r:r + k, c - 1] = 1
    elif d == 2:
        field[r - 1, c:c + k] = 1
    else:
        field[r:r + k, c + k] = 1


def _grow_route(rng, h, w, k, stride, target, turns, seed_length):
    """Insert keyed faces on actual pulse sites, then retrace after each insertion."""
    r = int(rng.integers(1, h - k))
    c = int(rng.integers(1, w - k))
    d = int(rng.integers(4))
    source = (r, c, d)
    field = np.zeros((h, w), dtype=np.int8)
    _tag(field, source, k)
    current = _trace(field, 0, 1, source, k, stride, turns)
    if len(current[0]) < seed_length + 1:
        return None
    blocks = []
    for _ in range(180):
        if len(current[1]) >= target:
            break
        existing_ticks = {e[0] for e in current[1]}
        choices = [(i, p) for i, p in enumerate(current[0]) if i >= seed_length and i not in existing_ticks]
        if not choices:
            break
        # Prefer a later leg when growing a recursive chain.
        late = [q for q in choices if q[0] > max(existing_ticks, default=-1)]
        if late and rng.random() < .7:
            choices = late
        i, (y, x, heading) = choices[int(rng.integers(len(choices)))]
        color = int(rng.choice(list(turns)))
        # Thin face, variable transverse extensions, occasionally boundary-hugging.
        before, after = int(rng.integers(0, stride)), int(rng.integers(0, stride))
        depth = int(rng.integers(1, stride - k + 1))
        if heading in (0, 2):
            a, b = (max(0, y - depth), y) if heading == 0 else (y + k, min(h, y + k + depth))
            e, f = max(0, x - before), min(w, x + k + after)
            if rng.random() < .18:
                e, f = (0, f) if rng.random() < .5 else (e, w)
        else:
            a, b = max(0, y - before), min(h, y + k + after)
            e, f = (x + k, min(w, x + k + depth)) if heading == 1 else (max(0, x - depth), x)
            if rng.random() < .18:
                a, b = (0, b) if rng.random() < .5 else (a, h)
        if not (a < b and e < f) or np.any(field[a:b, e:f] != 0):
            continue
        if np.any(field[max(0, a-1):min(h, b+1), max(0, e-1):min(w, f+1)] == color):
            continue
        candidate = field.copy()
        candidate[a:b, e:f] = color
        try:
            trace = _trace(candidate, 0, 1, source, k, stride, turns)
        except ValueError:
            continue
        if len(trace[0]) < seed_length + 2 or len(trace[1]) <= len(current[1]):
            continue
        field, current = candidate, trace
        blocks.append((a, e, b - a, f - e, color))
    if len(current[1]) < target:
        return None
    # Occasionally add one untouched keyed rectangle outside the swept corridor.
    if rng.random() < .35:
        for _ in range(25):
            a, e = int(rng.integers(h)), int(rng.integers(w))
            b, f = min(h, a + int(rng.integers(1, 4))), min(w, e + int(rng.integers(1, 5)))
            color = int(rng.choice(list(turns)))
            if np.any(current[2][a:b, e:f]) or np.any(field[a:b, e:f]):
                continue
            if np.any(field[max(0,a-1):min(h,b+1), max(0,e-1):min(w,f+1)] == color):
                continue
            field[a:b, e:f] = color
            break
    return field, source, current


def _sample_plan(rng, shape, payload_count, clue_length, pulse_count, lesson):
    for _ in range(100):
        plan = rng.choice(np.arange(5, 5 + payload_count), size=shape).astype(np.int8)
        # Blank ticks are present in every teaching plan, optional in other scenes.
        holes = int(rng.integers(1 if lesson else 0, max(2, plan.size // 3 + 1)))
        if holes:
            plan.flat[rng.choice(plan.size, size=holes, replace=False)] = 0
        if len(set(map(int, plan.flat)) - {0}) < 2:
            continue
        seq = _schedule(plan)
        phases = [p for p in range(len(seq))
                  if all(seq[(p+i) % len(seq)] != 0 for i in range(clue_length))
                  and len(_matches(seq, tuple(seq[(p+i) % len(seq)] for i in range(clue_length)))) == 1]
        if lesson:
            phases = [p for p in phases if p != 0 and any(seq[(p+i) % len(seq)] == 0 for i in range(clue_length, pulse_count))]
        if phases:
            return plan, int(rng.choice(phases))
    return None


def _make_pair(rng, k, plan_shape, target, lesson=False, min_pulses=0):
    stride = k + int(rng.integers(1, 4 if k == 2 else 3))
    key_count = int(rng.integers(2, 4))
    # Both meanings have visible keys, even when a particular route uses one.
    colors = list(range(2, 2 + key_count))
    sides = [-1, 1] + ([int(rng.choice([-1, 1]))] if key_count == 3 else [])
    rng.shuffle(sides)
    turns = dict(zip(colors, sides))
    header_height = 2 * plan_shape[0] + 1
    min_width = 2 * plan_shape[1] + 1 + 5 * key_count
    h = int(rng.integers(max(11, 3 * stride + k), 31 - header_height))
    w = int(rng.integers(max(min_width, 4 * stride + k), 31))
    clue_length = 2 if lesson else int(rng.choice([2, 2, 3]))
    route = _grow_route(rng, h, w, k, stride, target, turns, clue_length)
    if route is None:
        return None
    field, source, trace = route
    sites, events, corridor = trace
    if len(events) != target or len(sites) < max(min_pulses, clue_length + 2):
        return None
    sampled = _sample_plan(rng, plan_shape, int(rng.integers(2, 5)), clue_length, len(sites), lesson)
    if sampled is None:
        return None
    plan, phase = sampled
    seq = _schedule(plan)
    if not any(i > events[0][0] and seq[(phase + i) % len(seq)] != 0 for i in range(len(sites))):
        return None
    for i, (y, x, _) in enumerate(sites[:clue_length]):
        field[y:y + k, x:x + k] = seq[(phase + i) % len(seq)]
    info = dict(field=field, sequence=seq, phase=phase, k=k, background=0, events=events)
    out = _paint(info, sites)
    header = _make_header(rng, plan, w, turns)
    palette = rng.permutation(10)
    return dict(input=palette[np.vstack((header, field))].astype(np.int8), output=palette[out].astype(np.int8))


def _hypotheses(train):
    """Small, explicit alternative rule audit, not an independent solver."""
    survivors = []
    for order in ("snake", "raster", "reverse"):
        for left in (-1, 1):
            for right in (-1, 1):
                fits = True
                for pair in train:
                    try:
                        info = _parse(pair["input"])
                        seq = _schedule(info["plan"], order)
                        matches = _matches(seq, info["clue"])
                        if not matches:
                            fits = False
                            break
                        turns = {c: left if s == -1 else right for c, s in info["turns"].items()}
                        sites, events, _ = _trace(info["field"], info["background"], info["frame"], info["source"], info["k"], info["stride"], turns)
                        info["events"] = events
                        # Give the alternative every clue-compatible phase, even
                        # when its reading order would make the input ambiguous.
                        if not any(np.array_equal(_paint(info, sites, seq, phase), pair["output"]) for phase in matches):
                            fits = False
                            break
                    except ValueError:
                        fits = False
                        break
                if fits:
                    survivors.append((order, left, right))
    return survivors


def _teaching(train):
    infos = [_decode(p["input"])[1] for p in train]
    if len({q["background"] for q in infos}) < 2:
        return False
    sides = {side: {e[1] for q in infos for e in q["events"] if e[2] == side} for side in (-1, 1)}
    if not all(len(s) >= 2 for s in sides.values()):
        return False
    if not any(len(q["sites"]) >= len(q["sequence"]) + 2 for q in infos):
        return False
    if not any(max((e[0] for e in q["events"]), default=0) < len(q["sites"]) - 2 and len(q["events"]) >= 2 for q in infos):
        return False
    for mode in ("reset", "skip_blanks", "advance_turn"):
        if not any(not np.array_equal(_paint(q, q["sites"], **{mode: True}), p["output"]) for q, p in zip(infos, train)):
            return False
    if not any(q["phase"] != 0 and not np.array_equal(_paint(q, q["sites"], phase=0), p["output"]) for q, p in zip(infos, train)):
        return False
    if not any(not np.array_equal(_without_A(q), p["output"]) for q, p in zip(infos, train)):
        return False
    return _hypotheses(train) == [("snake", -1, 1)]


def _without_A(info):
    """Keep the known seed but extend it with its first color everywhere."""
    out = info["field"].copy()
    k = info["k"]
    for r, c, _ in info["sites"][len(info["clue"]):]:
        out[r:r + k, c:c + k] = info["clue"][0]
    return out


def _without_B(info):
    """Deposit the calibrated pattern only up to the first blocker."""
    tick = info["events"][0][0]
    return _paint(info, info["sites"][:tick + 1])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, count, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not lo <= count <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["larger", "longer_plan", "more_turns"]))
    seen = set()
    # Choose two explanatory scenes by observable evidence, then shuffle them.
    for _ in range(240):
        train = []
        for k in (2, 3):
            for attempt in range(160):
                p = _make_pair(rng, k, (2, 3), int(rng.integers(2, 4)), True, 8)
                if p is not None:
                    train.append(p)
                    break
            if len(train) != (1 if k == 2 else 2):
                break
        if len(train) == 2 and _teaching(train):
            break
    else:
        raise RuntimeError("bounded teaching construction exhausted")
    seen.update(_key(p["input"]) for p in train)
    puzzle = dict(train=train, test=[])
    for split, count in (("train", int(num_train) - 2), ("test", int(num_test))):
        for _ in range(count):
            for attempt in range(500):
                if split == "test":
                    k = 4 if shift == "larger" else int(rng.choice([2, 3]))
                    shape = (3, int(rng.integers(3, 5))) if shift == "longer_plan" else (2, int(rng.integers(3, 6)))
                    target = int(rng.integers(4, 6)) if shift == "more_turns" else int(rng.integers(1, 5))
                else:
                    k = int(rng.choice([2, 3]))
                    shape = (2, int(rng.integers(3, 6)))
                    target = int(rng.integers(1, 4))
                p = _make_pair(rng, k, shape, target)
                if p is not None and _key(p["input"]) not in seen:
                    seen.add(_key(p["input"]))
                    puzzle[split].append(p)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "bad counts")
    seen, infos = set(), dict(train=[], test=[])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, info = _decode(inp)
            _require(np.array_equal(expected, out), "incorrect beads, crop or preservation")
            _require(len(info["events"]) >= 1, "ray must turn")
            _require(not np.array_equal(out, info["field"]), "no bead completion")
            _require(any(i > info["events"][0][0] and info["sequence"][(info["phase"] + i) % len(info["sequence"])] != info["background"] for i in range(len(info["sites"]))), "turn has no visible consequence")
            _require(not np.array_equal(_without_B(info), out), "omitting turns changes nothing")
            infos[split].append(info)
    _require({q["k"] for q in infos["train"]} >= {2, 3}, "scale variation not demonstrated")
    _require(_teaching(puzzle["train"]), "insufficient teaching evidence")
    _require(all(len(q["events"]) <= 3 and q["plan"].shape[0] == 2 and q["k"] <= 3 for q in infos["train"]), "unexpected training regime")
    _require(all(q["k"] == 4 or q["plan"].shape[0] == 3 or len(q["events"]) >= 4 for q in infos["test"]), "test lacks structural shift")


def _extension_fixtures():
    """Hand-specified pulse coordinates exercise bounds beyond random sampling.

    Expected squares are drawn from these explicit coordinates, not _trace.
    """
    examples = []
    configurations = [
        (np.array([[5, 5, 6, 5, 6, 7], [7, 6, 0, 8, 5, 6],
                   [6, 5, 7, 8, 0, 5], [5, 8, 7, 6, 8, 7]], dtype=np.int8),
         2, 3, 4, 21, (16, 30), (12, 1, 1), (11, 15, 15, 17),
         [(12, 1), (12, 4), (12, 7), (12, 10), (12, 13), (9, 13), (6, 13), (3, 13), (0, 13)]),
        (np.array([[5, 6, 0], [7, 8, 5]], dtype=np.int8),
         4, 7, 2, 0, (17, 29), (10, 1, 1), (10, 14, 19, 22),
         [(10, 1), (10, 8), (10, 15), (3, 15)]),
    ]
    for plan, k, stride, length, phase, shape, source, rect, sites in configurations:
        field = np.zeros(shape, dtype=np.int8)
        _tag(field, source, k)
        a, b, e, f = rect
        field[a:b, e:f] = 2
        seq = _schedule(plan)
        clue = tuple(seq[(phase + i) % len(seq)] for i in range(length))
        assert _matches(seq, clue) == [phase]
        for i, (r, c) in enumerate(sites[:length]):
            field[r:r + k, c:c + k] = clue[i]
        out = field.copy()
        for i, (r, c) in enumerate(sites):
            out[r:r + k, c:c + k] = seq[(phase + i) % len(seq)]
        for reflect in (False, True):
            turns = {2: 1 if reflect else -1, 3: -1 if reflect else 1}
            inp_field = field[:, ::-1].copy() if reflect else field.copy()
            expected = out[:, ::-1].copy() if reflect else out.copy()
            header = _make_header(np.random.default_rng(71), plan, shape[1], turns)
            examples.append(dict(input=np.vstack((header, inp_field)), output=expected))
    return examples


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, features = set(), []
    for puzzle in puzzles:
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                before = inp.copy()
                out, q = _decode(inp)
                assert np.array_equal(out, pair["output"]) and np.array_equal(inp, before)
                features.append((inp.shape, q["k"], q["stride"], q["plan"].shape, q["phase"],
                                 q["source"][2], len(q["sites"]), len(q["events"]),
                                 len(q["blockers"]), q["background"], q["mini_column"]))
    assert len(signatures) == 42
    for axis, minimum in ((0, 25), (1, 3), (2, 4), (3, 5), (4, 6), (5, 4), (6, 8), (7, 4), (8, 4), (9, 9), (10, 8)):
        assert len({f[axis] for f in features}) >= minimum, (axis, minimum)
    assert tuple(_key(p["input"]) for p in generate_puzzle(0)["train"]) == tuple(_key(p["input"]) for p in puzzles[0]["train"])
    fixtures = _extension_fixtures()
    for pair in fixtures:
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # Deliberately break the k=4 fixture's complete face collision.
    partial = fixtures[2]["input"].copy()
    q = _parse(partial)
    r, c, d = q["source"]
    partial[q["crop"] + r + q["k"] - 1, 19:22] = q["background"]
    try:
        solve(partial)
    except ValueError:
        pass
    else:
        raise AssertionError("partial face accepted")
    # A visibly longer gap to the blocker must not create a fractional-clock turn.
    off_clock = fixtures[2]["input"].copy()
    field = off_clock[q["crop"]:]
    field[10:14, 19:22] = 0
    field[10:14, 20:23] = 2
    try:
        solve(off_clock)
    except ValueError:
        pass
    else:
        raise AssertionError("off-clock turn accepted")
    # The seed [5,6] matches two phases of this alternate miniature.
    ambiguous_plan = np.array([[5, 6, 5], [6, 5, 7]], dtype=np.int8)
    ambiguous_header = _make_header(np.random.default_rng(71), ambiguous_plan, 29, {2: -1, 3: 1})
    ambiguous = np.vstack((ambiguous_header, fixtures[2]["input"][5:]))
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous cycle phase accepted")
    # The host verifier exhausts all legal explicit count combinations. Keep
    # this additional audit bounded to extremes and an interior request.
    for nt, ns in ((2, 1), (3, 2), (6, 3), (10, 1), (10, 4)):
        generate_puzzle(1000 + 10 * nt + ns, nt, ns)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{n: np.asarray(g).copy() for n, g in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(signatures),
            "feature_counts": [len({f[i] for f in features}) for i in range(len(features[0]))]}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
