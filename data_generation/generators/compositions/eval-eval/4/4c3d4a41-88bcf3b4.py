"""A finite taut cable is a flexible key for a bank of roof-hung pins.

All semantics are recovered from input pixels. No parent generator is imported.
Tests are bounded and do not save generated corpora or execution traces.
"""
from itertools import combinations
import json
import numpy as np

_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,835873644,217358741,40224698,640793524,727809389,336446751,622840447,150545318,748189734,881058224,187400055,209702223,4508620,848638278,559141937,151414385,560317822,965423871,432571051,785376214,743088915,418109069,182475409,340154776,594179465,26613402,225232836,565346635,86613902,369966866,595841700,697715797,867823310,373163899,33935456,719627658,122250556,154822499,9605942,902893635,428244073)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _connected(cells, directions=_D8):
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _profile(obstacle, length, check=True):
    """Source-B digital clearance envelope, with the parallel tangent case.

    Coordinates are forward u and height v above the support endpoint.
    The cable starts at (0,1); each subsequent cell advances u by one.
    """
    peak = max(v for u, v in obstacle)
    assert min(u for u, v in obstacle) >= 2 and peak >= 0
    if check:
        assert max(v - u for u, v in obstacle) == 0 or peak == 0
    first_peak = min(u for u, v in obstacle if v == peak)
    size = max(length, max(u for u, v in obstacle) + peak + 4)
    h = [1 if peak == 0 and u <= first_peak else
         max(v + 1 - abs(u - t) for t, v in obstacle) for u in range(size)]
    steps = [y - x for x, y in zip(h, h[1:])]
    if check:
        assert h[0] == 1 and all(s in (-1, 0, 1) for s in steps)
        assert steps == sorted(steps, reverse=True), "unsupported nonconvex contour"
    assert not set(enumerate(h)) & obstacle
    return h[:length]


def _runs(row, color):
    mask = np.r_[False, row == color, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(mask) == 1),
                    np.flatnonzero(np.diff(mask) == -1) - 1))


def _read_canonical(a, bg, structure, roof, left, right):
    """Read a complete scene, rejecting unused pixels and ambiguous roles."""
    assert 10 <= right - left + 1 <= 22
    # The separate short rod is below all roof-connected structural cells.
    sr = int(np.max(np.argwhere(a == structure)[:, 0]))
    assert sr - roof >= 5
    rod = np.flatnonzero(a[sr] == structure)
    assert 3 <= len(rod) <= 6 and np.all(np.diff(rod) == 1)
    x0 = int(rod[-1])
    assert left == x0 + 1
    attachment = (sr - 1, x0)
    cable_color = int(a[attachment])
    assert cable_color not in (bg, structure)
    slack = {tuple(map(int, p)) for p in np.argwhere(a == cable_color)}
    assert 2 <= len(slack) <= 28 and _connected(slack)
    assert all(r < sr and c <= x0 for r, c in slack)
    # Track exactly which pixels belong to which object. Pins are read by
    # their prefix/suffix grammar, so adjacent differently colored obstacles
    # cannot be accidentally counted as pin material.
    used = {(roof, c) for c in range(left, right + 1)}
    used.update((sr, int(c)) for c in rod)
    used.update(slack)
    lanes = np.flatnonzero(a[roof + 1, left:right + 1] != bg) + left
    assert 2 <= len(lanes) <= 6 and np.all(np.diff(lanes) >= 2)
    assert lanes[0] >= left + 1 and lanes[-1] <= right - 1
    pins = []
    for col in lanes:
        row = roof + 1
        seq = []
        while row < sr and a[row, col] == structure:
            seq.append(structure)
            row += 1
        assert row < sr
        payload = int(a[row, col])
        assert payload not in (bg, structure, cable_color)
        while row < sr and a[row, col] == payload:
            seq.append(payload)
            row += 1
        assert seq and row <= sr
        cells = {(roof + 1 + j, int(col)) for j in range(len(seq))}
        assert not cells & used
        used.update(cells)
        pins.append((int(col), tuple(seq)))
    rest = {tuple(map(int, p)) for p in np.argwhere(a != bg)} - used
    assert rest and _connected(rest)
    colors = {int(a[p]) for p in rest}
    assert len(colors) == 1
    obstacle_color = colors.pop()
    assert obstacle_color not in (bg, structure, cable_color)
    assert all(obstacle_color not in seq for _, seq in pins)
    obstacle = {(c - x0, sr - r) for r, c in rest}
    assert min(u for u, v in obstacle) >= 2
    assert max(u for u, v in obstacle) <= right - x0 - 1
    assert min(r for r, c in rest) >= roof + 3
    heights = _profile(obstacle, len(slack))
    assert all(sr - h >= roof + 2 for h in heights)
    assert used | rest == {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    return dict(background=bg, structure=structure, roof=roof, support_row=sr,
                anchor_col=x0, roof_right=right, pins=pins, slack=slack,
                obstacle=obstacle, obstacle_cells=rest, obstacle_color=obstacle_color,
                cable_color=cable_color, heights=heights, rod={(sr, int(c)) for c in rod})


def _parse(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(counts.argmax())
    candidates = []
    for code in range(8):
        b = _unorient(a, code)
        # A roof has a long uninterrupted run; separated thin pin lanes do not.
        for color in np.unique(b):
            s = int(color)
            if s == bg:
                continue
            for row in range(b.shape[0] - 5):
                for left, right in _runs(b[row], s):
                    if right - left + 1 < 10:
                        continue
                    try:
                        f = _read_canonical(b, bg, s, row, int(left), int(right))
                    except (AssertionError, IndexError):
                        continue
                    f["orientation"] = code
                    candidates.append(f)
    assert len(candidates) == 1, "input must have exactly one mechanical reading"
    return candidates[0]


def _predict(a, f, *, push=True, finite=True, bounding_box=False):
    """Apply the rule to actual input pixels, using no construction state."""
    b = _unorient(a, f["orientation"])
    out = b.copy()
    bg, roof, sr, x0 = (f[k] for k in ("background", "roof", "support_row", "anchor_col"))
    obstacle = f["obstacle"]
    if bounding_box:
        us, vs = zip(*obstacle)
        obstacle = {(u, v) for u in range(min(us), max(us) + 1)
                    for v in range(min(vs), max(vs) + 1)}
    # For ablation checks an extended cable affects pins but the actual cable
    # drawing is still finite. This isolates the length-gating contribution.
    extent = max(len(f["slack"]), f["roof_right"] - x0 + 1) if not finite else len(f["slack"])
    h = _profile(obstacle, extent, check=not bounding_box)
    for p in f["slack"]:
        out[p] = bg
    if push:
        for col, seq in f["pins"]:
            u = col - x0
            if u >= len(h):
                continue
            cable_row = sr - h[u]
            # Move a stack only when it overlaps the final cable position.
            cells = [(roof + 1 + j, color) for j, color in enumerate(seq)]
            for row, _ in cells:
                out[row, col] = bg
            while cells[-1][0] >= cable_row:
                cells = [(row - 1, color) for row, color in cells if row - 1 > roof]
                assert cells, "one or more pin cells must survive"
            for row, color in cells:
                out[row, col] = color
    for u in range(len(f["slack"])):
        p = sr - h[u], x0 + u
        if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]:
            # The no-push ablation deliberately leaves pins covering the cable.
            if not push and out[p] != bg:
                continue
            assert out[p] == bg, "cable must not overwrite fixed material or pins"
            out[p] = f["cable_color"]
    return _orient(out, f["orientation"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any member of the specified grammar from its input alone."""
    a = _grid(input_grid)
    return _predict(a, _parse(a))


def _obstacle(rng, lesson):
    # Procedural convex caps: point, bar, rectangle, stepped band, filled cap, ribbon.
    kinds = (3, 4, 5) if lesson == "visible" else tuple(range(6))
    kind = int(rng.choice(kinds))
    if kind == 0:
        rise, plateau, fall, thickness = 0, 1, 0, 1
    elif kind in (1, 2):
        rise, plateau, fall = 0, int(rng.integers(2, 5)), 0
        thickness = 1 if kind == 1 else int(rng.integers(2, 4))
    elif kind == 3:
        rise, plateau, fall = 0, int(rng.integers(1, 4)), int(rng.integers(1, 3))
        thickness = int(rng.integers(2, 5))
    elif kind == 4:
        rise = fall = int(rng.integers(1, 3))
        plateau, thickness = int(rng.integers(1, 3)), 0
    else:
        rise, plateau, fall = int(rng.integers(0, 3)), int(rng.integers(1, 4)), int(rng.integers(1, 3))
        thickness = int(rng.integers(1, 3))
    size = rise + plateau + fall
    tops = [-max(rise - u, 0) - max(u - rise - plateau + 1, 0) for u in range(size)]
    raw = set()
    for u, v in enumerate(tops):
        bottom = min(tops) if kind == 4 else v - thickness + 1
        raw.update((u, w) for w in range(bottom, v + 1))
    approach = "diagonal" if lesson == "visible" or rng.random() < .72 else "parallel"
    near = int(rng.integers(4, 7)) if lesson == "visible" else int(rng.integers(2, 7))
    offset = near - max(v - u for u, v in raw) if approach == "diagonal" else 0
    obs = {(u + near, v + offset) for u, v in raw}
    _profile(obs, 2)
    return obs


def _slack(rng, length, width, depth, diagonal):
    # Anchor (0,1), with optional diagonal-only connection to the whole lobe.
    cells = {(0, 1)}
    if diagonal:
        cells.add((-1, 2))
    style = int(rng.integers(0, 3))
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in _D8
                           if -width <= u + du <= (-1 if diagonal else 0)
                           and (2 if diagonal else 1) <= v + dv <= depth} - cells)
        assert frontier
        if style == 0:
            scores = np.array([1 + sum((u + du, v + dv) in cells for du, dv in _D8)
                               for u, v in frontier], dtype=float) ** 2
        elif style == 1:
            scores = np.array([1 / (1 + sum((u + du, v + dv) in cells for du, dv in _D8))
                               for u, v in frontier], dtype=float)
        else:
            scores = np.ones(len(frontier))
        cells.add(frontier[int(rng.choice(len(frontier), p=scores / scores.sum()))])
    return cells


def _construct_pair(rng, n, orientation, colors, lesson="", short=False, engage=False):
    """Build pins by direct destination formulas, not by calling solve."""
    bg, structure, cable, oc = colors[:4]
    obs = _obstacle(rng, lesson)
    far, peak = max(u for u, v in obs), max(v for u, v in obs)
    field = int(rng.integers(max(12, far + 5), 22))
    # Keep enough width for the slack and independent margins under 30 cells.
    x0 = int(rng.integers(4, min(7, 28 - field) + 1))
    sr = int(rng.integers(max(8, peak + 5), max(8, peak + 5) + 4))
    roof = int(rng.integers(0, 3))
    sr += roof
    if lesson == "visible":
        length = far + 3
    elif lesson == "clipped":
        length = min(28, max(field + 2, far + peak + 5))
    elif short:
        length = int(rng.integers(3, max(4, far + 1)))
    else:
        length = int(rng.integers(max(4, far), min(28, field + 5) + 1))
    h = _profile(obs, max(length, field + 1))
    # Select separated lanes, with two active lanes and one visibly unreached
    # but potentially movable lane in the complete-route teaching example.
    combos = [c for c in combinations(range(2, field), n)
              if all(b - a >= 2 for a, b in zip(c, c[1:]))]
    if lesson == "visible":
        combos = [c for c in combos if sum(u < length and h[u] > 0 for u in c) >= 2
                  and any(u >= length and h[u] > 0 for u in c)]
    elif lesson == "clipped" or engage:
        combos = [c for c in combos if any(u < length and h[u] > 0 for u in c)]
    if not combos:
        return None
    lanes = combos[int(rng.integers(len(combos)))]
    candidates = [u for u in lanes if u < length and h[u] > 0]
    moving = int(rng.choice(candidates)) if candidates else None
    stationary = next((u for u in lanes if u != moving and u < length), None)
    # Prefer an off-obstacle lane to make influence-at-a-distance observable.
    absent = [u for u in candidates if not any(t == u for t, v in obs)]
    if lesson == "visible" and absent:
        moving = int(rng.choice(absent))
        stationary = next(u for u in lanes if u != moving and u < length)
    pin_palette = colors[4:4 + int(rng.integers(1, min(5, n) + 1))]
    pins = []
    for u in lanes:
        cap = sr - roof - 1
        at = [v for t, v in obs if t == u]
        if at:
            cap = min(cap, sr - roof - max(at) - 1)
        contact_capacity = sr - h[u] - roof - 1
        assert cap >= 1 and contact_capacity >= 1
        size = int(rng.integers(1, cap + 1))
        if (lesson or engage) and u == moving:
            size = int(rng.integers(contact_capacity + 1, cap + 1))
        if lesson == "visible" and u == stationary:
            size = int(rng.integers(1, min(cap, contact_capacity) + 1))
        if lesson == "visible" and u >= length and h[u] > 0:
            size = int(rng.integers(contact_capacity + 1, cap + 1))
        shift = max(0, size - contact_capacity) if u < length else 0
        prefix = int(rng.integers(0, size))
        if lesson == "visible" and u == moving:
            prefix = int(rng.integers(shift + 1, size))
        elif lesson == "clipped" and u == moving:
            prefix = int(rng.integers(0, shift))
        payload = int(rng.choice(pin_palette))
        pins.append((u, (structure,) * prefix + (payload,) * (size - prefix)))
    width = int(rng.integers(4, min(7, x0) + 1))
    depth = max(4, 1 + (length - 1 + width - 1) // width)
    depth += int(rng.integers(0, 2))
    slack = _slack(rng, length, width, depth, lesson == "visible" or rng.random() < .45)
    rod_length = int(rng.integers(3, min(6, x0 + 1) + 1))
    bottom_input = max(sr, max(sr - v for u, v in obs))
    top_input = min(roof, min(sr - v for u, v in slack))
    if top_input < 0:
        # Shift all material equally; this changes no physical relationship.
        delta = -top_input
        roof, sr, bottom_input = roof + delta, sr + delta, bottom_input + delta
    full_route = [(sr - h[u], x0 + u) for u in range(length)]
    bottom = bottom_input
    right = x0 + field
    if lesson == "visible":
        bottom = max(bottom, max(r for r, c in full_route))
        right = max(right, max(c for r, c in full_route))
    elif lesson != "clipped" and rng.random() < .55:
        bottom = max(bottom, max(r for r, c in full_route))
        right = max(right, max(c for r, c in full_route))
    if max(bottom, right) >= 30:
        return None
    if lesson != "clipped":
        bottom += int(rng.integers(0, 3))
        right += int(rng.integers(0, min(2, 29 - right) + 1))
    if max(bottom, right) >= 30:
        return None
    inp = np.full((bottom + 1, right + 1), bg, dtype=np.int8)
    inp[roof, x0 + 1:x0 + field + 1] = structure
    inp[sr, x0 - rod_length + 1:x0 + 1] = structure
    for u, v in obs:
        inp[sr - v, x0 + u] = oc
    for u, v in slack:
        assert inp[sr - v, x0 + u] == bg
        inp[sr - v, x0 + u] = cable
    for u, seq in pins:
        for j, color in enumerate(seq):
            assert inp[roof + 1 + j, x0 + u] == bg
            inp[roof + 1 + j, x0 + u] = color
    out = inp.copy()
    for u, v in slack:
        out[sr - v, x0 + u] = bg
    for u, seq in pins:
        out[roof + 1:roof + 1 + len(seq), x0 + u] = bg
        capacity = sr - h[u] - roof - 1
        shift = max(0, len(seq) - capacity) if u < length else 0
        for j, color in enumerate(seq):
            r = roof + 1 + j - shift
            if r > roof:
                out[r, x0 + u] = color
    for p in full_route:
        if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]:
            assert out[p] == bg
            out[p] = cable
    return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _measure(a, f=None):
    f = _parse(a) if f is None else f
    b = _unorient(a, f["orientation"])
    sr, x0, roof = f["support_row"], f["anchor_col"], f["roof"]
    shifts, retained, payload_clip = [], False, False
    extended = _profile(f["obstacle"], f["roof_right"] - x0 + 1)
    gated = distant = False
    for col, seq in f["pins"]:
        u = col - x0
        capacity = sr - extended[u] - roof - 1
        shift = max(0, len(seq) - capacity) if u < len(f["slack"]) else 0
        shifts.append(shift)
        prefix = next(i for i, color in enumerate(seq) if color != f["structure"])
        retained |= 0 < shift < prefix
        payload_clip |= shift > prefix
        gated |= u >= len(f["slack"]) and len(seq) > capacity
        distant |= shift > 0 and not any(t == u for t, v in f["obstacle"])
    path = [(sr - v, x0 + u) for u, v in enumerate(f["heights"])]
    visible = sum(0 <= r < b.shape[0] and 0 <= c < b.shape[1] for r, c in path)
    obs = f["obstacle"]
    us, vs = zip(*obs)
    rectangular = len(obs) == (max(us) - min(us) + 1) * (max(vs) - min(vs) + 1)
    return dict(lanes=len(f["pins"]), length=len(f["slack"]), shifts=shifts,
                moving_prefix=bool(retained), payload_clip=bool(payload_clip),
                gated=bool(gated), distant=bool(distant), clipped=visible < len(path),
                visible=visible, diagonal_slack=not _connected(f["slack"], _D4),
                full_departure=len(path) > max(us) + 2,
                irregular=not rectangular, parallel=max(vs) == 0,
                orientation=f["orientation"], background=f["background"],
                profile=tuple(f["heights"]), silhouette=tuple(sorted(obs)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"illegal {name}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Schedules select visible presentation and curriculum, never rule variants.
    regime = int(rng.integers(0, 3))  # orientation, lane-count, or short-reach transfer
    base = int(rng.integers(8))
    role_colors = tuple(map(int, rng.permutation(10)))
    changing_palette = bool(rng.random() < .7)
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = ("visible", "clipped")[i] if split == "train" and i < 2 else ""
            if regime == 0:
                orientation = base if split == "train" else (base + 1 + int(rng.integers(3))) % 4 + (4 if base >= 4 else 0)
            else:
                orientation = int(rng.integers(8))
            n = int(rng.integers(3, 5)) if lesson else int(rng.integers(2, 7))
            if regime == 1:
                n = int(rng.integers(3, 5)) if split == "train" else int(rng.integers(5, 7))
            colors = tuple(map(int, rng.permutation(10))) if changing_palette else role_colors
            for attempt in range(160):
                pair = _construct_pair(rng, n, orientation, colors, lesson,
                                       short=(regime == 2 and split == "test"),
                                       engage=(split == "test"))
                if pair is None or _key(pair["input"]) in seen:
                    continue
                f = _parse(pair["input"])
                m = _measure(pair["input"], f)
                if lesson == "visible" and not (m["moving_prefix"] and m["gated"] and m["distant"]
                        and 0 in m["shifts"] and m["diagonal_slack"] and not m["clipped"]):
                    continue
                if lesson == "clipped" and not (m["payload_clip"] and m["clipped"]):
                    continue
                seen.add(_key(pair["input"]))
                result[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction exhausted")
        rng.shuffle(result[split])
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            f = _parse(a)
            expected = _predict(a, f)
            assert np.array_equal(expected, out) and not np.array_equal(a, out)
            canonical = _unorient(out, f["orientation"])
            assert all(canonical[p] == f["obstacle_color"] for p in f["obstacle_cells"])
            assert all(canonical[p] == f["structure"] for p in f["rod"])
            assert np.all(canonical[f["roof"], f["anchor_col"] + 1:f["roof_right"] + 1] == f["structure"])
            m = _measure(a, f)
            if split == "test":
                assert max(m["shifts"]) > 0, "queries must require pin displacement"
            assert int(np.count_nonzero(out == f["cable_color"])) == m["visible"]
            # Each moved pin is exactly the retained suffix of its input stack.
            for (col, seq), shift in zip(f["pins"], m["shifts"]):
                suffix = seq[shift:]
                top = f["roof"] + 1
                assert tuple(map(int, canonical[top:top + len(suffix), col])) == suffix
            if split == "train":
                training.append((a, out, f, m))
    assert any(m["moving_prefix"] and m["gated"] and m["distant"]
               and 0 in m["shifts"] and m["diagonal_slack"] and not m["clipped"]
               and m["full_departure"] and m["irregular"] for a, out, f, m in training)
    assert any(m["payload_clip"] and m["clipped"] for a, out, f, m in training)
    # Explicit parent ablations: keep pins fixed, or ignore finite cable reach.
    assert any(not np.array_equal(_predict(a, f, push=False), out) for a, out, f, m in training)
    assert any(not np.array_equal(_predict(a, f, finite=False), out) for a, out, f, m in training)
    assert any(not np.array_equal(_predict(a, f, bounding_box=True), out)
               for a, out, f, m in training if m["irregular"])


def _regressions():
    """Literal route/contact answers, including the first unreached column."""
    assert _profile({(2, 0)}, 6) == [1, 1, 1, 0, -1, -2]
    assert _profile({(2, v) for v in range(3)} | {(3, v) for v in range(3)}, 6) == [1, 2, 3, 3, 2, 1]
    assert _profile({(3, 3), (4, 4), (5, 3)}, 8) == [1, 2, 3, 4, 5, 4, 3, 2]
    # Roof t=0, support endpoint (12,4), obstacle local (5,5).
    # At u=2, K=8 and L=8: stationary exact contact.
    # At u=4, K=6 and L=9: move three, retain SPPPPP.
    # At u=6, the same capacity would move the pin, but N=6 stops first.
    a = np.zeros((14, 19), dtype=np.int8)
    a[0, 5:19] = 5
    a[12, 2:5] = 5
    a[7, 9] = 8
    slack = {(0, 1), (-1, 2), (-2, 2), (-3, 3), (-2, 4), (-1, 4)}
    for u, v in slack:
        a[12 - v, 4 + u] = 2
    a[1:9, 6] = [5, 5, 3, 3, 3, 3, 3, 3]
    a[1:10, 8] = [5, 5, 5, 5, 4, 4, 4, 4, 4]
    a[1:10, 10] = [5, 6, 6, 6, 6, 6, 6, 6, 6]
    expected = a.copy()
    expected[expected == 2] = 0
    expected[1:10, 8] = 0
    expected[1:7, 8] = [5, 4, 4, 4, 4, 4]
    for u, row in enumerate((11, 10, 9, 8, 7, 6)):
        expected[row, 4 + u] = 2
    assert np.array_equal(solve(a), expected)
    # Move only a slack cell, preserving count and connectivity: identical
    # mechanical output, despite a different input shape.
    rearranged = a.copy()
    rearranged[8, 3], rearranged[9, 2] = 0, 2
    assert np.array_equal(solve(rearranged), expected)
    # A ten-cell cable reaches u=6. Its S + eight payload cells loses SPP,
    # so the six surviving payload cells occupy rows 1..6.
    longer = a.copy()
    for u, v in ((-4, 3), (-4, 4), (-3, 4), (-4, 5)):
        longer[12 - v, 4 + u] = 2
    long_answer = expected.copy()
    long_answer[1:10, 10] = 0
    long_answer[1:7, 10] = 6
    for u, row in enumerate((7, 8, 9, 10), 6):
        long_answer[row, 4 + u] = 2
    assert np.array_equal(solve(longer), long_answer)
    # Both semantic cable-length bounds, including extensive outer clipping.
    tiny = a.copy()
    tiny[tiny == 2] = 0
    tiny[11, 4], tiny[10, 3] = 2, 2
    tiny_answer = a.copy()
    tiny_answer[tiny_answer == 2] = 0
    tiny_answer[11, 4], tiny_answer[10, 5] = 2, 2
    assert np.array_equal(solve(tiny), tiny_answer)
    maximal = a.copy()
    maximal[maximal == 2] = 0
    maximal[11, 4] = 2
    lobe = [(u, v) for v in range(2, 9) for u in (-1, -2, -3, -4)][:27]
    for u, v in lobe:
        maximal[12 - v, 4 + u] = 2
    maximal_answer = long_answer.copy()
    maximal_answer[11, 14] = maximal_answer[12, 15] = maximal_answer[13, 16] = 2
    assert np.array_equal(solve(maximal), maximal_answer)
    assert np.count_nonzero(maximal == 2) == 28
    assert np.count_nonzero(maximal_answer == 2) == 13
    # Semantic roof bounds extend beyond the constructor's 12..21 schedule.
    short_roof, short_answer = a.copy(), expected.copy()
    short_roof[0, 15:19] = short_answer[0, 15:19] = 0
    assert np.array_equal(solve(short_roof), short_answer)
    wide_roof, wide_answer = np.pad(a, ((0, 0), (0, 8))), np.pad(expected, ((0, 0), (0, 8)))
    wide_roof[0, 5:27] = wide_answer[0, 5:27] = 5
    assert np.array_equal(solve(wide_roof), wide_answer)
    # No four-connected counting, seed, output label or sampled metadata is
    # available to solve. These literal answers differ at the gated pin.
    assert not _connected(slack, _D4)


def self_test():
    """42 host seeds; aggregate geometry evidence, no persisted full corpus."""
    _regressions()
    shapes, structures, silhouettes, profiles, lengths, lanes, directions, backgrounds = (set() for _ in range(8))
    flags = {k: 0 for k in ("moving_prefix", "payload_clip", "gated", "distant", "clipped", "diagonal_slack", "irregular", "parallel")}
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in p:
            for pair in p[split]:
                a = pair["input"]
                assert np.array_equal(solve(a.copy()), pair["output"])
                m = _measure(a)
                shapes.add(a.shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.ravel())
                structures.add((a.shape, norm))
                silhouettes.add(m["silhouette"])
                profiles.add(m["profile"])
                lengths.add(m["length"])
                lanes.add(m["lanes"])
                directions.add(m["orientation"])
                backgrounds.add(m["background"])
                for k in flags:
                    flags[k] += int(m[k])
    assert len(shapes) >= 40 and len(structures) >= 180
    assert len(silhouettes) >= 35 and len(profiles) >= 50 and len(lengths) >= 12
    assert lanes == set(range(2, 7)) and directions == set(range(8)) and backgrounds == set(range(10))
    assert min(flags.values()) > 0
    p, q = generate_puzzle(913, 2, 1), generate_puzzle(913, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Geometry/palette equivariance on one actual family member, independently
    # of the sampled orientation schedules.
    pair = p["train"][0]
    mapping = np.array([(3 * i + 7) % 10 for i in range(10)], dtype=np.int8)
    for code in range(8):
        assert np.array_equal(solve(_orient(mapping[pair["input"]], code)),
                              _orient(mapping[pair["output"]], code))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(291 + 10 * nt + nq, nt, nq))
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(72, 2, 1)
    f = _parse(damaged["test"][0]["input"])
    canonical = _unorient(damaged["test"][0]["output"], f["orientation"])
    canonical[next(iter(f["obstacle_cells"]))] = f["background"]
    damaged["test"][0]["output"] = _orient(canonical, f["orientation"])
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted damaged fixed material")
    return {"seeds":42,"pairs":210,"shapes":len(shapes),"structures":len(structures),
            "silhouettes":len(silhouettes),"profiles":len(profiles),"lengths":sorted(lengths),
            "lanes":sorted(lanes),"orientations":sorted(directions),"flags":flags}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
