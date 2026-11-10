"""Reciprocal mirror repair: jointly register and repair two damaged scales.

Only standard-library modules and NumPy are used. Generation draws ground truth
from a complete sampled plan; solve reconstructs it using input pixels alone.
"""
from functools import lru_cache
import numpy as np


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


@lru_cache(maxsize=64)
def _orbits(h, w):
    return tuple(tuple(sorted({(r, c), (h - 1 - r, c),
                               (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))
                 for r in range((h + 1) // 2) for c in range((w + 1) // 2))


def _orbit_data(raw, marker):
    groups = _orbits(*raw.shape)
    ids = {p: i for i, group in enumerate(groups) for p in group}
    values = []
    for group in groups:
        known = {int(raw[p]) for p in group if raw[p] != marker}
        if len(known) > 1:
            raise ValueError("contradictory miniature reflection orbit")
        values.append(next(iter(known)) if known else None)
    return groups, ids, values


def _registrations(raw, sequence, axis, marker, *, markers_only=False, require_witness=True):
    """Content-only matching: canvas bounds never select a registration."""
    groups, ids, initial = _orbit_data(raw, marker)
    h, w = raw.shape
    length = len(sequence)
    result = []
    for r in range(h - (length - 1 if axis == 0 else 0)):
        for c in range(w - (length - 1 if axis == 1 else 0)):
            values = initial.copy()
            valid = True
            for k, color in enumerate(sequence):
                p = (r + k * (axis == 0), c + k * (axis == 1))
                i = ids[p]
                if color == marker:
                    if raw[p] != marker:
                        valid = False
                        break
                elif markers_only:
                    # Track reachability of missing orbits, but ignore payload
                    # identities and their equality/inequality constraints.
                    values[i] = 0
                else:
                    if values[i] is not None and values[i] != color:
                        valid = False
                        break
                    values[i] = int(color)
            if valid and ((markers_only and not require_witness)
                          or all(v is not None for v in values)):
                plan = None
                if not markers_only:
                    plan = np.empty(raw.shape, dtype=np.int8)
                    for group, color in zip(groups, values):
                        for p in group:
                            plan[p] = color
                result.append(((r, c), plan))
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    r, c = a.min(axis=0)
    b, d = a.max(axis=0) + 1
    return int(r), int(c), int(b), int(d)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _parse(inp):
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= np.partition(counts, -2)[-2]:
        raise ValueError("background must be uniquely most frequent")
    components = _components(inp != bg)
    frames = []
    for cells in components:
        r, c, b, d = _box(cells)
        height, width = b - r, d - c
        if min(height, width) < 9 or height % 2 != 1 or width % 2 != 1:
            continue
        border = {(y, x) for y in range(r, b) for x in range(c, d)
                  if y in (r, b - 1) or x in (c, d - 1)}
        if set(cells) == border and len({int(inp[p]) for p in cells}) == 1:
            frames.append((r, c, b, d))
    if len(frames) != 1:
        raise ValueError("expected one hollow miniature frame")
    frame = frames[0]
    mr, mc, mb, md = frame
    marker = int(inp[mr, mc])
    raw = inp[mr + 2:mb - 2:2, mc + 2:md - 2:2].copy()
    h, w = raw.shape
    interior = inp[mr + 1:mb - 1, mc + 1:md - 1]
    sites_mask = np.zeros(interior.shape, dtype=bool)
    sites_mask[1:2 * h:2, 1:2 * w:2] = True
    if np.any(interior[~sites_mask] != bg):
        raise ValueError("nonbackground miniature padding")
    squares = []
    for cells in components:
        r, c, b, d = _box(cells)
        if mr <= r and mc <= c and b <= mb and d <= md:
            continue
        side = b - r
        color = int(inp[r, c])
        if (side < 2 or d - c != side or len(cells) != side * side
                or not np.all(inp[r:b, c:d] == color)):
            raise ValueError("outside components must be solid squares")
        squares.append((r, c, side, color))
    if len(squares) < 3:
        raise ValueError("expected at least three enlarged squares")
    if len({s[0] for s in squares}) == 1:
        axis = 1
    elif len({s[1] for s in squares}) == 1:
        axis = 0
    else:
        raise ValueError("enlarged clue must lie on one line")
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    pitch = squares[1][axis] - squares[0][axis]
    if (pitch <= side or any(s[2] != side for s in squares)
            or any(squares[k + 1][axis] - squares[k][axis] != pitch
                   for k in range(len(squares) - 1))):
        raise ValueError("nonuniform enlarged lattice")
    return bg, marker, frame, raw, squares, axis, side, pitch


def _recover(inp):
    bg, marker, frame, raw, squares, axis, side, pitch = _parse(inp)
    sequence = tuple(s[3] for s in squares)
    matches = _registrations(raw, sequence, axis, marker)
    if len(matches) != 1:
        raise ValueError("registration must be uniquely color-consistent and fully witnessed")
    match, plan = matches[0]
    h, w = raw.shape
    base = (squares[0][0] - match[0] * pitch, squares[0][1] - match[1] * pitch)
    br, bc = base
    enlarged = (br, bc, br + (h - 1) * pitch + side, bc + (w - 1) * pitch + side)
    if (br < 0 or bc < 0 or enlarged[2] > inp.shape[0] or enlarged[3] > inp.shape[1]
            or _overlap(frame, enlarged)):
        raise ValueError("enlarged rectangle must fit and avoid the miniature")
    seed_sites = [(match[0] + k * (axis == 0), match[1] + k * (axis == 1))
                  for k in range(len(sequence))]
    supplied = dict(zip(seed_sites, sequence))
    expected = inp.copy()
    mr, mc = frame[:2]
    for (r, c), value in np.ndenumerate(plan):
        y, x = br + r * pitch, bc + c * pitch
        region = inp[y:y + side, x:x + side]
        if not np.all(region == supplied.get((r, c), bg)):
            raise ValueError("unsupplied enlarged sites must be background")
        expected[y:y + side, x:x + side] = value
        expected[mr + 2 + 2 * r, mc + 2 + 2 * c] = value
    groups, ids, initial = _orbit_data(raw, marker)
    bridge = {i for i, value in enumerate(initial) if value is None}
    large_known = {ids[p] for p, color in supplied.items() if color != marker}
    reciprocal = sum(color == marker and initial[ids[p]] is not None
                     for p, color in supplied.items())
    diagonal = 0
    for r, c in np.argwhere(raw == marker):
        if (ids[(r, c)] not in large_known and raw[h - 1 - r, c] == marker
                and raw[r, w - 1 - c] == marker
                and raw[h - 1 - r, w - 1 - c] != marker):
            diagonal += 1
    stats = dict(background=bg, marker=marker, plan_shape=(h, w), axis=axis,
                 side=side, gutter=pitch - side, length=len(sequence), match=match,
                 base=base, frame=frame, canvas=inp.shape,
                 bridge_count=len(bridge), reciprocal=int(reciprocal),
                 diagonal=int(diagonal), erased=int(np.count_nonzero(raw == marker)),
                 erased_blank=int(np.count_nonzero((raw == marker) & (plan == bg))),
                 seed_blank=sum(color == marker and plan[p] == bg for p, color in supplied.items()),
                 direct=sum(raw[p] == color and color != marker for p, color in supplied.items()),
                 marker_candidates=len(_registrations(raw, sequence, axis, marker,
                                                       markers_only=True, require_witness=False)),
                 color_candidates=len(_registrations(raw, sequence, axis, marker, markers_only=True)),
                 blank_sites=int(np.count_nonzero(plan == bg)),
                 palette=tuple(int(v) for v in np.unique(plan) if v != bg),
                 partial=len(sequence) < plan.shape[axis])
    return expected, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; no examples, seed, or sampled state is used."""
    return _recover(_grid(input_grid))[0]


@lru_cache(maxsize=1)
def _geometry_options():
    result = []
    for h in range(3, 7):
        for w in range(3, 7):
            mini = (2 * h + 3, 2 * w + 3)
            for side in range(2, 6):
                for gutter in range(1, 4):
                    big = tuple((n - 1) * (side + gutter) + side for n in (h, w))
                    for sep in (0, 1):
                        required = [max(mini[d], big[d]) for d in (0, 1)]
                        required[sep] = mini[sep] + big[sep] + 1
                        if max(required) <= 30:
                            result.append((h, w, side, gutter, sep))
    return tuple(result)


def _logical_attempt(rng, h, w, axis, bg, marker, palette, hard, test, density):
    groups = _orbits(h, w)
    ids = {p: i for i, group in enumerate(groups) for p in group}
    colors = [bg] + list(palette)
    blank_weight = float(rng.uniform(.08, .25))
    probabilities = [blank_weight] + [(1 - blank_weight) / len(palette)] * len(palette)
    values = rng.choice(colors, size=len(groups), p=probabilities)
    if len(set(map(int, values)) - {bg}) < 2:
        return None
    plan = np.empty((h, w), dtype=np.int8)
    for group, color in zip(groups, values):
        for p in group:
            plan[p] = color
    lengths = np.arange(3, (h, w)[axis] + 1)
    weights = np.ones(len(lengths))
    if test:
        weights[:-1] *= 3
    length = int(rng.choice(lengths, p=weights / weights.sum()))
    match = (int(rng.integers(h - (length - 1 if axis == 0 else 0))),
             int(rng.integers(w - (length - 1 if axis == 1 else 0))))
    sites = [(match[0] + k * (axis == 0), match[1] + k * (axis == 1))
             for k in range(length)]
    hit = sorted({ids[p] for p in sites if plan[p] != bg})
    if len(hit) < 2:
        return None
    bridge = int(rng.choice(hit))
    mask = rng.random((h, w)) < density
    for i, group in enumerate(groups):
        if i == bridge:
            for p in group:
                mask[p] = True
        elif all(mask[p] for p in group):
            mask[group[int(rng.integers(len(group)))]] = False
    sequence = [int(plan[p]) for p in sites]
    for k, p in enumerate(sites):
        if mask[p] and (plan[p] == bg or (ids[p] != bridge and rng.random() < .65)):
            sequence[k] = marker
        if plan[p] == bg and sequence[k] != marker:
            return None
    if hard:
        options = [i for i, group in enumerate(groups)
                   if len(group) == 4 and i != bridge and any(p not in sites for p in group)]
        if not options:
            return None
        triple = int(rng.choice(options))
        group = groups[triple]
        possible = [p for p in group if p not in sites]
        survivor = possible[int(rng.integers(len(possible)))]
        for p in group:
            mask[p] = p != survivor
        for k, p in enumerate(sites):
            if ids[p] == triple:
                sequence[k] = marker
        if not np.any(mask & (plan == bg)):
            return None
    if (sum(v != marker for v in sequence) < 2 or marker not in sequence
            or not any(ids[p] == bridge and v != marker for p, v in zip(sites, sequence))
            or not any(not mask[p] and v != marker for p, v in zip(sites, sequence))
            or not any(v == marker and ids[p] != bridge for p, v in zip(sites, sequence))):
        return None
    raw = plan.copy()
    raw[mask] = marker
    matches = _registrations(raw, sequence, axis, marker)
    if len(matches) != 1 or matches[0][0] != match:
        return None
    if len(_registrations(raw, sequence, axis, marker,
                        markers_only=True, require_witness=False)) < 2:
        return None
    if hard and len(_registrations(raw, sequence, axis, marker, markers_only=True)) < 2:
        return None
    return plan, raw, match, tuple(sequence)


def _fallback_logic(bg, marker, palette, axis, avoid_shape=None):
    """Constructive termination guard, under exactly the ordinary rule."""
    raw = np.array([[9, 1, 3], [9, 9, 9], [9, 3, 9], [0, 1, 9], [9, 9, 3]])
    plan = np.array([[3, 1, 3], [0, 1, 0], [1, 3, 1], [0, 1, 0], [3, 1, 3]])
    seq, match = [1, 9, 3], (2, 2)
    shape = (5, 3) if axis == 0 else (3, 5)
    if shape == avoid_shape:
        raw = np.array([[0, 9, 9], [2, 4, 2], [9, 9, 9],
                        [9, 9, 0], [2, 9, 2], [0, 3, 0]])
        plan = np.array([[0, 3, 0], [2, 4, 2], [0, 4, 0],
                         [0, 4, 0], [2, 4, 2], [0, 3, 0]])
        seq, match = [9, 4, 4, 4], (0, 1)
    if axis == 1:
        raw, plan = raw.T, plan.T
        match = match[::-1]
    mapping = np.arange(10, dtype=np.int8)
    mapping[0], mapping[9] = bg, marker
    roles = sorted(set(map(int, plan.flat)) - {0})
    for role, color in zip(roles, palette):
        mapping[role] = color
    return mapping[plan], mapping[raw], match, tuple(int(mapping[v]) for v in seq)


def _place(rng, shape, side, gutter, sep):
    mini = tuple(2 * n + 3 for n in shape)
    big = tuple((n - 1) * (side + gutter) + side for n in shape)
    gap = int(rng.integers(1, min(3, 30 - mini[sep] - big[sep]) + 1))
    required = [max(mini[d], big[d]) for d in (0, 1)]
    required[sep] = mini[sep] + big[sep] + gap
    mode = int(rng.integers(5))
    if mode == 0:
        canvas = (30, 30)
    elif mode == 1:
        n = int(rng.integers(max(required), 31))
        canvas = (n, n)
    else:
        canvas = tuple(int(rng.integers(n, min(n + 5, 30) + 1)) for n in required)
    anchor, base = [0, 0], [0, 0]
    free = canvas[sep] - required[sep]
    first = int(rng.integers(free + 1))
    between = int(rng.integers(free - first + 1))
    if rng.integers(2):
        anchor[sep], base[sep] = first, first + mini[sep] + gap + between
    else:
        base[sep], anchor[sep] = first, first + big[sep] + gap + between
    other = 1 - sep
    anchor[other] = int(rng.integers(canvas[other] - mini[other] + 1))
    base[other] = int(rng.integers(canvas[other] - big[other] + 1))
    return canvas, tuple(anchor), tuple(base)


def _draw(plan, raw, match, sequence, axis, bg, marker, side, gutter, layout):
    shape, anchor, base = layout
    inp = np.full(shape, bg, dtype=np.int8)
    mr, mc = anchor
    mh, mw = (2 * n + 3 for n in plan.shape)
    inp[mr, mc:mc + mw] = inp[mr + mh - 1, mc:mc + mw] = marker
    inp[mr:mr + mh, mc] = inp[mr:mr + mh, mc + mw - 1] = marker
    for (r, c), value in np.ndenumerate(raw):
        inp[mr + 2 + 2 * r, mc + 2 + 2 * c] = value
    pitch = side + gutter
    for k, color in enumerate(sequence):
        r, c = match[0] + k * (axis == 0), match[1] + k * (axis == 1)
        y, x = base[0] + r * pitch, base[1] + c * pitch
        inp[y:y + side, x:x + side] = color
    out = inp.copy()
    for (r, c), value in np.ndenumerate(plan):
        out[mr + 2 + 2 * r, mc + 2 + 2 * c] = value
        y, x = base[0] + r * pitch, base[1] + c * pitch
        out[y:y + side, x:x + side] = value
    return {"input": inp, "output": out}


def _make_pair(rng, axis, hard, test, density, avoid=None):
    colors = rng.permutation(10)
    if avoid is not None and colors[0] == avoid["background"]:
        colors[0], colors[1] = colors[1], colors[0]
    bg, marker = int(colors[0]), int(colors[1])
    palette = colors[2:2 + int(rng.integers(3, 7))]
    options = [o for o in _geometry_options()
               if (not hard or o[axis] >= 4)
               and (avoid is None or ((o[0], o[1]) != avoid["plan_shape"] and o[2] != avoid["side"]))]
    logical = None
    for _ in range(800):
        h, w, side, gutter, sep = options[int(rng.integers(len(options)))]
        logical = _logical_attempt(rng, h, w, axis, bg, marker, palette, hard, test, density)
        if logical is not None:
            break
    if logical is None:
        logical = _fallback_logic(bg, marker, palette, axis,
                                  None if avoid is None else avoid["plan_shape"])
        plan_shape = logical[0].shape
        choices = [o for o in options if o[:2] == plan_shape]
        h, w, side, gutter, sep = choices[int(rng.integers(len(choices)))]
    plan, raw, match, sequence = logical
    layout = _place(rng, plan.shape, side, gutter, sep)
    return _draw(plan, raw, match, sequence, axis, bg, marker, side, gutter, layout)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer("num_train", num_train, 2, 10)
    ns = _integer("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    mode = int(rng.choice(3, p=[.30, .30, .40]))
    axes = [mode] if mode < 2 else [0, 1]
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    rng.shuffle(train_axes)
    # Establish both clue orientations in mixed puzzles even with two demos.
    if len(axes) == 2:
        train_axes[:2] = list(rng.permutation(axes))
    bias = float(rng.uniform(-.04, .04))
    puzzle, seen = {"train": [], "test": []}, set()
    first_stats = None
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            test = split == "test"
            hard = test or i == 1 or (i > 1 and rng.random() < .45)
            axis = int(rng.choice(axes)) if test else int(train_axes[i])
            density = (.51 if test else .34 if hard else .19) + bias
            avoid = first_stats if split == "train" and i == 1 else None
            for _ in range(40):
                pair = _make_pair(rng, axis, hard, test, density, avoid)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input attempts exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and i == 0:
                first_stats = _recover(pair["input"])[1]
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, stats = _recover(inp)
            assert np.array_equal(out, expected), "wrong repair, transfer, or preserved pixel"
            assert not np.array_equal(inp, out)
            assert stats["bridge_count"] == 1, "miniature must need a large witness"
            assert stats["reciprocal"] > 0, "large marker must need miniature evidence"
            assert stats["direct"] > 0 and stats["marker_candidates"] >= 2
            assert len(stats["palette"]) >= 2
            # The marker has semantic meaning only on the preserved frame now.
            mr, mc, mb, md = stats["frame"]
            frame_mask = np.zeros(inp.shape, dtype=bool)
            frame_mask[mr, mc:md] = frame_mask[mb - 1, mc:md] = True
            frame_mask[mr:mb, mc] = frame_mask[mr:mb, md - 1] = True
            assert np.array_equal(out == stats["marker"], frame_mask)
            if split == "test":
                assert stats["diagonal"] > 0 and stats["erased_blank"] > 0
                assert stats["color_candidates"] >= 2, "payload colors must select the registration"
            records[split].append(stats)
    assert any(m["diagonal"] > 0 and m["erased_blank"] > 0 and m["color_candidates"] >= 2
               for m in records["train"])
    for feature in ("plan_shape", "side", "background"):
        assert len({m[feature] for m in records["train"]}) >= 2, feature
    assert {m["axis"] for m in records["test"]} <= {m["axis"] for m in records["train"]}


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed structural study plus semantic negative checks."""
    signatures, records = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                prediction, stats = _recover(pair["input"])
                assert np.array_equal(prediction, pair["output"])
                records.append((seed, split, stats))
    assert len(signatures) == 42
    p = generate_puzzle(17)
    assert _signature(p) == _signature(generate_puzzle(17))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(7100 + 10 * nt + ns, nt, ns)
            assert len(q["train"]) == nt and len(q["test"]) == ns
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    all_stats = [m for _, _, m in records]
    for feature, minimum in (("plan_shape", 10), ("canvas", 35), ("match", 12),
                             ("background", 9), ("marker", 9), ("palette", 30),
                             ("side", 4), ("gutter", 3), ("length", 4)):
        assert len({m[feature] for m in all_stats}) >= minimum, feature
    assert {m["axis"] for m in all_stats} == {0, 1}
    assert {(h % 2, w % 2) for m in all_stats for h, w in [m["plan_shape"]]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert any(m["partial"] for _, s, m in records if s == "train")
    assert any(m["partial"] for _, s, m in records if s == "test")
    assert any(m["seed_blank"] > 0 for _, s, m in records if s == "train")
    assert any(m["seed_blank"] > 0 for _, s, m in records if s == "test")
    assert any(m["blank_sites"] == 0 for m in all_stats)
    assert any(m["blank_sites"] >= 4 for m in all_stats)
    assert any(m["canvas"] == (30, 30) for m in all_stats)
    for axis in (0, 1):
        for avoid in (None, (5, 3) if axis == 0 else (3, 5)):
            plan, raw, match, sequence = _fallback_logic(0, 9, [1, 2, 3], axis, avoid)
            candidates = _registrations(raw, sequence, axis, 9)
            assert len(candidates) == 1 and candidates[0][0] == match
            assert np.array_equal(candidates[0][1], plan)
            assert len(_registrations(raw, sequence, axis, 9, markers_only=True)) >= 2
    # All-marker clues cannot fill the missing miniature orbit.
    inp = p["test"][0]["input"]
    bg, marker, frame, raw, squares, axis, side, pitch = _parse(inp)
    assert not _registrations(raw, [marker] * len(squares), axis, marker)
    # Missing every miniature orbit removes the reciprocal witnesses.
    assert not _registrations(np.full(raw.shape, marker), [s[3] for s in squares], axis, marker)
    for point in ((0, 0), tuple(np.argwhere(inp == marker)[-1])):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
        out = bad["test"][0]["output"]
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
