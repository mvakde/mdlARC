"""Reflect ornament shapes, but color their missing cells from a global ruler.

Only NumPy and the standard library are required. Generation uses construction
geometry; solve reconstructs geometry and the footprint from input pixels.
No filesystem access, stored examples, generation state, or output lookup is
used by solve. Tests and generation keep reconstructible puzzles in memory.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INWARD = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALTERNATIVES = ("monochrome", "copy_colors", "reset_phase", "prefix_period",
                "rotate_shape", "bounding_box", "unmarked", "reverse_word")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _count(value, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"counts must be integers in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _components(mask):
    todo = set(zip(*np.nonzero(mask)))
    groups = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in DIRS:
                p = (y + dy, x + dx)
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _connected(points):
    if not points:
        return False
    remaining = set(points)
    stack = [min(remaining)]
    remaining.remove(stack[0])
    while stack:
        y, x = stack.pop()
        for dy, dx in DIRS:
            p = (y + dy, x + dx)
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _box(cells):
    ys, xs = zip(*cells)
    return tuple(map(int, (min(ys), min(xs), max(ys), max(xs))))


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _place(corner, p):
    y, x, q = corner
    sy, sx = INWARD[q]
    return y + sy * p[0], x + sx * p[1]


def _phase(point, origin, direction):
    return ((point[0] - origin[0]) * direction[0]
            + (point[1] - origin[1]) * direction[1])


def _color(point, origin, direction, word):
    return int(word[_phase(point, origin, direction) % len(word)])


def _word_period(word, whole=True):
    return next(p for p in range(1, len(word) + 1)
                if (not whole or len(word) % p == 0)
                and all(v == word[i % p] for i, v in enumerate(word)))


def _frame_roles(a, background, components):
    found = []
    for color, groups in components.items():
        rects = [_box(g) for g in groups]
        if not all(b - t >= 6 and r - l >= 6 and g == _outline(rect)
                   for g, rect in zip(groups, rects) for t, l, b, r in [rect]):
            continue
        # Disjoint closed rectangles, including their empty interiors.
        assert color != background
        if any(not (a1[2] + 1 < a2[0] or a2[2] + 1 < a1[0]
                        or a1[3] + 1 < a2[1] or a2[3] + 1 < a1[1])
               for a1, a2 in itertools.combinations(rects, 2)):
            continue
        found.append((color, sorted(rects)))
    return found


def _interpret(a, background, frame, rects, hub_color, hub_cells):
    t, l, b, r = _box(hub_cells)
    assert b - t + 1 in (3, 5) and r - l + 1 in (3, 5)
    assert len(hub_cells) == (b - t + 1) * (r - l + 1)
    assert all(b < rt or rb < t or r < rl or rr < l for rt, rl, rb, rr in rects)
    faces = ((t - 1, (l + r) // 2), (b + 1, (l + r) // 2),
             ((t + b) // 2, l - 1), ((t + b) // 2, r + 1))
    choices = [(d, o) for d, o in zip(DIRS, faces)
               if 0 <= o[0] < a.shape[0] and 0 <= o[1] < a.shape[1]
               and int(a[o]) not in (background, frame, hub_color)]
    assert len(choices) == 1, "hub needs one centered attached strip"
    direction, origin = choices[0]
    word, ruler_cells = [], set()
    y, x = origin
    while 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        v = int(a[y, x])
        if v == background:
            break
        assert v not in (frame, hub_color)
        word.append(v)
        ruler_cells.add((y, x))
        y, x = y + direction[0], x + direction[1]
    assert 1 <= len(word) <= 10
    assert 0 <= y < a.shape[0] and 0 <= x < a.shape[1], "ruler ends before boundary"
    assert _word_period(word) <= 5
    for rect in rects:
        assert all(_phase(c[:2], origin, direction) >= 0 for c in _corners(rect))
        assert not any(rect[0] <= y <= rect[2] and rect[1] <= x <= rect[3]
                       for y, x in ruler_cells)
    clues = {}
    for y, x in np.argwhere((a != background) & (a != frame) & (a != hub_color)):
        point = (int(y), int(x))
        if point in ruler_cells:
            continue
        owners = [rect for rect in rects if rect[0] < y < rect[2] and rect[1] < x < rect[3]]
        assert len(owners) == 1, "unattached color outside frame interiors"
        corners = _corners(owners[0])
        ds = [(y - cy) ** 2 + (x - cx) ** 2 for cy, cx, _ in corners]
        assert ds.count(min(ds)) == 1, "equidistant corner clue"
        corner = corners[ds.index(min(ds))]
        sy, sx = INWARD[corner[2]]
        uv = (int((y - corner[0]) * sy), int((x - corner[1]) * sx))
        assert all(1 <= v <= 4 for v in uv)
        assert int(a[point]) == _color(point, origin, direction, word), "clue contradicts ruler"
        clues.setdefault(corner, set()).add(uv)
    assert len(clues) >= 2
    maxima = {frozenset(p) for p in clues.values()
              if all(other <= p for other in clues.values())}
    assert len(maxima) == 1, "no unique intact compatible footprint"
    pattern = maxima.pop()
    assert 3 <= len(pattern) <= 12 and _connected(pattern)
    us, vs = zip(*pattern)
    assert len(set(us)) >= 2 and len(set(vs)) >= 2
    assert len(pattern) < (max(us) - min(us) + 1) * (max(vs) - min(vs) + 1)
    assert any(p != pattern for p in clues.values()), "nothing to repair"
    for rt, rl, rb, rr in rects:
        assert rb - rt + 1 >= 2 * max(us) + 3
        assert rr - rl + 1 >= 2 * max(vs) + 3
    references = sorted(c for c, p in clues.items() if p == pattern)
    assert references
    info = dict(background=background, frame=frame, hub_color=hub_color,
                hub=(t, l, b, r), hub_cells=hub_cells, rects=rects,
                direction=direction, origin=origin, word=tuple(word),
                ruler_cells=ruler_cells, clues=clues, pattern=pattern,
                reference=references[0], references=references)
    # Check every completed location against the structural geometry.
    for corner in clues:
        for uv in pattern:
            p = _place(corner, uv)
            assert int(a[p]) in (background, _color(p, origin, direction, word))
    return info


def _read_input(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert int(np.count_nonzero(counts == counts[background])) == 1
    components = {int(c): _components(a == c) for c in np.unique(a) if c != background}
    answers = []
    for frame, rects in _frame_roles(a, background, components):
        for hub_color, groups in components.items():
            if hub_color == frame or len(groups) != 1:
                continue
            try:
                answers.append(_interpret(a, background, frame, rects, hub_color, groups[0]))
            except AssertionError:
                continue
    assert len(answers) == 1, "missing or ambiguous input interpretation"
    return answers[0]


def _render(a, info, alternative=None):
    out = a.copy()
    pattern = info["pattern"]
    origin, direction, word = info["origin"], info["direction"], info["word"]
    if alternative == "prefix_period":
        word = word[:_word_period(word, whole=False)]
    if alternative == "reverse_word":
        word = word[::-1]
    ref = info["reference"]
    ref_colors = {uv: int(a[_place(ref, uv)]) for uv in pattern}
    corners = sorted(info["clues"])
    if alternative == "unmarked":
        corners = [c for rect in info["rects"] for c in _corners(rect)]
    for corner in corners:
        local = pattern
        if alternative == "rotate_shape" and ((corner[2] in (1, 2)) != (ref[2] in (1, 2))):
            local = frozenset((v, u) for u, v in pattern)
        if alternative == "bounding_box":
            local = {(u, v) for u in range(min(p[0] for p in pattern), max(p[0] for p in pattern) + 1)
                     for v in range(min(p[1] for p in pattern), max(p[1] for p in pattern) + 1)}
        clue = info["clues"].get(corner, set())
        local_color = int(a[_place(corner, min(clue))]) if clue else 0
        for uv in local:
            point = _place(corner, uv)
            # Alternatives preserve supplied evidence too: comparisons concern
            # their actual missing-cell predictions, not recoloring known clues.
            if int(a[point]) != info["background"]:
                continue
            if alternative == "monochrome":
                value = local_color
            elif alternative == "copy_colors":
                value = ref_colors[uv]
            else:
                phase_origin = corner[:2] if alternative == "reset_phase" else origin
                value = _color(point, phase_origin, direction, word)
            out[point] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the output using only the supplied input, without mutation."""
    a = _grid(input_grid)
    return _render(a, _read_input(a))


def _distinguished(pair, info=None):
    a, out = pair["input"], pair["output"]
    info = _read_input(a) if info is None else info
    return {name for name in ALTERNATIVES
            if not np.array_equal(_render(a, info, name), out)}


def _sample_pattern(rng, radius, asymmetric=False):
    for _ in range(200):
        h, w = int(rng.integers(2, radius + 1)), int(rng.integers(2, radius + 1))
        if max(h, w) != radius:
            continue
        n = int(rng.integers(max(3, h, w), min(12, h * w - 1) + 1))
        points = {(int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1)))}
        while len(points) < n:
            frontier = sorted({(u + du, v + dv) for u, v in points for du, dv in DIRS
                               if 1 <= u + du <= h and 1 <= v + dv <= w} - points)
            points.add(frontier[int(rng.integers(len(frontier)))])
        us, vs = zip(*points)
        if (min(us), min(vs), max(us), max(vs)) != (1, 1, h, w):
            continue
        if asymmetric and points == {(v, u) for u, v in points}:
            continue
        # A narrow footprint can sit farther inward on its shorter axis.
        # Its full extent still reaches the requested maximum coordinate.
        shift_u = int(rng.integers(radius - h + 1))
        shift_v = int(rng.integers(radius - w + 1))
        return frozenset((u + shift_u, v + shift_v) for u, v in points)
    # Connected nonrectangular L with an extra asymmetric cell when possible.
    points = {(1, v) for v in range(1, radius + 1)} | {(u, 1) for u in range(1, radius + 1)}
    if radius == 2:
        points = {(1, 1), (2, 1), (2, 2)}
    else:
        points.add((radius, 2))
    return frozenset(points)


def _sample_word(rng, period, palette, style=None):
    if style == "ABA":
        return (palette[0], palette[1], palette[0])
    if style == "AB":
        return tuple(palette[:2])
    for _ in range(40):
        word = tuple(map(int, rng.choice(palette, size=period)))
        if _word_period(word) == period:
            return word
    return (palette[0],) * (period - 1) + (palette[1],) if period > 1 else (palette[0],)


def _apart(a, b, gap=2):
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _layout(rng, radius, count, word_length, force_fallback=False):
    minimum = 2 * radius + 3
    for _ in range(0 if force_fallback else 28):
        hh, hw = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
        lower_h = max(12, minimum + 3, 2 * minimum + 5 if count >= 2 else 0)
        lower_w = max(15, minimum + hw + 4, 2 * minimum + hw + 6 if count >= 3 else 0)
        if lower_h > 30 or lower_w > 30:
            continue
        h, w = int(rng.integers(lower_h, 31)), int(rng.integers(lower_w, 31))
        ht, hl = int(rng.integers(0, h - hh + 1)), int(rng.integers(0, 2))
        hub = (ht, hl, ht + hh - 1, hl + hw - 1)
        seed_box = (ht + hh // 2, hl + hw, ht + hh // 2, hl + hw + word_length - 1)
        if seed_box[3] >= w - 1:
            continue
        rects = []
        for _index in range(count):
            for _try in range(80):
                fh = int(rng.integers(minimum, min(h - 2, minimum + 9) + 1))
                max_w = min(w - hub[3] - 3, minimum + 12)
                if max_w < minimum:
                    break
                fw = int(rng.integers(minimum, max_w + 1))
                top = int(rng.integers(0, h - fh + 1))
                left = int(rng.integers(hub[3] + 3, w - fw + 1))
                rect = (top, left, top + fh - 1, left + fw - 1)
                if _apart(rect, seed_box, 1) and all(_apart(rect, r, 1) for r in rects):
                    rects.append(rect)
                    break
            else:
                break
            if len(rects) != _index + 1:
                break
        if len(rects) == count:
            return (h, w), hub, rects
    # Partition only after free placement's finite budget is exhausted.
    hub = (0, 0, 2, 2)
    columns = 2 if count >= 3 else 1
    rows = (count + columns - 1) // columns
    cell_h, cell_w = 27 // rows, 25 // columns
    rects = []
    for i in range(count):
        row, col = divmod(i, columns)
        top, left = 3 + row * cell_h, 5 + col * cell_w
        available_h, available_w = cell_h - 2, cell_w - 2
        assert available_h >= minimum and available_w >= minimum
        fh = int(rng.integers(minimum, available_h + 1))
        fw = int(rng.integers(minimum, available_w + 1))
        top += int(rng.integers(available_h - fh + 1))
        left += int(rng.integers(available_w - fw + 1))
        rects.append((top, left, top + fh - 1, left + fw - 1))
    return (30, 30), hub, rects


def _construct(rng, radius, count, period, turns, witness=None, fallback=False,
               phase_witness=False):
    colors = list(map(int, rng.permutation(10)))
    background, frame, hub_color = colors[:3]
    palette = colors[3:3 + int(rng.integers(2, 6))]
    word = _sample_word(rng, period, palette, witness)
    copies = 1 if witness else int(rng.choice((1, 1, 1, 2)))
    visible = word * copies
    pattern = _sample_pattern(rng, radius, asymmetric=witness == "ABA")
    shape, hub, rects = _layout(rng, radius, count, len(visible), fallback)
    if fallback and witness:
        # This explicit geometry proves the named witness distinctions. It is
        # a construction escape path, never a special case in solve.
        shape, hub, rects = (20, 28), (0, 0, 2, 2), [(4, 8, 14, 22)]
        pattern = (frozenset({(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2)})
                   if witness == "ABA" else frozenset({(1, 1), (2, 1), (2, 2)}))
    if fallback and phase_witness:
        # An even-width frame and distinct cyclic colors guarantee that a
        # reflected shape needs newly registered colors, for every period 2..5.
        assert 2 <= period <= 5
        visible = tuple(colors[3:3 + period])
        pattern = (frozenset({(1, 1), (2, 1), (2, 2)}) if radius == 2 else
                   frozenset({(1, v) for v in range(1, radius + 1)}
                             | {(u, 1) for u in range(1, radius + 1)} | {(radius, 2)}))
        shape, hub = (30, 30), (0, 0, 2, 2)
        columns = 2 if count >= 3 else 1
        rects = []
        for i in range(count):
            row, col = divmod(i, columns)
            top, left = 3 + row * 13, 5 + col * 12
            rects.append((top, left, top + 2 * radius + 2,
                          left + (8 if radius == 2 else 12) - 1))
    a = np.full(shape, background, dtype=np.int8)
    t, l, b, r = hub
    a[t:b + 1, l:r + 1] = hub_color
    origin, direction = ((t + b) // 2, r + 1), (0, 1)
    for i, color in enumerate(visible):
        a[origin[0], origin[1] + i] = color
    for rect in rects:
        for p in _outline(rect):
            a[p] = frame
    marked = []
    teaching = bool(witness or (fallback and phase_witness))
    for i, rect in enumerate(rects):
        n = 3 if teaching and i == 0 else int(rng.integers(2 if i == 0 else 0, 5))
        order = list(map(int, rng.permutation(4)))
        if fallback and teaching:
            order = [0, 1, 2, 3]
        marked.extend(_corners(rect)[q] for q in order[:n])
    reference = marked[0] if teaching else marked[int(rng.integers(len(marked)))]
    partial = next(c for c in marked if c != reference)
    out = a.copy()
    for corner in marked:
        for uv in pattern:
            point = _place(corner, uv)
            out[point] = _color(point, origin, direction, visible)
        ordered = sorted(pattern)
        rng.shuffle(ordered)
        if corner == reference or (corner != partial and not teaching and rng.random() < .25):
            retained = ordered
        else:
            retained = ordered[:int(rng.integers(1, len(pattern)))]
            if fallback and teaching:
                retained = [(1, 1)]
        for uv in retained:
            point = _place(corner, uv)
            a[point] = out[point]
    return {"input": np.ascontiguousarray(np.rot90(a, turns)),
            "output": np.ascontiguousarray(np.rot90(out, turns))}


def _pair(rng, radius, count, period, turns, witness=None, phase_witness=False):
    required = set(ALTERNATIVES[:-1]) if witness == "ABA" else {"reverse_word"} if witness else set()
    if phase_witness:
        required |= {"copy_colors", "monochrome"}
    for _ in range(40):
        pair = _construct(rng, radius, count, period, turns, witness)
        info = _read_input(pair["input"])
        if required <= _distinguished(pair, info):
            return pair
    pair = _construct(rng, radius, count, period, turns, witness, fallback=True,
                      phase_witness=phase_witness)
    assert required <= _distinguished(pair)
    return pair


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    colors = list(map(int, np.unique(pair["input"])))
    # At most 13 earlier inputs; 32 different injective recolorings suffice.
    for target in itertools.islice(itertools.permutations(range(10), len(colors)), 32):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = target
        result = {k: lookup[v] for k, v in pair.items()}
        if _key(result["input"]) not in used:
            return result
    raise AssertionError("bounded duplicate repair failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(4))  # larger mask, longer word, more frames, new axis
    train_axis = int(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = ("ABA" if i == 0 else "AB" if i == 1 else None) if split == "train" else None
            radius = 3 if witness == "ABA" else 2 if witness == "AB" else int(rng.integers(2, 4 if regime == 0 and split == "train" else 5))
            period = 3 if witness == "ABA" else 2 if witness == "AB" else int(rng.integers(1, 4 if regime == 1 and split == "train" else 6))
            if split == "test":
                period = max(2, period)
            nframes = 1 if witness else int(rng.integers(1, 5 if radius == 2 else 3))
            if regime == 0 and split == "test":
                radius = 4
                nframes = min(nframes, 2)
            elif regime == 1 and split == "test":
                period = int(rng.integers(4, 6))
            elif regime == 2:
                nframes = 1 if split == "train" else int(rng.integers(2, 5))
                if split == "test":
                    radius = 2
            turns = int(rng.integers(4))
            if regime == 3:
                axis = train_axis if split == "train" else 1 - train_axis
                turns = axis + 2 * int(rng.integers(2))
            pair = _unique(_pair(rng, radius, nframes, period, turns, witness,
                                 phase_witness=split == "test"), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, shape):
    return dict(shape=tuple(shape), frames=len(info["rects"]),
                footprint=tuple(sorted(info["pattern"])),
                radius=max(max(p) for p in info["pattern"]),
                size=len(info["pattern"]), period=_word_period(info["word"]),
                word_length=len(info["word"]), direction=info["direction"],
                axis=0 if info["direction"][0] == 0 else 1,
                background=info["background"], reference_count=len(info["references"]),
                marks=len(info["clues"]),
                partial_sizes=tuple(sorted(len(p) for p in info["clues"].values() if p != info["pattern"])))


def _observed_shifts(features):
    train, test = features["train"], features["test"]
    shifts = set()
    for name, key in (("larger_stencil", "radius"), ("longer_word", "period"), ("more_frames", "frames")):
        if min(f[key] for f in test) > max(f[key] for f in train):
            shifts.add(name)
    if {f["axis"] for f in train}.isdisjoint({f["axis"] for f in test}):
        shifts.add("new_axis")
    return shifts


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    used, distinguished = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in used
            used.add(_key(a))
            info = _read_input(a)
            assert np.array_equal(_render(a, info), out), "input-only reconstruction differs"
            changed = a != out
            assert np.any(changed) and np.all(a[changed] == info["background"])
            expected_support = {_place(c, uv) for c in info["clues"] for uv in info["pattern"]}
            assert all(tuple(p) in expected_support for p in np.argwhere(changed))
            assert set(map(int, out[changed])) <= set(info["word"])
            assert np.all(out[a == info["frame"]] == info["frame"])
            assert all(out[p] == a[p] for p in info["hub_cells"] | info["ruler_cells"])
            for c in info["clues"]:
                for uv in info["pattern"]:
                    p = _place(c, uv)
                    assert int(out[p]) == _color(p, info["origin"], info["direction"], info["word"])
            if split == "train":
                distinguished |= _distinguished({"input": a, "output": out}, info)
            else:
                assert {"copy_colors", "monochrome"} <= _distinguished({"input": a, "output": out}, info)
            features[split].append(_features(info, a.shape))
    assert distinguished == set(ALTERNATIVES), "training admits a named simpler rule"
    assert len({f["footprint"] for f in features["train"]}) >= 2
    assert _observed_shifts(features), "no measured train/test shift"
    return features


def self_test():
    """Stream 42 seeds; retain aggregate features and hashes, never puzzle files."""
    records, signatures, shifts = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        features = validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        shifts |= _observed_shifts(features)
        records.extend(f for s in ("train", "test") for f in features[s])
        # World registration must commute with rotations and color renaming.
        pair = puzzle["test"][0]
        lookup = np.array([3, 7, 0, 9, 2, 5, 1, 8, 6, 4], dtype=np.int8)
        for turns in range(4):
            a = np.rot90(pair["input"], turns).copy()
            before = a.copy()
            assert np.array_equal(solve(a), np.rot90(pair["output"], turns))
            assert np.array_equal(a, before)
        assert np.array_equal(solve(lookup[pair["input"]]), lookup[pair["output"]])
    assert len(signatures) == 42
    assert len({f["shape"] for f in records}) >= 40
    assert len({f["footprint"] for f in records}) >= 35
    assert {f["radius"] for f in records} == {2, 3, 4}
    assert {f["frames"] for f in records} >= {1, 2, 3, 4}
    assert {f["period"] for f in records} == {1, 2, 3, 4, 5}
    assert {f["direction"] for f in records} == set(DIRS)
    assert len({f["background"] for f in records}) == 10
    assert any(f["reference_count"] > 1 for f in records)
    assert any(f["word_length"] > f["period"] for f in records)
    assert shifts == {"larger_stencil", "longer_word", "more_frames", "new_axis"}
    assert _word_period((2, 3, 2)) == 3
    assert _word_period((2, 3, 2), whole=False) == 2
    # Exercise both construction escape paths and the explicit teaching cases.
    for turns in range(4):
        for witness, radius, period in (("ABA", 3, 3), ("AB", 2, 2)):
            pair = _construct(np.random.default_rng(7), radius, 1, period, turns, witness, True)
            assert np.array_equal(solve(pair["input"]), pair["output"])
            required = set(ALTERNATIVES[:-1]) if witness == "ABA" else {"reverse_word"}
            assert required <= _distinguished(pair)
    for radius, count in ((2, 4), (3, 2), (4, 2)):
        pair = _construct(np.random.default_rng(99), radius, count, 5, 1, fallback=True)
        assert np.array_equal(solve(pair["input"]), pair["output"])
        for period in range(2, 6):
            pair = _construct(np.random.default_rng(99), radius, count, period, 1,
                              fallback=True, phase_witness=True)
            assert np.array_equal(solve(pair["input"]), pair["output"])
            assert {"copy_colors", "monochrome"} <= _distinguished(pair)
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(137, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output mutation accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                shapes=len({f["shape"] for f in records}),
                stencils=len({f["footprint"] for f in records}),
                sizes=sorted({f["size"] for f in records}),
                frame_counts=sorted({f["frames"] for f in records}),
                periods=sorted({f["period"] for f in records}),
                shifts=sorted(shifts), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
