"""Clock-addressed mirrors: periodic nominations constrain reflected support.

Construction uses coordinate sets and far-to-near pulse painting. The input-only
solver uses maximal-run parsing, modular arithmetic, and row convolutions.
Only the standard library and NumPy are used; no parent code is imported.
"""
from __future__ import annotations

import itertools
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _schedule(blocks, width, bg, priority="near"):
    """Cellwise decoding; source gaps have no phase effect."""
    rail = np.full(width, bg, dtype=np.int8)
    for x in range(width):
        firing = [b for b in blocks if x % b["length"] == 0]
        if not firing:
            continue
        if priority == "short":
            winner = min(firing, key=lambda b: (b["length"], b["near"]))
        elif priority == "long":
            winner = min(firing, key=lambda b: (-b["length"], b["near"]))
        elif priority == "far":
            winner = firing[-1]
        else:
            winner = firing[0]
        rail[x] = winner["color"]
    return rail


def _analyze(input_grid):
    """Find the visibly encoded frame among all eight grid isometries.

    Canonical frame: code in row 0 left of a full-height separator, a blank
    row 1 on its right, and figures on the right in rows 2 onward.
    """
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size // 2:
        raise ValueError("the background must occupy a strict majority")
    interpretations = []
    for rotation in range(4):
        rotated = np.rot90(grid, rotation)
        for flipped in (False, True):
            a = np.flipud(rotated) if flipped else rotated
            h, w = a.shape
            if h < 4 or w < 4:
                continue
            for s in range(1, w - 1):
                marker = int(a[0, s])
                if (marker == bg or not np.all(a[:, s] == marker)
                        or np.count_nonzero(a == marker) != h
                        or not np.all(a[1:, :s] == bg)
                        or not np.all(a[:2, s + 1:] == bg)):
                    continue
                blocks, cursor = [], s - 1
                while cursor >= 0:
                    color = int(a[0, cursor])
                    if color == bg:
                        cursor -= 1
                        continue
                    near, length = s - cursor, 0
                    while cursor >= 0 and int(a[0, cursor]) == color:
                        length += 1
                        cursor -= 1
                    blocks.append({"color": color, "length": length, "near": near})
                field = a[2:, s + 1:]
                colors = set(map(int, np.unique(field))) - {bg}
                if (not blocks or not colors or marker in colors
                        or not colors <= {b["color"] for b in blocks}):
                    continue
                rail = _schedule(blocks, field.shape[1], bg)
                if not all(np.any(rail == c) for c in colors):
                    continue
                interpretations.append(dict(canonical=a.copy(), rotation=rotation,
                                            flipped=flipped, bg=bg, marker=marker,
                                            separator=s, width=field.shape[1],
                                            blocks=blocks, rail=rail, colors=colors))
    if len(interpretations) != 1:
        raise ValueError("the code rail and separator must have a unique interpretation")
    return interpretations[0]


def _undo(a, info):
    if info["flipped"]:
        a = np.flipud(a)
    return np.rot90(a, -info["rotation"]).copy()


def _supported_mask(field, color, eligible):
    """Maximum support on nominated integral axes, without latent parameters."""
    mask = field == color
    width = field.shape[1]
    scores2 = sum((np.convolve(row.astype(int), row.astype(int)) for row in mask),
                  np.zeros(2 * width - 1, dtype=int))
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("a figure has no nominated mirror")
    scores = scores2[2 * candidates]
    winners = candidates[scores == scores.max()]
    if len(winners) != 1 or scores.max() == 0:
        raise ValueError("eligible reflection support must have a unique positive maximum")
    axis = int(winners[0])
    reflected = 2 * axis - np.arange(width)
    valid = (reflected >= 0) & (reflected < width)
    keep = np.zeros_like(mask)
    keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
    return axis, keep, scores2


def _prediction(input_grid, *, priority="near", gate="color", erase=True):
    info = _analyze(input_grid)
    a, s, bg = info["canonical"], info["separator"], info["bg"]
    rail = _schedule(info["blocks"], info["width"], bg, priority)
    out = a.copy()
    out[0, s + 1:] = rail
    if erase:
        field = a[2:, s + 1:]
        clean = np.full_like(field, bg)
        for color in info["colors"]:
            eligible = (rail == color if gate == "color" else rail != bg
                        if gate == "any" else np.ones(info["width"], dtype=bool))
            _, keep, _ = _supported_mask(field, color, eligible)
            clean[keep] = color
        out[2:, s + 1:] = clean
    return _undo(out, info)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reveal the clock rail and keep each color's strongest addressed mirror."""
    return _prediction(input_grid)


def _coord_best(cells, axes):
    """Construction audit: explicit reflected coordinates, unlike the solver."""
    choices = []
    for axis in axes:
        supported = {p for p in cells if (p[0], 2 * int(axis) - p[1]) in cells}
        choices.append((len(supported), int(axis), supported))
    maximum = max(n for n, _, _ in choices)
    best = [(axis, core) for n, axis, core in choices if n == maximum]
    if len(best) != 1 or maximum == 0:
        return None
    return best[0]


def _code(rng, lengths, colors, width, boundary_gap=0):
    minimums = [boundary_gap] + [int(a == b) for a, b in zip(colors, colors[1:])]
    required = sum(lengths) + sum(minimums)
    cap = 29 - width
    if required > cap:
        return None
    source = int(rng.integers(required, min(cap, required + 4) + 1))
    additions = rng.multinomial(source - required, [1 / (len(lengths) + 1)] * (len(lengths) + 1))
    gaps = [g + int(v) for g, v in zip(minimums, additions)]
    return source, gaps


def _shape(rng, axis, top, height, radius):
    """Mirrored filled profiles, contours, tips and separated supported islands."""
    family = int(rng.integers(4))
    core = set()
    for y in range(height):
        if family == 0:
            level = radius
        elif family == 1:
            level = max(0, radius - abs(2 * y - (height - 1)) // 2)
        elif family == 2:
            level = int(rng.integers(radius + 1))
        else:
            level = radius if y % 3 == 0 else int(rng.integers(0, radius + 1))
        if y == height // 2:
            level = radius
        if level == 0 or rng.random() < .6:
            core.add((top + y, axis))
        for distance in range(1, level + 1):
            present = (distance == level or family == 0 and y in (0, height - 1)
                       or rng.random() < (.25 if family == 0 else .7))
            if present:
                core.update(((top + y, axis - distance), (top + y, axis + distance)))
    if height >= 5 and rng.random() < .25:
        gap = top + int(rng.integers(1, height - 1))
        core = {p for p in core if p[0] != gap}
    if not any(c != axis for _, c in core):
        row = top + int(rng.integers(height))
        core.update(((row, axis - radius), (row, axis + radius)))
    return core


def _triples(rail, color, bg, require_other_color):
    width = len(rail)
    eligible = [x for x in range(1, width - 1) if rail[x] == color]
    wrong = [x for x in range(1, width - 1) if rail[x] != color
             and (not require_other_color or rail[x] != bg)]
    return [(t, e, d) for e in eligible for t in eligible if t - e >= 3
            for d in wrong if abs(d - t) >= 3 and abs(d - e) >= 3]


def _witness(rng, rail, color, bg, height, require_other_color):
    """Two eligible echoes compete with a larger ineligible symmetric decoy."""
    triples = _triples(rail, color, bg, require_other_color)
    if not triples:
        return None
    for _ in range(80):
        axis, echo_axis, decoy_axis = triples[int(rng.integers(len(triples)))]
        h = int(rng.integers(3, 5))
        patterns = []
        for _row in range(h):
            patterns.append(set((-1, 0, 1)) if rng.random() < .6 else set((-1, 1)))
        patterns[int(rng.integers(h))] = {-1, 0, 1}
        size = sum(map(len, patterns))
        decoy_h = size // 3 + 2
        parts = [("core", h), ("echo", 1), ("decoy", decoy_h)]
        rng.shuffle(parts)
        slack = height - 2 - sum(n for _, n in parts)
        if slack < 0:
            continue
        gaps = rng.multinomial(slack, [.25] * 4)
        row = 2 + int(gaps[0])
        core, noise = set(), set()
        for i, (kind, n) in enumerate(parts):
            if kind == "core":
                core = {(row + y, axis + dx) for y, pattern in enumerate(patterns) for dx in pattern}
            else:
                center = echo_axis if kind == "echo" else decoy_axis
                noise |= {(row + y, center + dx) for y in range(n) for dx in (-1, 0, 1)}
            row += n + int(gaps[i + 1])
        cells = core | noise
        # Also corrupt the winning figure itself. Removing whole decoy objects
        # alone would leave pixel-level reflection deletion underdemonstrated.
        attached = [(r, axis + dx) for r in sorted({r for r, _ in core}) for dx in (-2, 2)
                    if 0 <= axis + dx < len(rail)]
        for j in rng.permutation(len(attached)):
            candidate = cells | {attached[int(j)]}
            addressed = _coord_best(candidate, np.flatnonzero(rail == color))
            unrestricted = _coord_best(candidate, range(len(rail)))
            if (addressed is not None and addressed[0] == axis and addressed[1] == core
                    and unrestricted is not None and unrestricted[0] == decoy_axis
                    and unrestricted[1] != core):
                return dict(axis=axis, core=core, cells=candidate)
    return None


def _place_core(rng, rail, color, height, occupied):
    eligible = [x for x in range(1, len(rail) - 1) if rail[x] == color]
    if not eligible:
        return None
    for attempt in range(100):
        axis = int(rng.choice(eligible))
        h = int(rng.integers(2, min(8 if attempt < 60 else 4, height - 2) + 1))
        radius = int(rng.integers(1, min(4 if attempt < 60 else 1, axis, len(rail) - 1 - axis) + 1))
        top = int(rng.integers(2, height - h + 1))
        core = _shape(rng, axis, top, h, radius)
        if core & occupied:
            continue
        best = _coord_best(core, np.flatnonzero(rail == color))
        if best is not None and best[0] == axis and best[1] == core:
            return dict(axis=axis, core=core, cells=set(core))
    return None


def _noise(rng, figure, rail, color, height, occupied):
    cells, core, axis = set(figure["cells"]), figure["core"], figure["axis"]
    if rng.random() < .18:
        return cells
    amount = min(24, max(1, int(round(len(core) * rng.uniform(.12, .75)))))
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    candidates = [(r, c) for r in range(2, height) for c in range(len(rail))
                  if (r, c) not in occupied and c != axis and (r, 2 * axis - c) not in cells]
    # Local additions, enclosed-hole marks, short branches and distant dots
    # share one rule; geometry only affects their sampling probability.
    rng.shuffle(candidates)
    candidates.sort(key=lambda p: (not (r0 - 1 <= p[0] <= r1 + 1 and c0 - 2 <= p[1] <= c1 + 2)))
    for p in candidates[:250]:
        if amount == 0:
            break
        if (p[0], 2 * axis - p[1]) in cells:
            continue
        candidate = cells | {p}
        best = _coord_best(candidate, np.flatnonzero(rail == color))
        if best is not None and best[0] == axis and best[1] == core:
            cells = candidate
            amount -= 1
    return cells


def _construct(rng, lengths, colors, bg, marker, width, height, rotation,
               flipped, *, witness=False, addressed_decoy=False, boundary_gap=0):
    code = _code(rng, lengths, colors, width, boundary_gap)
    if code is None:
        return None
    source, gaps = code
    inp = np.full((height, source + 1 + width), bg, dtype=np.int8)
    inp[:, source] = marker
    cursor = source
    for length, color, gap in zip(lengths, colors, gaps):
        cursor -= gap
        inp[0, cursor - length:cursor] = color
        cursor -= length
    rail = np.full(width, bg, dtype=np.int8)
    # Construction's independent overwrite formulation of pulse collisions.
    for length, color in reversed(list(zip(lengths, colors))):
        rail[::length] = color
    available = [c for c in set(colors) if any(rail[x] == c for x in range(1, width - 1))]
    figures = {}
    if witness:
        target = colors[0]
        fig = _witness(rng, rail, target, bg, height, addressed_decoy)
        if fig is None:
            return None
        figures[target] = fig
        others = [c for c in available if c != target]
        count = int(rng.integers(0, min(2, len(others)) + 1))
        selected = list(map(int, rng.choice(others, count, replace=False)))
    else:
        count = int(rng.integers(1, min(3, len(available)) + 1))
        selected = list(map(int, rng.choice(available, count, replace=False)))
    occupied = set().union(*(f["cells"] for f in figures.values())) if figures else set()
    for color in selected:
        fig = _place_core(rng, rail, color, height, occupied)
        if fig is None:
            return None
        figures[color] = fig
        occupied |= fig["cells"]
    for color, fig in figures.items():
        # The witness already contains two deliberate, substantial distractors.
        if not (witness and color == colors[0]):
            cells = _noise(rng, fig, rail, color, height, occupied)
            occupied |= cells
            fig["cells"] = cells
    out = inp.copy()
    out[0, source + 1:] = rail
    for color, fig in figures.items():
        for r, c in fig["cells"]:
            inp[r, source + 1 + c] = color
        for r, c in fig["core"]:
            out[r, source + 1 + c] = color
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    def transform(a):
        return np.rot90(np.flipud(a) if flipped else a, rotation).copy()
    return {"input": transform(inp), "output": transform(out)}


def _random_code(rng, max_period, count, budget, bg, marker, repeated=True):
    # Bounded integer compositions, rather than a menu of stored rail patterns.
    count = min(count, budget // 2)
    if count < 1:
        return None
    lengths = []
    for i in range(count):
        room = budget - sum(lengths) - 2 * (count - i - 1)
        lengths.append(1 if count == 1 and rng.random() < .35
                       else int(rng.integers(2, min(max_period, room) + 1)))
    if count >= 3 and max_period >= 4 and rng.random() < .65:
        # Favor genuinely contributing mixtures sometimes, while keeping equal,
        # divisible and occluded periods in the ordinary random distribution.
        for _ in range(12):
            unique = list(map(int, rng.choice(range(2, max_period + 1),
                                             min(count, max_period - 1), replace=False)))
            candidate = unique + [int(rng.choice(unique)) for _ in range(count - len(unique))]
            if sum(candidate) > budget:
                continue
            room = min(21, 29 - sum(candidate))
            seen = {next(i for i, length in enumerate(candidate) if x % length == 0)
                    for x in range(room) if any(x % length == 0 for length in candidate)}
            if len(seen) >= 3:
                lengths = candidate
                break
    palette = list(map(int, rng.choice([c for c in range(10) if c not in (bg, marker)],
                                      count, replace=False)))
    colors = palette.copy()
    if repeated and count >= 3 and rng.random() < .3:
        i = int(rng.integers(1, count))
        colors[i] = colors[int(rng.integers(i))]
    return lengths, colors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    a, b = [(2, 3), (2, 5), (3, 4), (3, 5)][int(rng.integers(4))]
    witness_colors = list(map(int, rng.choice(10, 2, replace=False)))
    rotation, flipped = int(rng.integers(4)), bool(rng.integers(2))
    shift = str(rng.choice(["reverse", "more_runs", "longer"], p=[.45, .30, .25]))
    puzzle, seen, backgrounds = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = split == "train" and index < 2
            joint_query = split == "test" and index == 0
            for attempt in range(180):
                if lesson:
                    lengths = [a, b] if index == 0 else [b, a]
                    colors = witness_colors if index == 0 else witness_colors[::-1]
                    bg_pool = [c for c in range(10) if c not in witness_colors]
                    if index == 1 and backgrounds[0] in bg_pool:
                        bg_pool.remove(backgrounds[0])
                    bg = int(rng.choice(bg_pool))
                    marker = int(rng.choice([c for c in range(10) if c not in colors and c != bg]))
                    width = int(rng.integers(18, min(21, 28 - sum(lengths)) + 1))
                    boundary_gap = 1
                else:
                    bg = int(rng.integers(10))
                    marker = int(rng.choice([c for c in range(10) if c != bg]))
                    if split == "test" and shift == "more_runs":
                        runs, budget = int(rng.integers(3, 5)), 14 if joint_query else 17
                    elif split == "test" and shift == "longer":
                        runs, budget = int(rng.integers(1, 3)), 6
                    else:
                        cap = 2 if shift == "more_runs" and split == "train" else 4
                        runs, budget = int(rng.integers(1, cap + 1)), 17
                    code = _random_code(rng, b, runs, budget, bg, marker, repeated=not joint_query)
                    if code is None:
                        continue
                    lengths, colors = code
                    if joint_query and lengths[0] == 1:
                        continue
                    needed = sum(lengths) + sum(x == y for x, y in zip(colors, colors[1:]))
                    upper = min(27 if split == "test" and shift == "longer" else 21, 29 - needed)
                    lower = 22 if split == "test" and shift == "longer" else 15 if joint_query else 10
                    if upper < lower:
                        continue
                    width = int(rng.integers(lower, upper + 1))
                    boundary_gap = 0
                height = int(rng.integers(18 if lesson or joint_query else 8, 31))
                turn = (rotation + 2) % 4 if split == "test" and shift == "reverse" else rotation
                pair = _construct(rng, lengths, colors, bg, marker, width, height, turn,
                                  flipped, witness=lesson or joint_query,
                                  addressed_decoy=lesson, boundary_gap=boundary_gap)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                # Input-only checks also catch a geometric interpretation other
                # than the construction's intended frame.
                try:
                    predicted = solve(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(predicted, pair["output"]):
                    raise AssertionError("construction disagrees with input-only decoding")
                break
            else:
                raise RuntimeError("bounded placement budget exhausted")
            seen.add(_key(pair["input"]))
            backgrounds.append(bg)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_diff(pair, **kwargs):
    try:
        return not np.array_equal(_prediction(pair["input"], **kwargs), pair["output"])
    except ValueError:
        return True  # The rival cannot supply a unique answer for this input.


def _gate_witness(pair, gate):
    """Require two unique, different subsets, rather than counting a rival tie."""
    info = _analyze(pair["input"])
    field = info["canonical"][2:, info["separator"] + 1:]
    alternative = (info["rail"] != info["bg"] if gate == "any"
                   else np.ones(info["width"], dtype=bool))
    for color in info["colors"]:
        _, correct, _ = _supported_mask(field, color, info["rail"] == color)
        try:
            _, rival, _ = _supported_mask(field, color, alternative)
        except ValueError:
            continue
        if not np.array_equal(correct, rival):
            return True
    return False


def _features(pair):
    info = _analyze(pair["input"])
    canonical_out = np.rot90(_grid(pair["output"]), info["rotation"])
    if info["flipped"]:
        canonical_out = np.flipud(canonical_out)
    s = info["separator"]
    field = info["canonical"][2:, s + 1:]
    clean = canonical_out[2:, s + 1:]
    axes, positive_before, axis_cells, islands, attached_removal, extents = [], False, False, False, False, []
    for color in info["colors"]:
        axis, keep, scores2 = _supported_mask(field, color, info["rail"] == color)
        axes.append(axis)
        eligible = np.flatnonzero(info["rail"] == color)
        positive_before |= any(x < axis and scores2[2 * x] > 0 for x in eligible)
        axis_cells |= bool(np.any(keep[:, axis]))
        neighbors = np.zeros_like(keep)
        neighbors[1:] |= keep[:-1]
        neighbors[:-1] |= keep[1:]
        neighbors[:, 1:] |= keep[:, :-1]
        neighbors[:, :-1] |= keep[:, 1:]
        attached_removal |= bool(np.any((field == color) & ~keep & neighbors))
        rr, cc = np.nonzero(keep)
        extents.append((int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)))
        islands |= any(not np.any(keep[y]) for y in range(int(rr.min()), int(rr.max()) + 1))
    return dict(shape=info["canonical"].shape, original_shape=np.asarray(pair["input"]).shape,
                width=info["width"], runs=len(info["blocks"]), colors=len(info["colors"]),
                periods=tuple(b["length"] for b in info["blocks"]), bg=info["bg"],
                marker=info["marker"], rotation=info["rotation"], flipped=info["flipped"],
                removed=int(np.count_nonzero(field != clean)), axes=tuple(sorted(axes)),
                positive_before=positive_before, axis_cells=axis_cells, islands=islands,
                attached_removal=attached_removal,
                extents=tuple(sorted(extents)),
                repeated=len({b["color"] for b in info["blocks"]}) < len(info["blocks"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    short_near, long_near, source_gap = False, False, False
    priority_edges, visible_periods = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = solve(inp)
            assert np.array_equal(expected, out), "clock, addressed axis, reflection or preservation error"
            a, s = info["canonical"], info["separator"]
            o = np.rot90(out, info["rotation"])
            if info["flipped"]:
                o = np.flipud(o)
            assert np.array_equal(a[:, :s + 1], o[:, :s + 1])
            assert np.array_equal(a[1, s + 1:], o[1, s + 1:])
            field, clean = a[2:, s + 1:], o[2:, s + 1:]
            assert np.all((field == clean) | (clean == info["bg"])), "receivers only lose cells"
            assert np.array_equal(o[0, s + 1:], info["rail"])
            for color in info["colors"]:
                axis, keep, _ = _supported_mask(field, color, info["rail"] == color)
                assert np.array_equal(clean == color, keep)
                assert any(c != axis for _, c in np.argwhere(keep)), "core needs reflected pair evidence"
            f = _features(pair)
            features[split].append(f)
            assert 1 <= f["runs"] <= 4 and 1 <= f["colors"] <= 3
            if split == "train":
                source_gap |= info["blocks"][0]["near"] > 1
                for b in info["blocks"]:
                    visible = np.flatnonzero(info["rail"] == b["color"])
                    if any(x > 0 and x % b["length"] == 0 for x in visible):
                        visible_periods.add(b["length"])
                bs = info["blocks"]
                for near, far in itertools.combinations(bs, 2):
                    if near["color"] == far["color"]:
                        continue
                    collision = any(x % near["length"] == x % far["length"] == 0
                                    and info["rail"][x] == near["color"] for x in range(info["width"]))
                    far_visible = any(x % far["length"] == 0 and info["rail"][x] == far["color"]
                                      for x in range(1, info["width"]))
                    if collision and far_visible:
                        short_near |= near["length"] < far["length"]
                        long_near |= near["length"] > far["length"]
                        priority_edges.add((near["color"], far["color"]))
    assert short_near and long_near and source_gap and len(visible_periods) >= 2
    assert any((b, a) in priority_edges for a, b in priority_edges), "fixed color priority fits training"
    for priority in ("short", "long", "far"):
        assert any(not np.array_equal(_schedule(i["blocks"], i["width"], i["bg"], priority), i["rail"])
                   for i in (_analyze(p["input"]) for p in puzzle["train"])), "rival priority fits the training rails"
    for split in ("train", "test"):
        assert any(f["removed"] > 0 for f in features[split]), "symmetry must matter"
        assert any(f["attached_removal"] for f in features[split]), "demonstrate pixel deletion inside a retained figure"
        assert any(_gate_witness(p, "all") for p in puzzle[split]), "clock must change mirror selection"
    assert any(_gate_witness(p, "any") for p in puzzle["train"]), "matching the address color must matter"
    assert any(f["positive_before"] for f in features["train"]), "first positive nominated axis is insufficient"
    train, test = features["train"], features["test"]
    reverse = (len({f["rotation"] for f in train}) == 1 and
               all((f["rotation"] - train[0]["rotation"]) % 4 == 2 for f in test))
    more_runs = max(f["runs"] for f in train) <= 2 and min(f["runs"] for f in test) >= 3
    longer = min(f["width"] for f in test) > max(f["width"] for f in train)
    assert reverse or more_runs or longer, "missing observable train/test shift"


def _fixture():
    """Hand-calculated rail and receiver, not solver-generated expected values."""
    inp = np.zeros((13, 20), dtype=np.int8)
    inp[:, 6] = 9
    inp[0, :6] = list(map(int, "222011"))
    core = {(2, 5), (2, 6), (2, 7), (3, 5), (3, 7), (4, 6)}
    noise = {(3, 8)} | {(6, x) for x in (1, 2, 3)} | {(r, x) for r in (8, 9, 10) for x in (8, 9, 10)}
    for r, x in core | noise:
        inp[r, 7 + x] = 1
    out = np.zeros_like(inp)
    out[:, :7] = inp[:, :7]
    out[0, 7:] = list(map(int, "1012101012101"))
    for r, x in core:
        out[r, 7 + x] = 1
    return {"input": inp, "output": out}


def self_test():
    fixture = _fixture()
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    assert _alternative_diff(fixture, gate="all")
    assert _alternative_diff(fixture, gate="any")
    assert _alternative_diff(fixture, erase=False)
    # Rotate/reflect an explicit expectation. No seed or training enters solve.
    for turn in range(4):
        for flip in (False, True):
            transform = lambda a: np.rot90(np.flipud(a) if flip else a, turn).copy()
            assert np.array_equal(solve(transform(fixture["input"])), transform(fixture["output"]))
    # Shift code blocks away from the separator: the receiver answer and phase
    # stay fixed, while source cells are preserved in their new positions.
    shifted = fixture["input"].copy()
    shifted[0, :6] = list(map(int, "222110"))
    expected = fixture["output"].copy()
    expected[0, :6] = shifted[0, :6]
    assert np.array_equal(solve(shifted), expected)
    permutation = np.array([7, 4, 0, 8, 2, 3, 1, 5, 9, 6], dtype=np.int8)
    assert np.array_equal(solve(permutation[fixture["input"]]), permutation[fixture["output"]])
    tied = fixture["input"].copy()
    tied[2:, 7:] = 0
    tied[2, 8:11] = 1  # field axis 2
    tied[4, 12:15] = 1  # field axis 6; equal support at a second eligible axis
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("an eligible-axis tie was silently broken")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(p) for q in puzzles for s in ("train", "test") for p in q[s]]
    assert len({tuple(_key(p["input"]) for s in ("train", "test") for p in q[s]) for q in puzzles}) == 42
    assert len({f["original_shape"] for f in records}) >= 35
    assert {f["rotation"] for f in records} == set(range(4))
    assert {f["flipped"] for f in records} == {False, True}
    assert {f["bg"] for f in records} == set(range(10))
    assert {f["marker"] for f in records} == set(range(10))
    assert {f["runs"] for f in records} == {1, 2, 3, 4}
    assert {f["colors"] for f in records} == {1, 2, 3}
    assert set(range(1, 6)) <= {x for f in records for x in f["periods"]}
    assert any(f["repeated"] for f in records) and any(f["islands"] for f in records)
    assert any(f["axis_cells"] for f in records)
    assert len({f["extents"] for f in records}) >= 25
    assert min(f["width"] for f in records) <= 12 and max(f["width"] for f in records) >= 24
    for seed in (0, 1, 41):
        repeated = generate_puzzle(seed)
        assert all(np.array_equal(p[k], q[k]) for s in puzzles[seed]
                   for p, q in zip(puzzles[seed][s], repeated[s]) for k in ("input", "output"))
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 3)):
        q = generate_puzzle(100 + a, a, b)
        assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    info = _analyze(bad["test"][0]["input"])
    out = bad["test"][0]["output"]
    foreground = np.argwhere((out != info["bg"]) & (out != info["marker"]))
    r, c = foreground[-1]
    out[r, c] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted deletion of supported evidence")
    return dict(seeds=42, pairs=len(records), shapes=len({f["original_shape"] for f in records}),
                receiver_extents=len({f["extents"] for f in records}),
                directions=4, reflected_frames=2, backgrounds=10,
                run_counts=sorted({f["runs"] for f in records}),
                receiver_color_counts=sorted({f["colors"] for f in records}))
