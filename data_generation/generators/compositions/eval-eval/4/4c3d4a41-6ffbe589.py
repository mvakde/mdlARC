"""Ejected pin colors are unary clockwise rotation pulses for a shared display.

Only NumPy and the standard library are used. The pixel reader has no access
to random seeds, generation features, examples, or labels.
"""
from collections import Counter
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(a):
    """All maximal nontrivial horizontal runs; no candidate output is read."""
    runs = []
    for y, row in enumerate(a):
        ends = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), len(row)]
        for x, e in zip(ends, ends[1:]):
            if e - x >= 7:
                runs.append((int(row[x]), y, x, e - 1))
    return runs


def _read_candidate(a, bg, s, top, left, right, bottom):
    n = right - left - 1
    roof = top + n + 1
    assert 5 <= n <= 24 and roof < bottom < a.shape[0]
    assert np.all(a[top, left:right + 1] == s)
    assert np.all(a[roof, left:right + 1] == s)
    assert np.all(a[bottom, left:right + 1] == s)
    assert np.all(a[top:bottom + 1, right] == s)
    assert np.all(a[top:roof + 1, left] == s)
    edge = a[roof + 1:bottom, left]
    assert np.all((edge == s) | (edge == bg))
    opening = np.flatnonzero(edge == bg) + roof + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    base = int(opening[1])
    span = base - roof
    assert span >= 4 and bottom - base >= 2
    picture = a[top + 1:roof, left + 1:right].copy()
    assert not np.any(picture == s)
    colors = tuple(int(v) for v in np.unique(picture) if v != bg)
    assert 2 <= len(colors) <= 8

    # The key is the only exterior object at the opening's insertion level.
    cells = np.flatnonzero(a[base, :left] != bg)
    assert len(cells) >= 3
    kl, kr = int(cells[0]), int(cells[-1])
    assert kr < left - 1 and len(cells) == kr - kl + 1
    assert np.all(a[base, kl:kr + 1] == s)
    teeth = np.flatnonzero(a[base - 1, kl:kr + 1] == s) + kl
    assert len(teeth) >= 2 and teeth[0] == kl and teeth[-1] == kr
    assert np.all(np.diff(teeth) >= 2)
    heights = []
    for col in teeth:
        row = base - 1
        while row >= 0 and a[row, col] == s:
            row -= 1
        heights.append(base - 1 - row)
    assert all(1 <= h <= span - 2 for h in heights)

    lanes = np.flatnonzero(a[roof + 1, left + 1:right] != bg) + left + 1
    assert len(lanes) == len(teeth)
    assert np.all(lanes - teeth == lanes[0] - teeth[0])
    assert lanes[0] >= left + 2 and lanes[-1] <= right - 2
    stacks, shifts, emissions = [], [], Counter()
    for col, h in zip(lanes, heights):
        segment = a[roof + 1:base, col]
        used = np.flatnonzero(segment != bg)
        assert len(used) and np.array_equal(used, np.arange(len(used)))
        seq = tuple(int(v) for v in segment[:len(used)])
        assert set(seq) <= set(colors) | {s}
        capacity = span - h - 1
        d = len(seq) - capacity
        assert 0 <= d <= h and capacity >= 1
        stacks.append(seq)
        shifts.append(d)
        emissions.update(v for v in seq[:d] if v != s)

    # Reconstruct the grammar solely from pixels, rejecting unexplained noise.
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[top, left:right + 1] = s
    expected[roof, left:right + 1] = s
    expected[bottom, left:right + 1] = s
    expected[top:bottom + 1, left] = s
    expected[top:bottom + 1, right] = s
    expected[base - 1:base + 1, left] = bg
    expected[top + 1:roof, left + 1:right] = picture
    expected[base, kl:kr + 1] = s
    for col, h in zip(teeth, heights):
        expected[base - h:base, col] = s
    for col, seq in zip(lanes, stacks):
        expected[roof + 1:roof + 1 + len(seq), col] = seq
    assert np.array_equal(a, expected)
    return dict(background=bg, structure=s, top=top, roof=roof, base=base,
                bottom=bottom, left=left, right=right, key_left=kl,
                picture=picture, colors=colors, lanes=tuple(map(int, lanes)),
                heights=tuple(heights), stacks=tuple(stacks), shifts=tuple(shifts),
                emissions=dict(emissions), size=n, shape=a.shape)


def _read(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent color")
    bg = int(values[counts.argmax()])
    found = []
    for orientation in range(4):
        a = np.rot90(grid, -orientation)
        runs = _runs(a)
        endpoints = {}
        for s, y, left, right in runs:
            if s != bg:
                endpoints.setdefault((s, left, right), []).append(y)
        for (s, left, right), rows in endpoints.items():
            for top in rows:
                roof = top + right - left
                if roof not in rows:
                    continue
                for bottom in rows:
                    if bottom <= roof:
                        continue
                    try:
                        info = _read_candidate(a, bg, s, top, left, right, bottom)
                    except (AssertionError, IndexError):
                        continue
                    info["orientation"] = orientation
                    found.append(info)
    if len(found) != 1:
        raise ValueError("input must identify exactly one framed display, chamber and comb")
    return found[0]


def _rotate_layers(source, bg, counts):
    out = np.full(source.shape, bg, dtype=np.int8)
    occupied = np.zeros(source.shape, dtype=bool)
    for color in np.unique(source):
        if color == bg:
            continue
        mask = np.rot90(source == color, -int(counts.get(int(color), 0)) % 4)
        if np.any(occupied & mask):
            raise ValueError("rotated picture layers conflict")
        out[mask] = color
        occupied |= mask
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count the clipped prefixes, rotate matching layers, return the picture."""
    f = _read(input_grid)
    output = _rotate_layers(f["picture"], f["background"], f["emissions"])
    return np.rot90(output, f["orientation"]).copy()


def _orbits(n):
    unused = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while unused:
        r, c = min(unused)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n - 1 - r
        unused.difference_update(orbit)
        result.append(orbit)
    return result


def _symmetry(mask):
    if np.array_equal(mask, np.rot90(mask)):
        return "quarter"
    if np.array_equal(mask, np.rot90(mask, 2)):
        return "half"
    return "asymmetric"


def _picture(rng, n, bg, colors, all_asymmetric=False):
    """Sample shapes in disjoint rotation-invariant territories, never templates."""
    orbits = _orbits(n)
    four = [o for o in orbits if len(o) == 4]
    center = [o for o in orbits if len(o) == 1]
    k = len(colors)
    assert len(four) >= k
    style = int(rng.integers(3))
    middle = (n - 1) / 2
    anchor = rng.uniform(0, middle, 2)
    scores = []
    for o in four:
        r, c = o[0]
        if style == 0:
            score = min(r, c, n - 1 - r, n - 1 - c)
        elif style == 1:
            score = abs(abs(r - middle) - abs(c - middle))
        else:
            score = min((y - anchor[0]) ** 2 + (x - anchor[1]) ** 2 for y, x in o)
        scores.append(score + float(rng.uniform(0, .7)))
    order = np.argsort(scores)
    sizes = np.ones(k, dtype=int) + rng.multinomial(len(four) - k, rng.dirichlet(np.ones(k) * 2))
    owners, cursor = [[] for _ in colors], 0
    for i in rng.permutation(k):
        owners[int(i)] = [four[int(j)] for j in order[cursor:cursor + sizes[i]]]
        cursor += int(sizes[i])
    a = np.full((n, n), bg, dtype=np.int8)
    for i, color in enumerate(colors):
        kind = "asymmetric" if all_asymmetric else str(rng.choice(
            ("asymmetric", "half", "quarter"), p=(.5, .25, .25)))
        phase = int(rng.integers(4))
        density = float(rng.uniform(.3, .9))
        for j, orbit in enumerate(owners[i]):
            if j > 0 and rng.random() > density:
                continue
            if kind == "quarter":
                phases = list(range(4))
            elif kind == "half":
                phases = [phase, (phase + 2) % 4] if j == 0 or rng.random() < .65 else list(range(4))
            elif j == 0:
                phases = [phase]  # A guaranteed asymmetric orbit witness.
            else:
                phases = [(phase + t) % 4 for t in range(int(rng.integers(1, 5)))]
            for p in phases:
                a[orbit[p]] = color
        if kind == "quarter" and center and rng.random() < .4:
            a[center[0][0]] = color
            center = []
    return a


def _geometry(rng, lanes, minimum_size=5, minimum_span=5):
    n = int(rng.integers(max(minimum_size, 2 * lanes + 1), 15))
    width_cap = min(n - 2, 25 - n)
    gaps = [2] * (lanes - 1)
    slack = width_cap - (2 * lanes - 1)
    for i in rng.permutation(lanes - 1):
        add = int(rng.integers(0, min(2, slack) + 1))
        gaps[int(i)] += add
        slack -= add
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    kw = offsets[-1] + 1
    # Minimum exterior margins and a key/chamber gap, plus bounded random slack.
    horizontal = [1, 1, 1]
    budget = 30 - (n + kw + 5)
    for i in rng.permutation(3):
        extra = int(rng.integers(0, min((1, 2, 2)[i], budget) + 1))
        horizontal[int(i)] += extra
        budget -= extra
    kl, gap, margin_right = horizontal
    left = kl + kw + gap
    right = left + n + 1
    first = left + int(rng.integers(2, n - kw + 1))
    span = int(rng.integers(minimum_span, min(11, 24 - n) + 1))
    vertical = [1, 2, 1]  # top margin, floor clearance, bottom margin
    budget = 30 - (n + span + 6)
    for i in rng.permutation(3):
        extra = int(rng.integers(0, min(1, budget) + 1))
        vertical[int(i)] += extra
        budget -= extra
    top, clearance, margin_bottom = vertical
    roof = top + n + 1
    base = roof + span
    bottom = base + clearance
    return dict(size=n, shape=(bottom + margin_bottom + 1, right + margin_right + 1),
                top=top, roof=roof, base=base, bottom=bottom, left=left, right=right,
                key_left=kl, lanes=tuple(first + x for x in offsets), span=span)


def _render(f):
    bg, s = f["background"], f["structure"]
    a = np.full(f["shape"], bg, dtype=np.int8)
    t, roof, b, l, r = (f[v] for v in ("top", "roof", "bottom", "left", "right"))
    a[t, l:r + 1] = a[roof, l:r + 1] = a[b, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    base = f["base"]
    a[base - 1:base + 1, l] = bg
    a[t + 1:roof, l + 1:r] = f["picture"]
    delta = f["lanes"][0] - f["key_left"]
    key_end = f["lanes"][-1] - delta
    a[base, f["key_left"]:key_end + 1] = s
    for col, h, seq in zip(f["lanes"], f["heights"], f["stacks"]):
        a[base - h:base, col - delta] = s
        a[roof + 1:roof + 1 + len(seq), col] = seq
    return np.rot90(a, f["orientation"]).copy()


def _construct_pair(f):
    """Construction counts explicit destination cells, rather than reader shifts."""
    counts = Counter()
    for h, seq in zip(f["heights"], f["stacks"]):
        start = f["base"] - h - len(seq)
        for i, color in enumerate(seq):
            if start + i <= f["roof"] and color != f["structure"]:
                counts[color] += 1
    out = _rotate_layers(f["picture"], f["background"], counts)
    return {"input": _render(f), "output": np.rot90(out, f["orientation"]).copy()}


def _teaching_comparison(rng, orientation):
    lanes = int(rng.choice((3, 4)))
    f = _geometry(rng, lanes, minimum_size=7, minimum_span=7)
    palette = [int(x) for x in rng.choice(10, 6, replace=False)]
    bg, s, *colors = palette
    f.update(background=bg, structure=s, colors=tuple(colors), orientation=orientation)
    f["picture"] = _picture(rng, f["size"], bg, colors, all_asymmetric=True)
    # Counts 1,2,3,0; structural loss is an extra silent event. Split the
    # three-pulse color between lanes to teach aggregation rather than lane rank.
    chunks = [3, 4] if lanes == 3 else [2, 2, 3]
    emitted = [colors[0], colors[1], colors[1], colors[2], colors[2], colors[2], s]
    for _ in range(32):
        rng.shuffle(emitted)
        parts, cursor = [], 0
        for length in chunks:
            parts.append(tuple(emitted[cursor:cursor + length]))
            cursor += length
        if sum(colors[2] in p for p in parts) >= 2:
            break
    else:
        # Swap a three-pulse token with a token outside its only occupied part.
        home = next(i for i, p in enumerate(parts) if colors[2] in p)
        other = (home + 1) % len(parts)
        parts = [list(p) for p in parts]
        j = parts[home].index(colors[2])
        parts[home][j], parts[other][0] = parts[other][0], parts[home][j]
        parts = [tuple(p) for p in parts]
    stationary = int(rng.integers(lanes))
    active = [i for i in range(lanes) if i != stationary]
    rng.shuffle(active)
    changed = int(rng.choice(active))
    heights, stacks = [0] * lanes, [()] * lanes
    cap = f["span"] - 2
    for lane in range(lanes):
        if lane == stationary:
            h = int(rng.integers(1, cap + 1))
            prefix = ()
        else:
            prefix = parts[active.index(lane)]
            h = int(rng.integers(len(prefix), cap + 1 - (lane == changed)))
        capacity = f["span"] - h - 1
        tail = rng.choice([s, *colors], capacity).tolist()
        if lane == changed:
            tail[0] = colors[2]
        heights[lane], stacks[lane] = h, tuple(prefix) + tuple(map(int, tail))
    # Nonflat witnesses allow a genuine flat-profile holdout curriculum.
    if len(set(heights)) == 1:
        heights[stationary] = 1 if heights[stationary] != 1 else cap
        length = f["span"] - heights[stationary] - 1
        stacks[stationary] = tuple(map(int, rng.choice([s, *colors], length)))
    f.update(heights=tuple(heights), stacks=tuple(stacks))
    second = dict(f)
    h2 = list(heights)
    h2[changed] += 1
    # If this would become flat, change the stationary lane in both examples;
    # it remains a contact lane and doesn't change the pulse totals.
    if len(set(h2)) == 1:
        heights[stationary] = 1
        h2[stationary] = 1
        length = f["span"] - 2
        stacks[stationary] = tuple(map(int, rng.choice([s, *colors], length)))
        f.update(heights=tuple(heights), stacks=tuple(stacks))
        second = dict(f)
    second.update(heights=tuple(h2))
    return f, second


def _profile(rng, lanes, cap, flat=False, forbid_flat=False):
    mode = "flat" if flat else str(rng.choice(("flat", "levels", "rising", "falling", "irregular")))
    if mode == "flat":
        h = [int(rng.integers(1, cap + 1))] * lanes
    elif mode == "levels":
        low = int(rng.integers(1, cap))
        high = int(rng.integers(low + 1, cap + 1))
        h = [low, high] + rng.choice([low, high], lanes - 2).tolist()
        rng.shuffle(h)
    else:
        h = rng.integers(1, cap + 1, lanes).tolist()
        if mode in ("rising", "falling"):
            h.sort(reverse=mode == "falling")
    if forbid_flat and len(set(h)) == 1:
        h[int(rng.integers(lanes))] = 1 if h[0] != 1 else cap
    return h


def _random_features(rng, regime, split, orientation):
    lanes = 5 if regime == "lanes" and split == "test" else int(rng.integers(2, 5 if regime == "lanes" else 6))
    long_test = regime == "pulses" and split == "test"
    f = _geometry(rng, lanes, minimum_span=7 if long_test else 5)
    layer_count = int(rng.integers(2, min(6, len(_orbits(f["size"])) - 1) + 1))
    palette = [int(x) for x in rng.choice(10, layer_count + 2, replace=False)]
    bg, s, *colors = palette
    f.update(background=bg, structure=s, colors=tuple(colors), orientation=orientation)
    f["picture"] = _picture(rng, f["size"], bg, colors, all_asymmetric=long_test)
    heights = _profile(rng, lanes, f["span"] - 2,
                       flat=regime == "flat" and split == "test",
                       forbid_flat=regime == "flat" and split == "train")
    shifts = [0 if rng.random() < .3 else int(rng.integers(1, h + 1)) for h in heights]
    forced = int(rng.integers(5, 10)) if long_test else 0
    if long_test:
        # Fit the requested pulse count visibly; extra tooth growth is bounded
        # by the same geometric capacity as any ordinary scene.
        while sum(shifts) < forced:
            available = [i for i in range(lanes) if shifts[i] < f["span"] - 2]
            lane = int(rng.choice(available))
            heights[lane] = max(heights[lane], shifts[lane] + 1)
            shifts[lane] += 1
    slots = [(i, j) for i, d in enumerate(shifts) for j in range(d)]
    rng.shuffle(slots)
    prefixes = [[s] * d for d in shifts]
    counted = Counter()
    forced_color = int(rng.choice(colors))
    selected = rng.choice(colors, int(rng.integers(1, layer_count + 1)), replace=False).tolist()
    for index, (i, j) in enumerate(slots):
        if index < forced:
            color = forced_color
        else:
            permitted = [s, *[c for c in selected if not (regime == "pulses" and split == "train" and counted[c] >= 4)]]
            color = int(rng.choice(permitted))
        prefixes[i][j] = color
        counted[color] += 1
    stacks = []
    for i, h in enumerate(heights):
        capacity = f["span"] - h - 1
        # Multicolor runs, sometimes a structural run, rather than obligatory
        # monochromatic payloads or a first-minimum prefix recipe.
        tail = []
        while len(tail) < capacity:
            tail.extend([int(rng.choice([s, *colors]))] * int(rng.integers(1, 4)))
        stacks.append(tuple(prefixes[i] + tail[:capacity]))
    f.update(heights=tuple(heights), stacks=tuple(stacks))
    return f


def _ensure_unique(f, used):
    """Rare deterministic color-permutation fallback; never alter the rule."""
    if _key(_render(f)) not in used:
        return f
    # Ten background shifts, each with two structure/content assignments,
    # give 20 distinct inputs, more than the maximum 14 used in a puzzle.
    for shift in range(10):
        for swap in (False, True):
            table = (np.arange(10, dtype=np.int8) + shift) % 10
            if swap:
                s, c = f["structure"], f["colors"][0]
                table[s], table[c] = int(table[c]), int(table[s])
            g = dict(f, background=int(table[f["background"]]), structure=int(table[f["structure"]]),
                     colors=tuple(int(table[c]) for c in f["colors"]), picture=table[f["picture"]],
                     stacks=tuple(tuple(int(table[c]) for c in seq) for seq in f["stacks"]))
            if _key(_render(g)) not in used:
                return g
    raise AssertionError("uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("pulses", "lanes", "orientation", "flat")))
    orientation = int(rng.integers(4))
    first, second = _teaching_comparison(rng, orientation)
    puzzle = {"train": [], "test": []}
    used = set()
    for f in (first, second):
        pair = _construct_pair(f)
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    for split, number in (("train", num_train - 2), ("test", num_test)):
        for _ in range(number):
            code = orientation
            if regime == "orientation" and split == "test":
                code = (orientation + int(rng.integers(1, 4))) % 4
            elif regime != "orientation":
                code = int(rng.integers(4))
            f = _random_features(rng, regime, split, code)
            f = _ensure_unique(f, used)
            pair = _construct_pair(f)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _stepwise_check(f):
    """Authored alternate check: push individual cells until contact, emit loss."""
    counts = Counter()
    survivors = []
    for h, seq in zip(f["heights"], f["stacks"]):
        cells = [(f["roof"] + 1 + j, color) for j, color in enumerate(seq)]
        tooth_top = f["base"] - h
        while cells[-1][0] >= tooth_top:
            moved = []
            for row, color in cells:
                if row - 1 <= f["roof"]:
                    if color != f["structure"]:
                        counts[color] += 1
                else:
                    moved.append((row - 1, color))
            cells = moved
            assert cells
        assert cells[-1][0] == tooth_top - 1
        survivors.append(tuple(color for _, color in cells))
    assert dict(counts) == f["emissions"]
    for seq, d, survivor in zip(f["stacks"], f["shifts"], survivors):
        assert seq[d:] == survivor
    # Coordinates implement rotation without calling np.rot90 for each layer.
    n, bg = f["size"], f["background"]
    out = np.full((n, n), bg, dtype=np.int8)
    for row, col in np.argwhere(f["picture"] != bg):
        color = int(f["picture"][row, col])
        y, x = int(row), int(col)
        for _ in range(counts[color] % 4):
            y, x = x, n - 1 - y
        assert out[y, x] == bg, "transformed layers must be disjoint"
        out[y, x] = color
    return np.rot90(out, f["orientation"]).copy()


def _alternative_counts(f, kind):
    if kind == "no_A":
        return Counter(v for seq in f["stacks"] for v in seq if v != f["structure"])
    if kind == "surviving":
        return Counter(v for seq, d in zip(f["stacks"], f["shifts"])
                       for v in seq[d:] if v != f["structure"])
    if kind == "height":
        result = Counter()
        for seq, h in zip(f["stacks"], f["heights"]):
            color = next((v for v in seq if v != f["structure"]), None)
            if color is not None:
                result[color] += h
        return result
    raise ValueError("unknown alternative")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _read(inp)
            assert out.shape == (f["size"], f["size"])
            assert np.array_equal(out, _stepwise_check(f))
            assert np.array_equal(out, solve(inp))
            for color in f["colors"]:
                assert np.count_nonzero(f["picture"] == color) == np.count_nonzero(out == color)
            assert not np.any(out == f["structure"])
            if split == "train":
                training.append((f, out))
    # Require evidence, without restricting the input-only semantic parser to
    # the generator's curricula, palette sizes, lane counts or asymmetry mix.
    informative = set()
    for f, _ in training:
        for c in f["colors"]:
            if _symmetry(f["picture"] == c) == "asymmetric":
                informative.add(f["emissions"].get(c, 0))
    assert {0, 1, 2, 3, 4} <= informative
    assert any(min(f["shifts"]) == 0 < max(f["shifts"]) for f, _ in training)
    assert any(any(f["structure"] in seq[:d] for seq, d in zip(f["stacks"], f["shifts"])) for f, _ in training)
    assert any(any(sum(c in seq[:d] for seq, d in zip(f["stacks"], f["shifts"])) >= 2
                       for c in f["colors"]) for f, _ in training)
    for kind in ("no_A", "surviving", "height"):
        assert any(not np.array_equal(
            _rotate_layers(f["picture"], f["background"], _alternative_counts(f, kind)),
            np.rot90(out, -f["orientation"])) for f, out in training), kind
    assert any(not np.array_equal(f["picture"], np.rot90(out, -f["orientation"])) for f, out in training)
    # Reconstruct the controlled comparison without depending on pair order.
    comparisons = []
    for i, (a, oa) in enumerate(training):
        for b, ob in training[i + 1:]:
            hardware = ("shape", "background", "structure", "orientation", "top", "roof",
                        "base", "bottom", "left", "right", "key_left", "lanes", "stacks")
            if all(a[k] == b[k] for k in hardware) and np.array_equal(a["picture"], b["picture"]):
                delta = np.asarray(b["heights"]) - np.asarray(a["heights"])
                if np.count_nonzero(delta) == 1 and np.max(np.abs(delta)) == 1:
                    assert not np.array_equal(oa, ob)
                    comparisons.append((a, b))
    assert comparisons, "training needs the one-tooth controlled comparison"


def self_test():
    """Bounded 42-seed semantic, evidence, diversity and API regression sweep."""
    puzzles, shapes, normalized, sizes, lane_counts = set(), set(), set(), set(), set()
    orientations, backgrounds, structural, symmetries, profiles, pulse_values = set(), set(), set(), set(), set(), set()
    irregular_spacing = stationary = silent = long = repeated = identities = 0
    shifts_observed = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(tuple(_key(x["input"]) for s in ("train", "test") for x in p[s]))
        infos = {s: [_read(x["input"]) for x in p[s]] for s in ("train", "test")}
        train, test = infos["train"], infos["test"]
        max_train = max(max(x["emissions"].values(), default=0) for x in train)
        if max_train <= 4 and any(max(x["emissions"].values(), default=0) >= 5 for x in test):
            shifts_observed.add("pulses")
        if max(len(x["lanes"]) for x in train) < 5 and all(len(x["lanes"]) == 5 for x in test):
            shifts_observed.add("lanes")
        if len({x["orientation"] for x in train}) == 1 and any(
                x["orientation"] != train[0]["orientation"] for x in test):
            shifts_observed.add("orientation")
        if all(len(set(x["heights"])) > 1 for x in train) and all(
                len(set(x["heights"])) == 1 for x in test):
            shifts_observed.add("flat")
        for split in ("train", "test"):
            for pair in p[split]:
                f = _read(pair["input"])
                shapes.add(pair["input"].shape)
                roles = {}
                normalized.add(tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat))
                sizes.add(f["size"])
                lane_counts.add(len(f["lanes"]))
                orientations.add(f["orientation"])
                backgrounds.add(f["background"])
                structural.add(f["structure"])
                profiles.add(f["heights"])
                symmetries.update(_symmetry(f["picture"] == c) for c in f["colors"])
                pulse_values.update(f["emissions"].get(c, 0) for c in f["colors"])
                irregular_spacing += len(set(np.diff(f["lanes"]))) > 1
                stationary += sum(d == 0 for d in f["shifts"])
                silent += sum(seq[:d].count(f["structure"]) for seq, d in zip(f["stacks"], f["shifts"]))
                long += max(f["emissions"].values(), default=0) >= 5
                repeated += any(sum(c in seq[:d] for seq, d in zip(f["stacks"], f["shifts"])) >= 2 for c in f["colors"])
                identities += np.array_equal(np.rot90(f["picture"], f["orientation"]), pair["output"])
                assert np.array_equal(pair["output"], solve(pair["input"].tolist()))
    assert len(puzzles) == 42 and len(shapes) >= 45 and len(normalized) >= 150
    assert sizes == set(range(5, 15)) and lane_counts == set(range(2, 6))
    assert orientations == set(range(4)) and backgrounds == structural == set(range(10))
    assert symmetries == {"asymmetric", "half", "quarter"}
    assert len(profiles) >= 80 and set(range(6)) <= pulse_values
    assert irregular_spacing >= 10 and stationary >= 80 and silent >= 100 and long >= 8 and repeated >= 50
    assert identities > 0
    assert shifts_observed == {"pulses", "lanes", "orientation", "flat"}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(103 + a + b, a, b)
            validate_generated_puzzle(p)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid example counts accepted")
    assert [_key(x["input"]) for x in generate_puzzle(77)["train"]] == [_key(x["input"]) for x in generate_puzzle(77)["train"]]
    assert [_key(x["input"]) for x in generate_puzzle(77)["train"]] != [_key(x["input"]) for x in generate_puzzle(78)["train"]]
    # The finite uniqueness fallback works even when every first choice is
    # already used. It preserves the physical and color-matching semantics.
    f, _ = _teaching_comparison(np.random.default_rng(5), 0)
    used = set()
    for _ in range(14):
        fresh = _ensure_unique(f, used)
        pair = _construct_pair(fresh)
        used.add(_key(pair["input"]))
        assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(used) == 14
    # Focused corruption checks demonstrate that the validator isn't only
    # comparing dimensions or trusting generator metadata.
    p = generate_puzzle(0, 2, 1)
    pair = p["test"][0]
    pair["output"] = pair["output"].copy()
    pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    p = generate_puzzle(1, 2, 1)
    f = _read(p["test"][0]["input"])
    canonical = np.rot90(p["test"][0]["input"], -f["orientation"]).copy()
    canonical[f["roof"], f["left"] + 1] = f["background"]
    try:
        solve(np.rot90(canonical, f["orientation"]))
    except ValueError:
        pass
    else:
        raise AssertionError("broken shared wall accepted")
