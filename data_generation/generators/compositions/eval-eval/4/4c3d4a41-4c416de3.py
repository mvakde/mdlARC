"""Self-repairing corner cams: infer a footprint, then resolve its pin collisions.

No parent module is imported. Construction uses destination formulas; solve
reads the input and moves its actual pin cells one step at a time.
"""

from itertools import combinations
import numpy as np


HOST_SEEDS = (0, 869532610, 471018814, 674558363, 354512163, 335253123,
              928941647, 80479807, 270663409, 494221232, 520654534,
              706734926, 15225948, 473149255, 417537093, 13607187,
              550720977, 779694988, 690000339, 930198120, 780819783,
              763268111, 778913832, 527853157, 213229398, 695829536,
              913858149, 554395733, 598594579, 465416692, 573994828,
              775977169, 131392699, 563086209, 476568604, 627007276,
              422030960, 932423560, 144949748, 145768534, 510764584,
              466323950)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer ARC colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(points):
    if not points:
        return False
    todo = [min(points)]
    seen = set(todo)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return seen == set(points)


def _positions(f, side, pattern):
    b, l, r = f["bottom"], f["left"], f["right"]
    return {(b - u, l + v if side == 0 else r - v) for u, v in pattern}


def _support(f, pattern=None):
    pattern = f["pattern"] if pattern is None else pattern
    depths = {}
    for side in (0, 1):
        local = pattern if not isinstance(pattern, tuple) else pattern[side]
        for u, v in local:
            col = f["left"] + v if side == 0 else f["right"] - v
            depths[col] = max(depths.get(col, 0), u)
    return depths


def _read_canonical(a, bg, structural, bounds, cam_colors):
    t, l, b, r = bounds
    d = b - t
    _require(d >= 8 and r - l >= 9, "chamber too small")
    _require(all(a[p] == structural for p in
                 [(t, c) for c in range(l, r + 1)] +
                 [(b, c) for c in range(l, r + 1)] +
                 [(y, l) for y in range(t, b + 1)] +
                 [(y, r) for y in range(t, b + 1)]), "broken chamber")
    pin_mask = np.zeros(a.shape, dtype=bool)
    pins = {}
    for col in range(l + 1, r):
        row = t + 1
        seq = []
        while row < b and int(a[row, col]) not in (bg, *cam_colors):
            seq.append(int(a[row, col]))
            pin_mask[row, col] = True
            row += 1
        if seq:
            prefix = 0
            while prefix < len(seq) and seq[prefix] == structural:
                prefix += 1
            _require(prefix < len(seq) and structural not in seq[prefix:],
                     "pins have a structural prefix followed by payload")
            pins[col] = tuple(seq)
    _require(len(pins) >= 2, "not enough roof-attached pins")
    _require(any(seq[0] == structural for seq in pins.values()),
             "one visible structural prefix must identify the pin wall")
    clues = [set(), set()]
    colors = [set(), set()]
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[t, l:r + 1] = expected[b, l:r + 1] = structural
    expected[t:b + 1, l] = expected[t:b + 1, r] = structural
    expected[pin_mask] = a[pin_mask]
    for y, x in np.argwhere(np.isin(a, cam_colors)):
        y, x = int(y), int(x)
        _require(t < y < b and l < x < r, "cam outside chamber")
        u = b - y
        side = 0 if x - l < r - x else 1
        v = x - l if side == 0 else r - x
        _require(x - l != r - x and 1 <= u <= min(8, (d - 2) // 2)
                 and 1 <= v <= min(10, (r - l - 2) // 2), "cam outside corner zone")
        clues[side].add((u, v))
        colors[side].add(int(a[y, x]))
        expected[y, x] = a[y, x]
    _require(all(clues) and all(len(cs) == 1 for cs in colors),
             "each lower corner has one monochrome cam clue")
    _require(np.array_equal(expected, a), "unexplained cells")
    maximal = {frozenset(p) for p in clues if all(other <= p for other in clues)}
    _require(len(maximal) == 1, "no intact common cam")
    pattern = maximal.pop()
    _require(_connected(pattern) and len(pattern) >= 4, "invalid cam footprint")
    _require(min(v for _, v in pattern) == 1 and
             {v for _, v in pattern} == set(range(1, max(v for _, v in pattern) + 1)),
             "cam footprint must occupy consecutive local columns")
    f = dict(top=t, bottom=b, left=l, right=r, background=bg,
             structure=structural, pins=pins, clues=tuple(map(frozenset, clues)),
             cam_colors=tuple(next(iter(cs)) for cs in colors), pattern=pattern)
    # A completed cam may occupy input pin cells: that is the intended collision.
    # Neither cam may reach the roof, the other cam, or the outer frame.
    _require(not _positions(f, 0, pattern) & _positions(f, 1, pattern), "cams collide")
    return f


def _parse(input_grid):
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    _require(np.all(inp[0] == bg) and np.all(inp[-1] == bg) and
             np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg),
             "the outer canvas ring is background")
    readings = []
    for turn in range(4):
        a = np.rot90(inp, turn).copy()
        for s in map(int, np.unique(a)):
            if s == bg:
                continue
            coords = np.argwhere(a == s)
            t, l = coords.min(axis=0).tolist()
            b, r = coords.max(axis=0).tolist()
            if b - t < 8 or r - l < 9:
                continue
            if not (np.all(a[t, l:r + 1] == s) and np.all(a[b, l:r + 1] == s)
                    and np.all(a[t:b + 1, l] == s) and np.all(a[t:b + 1, r] == s)):
                continue
            # Cam colors occur wholly within the bottom corner band. Pin
            # suffixes may also occur there, so they remain candidates until
            # exact geometric reconstruction and template agreement settle it.
            possible = []
            for c in map(int, np.unique(a)):
                if c in (bg, s):
                    continue
                cells = np.argwhere(a == c)
                if all(t < y < b and l < x < r and
                       1 <= b - y <= min(8, (b - t - 2) // 2) and
                       min(x - l, r - x) <= min(10, (r - l - 2) // 2)
                       for y, x in cells):
                    possible.append(c)
            for count in (1, 2):
                for cams in combinations(possible, count):
                    try:
                        f = _read_canonical(a, bg, s, (t, l, b, r), cams)
                    except ValueError:
                        continue
                    f["turn"] = turn
                    readings.append(f)
    _require(len(readings) == 1, "input must determine one chamber/cam interpretation")
    return inp, readings[0]


def _simulate(inp, f, *, complete=True, move=True, visible_support=False,
              uniform_support=False):
    a = np.rot90(inp, f["turn"]).copy()
    out = a.copy()
    bg, t, b = f["background"], f["top"], f["bottom"]
    depths = _support(f, f["clues"] if visible_support else None)
    if uniform_support:
        depths = {col: max(depths.values()) for col in depths}
    for col, seq in f["pins"].items():
        out[t + 1:t + 1 + len(seq), col] = bg
    for side in (0, 1):
        for p in _positions(f, side, f["pattern"] if complete else f["clues"][side]):
            out[p] = f["cam_colors"][side]
    for col, seq in f["pins"].items():
        cells = [(t + 1 + i, color) for i, color in enumerate(seq)]
        h = depths.get(col, 0)
        if move and h:
            contact = b - h - 1
            while cells[-1][0] > contact:
                cells = [(row - 1, color) for row, color in cells if row - 1 > t]
                _require(bool(cells), "cam consumed entire pin")
        for row, color in cells:
            out[row, col] = color
    return np.rot90(out, -f["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, a reflected reference footprint and pins from this input."""
    inp, f = _parse(input_grid)
    return _simulate(inp, f)


def _sample_pattern(rng, width, height, witness=False):
    mode = str(rng.choice(("flat", "levels", "rising", "falling", "valley", "random")))
    low = int(rng.integers(1, height))
    if mode == "flat":
        tops = [height] * width
    elif mode == "levels":
        tops = rng.choice((low, height), width).tolist()
    elif mode in ("rising", "falling"):
        tops = sorted(rng.integers(low, height + 1, width).tolist(),
                      reverse=mode == "falling")
    elif mode == "valley":
        tops = [height] + rng.integers(low, height, width - 2).tolist() + [height]
    else:
        tops = rng.integers(low, height + 1, width).tolist()
    tops[int(rng.integers(width))] = height
    offset = 1 if witness else int(rng.integers(1, min(tops) + 1))
    if witness:
        tops = [max(3, h) for h in tops]
        i, j = rng.choice(width, 2, replace=False)
        tops[int(i)], tops[int(j)] = height, height - 1
    pattern = {(u, v) for v, h in enumerate(tops, 1) for u in range(offset, h + 1)}
    removable = sorted((u, v) for u, v in pattern if offset < u < tops[v - 1])
    rng.shuffle(removable)
    for cell in removable:
        if rng.random() < .42 and _connected(pattern - {cell}):
            pattern.remove(cell)
    if witness:
        # A real cavity distinguishes the footprint from a filled height map.
        cell = (2, 2)
        if cell in pattern and _connected(pattern - {cell}):
            pattern.remove(cell)
    return frozenset(pattern)


def _payload_runs(rng, palette, length, ordered_witness=False):
    """Short color runs keep column order legible instead of adding noise."""
    runs = min(length, 2 if ordered_witness else int(rng.choice((1, 1, 2, 3))))
    cuts = sorted(rng.choice(range(1, length), runs - 1, replace=False).tolist())
    ends = [0, *cuts, length]
    colors = rng.choice(palette, runs).tolist()
    if ordered_witness:
        colors = [palette[0], palette[1]]
    return tuple(int(colors[i]) for i in range(runs) for _ in range(ends[i + 1] - ends[i]))


def _draw_input(f):
    g = np.full(f["shape"], f["background"], dtype=np.int8)
    t, l, b, r, s = f["top"], f["left"], f["bottom"], f["right"], f["structure"]
    g[t, l:r + 1] = g[b, l:r + 1] = s
    g[t:b + 1, l] = g[t:b + 1, r] = s
    for side in (0, 1):
        for point in _positions(f, side, f["clues"][side]):
            g[point] = f["cam_colors"][side]
    for col, seq in f["pins"].items():
        _require(np.all(g[t + 1:t + 1 + len(seq), col] == f["background"]),
                 "visible cam clue overlaps an input pin")
        g[t + 1:t + 1 + len(seq), col] = seq
    return g


def _construction_answer(f):
    """Direct cell destinations, rather than solve's iterative collision path."""
    full = dict(f, clues=(f["pattern"], f["pattern"]), pins={})
    g = _draw_input(full)
    depths = _support(f)
    for col, seq in f["pins"].items():
        original_bottom = f["top"] + len(seq)
        shift = max(0, original_bottom - (f["bottom"] - depths[col] - 1)) if col in depths else 0
        for i, color in enumerate(seq):
            dest = f["top"] + 1 + i - shift
            if dest > f["top"]:
                g[dest, col] = color
    return g


def _damage(rng, f, previous=None):
    target = 1 - f["reference"]
    possible = []
    for u, v in sorted(f["pattern"]):
        col = f["left"] + v if target == 0 else f["right"] - v
        # Keep a blank separator beneath a moving input pin wherever possible.
        pin_bottom = f["top"] + len(f["pins"].get(col, ()))
        if f["bottom"] - u > pin_bottom + 1:
            possible.append((u, v))
    _require(len(possible) >= 2, "no readable surviving cam clue")
    rng.shuffle(possible)
    keep = int(rng.integers(1, len(possible) + 1))
    clue = frozenset(possible[:keep])
    if previous is not None and clue == previous:
        clue = frozenset(possible[:1]) if len(previous) != 1 else frozenset(possible[:2])
    clues = [f["pattern"], f["pattern"]]
    clues[target] = clue
    return dict(f, clues=tuple(clues))


def _sample_features(rng, *, witness=False, shift_mode=0, test=False, turn=0,
                     fixed_roles=None):
    width = int(rng.integers(7, 10) if test and shift_mode == 0 else rng.integers(3, 7))
    height = 7 if test and shift_mode == 1 else int(rng.integers(4, 7))
    pattern = _sample_pattern(rng, width, height, witness)
    if witness:
        _require(pattern != frozenset((v, u) for u, v in pattern),
                 "first demonstration must distinguish reflection from rotation")
    hmap = {v: max(u for u, x in pattern if x == v) for v in range(1, width + 1)}
    high = [v for v, h in hmap.items() if h >= 3]
    _require(len(high) >= 2, "need two pressure lanes")
    d = 2 * height + int(rng.integers(2, 6))
    t, l = map(int, rng.integers(1, 3, 2))
    b = t + d
    r = l + 2 * width + 3 + int(rng.integers(0, 4))
    shape = (b + int(rng.integers(2, 4)), r + int(rng.integers(2, min(4, 31 - r))))
    _require(all(x <= 30 for x in shape), "canvas overflow")
    palette = rng.permutation(10).tolist()
    bg, structure = (palette.pop(), palette.pop()) if fixed_roles is None else fixed_roles
    palette = [c for c in palette if c not in (bg, structure)]
    rng.shuffle(palette)
    cam_a = palette.pop()
    cam_b = cam_a if rng.random() < .45 else palette.pop()
    pin_palette = palette[:int(rng.integers(2, min(5, len(palette) + 1)))]
    ref = int(rng.integers(2))
    target = 1 - ref
    if witness:
        pairs = [(i, j) for i in high for j in high if hmap[i] != hmap[j]]
        first, second = pairs[int(rng.integers(len(pairs)))]
    else:
        first, second = map(int, rng.choice(high, 2, replace=False))
    # One spare column keeps a visible clue regardless of the pressures.
    spare = int(rng.choice([v for v in hmap if v not in (first, second)]))
    target_vs = [first, second]
    others = [v for v in hmap if v not in (first, second, spare)]
    rng.shuffle(others)
    target_vs += others[:int(rng.integers(len(others) + 1))]
    ref_vs = rng.choice(list(hmap), int(rng.integers(1, min(3, width) + 1)), replace=False).tolist()
    pins = {}
    for side, columns in ((ref, ref_vs), (target, target_vs)):
        for v in columns:
            h = hmap[v]
            capacity = d - h - 1
            shift = 0 if side == ref else int(rng.integers(1, h + 1))
            if side == target and v == first:
                shift = 1
            if side == target and v == second:
                shift = min(h, int(rng.integers(2, h + 1)))
            length = capacity + shift
            prefix = int(rng.integers(0, length))
            if side == target and v == first:
                prefix = shift + 1  # visibly retained moving structural prefix
            elif side == target and v == second:
                prefix = int(rng.integers(0, shift))  # clips some payload
            payload = _payload_runs(rng, pin_palette, length - prefix,
                                    ordered_witness=side == target and v == first)
            seq = (structure,) * prefix + payload
            col = l + v if side == 0 else r - v
            pins[col] = seq
    # Optional central control lanes have no cam underneath: they never move.
    central = list(range(l + width + 1, r - width))
    rng.shuffle(central)
    for col in central[:int(rng.integers(min(2, len(central)) + 1))]:
        length = int(rng.integers(2, d))
        p = int(rng.integers(length))
        pins[col] = (structure,) * p + _payload_runs(rng, pin_palette, length - p)
    f = dict(shape=shape, top=t, bottom=b, left=l, right=r, background=bg,
             structure=structure, cam_colors=(cam_a, cam_b), pattern=pattern,
             pins=pins, reference=ref, turn=turn)
    return _damage(rng, f)


def _pair(f):
    return {"input": np.rot90(_draw_input(f), -f["turn"]).copy(),
            "output": np.rot90(_construction_answer(f), -f["turn"]).copy()}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(name + " is outside the legal range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # wider cam, taller cam, or orientation transfer
    turn = int(rng.integers(4))
    mixed = bool(rng.random() < .4)
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .35 else None
    puzzle, seen = {"train": [], "test": []}, set()
    first_f = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 1:
                f = _damage(rng, first_f, first_f["clues"][1 - first_f["reference"]])
                pair = _pair(f)
            else:
                orientation = int(rng.integers(4)) if mixed and not (split == "train" and index == 0) else turn
                if split == "test" and mode == 2:
                    orientation = (turn + int(rng.integers(1, 4))) % 4
                for _ in range(120):
                    try:
                        f = _sample_features(rng, witness=split == "train" and index == 0,
                                             shift_mode=mode, test=split == "test",
                                             turn=orientation, fixed_roles=roles)
                        pair = _pair(f)
                        _, reading = _parse(pair["input"])
                        _require(np.array_equal(_simulate(pair["input"], reading), pair["output"]),
                                 "construction/reading disagreement")
                        _require(_key(pair["input"]) not in seen, "duplicate input")
                        break
                    except ValueError:
                        continue
                else:
                    raise RuntimeError("bounded cam construction exhausted")
            if split == "train" and index == 0:
                first_f = f
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _features(inp):
    inp, f = _parse(inp)
    depths = _support(f)
    shifts, prefixes, clipped_payload, retained_prefix, ordered_moving = [], [], 0, 0, 0
    for col, seq in f["pins"].items():
        s = max(0, f["top"] + len(seq) - (f["bottom"] - depths[col] - 1)) if col in depths else 0
        p = next(i for i, c in enumerate(seq) if c != f["structure"])
        shifts.append(s)
        prefixes.append(p)
        clipped_payload += s > p
        retained_prefix += 0 < s < p
        ordered_moving += s > 0 and len(set(seq[max(p, s):])) > 1
    profile = tuple(max(u for u, x in f["pattern"] if x == v)
                    for v in range(1, max(v for _, v in f["pattern"]) + 1))
    filled = {(u, v) for v, top in enumerate(profile, 1) for u in
              range(min(u for u, x in f["pattern"] if x == v), top + 1)}
    return dict(shape=inp.shape, turn=f["turn"], pin_count=len(f["pins"]),
                width=len(profile), height=max(profile), profile=profile,
                footprint=tuple(sorted(f["pattern"])), holes=len(filled - f["pattern"]),
                clue_sizes=tuple(map(len, f["clues"])), size=len(f["pattern"]),
                moving=sum(s > 0 for s in shifts), stationary=sum(s == 0 for s in shifts),
                clipped_payload=clipped_payload, retained_prefix=retained_prefix,
                ordered_moving=ordered_moving,
                shifts=tuple(shifts), bg=f["background"], structure=f["structure"],
                same_cam_color=f["cam_colors"][0] == f["cam_colors"][1])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, train_readings = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape or uniqueness error")
            seen.add(_key(inp))
            _, f = _parse(inp)
            _require(np.array_equal(_simulate(inp, f), out), "input-only solution mismatch")
            _require(np.any(inp != out), "identity pair")
            _require(any(p != f["pattern"] for p in f["clues"]), "no damaged cam")
            _require(not np.array_equal(_simulate(inp, f, complete=False, visible_support=True), out),
                     "omitting reference completion must change the answer")
            _require(not np.array_equal(_simulate(inp, f, visible_support=True), out),
                     "the inferred hidden surface must change pin motion itself")
            _require(not np.array_equal(_simulate(inp, f, move=False), out),
                     "omitting pin motion must change the answer")
            if split == "train":
                train_infos.append(_features(inp))
                train_readings.append(f)
    _require(any(x["retained_prefix"] for x in train_infos), "missing prefix-motion witness")
    _require(any(x["clipped_payload"] for x in train_infos), "missing clipping witness")
    _require(any(x["ordered_moving"] for x in train_infos), "missing surviving ordered-payload witness")
    _require(any(x["stationary"] and len(set(x["shifts"])) > 1 for x in train_infos),
             "missing local-motion witness")
    # The first two examples intentionally differ only in fragment loss.
    _require(np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
             "missing deletion-invariance controlled comparison")
    first, second = train_readings[:2]
    _require(all(first[k] == second[k] for k in
                 ("top", "bottom", "left", "right", "background", "structure", "pins",
                  "cam_colors", "pattern", "turn")), "comparison changed hardware or template")
    _require(first["clues"] != second["clues"], "comparison did not change the deletion mask")
    _require(first["pattern"] != frozenset((v, u) for u, v in first["pattern"]),
             "adjacent-corner reflection witness missing")


def _regression_test():
    """Small hand-specified specimen with two different shifts and a true void."""
    pattern = frozenset({(u, v) for v, h in enumerate((5, 3, 4, 4), 1)
                         for u in range(1, h + 1)} - {(2, 2)})
    f = dict(shape=(19, 17), top=2, bottom=16, left=2, right=14,
             background=0, structure=5, cam_colors=(2, 3), pattern=pattern,
             clues=(pattern, frozenset({(1, 1), (1, 3), (2, 4)})),
             pins={13: (5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7),
                   11: (6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7),
                   4: (5, 6, 6, 6, 6, 6, 6, 7, 7, 7),
                   8: (5, 6, 6, 7, 7)}, turn=0)
    inp = _draw_input(f)
    # Explicit expected rows, independently of the production answer renderer.
    expected = inp.copy()
    expected[3:14, 13] = 0
    expected[3:14, 11] = 0
    for u, v in pattern:
        expected[16 - u, 14 - v] = 3
    expected[3:11, 13] = (5, 6, 6, 7, 7, 7, 7, 7)
    expected[3:12, 11] = (6, 7, 7, 7, 7, 7, 7, 7, 7)
    actual = solve(inp)
    _require(np.array_equal(actual, expected), "literal contact/order/clipping regression failed")
    _require(actual[14, 12] == 0, "cam cavity was filled")
    _require(np.array_equal(actual[3:13, 4], inp[3:13, 4]), "contact pin moved")
    _require(np.array_equal(actual[3:8, 8], inp[3:8, 8]), "unactuated pin moved")
    _, reading = _parse(inp)
    _require(not np.array_equal(_simulate(inp, reading, visible_support=True), expected),
             "visible-profile countermodel not distinguished")
    _require(not np.array_equal(_simulate(inp, reading, uniform_support=True), expected),
             "uniform-height countermodel not distinguished")
    # Structural damage destroys the visible convention; it is not repaired.
    broken = inp.copy()
    broken[2, 7] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("broken wall accepted")


def self_test():
    """Bounded input-based semantic and diversity checks on the 42 host seeds."""
    _regression_test()
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for split in ("train", "test")
                          for p in puzzle[split] for k in ("input", "output"))
        _require(signature not in signatures, "repeated seed puzzle")
        signatures.add(signature)
        for split in puzzle:
            for pair in puzzle[split]:
                records.append(_features(pair["input"]))
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list solve failed")
        # Solving must be covariant under geometric and palette transformations.
        inp, out = puzzle["test"][0]["input"], puzzle["test"][0]["output"]
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn)), "rotation failure")
        mapping = np.array([(c + 3) % 10 for c in range(10)], dtype=np.int8)
        _require(np.array_equal(solve(mapping[inp]), mapping[out]), "recolor failure")
        _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "reflection failure")
    _require(len(signatures) == 42, "42-seed coverage missing")
    _require(len({r["shape"] for r in records}) >= 50, "insufficient geometric diversity")
    _require(len({r["footprint"] for r in records}) >= 80, "insufficient stencil diversity")
    _require(len({r["profile"] for r in records}) >= 60, "insufficient pressure diversity")
    _require({r["turn"] for r in records} == set(range(4)), "orientation diversity missing")
    _require({r["width"] for r in records} >= {3, 4, 5, 6, 7, 8, 9}, "width coverage missing")
    _require({r["bg"] for r in records} == set(range(10)) and
             {r["structure"] for r in records} == set(range(10)), "palette-role coverage missing")
    _require(len({r["pin_count"] for r in records}) >= 7, "load-count diversity missing")
    _require(sum(r["holes"] > 0 for r in records) >= 60, "cavity evidence missing")
    _require(any(r["same_cam_color"] for r in records) and
             any(not r["same_cam_color"] for r in records), "cam-color reuse coverage missing")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(0, nt, nq)
            validate_generated_puzzle(p)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts lost")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(17), generate_puzzle(17)
    _require(all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split])
                 for k in ("input", "output")), "nondeterministic generation")
    # A wrong label in an untouched outer-background cell must be rejected.
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unrelated output corruption accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, legal counts, local collision and template checks")
