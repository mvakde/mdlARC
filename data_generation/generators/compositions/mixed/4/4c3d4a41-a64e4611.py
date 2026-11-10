"""Insert a collared-pin comb, then inspect every clipped 3x3 blank window.

Only NumPy and the standard library are used. No files, generation metadata,
examples, or random state are consulted by solve(). Highlight is always 3.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HIGHLIGHT = 3
HOST_SEEDS = (0, 779477769, 234834131, 475296706, 384224030, 263193845,
              86444987, 520743678, 496681585, 538114510, 559380915, 520798658,
              322055502, 453812686, 277879245, 894858649, 625608097, 63910178,
              310575972, 555559381, 261746224, 973305739, 142027112, 719229222,
              623226451, 340610352, 348137955, 381144931, 79525184, 429376881,
              761501288, 613348596, 210716495, 902723981, 156321919, 642235815,
              103632327, 457739766, 545728847, 365223670, 342367899, 140616971)


def _integer(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _key(a):
    return a.shape, a.tobytes()


def _outline(f):
    a = np.full(f["shape"], f["bg"], dtype=np.int8)
    t, b, l, r, s = (f[k] for k in ("top", "bottom", "left", "right", "structure"))
    a[t, l:r + 1] = a[b, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    a[f["base"] - 1:f["base"] + 1, l] = f["bg"]
    return a


def _comb(a, f, start):
    base, s, spacing = f["base"], f["structure"], f["spacing"]
    end = start + spacing * (len(f["heights"]) - 1)
    a[base, start:end + 1] = s
    for i, height in enumerate(f["heights"]):
        a[base - height:base, start + i * spacing] = s


def _render_input(f):
    a = _outline(f)
    _comb(a, f, f["key_left"])
    for i, stack in enumerate(f["stacks"]):
        c = f["first"] + i * f["spacing"]
        for j, (color, half) in enumerate(stack):
            a[f["top"] + 1 + j, c - half:c + half + 1] = color
    return a


def _construct_mechanical(f, move=True):
    """Constructor: direct destination arithmetic, no input parser or solver."""
    a = _outline(f)
    for i, (height, stack) in enumerate(zip(f["heights"], f["stacks"])):
        c = f["first"] + i * f["spacing"]
        shift = len(stack) - (f["base"] - f["top"] - height - 1) if move else 0
        for j, (color, half) in enumerate(stack):
            row = f["top"] + 1 + j - shift
            if row > f["top"]:
                a[row, c - half:c + half + 1] = color
    _comb(a, f, f["first"])
    return a


def _clear_centers(a, bg):
    """False outside padding means an actually clipped neighborhood."""
    h, w = a.shape
    padded = np.pad(a != bg, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy + h, dx:dx + w]
    return ~near


def _paint(a, bg):
    result = a.copy()
    result[_clear_centers(a, bg)] = HIGHLIGHT
    return result


def _wall_candidates(a, bg):
    runs = {}
    for row, values in enumerate(a):
        cuts = [0, *(np.flatnonzero(values[1:] != values[:-1]) + 1).tolist(), len(values)]
        for l, stop in zip(cuts, cuts[1:]):
            s = int(values[l])
            if s != bg and stop - l >= 9:
                runs.setdefault((s, l, stop - 1), []).append(row)
    for (s, l, r), rows in runs.items():
        for t in rows:
            for b in rows:
                if b - t >= 6:
                    yield s, l, r, t, b


def _read_canonical(a, bg, wall):
    s, l, r, t, b = wall
    assert np.all(a[t:b + 1, r] == s)
    opening = np.flatnonzero(a[t + 1:b, l] == bg) + t + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    base = int(opening[1])
    assert base - t >= 4 and b - base >= 2
    outside = np.flatnonzero(a[base, :l] != bg)
    assert len(outside) >= 4
    kl, kr = int(outside[0]), int(outside[-1])
    assert kr <= l - 2 and len(outside) == kr - kl + 1
    assert np.all(a[base, kl:kr + 1] == s)
    teeth = np.flatnonzero(a[base - 1, kl:kr + 1] == s) + kl
    assert 2 <= len(teeth) <= 4 and teeth[0] == kl and teeth[-1] == kr
    spacing = int(teeth[1] - teeth[0])
    assert spacing >= 3 and np.all(np.diff(teeth) == spacing)
    heights = []
    for c in teeth:
        y = base - 1
        while y >= 0 and a[y, c] == s:
            y -= 1
        heights.append(base - y - 1)
    assert all(1 <= h <= base - t - 2 for h in heights)
    # Input root rows are thin; rows below may have lateral collars.
    lanes = np.flatnonzero(a[t + 1, l + 1:r] != bg) + l + 1
    assert len(lanes) == len(teeth) and np.all(lanes - teeth == lanes[0] - teeth[0])
    assert lanes[0] >= l + 3 and lanes[-1] <= r - 3
    stacks = []
    for c, height in zip(lanes, heights):
        center = a[t + 1:base, c]
        occupied = np.flatnonzero(center != bg)
        assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
        stack = []
        max_half = min(2, (spacing - 1) // 2, int(c - l - 2), int(r - c - 2))
        for j in range(len(occupied)):
            row, color = t + 1 + j, int(center[j])
            half = 0
            while half < max_half and a[row, c - half - 1] == color and a[row, c + half + 1] == color:
                half += 1
            stack.append((color, half))
        assert stack[0][1] == 0
        colors = [v for v, _ in stack]
        p = next((i for i, v in enumerate(colors) if v != s), len(colors))
        assert p < len(colors) and len(set(colors[p:])) == 1
        assert colors[p] not in (bg, s, HIGHLIGHT)
        assert len(stack) >= base - t - height - 1
        stacks.append(tuple(stack))
    f = dict(shape=a.shape, bg=bg, structure=s, top=t, bottom=b, left=l, right=r,
             base=base, key_left=kl, first=int(lanes[0]), spacing=spacing,
             heights=tuple(heights), stacks=tuple(stacks))
    assert np.array_equal(a, _render_input(f)), "extra or malformed cells"
    return f


def _parse(a):
    assert not np.any(a == HIGHLIGHT), "highlight 3 is reserved for the output"
    values, counts = np.unique(a, return_counts=True)
    assert counts.max() > a.size // 2, "background must be a strict majority"
    bg = int(values[np.argmax(counts)])
    readings = []
    for code in range(8):
        canonical = _unorient(a, code)
        for wall in _wall_candidates(canonical, bg):
            try:
                f = _read_canonical(canonical, bg, wall)
            except (AssertionError, IndexError):
                continue
            f["orientation"] = code
            readings.append(f)
    assert len(readings) == 1, "expected one complete comb/housing interpretation"
    return readings[0]


def _move_pixels(a, f):
    """Solver: copy visible rows and push one step at a time to tooth contact."""
    canonical = _unorient(a, f["orientation"])
    result = canonical.copy()
    bg, t, base = f["bg"], f["top"], f["base"]
    external = np.argwhere(canonical[:, :f["left"]] != bg)
    for row, c in external:
        result[row, c] = bg
    for i, stack in enumerate(f["stacks"]):
        c = f["first"] + i * f["spacing"]
        for j, (_, half) in enumerate(stack):
            result[t + 1 + j, c - half:c + half + 1] = bg
    translation = f["first"] - f["key_left"]
    for row, c in external:
        result[row, c + translation] = canonical[row, c]
    for i, stack in enumerate(f["stacks"]):
        c = f["first"] + i * f["spacing"]
        top_of_tooth = base
        while result[top_of_tooth - 1, c] == f["structure"]:
            top_of_tooth -= 1
        rows = [(t + 1 + j, half, canonical[t + 1 + j, c - half:c + half + 1].copy())
                for j, (_, half) in enumerate(stack)]
        while rows[-1][0] >= top_of_tooth:
            rows = [(row - 1, half, pixels) for row, half, pixels in rows if row - 1 > t]
            assert rows, "a payload row must survive"
        assert rows[-1][0] == top_of_tooth - 1
        for row, half, pixels in rows:
            result[row, c - half:c + half + 1] = pixels
    return _orient(result, f["orientation"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    f = _parse(a)
    return _paint(_move_pixels(a, f), f["bg"])


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = unseen.pop()
        group, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in unseen:
                    unseen.remove(point)
                    group.append(point)
                    pending.append(point)
        result.append(group)
    return result


def _measure(f):
    mech = _construct_mechanical(f)
    bg = f["bg"]
    centers = _clear_centers(mech, bg)
    unmoved = _clear_centers(_construct_mechanical(f, move=False), bg)
    inside = np.zeros(mech.shape, dtype=bool)
    inside[f["top"] + 1:f["base"], f["left"] + 1:f["right"]] = True
    groups = _components(centers)
    span = f["base"] - f["top"]
    shifts = tuple(len(stack) - (span - h - 1) for stack, h in zip(f["stacks"], f["heights"]))
    prefix = [next(j for j, (color, _) in enumerate(stack) if color != f["structure"])
              for stack in f["stacks"]]
    orth_near = mech != bg
    padded = np.pad(orth_near, 1, constant_values=False)
    h, w = mech.shape
    orth_near = orth_near.copy()
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        orth_near |= padded[dy:dy + h, dx:dx + w]
    return dict(lanes=len(shifts), span=span, spacing=f["spacing"],
                heights=f["heights"], shifts=shifts,
                gained=int(np.sum(centers & ~unmoved & inside)),
                lost=int(np.sum(unmoved & ~centers & inside)),
                internal=int(np.sum(centers & inside)), total=int(centers.sum()),
                components=len(groups), small=any(1 <= len(g) <= 4 for g in groups),
                prefix_motion=any(0 < d < p for d, p in zip(shifts, prefix)),
                payload_clip=any(d > p for d, p in zip(shifts, prefix)),
                collar_clip=any(any(half for _, half in stack[:d])
                                for stack, d in zip(f["stacks"], shifts)),
                stationary=any(d == 0 for d in shifts),
                diagonal=bool(np.any((mech == bg) & ~orth_near & ~centers)),
                edge=bool(centers[0].any() or centers[-1].any() or
                          centers[:, 0].any() or centers[:, -1].any()),
                max_half=max(half for stack in f["stacks"] for _, half in stack))


def _profile(rng, n, span, mode):
    low = int(rng.integers(1, max(2, span // 2)))
    high = int(rng.integers(max(low + 1, 2), span - 1))
    if mode == 0:
        return [int(rng.integers(1, span - 2))] * n
    if mode in (1, 2):
        h = sorted(rng.integers(1, span - 1, n).tolist(), reverse=mode == 2)
    elif mode == 3:
        h = [high] + [low] * (n - 2) + [high]
    elif mode == 4:
        h = [int(rng.choice((low, high))) for _ in range(n)]
    else:
        h = rng.integers(1, span - 1, n).tolist()
    return h


def _sample_features(rng, n, spans, collar_cap, colors, code, demo):
    bg, s = colors
    span = int(rng.integers(spans[0], spans[1] + 1))
    # Allocate real geometric slack, never padding past the 30-cell bound.
    choices = []
    for spacing in range(3, 9):
        half = min(collar_cap, (spacing - 1) // 2)
        if collar_cap == 2 and half < 2:
            continue
        inset = max(3, half + 2)
        kw = spacing * (n - 1) + 1
        if 2 * kw + 1 + 2 * inset <= 30:
            choices.append((spacing, half, inset, kw))
    spacing, half_cap, inset, kw = choices[int(rng.integers(len(choices)))]
    # For spacing three, internal gaps are thin; keep a side clearance core.
    extras = [0, 0, 0, 0, 0]
    budget = 30 - (2 * kw + 1 + 2 * inset)
    if spacing == 3:
        extras[int(rng.choice((2, 3)))] = 1
        budget -= 1
    for i in rng.permutation(5):
        add = int(rng.integers(0, min((2, 2, 2, 2, 2)[i], budget) + 1))
        extras[i] += add
        budget -= add
    kl, gap_extra, il, ir, right_pad = extras
    l = kl + kw + 1 + gap_extra
    first = l + inset + il
    r = first + kw - 1 + inset + ir
    top = int(rng.integers(0, 4))
    base = top + span
    bottom = base + int(rng.integers(2, 6))
    lower_pad = int(rng.integers(0, min(3, 29 - bottom) + 1))
    heights = _profile(rng, n, span, int(rng.integers(6)))
    witness = int(rng.integers(n))
    if demo == "prefix":
        heights[witness] = min(heights[witness], span - 3)
    elif demo == "clip":
        heights[witness] = max(2, heights[witness])
    shifts = [int(rng.integers(0, h + 1)) for h in heights]
    if demo == "prefix":
        shifts[witness] = int(rng.integers(1, heights[witness] + 1))
    elif demo == "clip":
        shifts[witness] = int(rng.integers(2, heights[witness] + 1))
        shifts[(witness + 1) % n] = 0
    payloads = [c for c in range(10) if c not in (bg, s, HIGHLIGHT)]
    palette = rng.choice(payloads, int(rng.integers(1, min(n, 4) + 1)), replace=False).tolist()
    prefix_mode = int(rng.integers(3))
    stacks = []
    for i, (height, shift) in enumerate(zip(heights, shifts)):
        length = span - height - 1 + shift
        prefix = 0 if prefix_mode == 0 else int(rng.integers(1 if prefix_mode == 1 and length > 1 else 0, length))
        if i == witness and demo == "prefix":
            prefix = int(rng.integers(shift + 1, length))
        elif i == witness and demo == "clip":
            prefix = int(rng.integers(0, shift))
        payload = int(rng.choice(palette))
        widths = [0] + [int(rng.integers(1, half_cap + 1)) if rng.random() < .30 else 0
                        for _ in range(length - 1)]
        if length > 1:
            widths[int(rng.integers(1, length))] = half_cap
        if i == witness and demo == "prefix":
            widths[shift] = half_cap
        elif i == witness and demo == "clip":
            widths[1] = half_cap
        stacks.append(tuple((s if j < prefix else payload, widths[j]) for j in range(length)))
    return dict(shape=(bottom + lower_pad + 1, r + right_pad + 1), bg=bg, structure=s,
                top=top, bottom=bottom, left=l, right=r, base=base, key_left=kl,
                first=first, spacing=spacing, heights=tuple(heights),
                stacks=tuple(stacks), orientation=code)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _integer(num_train, 2, 10, "num_train")
    ntest = _integer(num_test, 1, 4, "num_test")
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    plan = int(rng.integers(3))  # depth, lane-count, or collar-width transfer
    orientation_plan = int(rng.integers(3))  # fixed, mixed, or quarter-turn transfer
    code0 = int(rng.integers(8))
    colors = [c for c in range(10) if c != HIGHLIGHT]
    fixed_colors = tuple(int(c) for c in rng.choice(colors, 2, replace=False))
    recolor = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            demo = ("prefix", "clip")[i] if split == "train" and i < 2 else ""
            for attempt in range(96):
                is_test = split == "test"
                spans = (12, 17) if is_test and plan == 0 else (7, 10) if plan == 0 else (7, 15)
                n = (4 if is_test else int(rng.integers(2, 4))) if plan == 1 else int(rng.integers(2, 5))
                collar_cap = 2 if is_test and plan == 2 else 1 if plan == 2 else int(rng.integers(1, 3))
                if collar_cap == 2:
                    n = 2  # wide collars and both key copies must fit the canvas
                if plan == 1 and is_test:
                    collar_cap = 1
                    n = 4
                code = code0
                if orientation_plan == 1:
                    code = (code0 + i + (1 if is_test else 0)) % 8
                elif orientation_plan == 2:
                    # Retain the reflection bit while rotating. Adding modulo
                    # eight would sometimes change reflection at a wraparound.
                    offset = 1 if is_test else 2 * (i % 2)
                    code = 4 * (code0 // 4) + (code0 % 4 + offset) % 4
                role_colors = tuple(int(c) for c in rng.choice(colors, 2, replace=False)) if recolor else fixed_colors
                f = _sample_features(rng, n, spans, collar_cap, role_colors, code, demo)
                inp = _orient(_render_input(f), code)
                if np.count_nonzero(inp == f["bg"]) <= inp.size // 2 or _key(inp) in seen:
                    continue
                metrics = _measure(f)
                if (metrics["gained"] < 1 or metrics["internal"] < 2 or
                        metrics["total"] < 8 or metrics["components"] < 2):
                    continue
                if demo == "prefix" and not (metrics["prefix_motion"] and metrics["small"]):
                    continue
                if demo == "clip" and not (metrics["payload_clip"] and metrics["collar_clip"]):
                    continue
                # Check geometric uniqueness without giving the reader features.
                try:
                    read = _parse(inp)
                except AssertionError:
                    continue
                assert read["orientation"] == code
                out = _orient(_paint(_construct_mechanical(f), f["bg"]), code)
                puzzle[split].append({"input": inp, "output": out})
                seen.add(_key(inp))
                break
            else:
                raise RuntimeError(f"bounded construction failed at {seed}/{split}/{i}")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntr <= 10 and 1 <= nte <= 4 and ntr > nte
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _parse(inp)
            mechanical = _move_pixels(inp, f)
            direct = _orient(_construct_mechanical(f), f["orientation"])
            assert np.array_equal(mechanical, direct)
            assert np.array_equal(out, solve(inp))
            obstacles = mechanical != f["bg"]
            assert np.array_equal(out[obstacles], mechanical[obstacles])
            # Literal clipped-window scan verifies every center and fringe.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    window = mechanical[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    expected = HIGHLIGHT if np.all(window == f["bg"]) else mechanical[r, c]
                    assert out[r, c] == expected
            m = _measure(f)
            assert m["gained"] >= 1 and m["internal"] >= 2 and m["components"] >= 2
            assert not np.array_equal(out, mechanical), "B ablation must change answer"
            # Omitting A changes both the answer and the highlight mask.
            assert not np.array_equal(out, _paint(inp, f["bg"]))
            no_motion = _orient(_paint(_construct_mechanical(f, move=False), f["bg"]), f["orientation"])
            assert not np.array_equal(out == HIGHLIGHT, no_motion == HIGHLIGHT)
            if split == "train":
                training.append(m)
    for feature in ("prefix_motion", "payload_clip", "collar_clip", "stationary", "small"):
        assert any(m[feature] for m in training), f"missing training witness: {feature}"


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _focused_regressions():
    # Hand-specified rows: lane zero loses two rows, including a collar;
    # its next structural row and all three ordered payload shapes survive.
    # Lane one already touches its tooth and is completely stationary.
    f = dict(shape=(15, 23), bg=0, structure=5, top=2, bottom=14, left=8,
             right=21, base=11, key_left=1, first=12, spacing=5,
             heights=(4, 2), orientation=0,
             stacks=(((5, 0), (5, 1), (5, 0), (8, 2), (8, 0), (8, 1)),
                     ((9, 0), (9, 1), (9, 0), (9, 0), (9, 2), (9, 0))))
    inp = _render_input(f)
    expected = np.zeros((15, 23), dtype=np.int8)
    expected[2, 8:22] = expected[14, 8:22] = 5
    expected[2:15, 8] = expected[2:15, 21] = 5
    expected[10:12, 8] = 0
    expected[11, 12:18] = 5
    expected[7:11, 12] = expected[9:11, 17] = 5
    for row, half, color in ((3, 0, 5), (4, 2, 8), (5, 0, 8), (6, 1, 8)):
        expected[row, 12 - half:13 + half] = color
    for row, half in ((3, 0), (4, 1), (5, 0), (6, 0), (7, 2), (8, 0)):
        expected[row, 17 - half:18 + half] = 9
    assert np.array_equal(_move_pixels(inp, _parse(inp)), expected)
    # Recolor all visible roles, reserving 3, and transform the entire scene.
    colors = [c for c in range(10) if c != HIGHLIGHT]
    for code in range(8):
        mapping = np.arange(10, dtype=np.int8)
        for i, color in enumerate(colors):
            mapping[color] = colors[(i + code) % len(colors)]
        a = _orient(mapping[inp], code)
        mech = _orient(mapping[expected], code)
        assert np.array_equal(_move_pixels(a, _parse(a)), mech)
        bg = int(mapping[0])
        label = mech.copy()
        for r in range(mech.shape[0]):
            for c in range(mech.shape[1]):
                if np.all(mech[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == bg):
                    label[r, c] = HIGHLIGHT
        assert np.array_equal(solve(a), label)
    malformed = inp.copy()
    malformed[2, 10] = 0
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken roof was accepted")


def self_test():
    """Bounded streaming sweep: 42 puzzles, 30 count cases, local regressions."""
    _focused_regressions()
    signatures, shapes, profiles, orientations, palettes = set(), set(), set(), set(), set()
    tallies = Counter()
    lane_counts, half_widths, backgrounds, structures = set(), set(), set(), set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        assert _signature(p) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _parse(pair["input"])
                m = _measure(f)
                shapes.add(pair["input"].shape)
                profiles.add(f["heights"])
                orientations.add(f["orientation"])
                backgrounds.add(f["bg"])
                structures.add(f["structure"])
                palettes.add(tuple(np.unique(pair["input"]).tolist()))
                lane_counts.add(m["lanes"])
                half_widths.add(m["max_half"])
                for flag in ("small", "diagonal", "edge", "prefix_motion", "payload_clip", "collar_clip", "stationary"):
                    tallies[flag] += int(m[flag])
                tallies["pairs"] += 1
    assert len(signatures) == 42 and len(shapes) >= 35 and len(profiles) >= 40
    assert orientations == set(range(8)) and lane_counts == {2, 3, 4} and half_widths == {1, 2}
    assert len(backgrounds) >= 7 and len(structures) >= 7 and len(palettes) >= 25
    assert all(tallies[flag] >= 10 for flag in ("small", "diagonal", "edge", "prefix_motion", "payload_clip", "collar_clip", "stationary"))
    for ntr in range(2, 11):
        for nte in range(1, min(4, ntr - 1) + 1):
            p = generate_puzzle(20261023, ntr, nte)
            assert len(p["train"]) == ntr and len(p["test"]) == nte
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    # Boundary regression: a corner has a clipped 2x2 neighborhood; a
    # diagonal obstacle disqualifies its center even with clear orthogonals.
    a = np.zeros((5, 6), dtype=np.int8)
    a[2, 2] = 8
    mask = _clear_centers(a, 0)
    assert mask[0, 0] and mask[-1, -1] and not mask[1, 1]
    assert mask.sum() == 21
    pocket = np.full((5, 5), 8, dtype=np.int8)
    pocket[1:4, 1:4] = 0
    assert _paint(pocket, 0)[2, 2] == HIGHLIGHT
    assert np.count_nonzero(_paint(pocket, 0) == HIGHLIGHT) == 1
    return {"seeds": 42, "pairs": tallies["pairs"], "shapes": len(shapes),
            "profiles": len(profiles), "orientations": sorted(orientations),
            "lane_counts": sorted(lane_counts), "collar_half_widths": sorted(half_widths),
            "features": dict(tallies), "independent_solver": False}
