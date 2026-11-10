"""A short comb finds its seat by the mirror support of displaced color pins.

The constructor uses suffix arithmetic. The input-only solver reads pixels and
transports pins one step at a time. No parent generator or saved pair is used.
"""

from itertools import permutations

import numpy as np


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _support(block):
    return (block >= 0) & (block == block[:, ::-1])


def _raw_block(stacks, depth):
    block = np.full((depth - 1, len(stacks)), -1, dtype=int)
    for i, seq in enumerate(stacks):
        block[:len(seq), i] = seq
    return block


def _direct_candidates(stacks, heights, depth):
    """Constructor arithmetic: a push removes a prefix, without reordering."""
    n, m = len(stacks), len(heights)
    records = []
    for seat in range(n - m + 1):
        drops = [0] * n
        for k, h in enumerate(heights):
            drops[seat + k] = len(stacks[seat + k]) - (depth - h - 1)
        if min(drops) < 0:
            continue
        shifted = _raw_block([seq[d:] for seq, d in zip(stacks, drops)], depth)
        mask = _support(shifted)
        kept = np.where(mask, shifted, -1)
        off = mask.copy()
        if n % 2:
            off[:, n // 2] = False
        records.append(dict(seat=seat, drops=tuple(drops), shifted=shifted,
                            kept=kept, score=int(mask.sum()), off=int(off.sum())))
    return sorted(records, key=lambda r: (-r["score"], r["seat"]))


def _winner(records):
    if len(records) < 2:
        raise ValueError("at least two contact-feasible seats are required")
    if records[0]["score"] == records[1]["score"]:
        raise ValueError("equal maximum-support seats have no family tie-break")
    return records[0]


def _diagnostics(stacks, heights, depth, records):
    w = _winner(records)
    raw = _raw_block(stacks, depth)
    frozen = np.where(_support(raw), raw, -1)
    # Erase raw unsupported pixels, then translate remaining positions without
    # closing holes. This is the plausible but incorrect reversed-order rule.
    pre = np.full_like(raw, -1)
    for i, d in enumerate(w["drops"]):
        pre[:depth - 1 - d, i] = frozen[d:, i]
    raw_mask = _support(raw)
    moved_mask = _support(w["shifted"])
    clipped_mate = False
    for i, d in enumerate(w["drops"]):
        if d:
            continue
        other_drop = w["drops"][len(stacks) - 1 - i]
        for row in range(min(other_drop, len(stacks[i]))):
            if raw_mask[row, i] and not moved_mask[row, i]:
                clipped_mate = True
    return dict(
        omit_a=not np.array_equal(frozen, w["kept"]),
        omit_b=not np.array_equal(w["shifted"], w["kept"]),
        reversed_order=not np.array_equal(pre, w["kept"]),
        not_minimum_clip=sum(w["drops"]) > min(sum(r["drops"]) for r in records),
        clipped_mate=clipped_mate,
        color_sensitive=bool(np.any((w["shifted"] >= 0) & (w["shifted"][:, ::-1] >= 0)
                                    & (w["shifted"] != w["shifted"][:, ::-1]))),
        stationary_active=any(w["drops"][w["seat"] + k] == 0
                              for k in range(len(heights))),
        moving=max(w["drops"]) > 0,
        deleted=int(np.count_nonzero(w["shifted"] >= 0) - w["score"]),
    )


def _acceptable(stacks, heights, depth):
    records = _direct_candidates(stacks, heights, depth)
    try:
        w = _winner(records)
    except ValueError:
        return None
    d = _diagnostics(stacks, heights, depth, records)
    if w["off"] < 4 or not d["moving"] or not d["omit_a"] or not d["omit_b"]:
        return None
    return records


def _draw_walls(a, f):
    t, b, l, r, s = f["top"], f["floor"], f["left"], f["right"], f["structure"]
    a[t, l:r + 1] = a[b, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    a[f["base"] - 1:f["base"] + 1, l] = f["background"]


def _draw_key(a, f, left):
    b, s, spacing = f["base"], f["structure"], f["spacing"]
    a[b, left:left + spacing * (len(f["heights"]) - 1) + 1] = s
    for k, h in enumerate(f["heights"]):
        a[b - h:b, left + spacing * k] = s


def _render_input(f):
    a = np.full(f["shape"], f["background"], dtype=int)
    _draw_walls(a, f)
    _draw_key(a, f, f["key_left"])
    for i, seq in enumerate(f["stacks"]):
        a[f["top"] + 1:f["top"] + 1 + len(seq), f["lanes"][i]] = seq
    return a


def _render_output(f, record):
    a = np.full(f["shape"], f["background"], dtype=int)
    _draw_walls(a, f)
    _draw_key(a, f, f["lanes"][record["seat"]])
    for row, lane in np.argwhere(record["kept"] >= 0):
        a[f["top"] + 1 + row, f["lanes"][lane]] = record["kept"][row, lane]
    return a


def _wall_candidates(a, bg):
    runs = {}
    for row, values in enumerate(a):
        cuts = [0, *(np.flatnonzero(values[1:] != values[:-1]) + 1).tolist(), a.shape[1]]
        for start, stop in zip(cuts, cuts[1:]):
            color = int(values[start])
            if color != bg and stop - start >= 7:
                runs.setdefault((color, start, stop - 1), []).append(row)
    return [(s, l, r, t, b) for (s, l, r), rows in runs.items()
            for t in rows for b in rows if b > t]


def _read_canonical(a, bg, wall):
    s, l, r, top, floor = wall
    assert np.all(a[top:floor + 1, r] == s)
    edge = a[top + 1:floor, l]
    assert np.all((edge == bg) | (edge == s))
    opening = np.flatnonzero(edge == bg) + top + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    base = int(opening[-1])
    depth = base - top
    assert depth >= 3 and floor - base >= 2
    external = np.flatnonzero(a[base, :l] != bg)
    assert len(external) >= 3
    kl, kr = int(external[0]), int(external[-1])
    assert kr < l - 1 and len(external) == kr - kl + 1
    assert np.all(a[base, kl:kr + 1] == s)
    teeth = np.flatnonzero(a[base - 1, kl:kr + 1] == s) + kl
    assert len(teeth) >= 2 and teeth[0] == kl and teeth[-1] == kr
    steps = np.diff(teeth)
    assert np.all(steps == steps[0]) and steps[0] >= 2
    spacing = int(steps[0])
    heights = []
    for col in teeth:
        row = base - 1
        while row >= 0 and a[row, col] == s:
            row -= 1
        heights.append(base - row - 1)
    assert all(1 <= h <= depth - 2 for h in heights)
    lanes = np.flatnonzero(a[top + 1, l + 1:r] != bg) + l + 1
    assert len(lanes) > len(teeth)
    assert np.all(np.diff(lanes) == spacing)
    assert lanes[0] - l == r - lanes[-1] and lanes[0] - l >= 2
    stacks = []
    for col in lanes:
        segment = a[top + 1:base, col]
        occupied = np.flatnonzero(segment != bg)
        assert len(occupied) > 0 and np.array_equal(occupied, np.arange(len(occupied)))
        seq = tuple(map(int, segment[:len(occupied)]))
        assert s not in seq
        stacks.append(seq)
    f = dict(shape=a.shape, background=bg, structure=s, top=top, floor=floor,
             base=base, left=l, right=r, key_left=kl, spacing=spacing,
             heights=tuple(heights), lanes=tuple(map(int, lanes)), stacks=tuple(stacks))
    assert np.array_equal(a, _render_input(f)), "extra cells or malformed geometry"
    return f


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size / 2:
        raise ValueError("background must occupy more than half the input")
    readings = []
    for code in range(8):
        canonical = _unorient(a, code)
        for wall in _wall_candidates(canonical, bg):
            try:
                f = _read_canonical(canonical, bg, wall)
            except AssertionError:
                continue
            f["orientation"] = code
            readings.append(f)
    if len(readings) != 1:
        raise ValueError("scene must have exactly one housing/key interpretation")
    return a, readings[0]


def _simulate_candidates(canonical, f):
    """Pixel transport, separate from the constructor's prefix slicing."""
    records = []
    depth = f["base"] - f["top"]
    n, m = len(f["lanes"]), len(f["heights"])
    for seat in range(n - m + 1):
        block = np.full((depth - 1, n), -1, dtype=int)
        valid = True
        for i, col in enumerate(f["lanes"]):
            cells = [(row, int(canonical[row, col]))
                     for row in range(f["top"] + 1, f["base"])
                     if canonical[row, col] != f["background"]]
            if seat <= i < seat + m:
                tooth_top = f["base"] - f["heights"][i - seat]
                if cells[-1][0] < tooth_top - 1:
                    valid = False
                    break
                while cells[-1][0] >= tooth_top:
                    cells = [(row - 1, color) for row, color in cells if row - 1 > f["top"]]
                assert cells[-1][0] == tooth_top - 1
            for row, color in cells:
                block[row - f["top"] - 1, i] = color
        if not valid:
            continue
        keep = np.full_like(block, -1)
        # Explicit coordinate/color comparison instead of constructor masking.
        for row in range(depth - 1):
            for i in range(n):
                color = int(block[row, i])
                if color >= 0 and color == int(block[row, n - 1 - i]):
                    keep[row, i] = color
        records.append(dict(seat=seat, score=int(np.count_nonzero(keep >= 0)), kept=keep))
    return sorted(records, key=lambda r: (-r["score"], r["seat"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the winning seat and supported payload exclusively from input pixels."""
    a, f = _parse(input_grid)
    canonical = _unorient(a, f["orientation"])
    winner = _winner(_simulate_candidates(canonical, f))
    result = canonical.copy()
    bg = f["background"]
    key_pixels = [(int(row), int(col), int(canonical[row, col]))
                  for row, col in np.argwhere(canonical[:, :f["left"]] != bg)]
    for row, col, _ in key_pixels:
        result[row, col] = bg
    for col in f["lanes"]:
        result[f["top"] + 1:f["base"], col] = bg
    translation = f["lanes"][winner["seat"]] - f["key_left"]
    for row, col, color in key_pixels:
        result[row, col + translation] = color
    for row, lane in np.argwhere(winner["kept"] >= 0):
        result[f["top"] + 1 + row, f["lanes"][lane]] = winner["kept"][row, lane]
    return _orient(result, f["orientation"])


def _sample_hardware(rng, n, m, depth, palette):
    heights = tuple(map(int, rng.integers(1, depth - 1, m)))
    target = int(rng.integers(n - m + 1))
    lengths = list(map(int, rng.integers(depth - min(heights) - 1, depth, n)))
    drops = [0] * n
    for k, h in enumerate(heights):
        drops[target + k] = lengths[target + k] - (depth - h - 1)
    arrays = [list(map(int, rng.choice(palette, length - drop)))
              for length, drop in zip(lengths, drops)]
    corruption = float(rng.uniform(.05, .28))
    for i in range(n // 2):
        j = n - 1 - i
        for row in range(min(len(arrays[i]), len(arrays[j]))):
            color = int(rng.choice(palette))
            arrays[i][row] = arrays[j][row] = color
            if rng.random() < corruption:
                arrays[j][row] = int(rng.choice([c for c in palette if c != color]))
    stacks = tuple(tuple(map(int, rng.choice(palette, drop))) + tuple(seq)
                   for drop, seq in zip(drops, arrays))
    return stacks, heights


def _geometry(rng, stacks, heights, depth, roles, orientation):
    n, m = len(stacks), len(heights)
    allowed = [s for s in (2, 3) if s * (n + m - 2) + 7 <= 30]
    spacing = int(rng.choice(allowed))
    # Allocate slack rather than relying on unbounded placement rejection.
    budget = 30 - (spacing * (n + m - 2) + 7)
    extras = [0] * 5
    for i in rng.permutation(5):
        unit = 2 if i == 3 else 1
        cap = 1 if i == 3 else 2
        extras[i] = int(rng.integers(min(cap, budget // unit) + 1))
        budget -= unit * extras[i]
    kl, extra_gap, right_margin, extra_inset, _ = extras
    # Fifth slack term supplies another independent right margin contribution.
    right_margin += extras[4]
    width_key = spacing * (m - 1) + 1
    left = kl + width_key + 1 + extra_gap
    inset = 2 + extra_inset
    lanes = tuple(left + inset + spacing * i for i in range(n))
    right = lanes[-1] + inset
    top = int(rng.integers(3))
    base = top + depth
    floor = base + int(rng.integers(2, 4))
    shape = (floor + 1 + int(rng.integers(3)), right + 1 + right_margin)
    f = dict(shape=shape, background=roles[0], structure=roles[1], top=top,
             base=base, floor=floor, left=left, right=right, key_left=kl,
             spacing=spacing, lanes=lanes, stacks=stacks, heights=heights,
             orientation=orientation)
    # Rare dense compact instances get ordinary blank vertical space so the
    # visible background convention remains true, without changing hardware.
    a = _render_input(f)
    while np.count_nonzero(a == roles[0]) <= a.size / 2:
        f["shape"] = (f["shape"][0] + 1, f["shape"][1])
        assert f["shape"][0] <= 30
        a = _render_input(f)
    assert all(d <= 30 for d in f["shape"])
    return f


def _ranges(shift, test):
    ns = (6, 7) if shift == "teeth" or (shift == "lanes" and test) else (4, 5)
    if shift == "depth":
        ns = (4, 5, 6, 7)
    depths = (10, 14) if shift == "depth" and test else (6, 9)
    ms = (4, 5) if shift == "teeth" and test else (2, 3)
    if shift != "teeth":
        ms = (2, 3, 4)
    return ns, ms, depths


def _palette(rng, roles, minimum=3):
    available = [c for c in range(10) if c not in roles]
    return tuple(map(int, rng.choice(available, int(rng.integers(minimum, 7)), replace=False)))


_FALLBACK = {
    4: (((1, 1, 2, 1), (3, 0, 3, 2, 0, 2), (3, 3, 1, 2, 3, 0), (3, 2, 3, 2, 0, 1)),
        (4, 5), (3, 2)),
    6: (((0, 2, 3, 3), (0, 3, 3, 3, 2, 1), (1, 1, 1, 1), (1, 2, 1, 1, 1),
         (0, 3, 1, 2), (0, 2, 3, 3)), (3, 3), (1, 3)),
}


def _comparison(rng, shift, roles, orientation):
    ns, _, depths = _ranges(shift, False)
    n = int(rng.choice(ns))
    m = int(rng.choice((2, 3)))
    depth = int(rng.integers(depths[0], depths[1] + 1))
    palette = _palette(rng, roles, 4)
    for _ in range(80):
        stacks, heights = _sample_hardware(rng, n, m, depth, palette)
        first = _acceptable(stacks, heights, depth)
        if first is None:
            continue
        d1 = _diagnostics(stacks, heights, depth, first)
        for _ in range(100):
            other_h = tuple(map(int, rng.integers(1, depth - 1, m)))
            second = _acceptable(stacks, other_h, depth)
            if second is None or second[0]["seat"] == first[0]["seat"]:
                continue
            d2 = _diagnostics(stacks, other_h, depth, second)
            if not (d1["not_minimum_clip"] or d2["not_minimum_clip"]):
                continue
            if not (d1["reversed_order"] or d2["reversed_order"]):
                continue
            if not (d1["color_sensitive"] or d2["color_sensitive"]):
                continue
            f = _geometry(rng, stacks, heights, depth, roles, orientation)
            return [f, dict(f, heights=other_h)]
    # Certified comparisons preserve the declared split range and every
    # mandatory evidence property. They are only a bounded construction fallback.
    n = 6 if shift == "teeth" else 4
    symbolic, h1, h2 = _FALLBACK[n]
    stacks = tuple(tuple(palette[c] for c in seq) for seq in symbolic)
    f = _geometry(rng, stacks, h1, 7, roles, orientation)
    return [f, dict(f, heights=h2)]


def _ordinary(rng, shift, test, roles, orientation):
    ns, ms, depths = _ranges(shift, test)
    n = int(rng.choice(ns))
    m = int(rng.choice([m for m in ms if m < n]))
    depth = int(rng.integers(depths[0], depths[1] + 1))
    palette = _palette(rng, roles)
    for _ in range(120):
        stacks, heights = _sample_hardware(rng, n, m, depth, palette)
        if _acceptable(stacks, heights, depth) is not None:
            return _geometry(rng, stacks, heights, depth, roles, orientation)
    stacks, heights, _ = _fallback_hardware(n, m, depth, palette)
    return _geometry(rng, stacks, heights, depth, roles, orientation)


def _fallback_hardware(n, m, depth, palette):
    # A finite deterministic search, certified by self_test for every sampled
    # count/depth/alphabet-size combination. Literal palette bijections preserve
    # feasibility and all scores, so that finite check covers all role colors.
    for fallback_seed in range(256):
        fallback_rng = np.random.default_rng(fallback_seed)
        stacks, heights = _sample_hardware(fallback_rng, n, m, depth, palette)
        if _acceptable(stacks, heights, depth) is not None:
            return stacks, heights, fallback_seed
    raise RuntimeError("bounded hardware construction exhausted")


def _pair(f):
    records = _direct_candidates(f["stacks"], f["heights"], f["base"] - f["top"])
    return {"input": _orient(_render_input(f), f["orientation"]),
            "output": _orient(_render_output(f, _winner(records)), f["orientation"])}


def _fresh_pair(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    used = sorted(map(int, np.unique(pair["input"])))
    # Every assignment is distinct because every used color occurs. At most
    # thirteen previously seen inputs can forbid these injective recolorings.
    for colors in permutations(range(10), len(used)):
        mapping = np.arange(10)
        mapping[used] = colors
        candidate = {role: mapping[a] for role, a in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite palette capacity exceeds the fourteen-pair bound")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("lanes", "depth", "teeth")))
    fixed_orientation = int(rng.integers(8))
    mixed_orientation = bool(rng.random() < .35)
    fixed_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    changing_roles = bool(rng.random() < .7)
    result = {"train": [], "test": []}
    seen = set()
    comparisons = _comparison(rng, shift, fixed_roles, fixed_orientation)
    for f in comparisons:
        pair = _pair(f)
        assert _key(pair["input"]) not in seen
        seen.add(_key(pair["input"]))
        result["train"].append(pair)
    for split, remaining in (("train", num_train - 2), ("test", num_test)):
        for _ in range(remaining):
            roles = tuple(map(int, rng.choice(10, 2, replace=False))) if changing_roles else fixed_roles
            orientation = int(rng.integers(8)) if mixed_orientation else fixed_orientation
            f = _ordinary(rng, shift, split == "test", roles, orientation)
            pair = _fresh_pair(_pair(f), seen)
            seen.add(_key(pair["input"]))
            result[split].append(pair)
    rng.shuffle(result["train"])
    return result


def _hardware_key(f):
    return tuple((k, f[k]) for k in sorted(f) if k != "heights")


def validate_generated_puzzle(puzzle):
    """Check geometry, placement uniqueness, support, transport, and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, readings = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            _, f = _parse(inp)
            depth = f["base"] - f["top"]
            records = _direct_candidates(f["stacks"], f["heights"], depth)
            winner = _winner(records)
            d = _diagnostics(f["stacks"], f["heights"], depth, records)
            assert winner["off"] >= 4 and d["moving"] and d["omit_a"] and d["omit_b"]
            expected = _orient(_render_output(f, winner), f["orientation"])
            assert np.array_equal(out, expected), "incorrect support, key placement, or wall preservation"
            assert np.array_equal(solve(inp), out), "input-only pixel transport disagrees"
            canonical_out = _unorient(out, f["orientation"])
            payload = np.full((depth - 1, len(f["lanes"])), -1, dtype=int)
            for i, col in enumerate(f["lanes"]):
                segment = canonical_out[f["top"] + 1:f["base"], col]
                good = (segment != f["background"]) & (segment != f["structure"])
                payload[good, i] = segment[good]
            assert np.array_equal(payload, payload[:, ::-1]), "payload is not color-reflected"
            assert np.count_nonzero(payload >= 0) == winner["score"]
            assert np.all((payload < 0) | (payload == winner["shifted"])), "completion or reordering"
            readings[split].append(f)
            if split == "train":
                training.append((f, winner, d))
    comparisons = [(a, b) for i, a in enumerate(training) for b in training[i + 1:]
                   if _hardware_key(a[0]) == _hardware_key(b[0])
                   and a[0]["heights"] != b[0]["heights"]
                   and a[1]["seat"] != b[1]["seat"]]
    assert comparisons, "training must contain a profile-only seat-changing comparison"
    assert any(a[2]["not_minimum_clip"] or b[2]["not_minimum_clip"] for a, b in comparisons)
    assert any(a[2]["reversed_order"] or b[2]["reversed_order"] for a, b in comparisons)
    assert any(a[2]["color_sensitive"] or b[2]["color_sensitive"] for a, b in comparisons), \
        "training must reject occupancy-only reflection"
    # Validate an actual held-out geometric range, without a latent schedule tag.
    for measure in (lambda f: len(f["lanes"]), lambda f: f["base"] - f["top"],
                    lambda f: len(f["heights"])):
        if max(map(measure, readings["train"])) < min(map(measure, readings["test"])):
            break
    else:
        raise AssertionError("no promised geometric train/test shift")


def self_test():
    """Bounded 42-seed semantic/variation checks plus focused failure probes."""
    shapes, profiles, structures, signatures = set(), set(), set(), set()
    lanes, teeth, depths, spacings, orientations, backgrounds = (set() for _ in range(6))
    scores, positions, parities = set(), set(), set()
    flags = {k: 0 for k in ("stationary_active", "clipped_mate", "reversed_order", "not_minimum_clip",
                           "color_sensitive")}
    pairs = 0
    for seed in range(42):
        num_train = 2 + seed % 9
        num_test = 1 + seed % min(4, num_train - 1)
        puzzle = generate_puzzle(seed, num_train, num_test)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                a = pair["input"]
                untouched = a.copy()
                assert np.array_equal(solve(a), pair["output"]) and np.array_equal(a, untouched)
                _, f = _parse(a)
                depth = f["base"] - f["top"]
                rs = _direct_candidates(f["stacks"], f["heights"], depth)
                d = _diagnostics(f["stacks"], f["heights"], depth, rs)
                for k in flags:
                    flags[k] += int(d[k])
                shapes.add(a.shape)
                profiles.add(f["heights"])
                # Observable geometry/sequence patterns, not palette hashes alone.
                color_roles = {}
                norm = tuple(color_roles.setdefault(int(c), len(color_roles)) for c in a.ravel())
                structures.add((a.shape, norm))
                lanes.add(len(f["lanes"]))
                teeth.add(len(f["heights"]))
                depths.add(depth)
                spacings.add(f["spacing"])
                orientations.add(f["orientation"])
                backgrounds.add(f["background"])
                scores.add(rs[0]["score"])
                positions.add((rs[0]["seat"], len(f["lanes"]) - len(f["heights"])))
                parities.add((f["left"] + f["right"]) % 2)
        if seed < 3:
            again = generate_puzzle(seed, num_train, num_test)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in ("input", "output"))
    assert len(signatures) == 42 and len(structures) >= 160 and len(shapes) >= 40
    assert lanes == {4, 5, 6, 7} and teeth == {2, 3, 4, 5}
    assert len(profiles) >= 80 and depths.issuperset(range(6, 15))
    assert spacings == {2, 3} and orientations == set(range(8))
    assert backgrounds == set(range(10)) and parities == {0, 1}
    assert len(scores) >= 15 and any(seat == 0 for seat, last in positions)
    assert any(seat == last for seat, last in positions) and any(0 < seat < last for seat, last in positions)
    assert all(v > 0 for v in flags.values())
    certified_geometries, largest_fallback_seed = 0, 0
    for n in range(4, 8):
        for m in range(2, min(5, n - 1) + 1):
            for depth in range(6, 15):
                for alphabet_size in range(3, 7):
                    stacks, heights, fallback_seed = _fallback_hardware(
                        n, m, depth, tuple(range(2, 2 + alphabet_size)))
                    assert _acceptable(stacks, heights, depth) is not None
                    certified_geometries += 1
                    largest_fallback_seed = max(largest_fallback_seed, fallback_seed)
    # Certified comparisons, all scene orientations, and role permutation.
    for n, (symbolic, h1, h2) in _FALLBACK.items():
        stacks = tuple(tuple(c + 2 for c in seq) for seq in symbolic)
        f = _geometry(np.random.default_rng(n), stacks, h1, 7, (0, 1), 0)
        seats, comparison_evidence = [], []
        for h in (h1, h2):
            f = dict(f, heights=h)
            pair = _pair(f)
            records = _acceptable(stacks, h, 7)
            assert records is not None, "fallback must expose both parent concepts"
            seats.append(records[0]["seat"])
            comparison_evidence.append(_diagnostics(stacks, h, 7, records))
            for code in range(8):
                mapping = np.array([(3 * c + code) % 10 for c in range(10)])
                assert np.array_equal(solve(_orient(mapping[pair["input"]], code)),
                                      _orient(mapping[pair["output"]], code))
        assert seats[0] != seats[1]
        assert any(d["not_minimum_clip"] for d in comparison_evidence)
        assert any(d["reversed_order"] for d in comparison_evidence)
        assert any(d["color_sensitive"] for d in comparison_evidence)
    # A malformed key, an equal-score lock, and a corrupted output are rejected.
    f = _geometry(np.random.default_rng(0), ((2,) * 4,) * 4, (1, 1), 6, (0, 1), 0)
    for bad in (_render_input(f),):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("equal-support seats accepted")
    bad = generate_puzzle(300, 2, 1)
    a, f = _parse(bad["test"][0]["input"])
    work = _unorient(bad["test"][0]["output"], f["orientation"])
    payload_cells = np.argwhere((work != f["background"]) & (work != f["structure"]))
    work[tuple(payload_cells[0])] = f["background"]
    bad["test"][0]["output"] = _orient(work, f["orientation"])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted deletion of a supported cell")
    broken = a.copy()
    canonical = _unorient(broken, f["orientation"])
    canonical[f["base"], f["key_left"] + 1] = f["background"]
    try:
        solve(_orient(canonical, f["orientation"]))
    except ValueError:
        pass
    else:
        raise AssertionError("broken comb accepted")
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(421, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), profiles=len(profiles),
                color_normalized_structures=len(structures), housing_lanes=sorted(lanes),
                key_teeth=sorted(teeth), depths=sorted(depths), spacings=sorted(spacings),
                orientations=sorted(orientations), axis_parities=sorted(parities),
                winner_scores=sorted(scores), evidence_counts=flags,
                fallback_geometries_certified=certified_geometries,
                largest_fallback_seed=largest_fallback_seed,
                verification="generator-owned; no independent solver or blind prediction")
