"""Modular cam rail: keyed profile union pushes and clips ordered pin stacks.

Only NumPy and the standard library are used. No files, generation state or
training labels are consulted by solve(). See task_spec.md for the grammar.
"""
from functools import lru_cache
from hashlib import sha256
from itertools import combinations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "bad colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows)+1, max(cols)+1


def _border(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _skyline(width, templates, anchors):
    """At a shared column use the upper envelope, never sum or overwrite."""
    contributions = [[] for _ in range(width)]
    for col, key in anchors:
        heights, offset = templates[key]
        left = col-offset
        _require(1 <= left and left+len(heights) <= width-1, "template crosses side wall")
        for i, height in enumerate(heights):
            if height:
                contributions[left+i].append((height, key))
    heights = [max((h for h, _ in parts), default=0) for parts in contributions]
    return heights, contributions


def _read_canonical(a):
    bg = int(a[0, 0])
    _require(_border(a, bg), "a complete background margin is required")
    components = _components(a != bg)
    frames = []
    for cells in components:
        t, l, b, r = _box(cells)
        region = a[t:b, l:r]
        s = int(region[0, 0])
        if min(region.shape) >= 8 and _border(region, s):
            frames.append((t, l, b, r, s))
    _require(len(frames) == 1, "one complete chamber is required")
    t, l, b, r, s = frames[0]
    chamber = a[t:b, l:r].copy()
    height, width = chamber.shape
    base = height-3
    _require(s != bg and np.all(chamber[base+1, 1:-1] == bg), "missing rail clearance")
    _require(np.all(chamber[base, 1:-1] != bg), "broken rail")

    outside = a.copy()
    outside[t:b, l:r] = bg
    templates, repeats = {}, {}
    for cells in _components(outside != bg):
        top, left, bottom, right = _box(cells)
        motif = outside[top:bottom, left:right]
        keys = [(rr, cc) for rr, cc in cells if int(outside[rr, cc]) != s]
        _require(len(keys) == 1, "prototype needs exactly one colored key")
        ar, ac = keys[0]
        key = int(outside[ar, ac])
        _require(ar == bottom-1 and right-left >= 2, "key must lie on the prototype base")
        heights = []
        expected = np.full(motif.shape, bg, dtype=np.int8)
        expected[-1] = s
        expected[-1, ac-left] = key
        for cc in range(right-left):
            h = 0
            while h+1 < len(motif) and motif[-2-h, cc] == s:
                h += 1
            heights.append(h)
            if h:
                expected[-1-h:-1, cc] = s
        _require(max(heights) > 0 and np.array_equal(motif, expected), "not a solid-column profile")
        signature = (tuple(heights), ac-left)
        _require(key not in templates or templates[key] == signature, "conflicting prototypes")
        templates[key] = signature
        repeats[key] = repeats.get(key, 0)+1
    _require(bool(templates), "no dictionary")
    anchors = [(c, int(chamber[base, c])) for c in range(1, width-1)
               if int(chamber[base, c]) != s]
    _require(bool(anchors) and all(k in templates for _, k in anchors), "unknown rail key")
    heights, contributions = _skyline(width, templates, anchors)
    _require(max(heights) <= base-3, "leave room for at least two surviving pin cells")
    lanes = np.flatnonzero(chamber[1, 1:-1] != bg)+1
    _require(len(lanes) >= 2 and np.all(np.diff(lanes) >= 2), "pins must be separated thin lanes")
    expected = np.full(chamber.shape, bg, dtype=np.int8)
    expected[0] = expected[-1] = s
    expected[:, 0] = expected[:, -1] = s
    expected[base, 1:-1] = chamber[base, 1:-1]
    stacks = []
    for c in lanes:
        segment = chamber[1:base, c]
        occupied = np.flatnonzero(segment != bg)
        _require(np.array_equal(occupied, np.arange(len(occupied))), "pin contains a gap")
        seq = tuple(int(v) for v in segment[:len(occupied)])
        prefix = next((i for i, v in enumerate(seq) if v != s), len(seq))
        _require(prefix < len(seq) and s not in seq[prefix:], "structural cells must be a prefix")
        _require(len(seq) >= base-heights[c]-1, "pin would need an unsupported downward motion")
        expected[1:1+len(seq), c] = seq
        stacks.append(seq)
    _require(np.array_equal(chamber, expected), "unexpected chamber cells")
    return dict(background=bg, structure=s, chamber=chamber, base=base,
                templates=templates, repeats=repeats, anchors=anchors,
                heights=heights, contributions=contributions,
                lanes=tuple(int(c) for c in lanes), stacks=tuple(stacks))


def _read(input_grid):
    a = _grid(input_grid)
    readings = []
    for turn in range(4):
        try:
            f = _read_canonical(np.rot90(a, turn))
            f["turn"] = turn
            readings.append(f)
        except ValueError:
            continue
    _require(len(readings) == 1, "input must have one oriented interpretation")
    return readings[0]


def _simulate(f):
    """Pixel movement; intentionally separate from the label constructor."""
    result = f["chamber"].copy()
    bg, s, base = f["background"], f["structure"], f["base"]
    result[1:base, 1:-1] = bg
    for col, key in f["anchors"]:
        profile, offset = f["templates"][key]
        for dc, height in enumerate(profile):
            for dr in range(1, height+1):
                result[base-dr, col-offset+dc] = s
    for col, seq in zip(f["lanes"], f["stacks"]):
        tooth_top = base
        while result[tooth_top-1, col] == s:
            tooth_top -= 1
        cells = list(enumerate(seq, 1))
        while cells[-1][0] >= tooth_top:
            cells = [(row-1, color) for row, color in cells if row > 1]
        for row, color in cells:
            result[row, col] = color
    return np.rot90(result, -f["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive dictionary, union, movement, clipping and crop from this input."""
    return _simulate(_read(input_grid))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _spaced_lanes(rng, width, count, forced):
    """Sample nonadjacent columns with required witnesses, without retries."""
    forced = frozenset(forced)

    @lru_cache(None)
    def ways(pos, remaining):
        if remaining == 0:
            return int(not any(c >= pos for c in forced))
        if pos > width-2:
            return 0
        skip = ways(pos+1, remaining) if pos not in forced else 0
        take = ways(pos+2, remaining-1) if pos+1 not in forced else 0
        return skip+take

    _require(ways(1, count) > 0, "pin witnesses do not fit")
    pos, remaining, result = 1, count, []
    while remaining:
        skip = ways(pos+1, remaining) if pos not in forced else 0
        take = ways(pos+2, remaining-1) if pos+1 not in forced else 0
        if int(rng.integers(skip+take)) < take:
            result.append(pos)
            remaining -= 1
            pos += 2
        else:
            pos += 1
    return tuple(result)


def _profile(rng, width, cap, nonflat=False):
    if not nonflat and rng.random() < .18:
        heights = [int(rng.integers(1, cap+1))]*width
    else:
        low = 0 if rng.random() < .25 else 1
        heights = rng.integers(low, cap+1, width).tolist()
        heights[0] = int(rng.integers(1, cap+1))
        heights[-1] = int(rng.integers(1, cap+1))
        if nonflat and len(set(heights)) == 1:
            heights[1] = (heights[1] % cap)+1
    return tuple(heights), int(rng.integers(width))


def _reference_layout(rng, templates, repeats):
    items = [k for k in templates for _ in range(repeats[k])]
    rng.shuffle(items)
    if sum(len(templates[k][0])+1 for k in items)-1 <= 28 and rng.random() < .65:
        groups = [items]
    else:
        cut = (len(items)+1)//2
        groups = [g for g in (items[:cut], items[cut:]) if g]
    rows = []
    for group in groups:
        x, places = 0, []
        row_height = max(max(templates[k][0])+1 for k in group)
        for k in group:
            h, offset = templates[k]
            y = int(rng.integers(row_height-max(h)))
            places.append((k, y+max(h), x+offset))
            x += len(h)+1
        rows.append((row_height, x-1, places))
    return rows


def _construct_label(chamber, base, heights, lanes, stacks, bg, s):
    """Direct destination formula, using construction data, not solve()."""
    out = chamber.copy()
    out[1:base, 1:-1] = bg
    for col, h in enumerate(heights):
        if h:
            out[base-h:base, col] = s
    for col, seq in zip(lanes, stacks):
        destination = base-heights[col]-len(seq)
        for i, color in enumerate(seq):
            if destination+i > 0:
                out[destination+i, col] = color
    return out


def _make_pair(rng, config, stage, regime):
    # stage 0: separated repeated module; 1: two different overlapping modules;
    # 2: additional training; 3: held-out arrangement/count transfer.
    ntypes = 1 if stage == 0 else 2 if stage == 1 else int(rng.integers(1 if stage == 2 else 2, 4))
    palette = rng.permutation(10).tolist()
    bg, s = config["roles"] if config["fixed_roles"] else palette[:2]
    available = [c for c in palette if c not in (bg, s)]
    keys = available[:ntypes]
    cap = int(rng.integers(2, 6))
    common_width = int(rng.integers(4, 7))
    templates = {k: _profile(rng, common_width if stage == 1 else int(rng.integers(3, 7)),
                             cap, stage < 2) for k in keys}
    repeats = {k: 2 if stage < 2 else int(rng.integers(1, 3)) for k in keys}
    if stage == 0:
        groups = [[keys[0]], [keys[0]]]
    elif stage == 1:
        groups = [[keys[0], keys[1]]]
    elif stage == 3 and regime == "overlap":
        total = int(rng.integers(3, 7))
        ids = keys+[_pick(rng, keys) for _ in range(total-len(keys))]
        rng.shuffle(ids)
        groups = [ids[:3]] + ([ids[3:]] if total > 3 else [])
    else:
        total = int(rng.integers(1 if stage == 2 else 2, 5))
        ids = [_pick(rng, keys) for _ in range(total)]
        groups = [ids[i:i+2] for i in range(0, total, 2)]
    left, anchors = 1, []
    for group in groups:
        for j, k in enumerate(group):
            anchors.append((left+j+templates[k][1], k))
        extent = max(j+len(templates[k][0]) for j, k in enumerate(group))
        left += extent+int(rng.integers(1, 4))
    extent = max(c-templates[k][1]+len(templates[k][0])-1 for c, k in anchors)
    _require(len({c for c, _ in anchors}) == len(anchors), "two keys occupy the same rail cell")
    if stage == 3 and regime == "lanes":
        nlanes = int(rng.integers(6, 10))
    else:
        nlanes = int(rng.integers(3, 6 if regime == "lanes" else 7))
    width = int(rng.integers(max(13, extent+2, 2*nlanes+2), 29))
    dx = int(rng.integers(width-1-extent))
    anchors = [(c+dx, k) for c, k in anchors]
    heights, contributions = _skyline(width, templates, anchors)
    positive = [c for c, h in enumerate(heights) if h]
    forced = []
    if stage == 1:
        forced = [min(positive), max(positive)]
        middle = [c for c in positive if len(contributions[c]) == 2
                  and c >= forced[0]+2 and c <= forced[1]-2]
        _require(bool(middle), "need an observable overlap")
        forced.append(_pick(rng, middle))
    elif stage == 3 and regime == "overlap":
        triple = [c for c in positive if len(contributions[c]) >= 3]
        _require(bool(triple), "need a three-way envelope query")
        forced = [_pick(rng, triple)]
    else:
        forced = [_pick(rng, positive)]
    lanes = _spaced_lanes(rng, width, nlanes, forced)
    moving_choices = [c for c in lanes if heights[c] > 0]
    _require(bool(moving_choices), "need a moving pin")
    moving = _pick(rng, moving_choices)
    stationary = _pick(rng, [c for c in lanes if c != moving])
    rows = _reference_layout(rng, templates, repeats)
    ref_height = sum(h for h, _, _ in rows)+len(rows)-1
    base_low = max(7, max(heights)+4)
    base = int(rng.integers(base_low, min(15, 24-ref_height)+1))
    payloads = available[:int(rng.integers(max(ntypes, 2), 6))]
    stacks = []
    for c in lanes:
        capacity = base-heights[c]-1
        shift = 0 if c == stationary else int(rng.integers(heights[c]+1))
        if c == moving:
            shift = int(rng.integers(1, heights[c]+1))
        length = capacity+shift
        prefix = int(rng.integers(min(4, length-1)))
        if c == moving and stage == 0:
            prefix = shift+1
        elif c == moving and stage == 1:
            prefix = 0
        seq = [s]*prefix+[_pick(rng, payloads) for _ in range(length-prefix)]
        # Two surviving payload colors reveal exact sequence preservation.
        if c == moving and stage < 2:
            seq[-2:] = [payloads[0], payloads[1]]
        stacks.append(tuple(seq))
    chamber = np.full((base+3, width), bg, dtype=np.int8)
    chamber[0] = chamber[-1] = s
    chamber[:, 0] = chamber[:, -1] = s
    chamber[base, 1:-1] = s
    for col, key in anchors:
        chamber[base, col] = key
    for col, seq in zip(lanes, stacks):
        chamber[1:1+len(seq), col] = seq
    out = _construct_label(chamber, base, heights, lanes, stacks, bg, s)
    canvas_width = int(rng.integers(max(width, max(w for _, w, _ in rows))+2, 31))
    min_height = ref_height+len(chamber)+3
    canvas_height = int(rng.integers(min_height, 31))
    slack = canvas_height-min_height
    top = 1+int(rng.integers(slack+1))
    gap = 1+int(rng.integers(slack-(top-1)+1))
    if rng.random() < .5:
        ry, cy = top, top+ref_height+gap
    else:
        cy, ry = top, top+len(chamber)+gap
    cx = int(rng.integers(1, canvas_width-width))
    canvas = np.full((canvas_height, canvas_width), bg, dtype=np.int8)
    canvas[cy:cy+len(chamber), cx:cx+width] = chamber
    y = ry
    for row_height, row_width, places in rows:
        x = int(rng.integers(1, canvas_width-row_width))
        for key, ar, ac in places:
            profile, offset = templates[key]
            for dc, h in enumerate(profile):
                canvas[y+ar-h:y+ar+1, x+ac-offset+dc] = s
            canvas[y+ar, x+ac] = key
        y += row_height+1
    turn = int(rng.integers(4)) if config["mixed_orientation"] else config["orientation"]
    pair = {"input": np.rot90(canvas, turn).copy(), "output": np.rot90(out, turn).copy()}
    # Check the intended witnesses before accepting, entirely from input cells.
    f = _read(pair["input"])
    m = _features(f)
    if stage == 1:
        _require(m["winning_types"] >= 2 and m["overlap_pins"] > 0
                 and m["overlap_winning_types"] >= 2
                 and m["rectangle_erasers"] >= 2, "need crossing and transparent-union witnesses")
    if stage < 2:
        _require(m["off_anchor_motion"] > 0 and m["ordered_survivor"], "missing copying/order witness")
    return pair


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    regime = _pick(rng, ("overlap", "lanes"))
    config = dict(roles=rng.choice(10, 2, replace=False).tolist(), fixed_roles=bool(rng.integers(2)),
                  mixed_orientation=bool(rng.integers(2)), orientation=int(rng.integers(4)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            stage = min(index, 2) if split == "train" else 3
            for attempt in range(96):
                try:
                    pair = _make_pair(rng, config, stage, regime)
                    break
                except ValueError:
                    if attempt == 95:
                        raise RuntimeError("bounded construction failed") from None
            # At least three used colors yield >14 distinct images under the
            # 45 transpositions; this is a finite uniqueness fallback.
            if _key(pair["input"]) in seen:
                for x, y in combinations(range(10), 2):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[x], mapping[y] = y, x
                    candidate = {role: mapping[a] for role, a in pair.items()}
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
                else:
                    raise AssertionError("uniqueness bound violated")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _features(f):
    shifts, prefixes, capacities = [], [], []
    winners = set()
    for c, seq in zip(f["lanes"], f["stacks"]):
        capacity = f["base"]-f["heights"][c]-1
        shift = len(seq)-capacity
        capacities.append(capacity)
        shifts.append(shift)
        prefixes.append(next(i for i, color in enumerate(seq) if color != f["structure"]))
        top = {key for h, key in f["contributions"][c] if h == f["heights"][c]}
        if len(top) == 1:
            winners.update(top)
    anchor_cols = {c for c, _ in f["anchors"]}
    overlap_winners = set()
    for parts in f["contributions"]:
        if len(parts) >= 2:
            highest = max(h for h, _ in parts)
            top_keys = {key for h, key in parts if h == highest}
            if len(top_keys) == 1:
                overlap_winners.update(top_keys)
    rectangle_erasers = set()
    for col, key in f["anchors"]:
        profile, offset = f["templates"][key]
        for dc, height in enumerate(profile):
            if height < min(max(profile), f["heights"][col-offset+dc]):
                rectangle_erasers.add(key)
    return dict(lanes=len(f["lanes"]), types=len(f["templates"]), targets=len(f["anchors"]),
                depth=max(map(len, f["contributions"])),
                overlap_pins=sum(len(f["contributions"][c]) >= 2 for c in f["lanes"]),
                overlap_winning_types=len(overlap_winners),
                rectangle_erasers=len(rectangle_erasers),
                triple_pins=sum(len(f["contributions"][c]) >= 3 for c in f["lanes"]),
                winning_types=len(winners), shifts=tuple(shifts),
                stationary=sum(s == 0 for s in shifts), moving=sum(s > 0 for s in shifts),
                off_anchor_motion=sum(c not in anchor_cols and shift > 0 for c, shift in zip(f["lanes"], shifts)),
                retained_prefix=any(0 < shift < p for shift, p in zip(shifts, prefixes)),
                payload_clip=any(shift > p for shift, p in zip(shifts, prefixes)),
                ordered_survivor=any(shift > 0 and len(set(seq[shift:])-{f["structure"]}) >= 2
                                     for seq, shift in zip(f["stacks"], shifts)),
                min_capacity=min(capacities))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _read(inp)
            predicted = _simulate(f)
            _require(np.array_equal(predicted, out), "wrong union, motion, clipping or crop")
            canonical_out = np.rot90(out, f["turn"])
            _require(np.array_equal(canonical_out[[0, -1]], f["chamber"][[0, -1]]), "roof/floor changed")
            _require(np.array_equal(canonical_out[f["base"]:], f["chamber"][f["base"]:]), "rail changed")
            for c, seq in zip(f["lanes"], f["stacks"]):
                capacity = f["base"]-f["heights"][c]-1
                _require(tuple(canonical_out[1:capacity+1, c]) == seq[-capacity:], "pin order changed")
            records[split].append(_features(f))
    train, test = records["train"], records["test"]
    _require(any(m["depth"] == 1 for m in train), "missing separated stamp demonstration")
    _require(any(m["overlap_pins"] and m["winning_types"] >= 2
                 and m["overlap_winning_types"] >= 2 and m["rectangle_erasers"] >= 2
                 for m in train), "missing envelope demonstration")
    for flag in ("retained_prefix", "payload_clip", "ordered_survivor", "off_anchor_motion"):
        _require(any(m[flag] for m in train), f"missing {flag} demonstration")
    _require(all(m["stationary"] and m["moving"] for m in train+test), "missing contact contrast")
    overlap_transfer = max(m["depth"] for m in train) <= 2 and all(m["triple_pins"] for m in test)
    lane_transfer = max(m["lanes"] for m in train) < min(m["lanes"] for m in test)
    _require(overlap_transfer or lane_transfer, "missing documented structural transfer")


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _focused_regression():
    # Two overlapping profiles. At shared column 3, max(2,3)=3, not 5;
    # the second module reaches a pin away from either anchor.
    templates = {2: ((1, 4, 2, 1, 2), 1), 3: ((3, 0, 2, 1), 1)}
    anchors = [(2, 2), (4, 3)]
    h, _ = _skyline(10, templates, anchors)
    assert h == [0, 1, 4, 3, 1, 2, 1, 0, 0, 0]
    base = 9
    a = np.zeros((17, 14), dtype=np.int8)
    # Complete prototypes above chamber, each with one colored base key.
    for key, ar, ac in ((2, 5, 2), (3, 5, 9)):
        profile, offset = templates[key]
        for dc, height in enumerate(profile):
            a[ar-height:ar+1, ac-offset+dc] = 5
        a[ar, ac] = key
    a = np.pad(a, ((0, 4), (0, 0)))
    ch = a[7:19, 2:12]
    ch[0] = ch[-1] = 5
    ch[:, 0] = ch[:, -1] = 5
    ch[base, 1:-1] = 5
    for c, key in anchors:
        ch[base, c] = key
    sequences = {1: (5, 5, 7, 8, 7, 8, 7), 3: (7, 8, 7, 8, 7, 8, 7, 8),
                 5: (8, 7, 8, 7, 8, 7), 7: (7, 8, 7, 8, 7, 8, 7, 8)}
    for c, seq in sequences.items():
        ch[1:1+len(seq), c] = seq
    prediction = solve(a)
    assert tuple(prediction[1:6, 3]) == (8, 7, 8, 7, 8)
    assert tuple(prediction[6:9, 3]) == (5, 5, 5)
    assert tuple(prediction[1:8, 1]) == sequences[1]
    assert prediction.shape == (12, 10)
    # Omitting A: stamp the teeth while leaving the original ordered pins still.
    stamp_only = ch.copy()
    for c, height in enumerate(h):
        if height:
            stamp_only[base-height:base, c] = 5
    assert stamp_only[1, 3] != prediction[1, 3]
    # Omitting B: the chamber is identical but a changed dictionary profile
    # changes the answer at an off-anchor pin (column 3).
    changed_dictionary = a.copy()
    changed_dictionary[2, 8] = 0
    altered = solve(changed_dictionary)
    assert altered[1, 3] != prediction[1, 3]
    assert np.array_equal(changed_dictionary[7:19, 2:12], ch)
    # Max-versus-addition and a tied contact are checked by literal heights.
    assert sum(v for v, _ in _skyline(10, templates, anchors)[1][3]) == 5
    assert h[5] == 2  # both modules contribute height two here
    key_only = np.zeros(ch.shape, dtype=np.int8)
    for c, height in enumerate(h):
        if height:
            key_only[base-height:base, c] = 5
    for order in (anchors, list(reversed(anchors))):
        rectangle_copy = np.zeros(ch.shape, dtype=np.int8)
        for c, key in order:
            profile, offset = templates[key]
            rectangle_copy[base-max(profile):base, c-offset:c-offset+len(profile)] = 0
            for dc, height in enumerate(profile):
                if height:
                    rectangle_copy[base-height:base, c-offset+dc] = 5
        assert not np.array_equal(rectangle_copy, key_only)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(a, turn)), np.rot90(prediction, turn))
    mapping = np.array([9, 0, 6, 4, 1, 2, 3, 5, 7, 8], dtype=np.int8)
    assert np.array_equal(solve(mapping[a]), mapping[prediction])
    # A gap in an otherwise valid ordered pin is outside the family.
    broken = a.copy()
    broken[10, 5] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("pin gap accepted")
    contradictory = np.pad(a, ((0, 6), (0, 0)))
    for dc, height in enumerate((1, 3, 2, 1, 2)):
        contradictory[24-height:25, 1+dc] = 5
    contradictory[24, 2] = 2
    try:
        solve(contradictory)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting repeated prototypes accepted")


def self_test():
    """42 seeds, one puzzle resident at a time; no disk writes or trace logs."""
    _focused_regression()
    digests, shapes, profiles, anchors, rotations, lanes, depths, roles = (set() for _ in range(8))
    totals = {"pairs": 0, "overlap_pins": 0, "triple_pins": 0, "retained_prefix": 0,
              "payload_clip": 0, "off_anchor_motion": 0}
    regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        digests.add(digest)
        readings = {split: [_read(p["input"]) for p in puzzle[split]] for split in puzzle}
        for split in puzzle:
            for pair, f in zip(puzzle[split], readings[split]):
                assert np.array_equal(solve(pair["input"]), pair["output"])
                m = _features(f)
                shapes.add(pair["input"].shape)
                profiles.update(h for h, _ in f["templates"].values())
                anchors.update(offset for _, offset in f["templates"].values())
                rotations.add(f["turn"])
                lanes.add(m["lanes"])
                depths.add(m["depth"])
                roles.add((f["background"], f["structure"]))
                totals["pairs"] += 1
                for flag in totals.keys()-{"pairs"}:
                    totals[flag] += int(m[flag])
        if min(len(f["lanes"]) for f in readings["test"]) > max(len(f["lanes"]) for f in readings["train"]):
            regimes.add("lanes")
        if all(_features(f)["triple_pins"] for f in readings["test"]):
            regimes.add("overlap")
    assert len(digests) == 42 and len(shapes) >= 30 and len(profiles) >= 100
    assert rotations == set(range(4)) and len(roles) >= 25 and anchors.issuperset({0, 1, 2, 3})
    assert lanes.issuperset(range(3, 10)) and depths.issuperset({1, 2, 3}) and regimes == {"lanes", "overlap"}
    assert all(totals[k] >= (10 if k == "triple_pins" else 20) for k in totals if k != "pairs")
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(731+ntrain*10+ntest, ntrain, ntest)
            validate_generated_puzzle(p)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    malformed = generate_puzzle(0)
    malformed["test"][0]["output"][0, 0] = (int(malformed["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, distinct_puzzles=len(digests), input_shapes=len(shapes),
                profiles=len(profiles), anchor_offsets=sorted(anchors), orientations=sorted(rotations),
                lane_counts=sorted(lanes), overlap_depths=sorted(depths), role_pairs=len(roles),
                regimes=sorted(regimes), **totals)
