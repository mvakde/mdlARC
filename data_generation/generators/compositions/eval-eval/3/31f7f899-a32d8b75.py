"""Sorting repair loom: rank-transfer spans, then weave the signed changes.

The frame contains a complete binary tile in scene-background/carrier colors.
Loss blocks use its pattern; gain blocks use its complement, recolored to the
fixed destination slot. Persistent and persistently empty macro cells cancel.
No files, parent generators, labels, seeds, or training examples are read by solve.
"""
from collections import Counter
from itertools import permutations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _tile_panel(panel, divider):
    """A single rectangular frame, with explicitly complete binary contents."""
    coords = np.argwhere(panel == divider)
    assert len(coords)
    lo, hi = coords.min(0), coords.max(0) + 1
    r, c = map(int, lo)
    h, w = map(int, hi - lo)
    assert h == w and 4 <= h <= 6
    outside = np.ones(panel.shape, bool)
    outside[r:r + h, c:c + w] = False
    ground_values = np.unique(panel[outside])
    assert len(ground_values) == 1
    bg = int(ground_values[0])
    assert bg != divider
    box = panel[r:r + h, c:c + w]
    assert all(np.all(edge == divider) for edge in (box[0], box[-1], box[:, 0], box[:, -1]))
    tile = box[1:-1, 1:-1].copy()
    values = np.unique(tile)
    assert len(values) == 2 and bg in values and divider not in values
    carrier = int(values[values != bg][0])
    expected = np.full(panel.shape, bg, int)
    expected[r:r + h, c:c + w] = divider
    expected[r + 1:r + h - 1, c + 1:c + w - 1] = tile
    assert np.array_equal(expected, panel)
    return bg, carrier, tile, (r, c)


def _scene_info(scene, bg, carrier):
    rows = np.flatnonzero(np.all(scene != bg, axis=1))
    cols = np.flatnonzero(np.all(scene != bg, axis=0))
    assert len(rows) + len(cols) == 1
    vertical = bool(len(cols))
    a = scene.T if vertical else scene
    row = int(cols[0] if vertical else rows[0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert a[row, 0] == carrier and a[row, -1] == carrier
    line = a[row]
    expected = np.full(a.shape, bg, int)
    expected[row] = carrier
    records = []
    x = 0
    while x < len(line):
        color = int(line[x])
        end = x + 1
        while end < len(line) and line[end] == color:
            end += 1
        if color != carrier:
            assert color != bg
            ys = np.flatnonzero(np.any(a[:, x:end] != bg, axis=1))
            top, bottom = int(ys[0]), int(ys[-1])
            assert 0 < top <= row <= bottom < len(a) - 1
            assert np.all(a[top:bottom + 1, x:end] == color)
            expected[top:bottom + 1, x:end] = color
            records.append((x, end - x, color, top - row, bottom - row))
        x = end
    assert 2 <= len(records) <= 5 and len({r[2] for r in records}) >= 2
    assert np.array_equal(expected, a), "scene has noise or nonsolid bars"
    by_length = {}
    for _, _, _, low, high in records:
        assert by_length.setdefault(high - low + 1, (low, high)) == (low, high), "ambiguous tie"
    return dict(scene=scene.copy(), shape=a.shape, vertical=vertical, row=row,
                records=records, bg=bg, carrier=carrier)


def _decode_input(input_grid):
    g = _grid(input_grid)
    candidates = []
    # A visible uniform divider separates the tile frame from the bar scene.
    # Blank full lines are tried too, but fail exact frame/scene reconstruction.
    for axis in (0, 1):
        a = g if axis == 1 else g.T
        for cut in range(1, a.shape[1] - 1):
            divider = int(a[0, cut])
            if not np.all(a[:, cut] == divider):
                continue
            for before in (True, False):
                panel = a[:, :cut] if before else a[:, cut + 1:]
                scene = a[:, cut + 1:] if before else a[:, :cut]
                # Restore physical coordinates; the tile never follows the bar axis.
                panel = panel if axis == 1 else panel.T
                scene = scene if axis == 1 else scene.T
                try:
                    bg, carrier, tile, origin = _tile_panel(panel, divider)
                    d = _scene_info(scene, bg, carrier)
                    assert divider not in scene
                    n = len(tile)
                    assert all(n * v <= 30 for v in scene.shape)
                    shared = scene.shape[0] if axis == 1 else scene.shape[1]
                    assert shared >= n + 3
                except (AssertionError, IndexError, ValueError):
                    continue
                side = ("left" if before else "right") if axis == 1 else ("top" if before else "bottom")
                d.update(tile=tile, bits=tile == carrier, n=n, divider=divider,
                         side=side, tile_origin=origin)
                candidates.append(d)
    assert len(candidates) == 1, "input must have one readable tile panel and carrier scene"
    return candidates[0]


def _sorted_spans(records):
    return [r[3:] for r in sorted(records, key=lambda r: r[4] - r[3])]


def _render_edits(d, targets, behavior="normal"):
    """Forward block substitution; also supports explicit counterfactual checks."""
    n, bg, row = d["n"], d["bg"], d["row"]
    out = np.full(tuple(v * n for v in d["scene"].shape), bg, dtype=np.int8)
    bits = d["bits"]
    if behavior == "rotate_tile":
        bits = np.rot90(bits)
    for record, (low, high) in zip(d["records"], targets):
        start, width, color, old_low, old_high = record
        for y in range(d["shape"][0]):
            offset = y - row
            old, new = old_low <= offset <= old_high, low <= offset <= high
            if behavior == "sorted_silhouette":
                if not new:
                    continue
                pattern = ~bits
            else:
                if old == new:
                    continue
                pattern = bits if old else ~bits
                if behavior == "no_exchange":
                    pattern = bits
                elif behavior == "exchange_everywhere":
                    pattern = ~bits
                elif behavior == "filled_changes":
                    pattern = np.ones(bits.shape, bool)
                elif behavior == "loss_only" and new:
                    continue
                elif behavior == "gain_only" and old:
                    continue
            tile = np.where(pattern, color, bg)
            for x in range(start, start + width):
                r, c = (x, y) if d["vertical"] else (y, x)
                out[r * n:(r + 1) * n, c * n:(c + 1) * n] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode_input(input_grid)
    return _render_edits(d, _sorted_spans(d["records"]))


def _check_output(d, output):
    """Invert edit blocks, then check conservation/order without sorting/painting.

    A blank block preserves its input occupancy. Loss/gain blocks reconstruct
    the new occupancy, which must form fixed-slot rectangles with increasing
    lengths and exactly the original multiset of relative intervals.
    """
    out = _grid(output)
    n, bg, row = d["n"], d["bg"], d["row"]
    assert out.shape == tuple(v * n for v in d["scene"].shape)
    h, w = d["shape"]
    owner = np.full(w, -1, int)
    for i, (start, width, _, _, _) in enumerate(d["records"]):
        owner[start:start + width] = i
    recovered = np.zeros((h, w), bool)
    losses = gains = 0
    for y in range(h):
        for x in range(w):
            r, c = (x, y) if d["vertical"] else (y, x)
            block = out[r * n:(r + 1) * n, c * n:(c + 1) * n]
            if owner[x] < 0:
                assert np.all(block == bg), "paint outside destination slots"
                continue
            _, _, color, low, high = d["records"][owner[x]]
            old = low <= y - row <= high
            if np.all(block == bg):
                recovered[y, x] = old
            elif np.array_equal(block, np.where(d["bits"], color, bg)):
                assert old, "loss block outside original bar"
                losses += 1
            else:
                assert np.array_equal(block, np.where(~d["bits"], color, bg)), "wrong tile or slot color"
                assert not old, "gain block inside original bar"
                recovered[y, x] = True
                gains += 1
    intervals, lengths = [], []
    for start, width, _, _, _ in d["records"]:
        ys = np.flatnonzero(recovered[:, start])
        assert len(ys) and row in ys and np.all(np.diff(ys) == 1)
        assert np.all(recovered[:, start:start + width] == recovered[:, start:start + 1])
        low, high = int(ys[0]) - row, int(ys[-1]) - row
        intervals.append((low, high))
        lengths.append(high - low + 1)
    assert all(a <= b for a, b in zip(lengths, lengths[1:])), "reconstructed lengths are not increasing"
    assert Counter(intervals) == Counter(r[3:] for r in d["records"]), "relative spans were not conserved"
    assert (losses == 0) == (gains == 0)
    return dict(losses=losses, gains=gains, targets=intervals)


_ALTERNATIVES = ("no_sort", "reverse", "descending", "area", "recenter",
                 "no_exchange", "exchange_everywhere", "rotate_tile",
                 "filled_changes", "sorted_silhouette", "loss_only", "gain_only")


def _alternative(d, name):
    rs = d["records"]
    spans = _sorted_spans(rs)
    if name == "no_sort":
        spans = [r[3:] for r in rs]
    elif name == "reverse":
        spans = [r[3:] for r in rs[::-1]]
    elif name == "descending":
        spans = spans[::-1]
    elif name == "area":
        spans = [r[3:] for r in sorted(rs, key=lambda r: r[1] * (r[4] - r[3] + 1))]
    elif name == "recenter":
        spans = [(-(length // 2), (length - 1) // 2) for length in (hi - lo + 1 for lo, hi in spans)]
    return _render_edits(d, spans, name)


def _bits(rng, n, witness):
    for _ in range(32):
        b = rng.random((n, n)) < rng.uniform(.18, .82)
        if b.any() and not b.all() and (not witness or
                (b.sum() * 2 != b.size and all(not np.array_equal(b, np.rot90(b, k)) for k in (1, 2, 3)))):
            return b
    b = np.zeros((n, n), bool)
    b[0, 0] = True
    return b


def _make_pair(rng, n, count, vertical, side, roles, witness=0):
    bg, carrier, divider = roles
    cap = 30 // n
    pool = [c for c in range(10) if c not in roles]
    if witness:
        lengths = [2, 3, 4] if count == 3 else [1, 2, 3, 4]
        sign = 1 if witness > 0 else -1
        templates = {1: (0, 0), 2: ((-1, 0) if sign < 0 else (0, 1)), 3: (-1, 1)}
        if count == 3:
            templates[4] = (-3, 0) if sign < 0 else (0, 3)
        else:
            templates[3] = (-2, 0) if sign < 0 else (0, 2)
            templates[4] = (-2, 1) if sign < 0 else (-1, 2)
        for _ in range(24):
            rng.shuffle(lengths)
            if all(a != b for a, b in zip(lengths, sorted(lengths))) and lengths[::-1] != sorted(lengths):
                break
        else:
            lengths = sorted(lengths)[1:] + sorted(lengths)[:1]
        widths = [2 if length == 2 else 1 for length in lengths]
        colors = [int(v) for v in rng.choice(pool, count, replace=False)]
    else:
        extent = int(rng.integers(3, cap - 1))
        pivot = int(rng.integers(extent))
        distinct = min(count, extent)
        if count > 2 and rng.random() < .4:
            distinct = int(rng.integers(2, distinct + 1))
        selected = [int(v) for v in rng.choice(np.arange(1, extent + 1), distinct, replace=False)]
        lengths = selected + [int(v) for v in rng.choice(selected, count - distinct)]
        templates = {}
        for length in selected:
            low = int(rng.integers(max(-pivot, 1 - length), min(0, extent - pivot - length) + 1))
            templates[length] = (low, low + length - 1)
        lengths.sort()
        order = int(rng.choice(4, p=[.6, .2, .1, .1]))
        if order == 0:
            rng.shuffle(lengths)
        elif order == 1:
            lengths.reverse()
        elif order == 2:
            i = int(rng.integers(count - 1))
            lengths[i], lengths[i + 1] = lengths[i + 1], lengths[i]
        widths = [1] * count
        colors = [int(v) for v in rng.choice(pool, count, replace=False)]
        if rng.random() < .4 and count >= 3 and cap >= count + 3:
            i = int(rng.integers(1, count))
            colors[i] = colors[i - 1]
    spans = [templates[length] for length in lengths]
    gaps = [int(a == b) for a, b in zip(colors, colors[1:])]
    spare = cap - sum(widths) - sum(gaps) - 2
    assert spare >= 0
    # Widths and carrier gaps are individual, independent geometric factors.
    for _ in range(int(rng.integers(spare + 1))):
        if rng.random() < .55:
            eligible = [i for i in range(count) if not witness or lengths[i] not in (2, 3)]
            widths[int(rng.choice(eligible))] += 1
        else:
            gaps[int(rng.integers(count - 1))] += 1
    min_y, max_y = min(lo for lo, _ in spans), max(hi for _, hi in spans)
    # Only the physical dimension alongside the panel must fit the framed tile.
    # The other scene dimension is limited solely by the bars and their margins.
    shared_is_h = (side in ("left", "right")) != vertical
    h_min = max(max_y - min_y + 3, n + 3 if shared_is_h else 3)
    w_min = max(sum(widths) + sum(gaps) + 2, n + 3 if not shared_is_h else 4)
    h = int(rng.integers(h_min, cap + 1))
    w = int(rng.integers(w_min, cap + 1))
    row = int(rng.integers(1 - min_y, h - 1 - max_y))
    # Randomly allocate unused carrier space, including both outside margins.
    margin = rng.multinomial(w - sum(widths) - sum(gaps) - 2, [1 / (count + 1)] * (count + 1))
    starts, x = [], 1 + int(margin[0])
    for i, width in enumerate(widths):
        starts.append(x)
        if i < count - 1:
            x += width + gaps[i] + int(margin[i + 1])
    scene = np.full((h, w), bg, dtype=np.int8)
    scene[row] = carrier
    records = []
    for x, width, color, (low, high) in zip(starts, widths, colors, spans):
        scene[row + low:row + high + 1, x:x + width] = color
        records.append((x, width, color, low, high))
    physical = scene.T.copy() if vertical else scene
    bits = _bits(rng, n, bool(witness))
    tile = np.where(bits, carrier, bg)
    along = physical.shape[0] if side in ("left", "right") else physical.shape[1]
    depth = n + 2 + int(rng.integers(3))
    pshape = (along, depth) if side in ("left", "right") else (depth, along)
    panel = np.full(pshape, bg, dtype=np.int8)
    tr, tc = [int(rng.integers(v - n - 1)) for v in pshape]
    panel[tr:tr + n + 2, tc:tc + n + 2] = divider
    panel[tr + 1:tr + n + 1, tc + 1:tc + n + 1] = tile
    if side in ("left", "right"):
        sep = np.full((along, 1), divider, dtype=np.int8)
        parts = (panel, sep, physical) if side == "left" else (physical, sep, panel)
        inp = np.concatenate(parts, axis=1)
    else:
        sep = np.full((1, along), divider, dtype=np.int8)
        parts = (panel, sep, physical) if side == "top" else (physical, sep, panel)
        inp = np.concatenate(parts, axis=0)
    d = dict(scene=physical, shape=scene.shape, vertical=vertical, row=row,
             records=records, bg=bg, carrier=carrier, n=n, bits=bits)
    # Generation orders numeric lengths; solve instead recovers every quantity
    # from visible input. The relational validator does neither operation.
    targets = [spans[int(i)] for i in np.argsort(lengths, kind="stable")]
    return {"input": inp, "output": _render_edits(d, targets)}


def _distinct_coloring(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    d = _decode_input(pair["input"])
    old = [d["bg"], d["carrier"], d["divider"]]
    old += [v for v in range(10) if v not in old]
    # 720 distinct role assignments exceed the maximum 13 previous exclusions.
    for roles in permutations(range(10), 3):
        new = list(roles) + [v for v in range(10) if v not in roles]
        mapping = np.zeros(10, dtype=np.int8)
        mapping[old] = new
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite uniqueness fallback exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    local_roles = bool(rng.random() < .5)
    shift = str(rng.choice(["count", "scale"]))
    first_n = int(rng.choice([2, 3]))
    first_axis = bool(rng.integers(2))
    sign = int(rng.choice([-1, 1]))
    sides = ["left", "right", "top", "bottom"]
    preferred_side = str(rng.choice(sides))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = sign if split == "train" and i == 0 else -sign if split == "train" and i == 1 else 0
            if witness:
                n, k = (first_n, 3) if i == 0 else (5 - first_n, 4)
                vertical = first_axis if i == 0 else not first_axis
            else:
                n = 4 if split == "test" and shift == "scale" else int(rng.choice([2, 3]))
                k = 5 if split == "test" and shift == "count" else int(rng.integers(2, 5))
                vertical = bool(rng.integers(2))
            roles = tuple(int(v) for v in rng.choice(10, 3, replace=False)) if local_roles else base_roles
            side = preferred_side if rng.random() < .4 else str(rng.choice(sides))
            pair = _make_pair(rng, n, k, vertical, side, roles, witness)
            pair = _distinct_coloring(pair, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            d = _decode_input(inp)
            stats = _check_output(d, out)
            if split == "train":
                train.append((d, out, stats))
    # Evidence requirements describe this generator's curriculum, not additional
    # hidden conventions used by solve. Both orientations and phases are shown.
    assert {d["vertical"] for d, _, s in train if s["losses"] and s["gains"]} == {False, True}
    assert {d["n"] for d, _, _ in train} >= {2, 3}
    for name in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(d, name), out) for d, out, _ in train), f"training leaves {name} unresolved"
    # Every signed macro edit can be read back; aggregate loss/gain *widths*
    # need not match, because destination widths are preserved independently.
    return None


def self_test():
    """Bounded streaming sweep: 42 seeds, no saved grids or execution traces."""
    signatures, shapes, layouts, axes, sizes, counts = set(), set(), set(), set(), set(), set()
    role_values = [set(), set(), set()]
    flags = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple((_key(pair["input"]), _key(pair["output"])) for s in p for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve mutated its input"
                d = _decode_input(original)
                records = d["records"]
                shapes.add((original.shape, pair["output"].shape))
                layouts.add(d["side"])
                axes.add(d["vertical"])
                sizes.add(d["n"])
                counts.add(len(records))
                for bag, value in zip(role_values, (d["bg"], d["carrier"], d["divider"])):
                    bag.add(value)
                lengths = [r[4] - r[3] + 1 for r in records]
                flags["mixed_widths"] += len({r[1] for r in records}) > 1
                flags["ties"] += len(set(lengths)) < len(lengths)
                flags["unit"] += 1 in lengths
                flags["sorted"] += lengths == sorted(lengths)
                flags["repeated_color"] += len({r[2] for r in records}) < len(records)
                flags["touching"] += any(a[0] + a[1] == b[0] for a, b in zip(records, records[1:]))
                flags["one_sided"] += any((lo == 0 or hi == 0) and hi - lo >= 2 for _, _, _, lo, hi in records)
                flags["even"] += any(v % 2 == 0 for v in lengths)
                flags["blank_output"] += np.all(pair["output"] == d["bg"])
        if seed == 0:
            q = generate_puzzle(0)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    assert len(signatures) == 42 and len(shapes) >= 60
    assert layouts == {"left", "right", "top", "bottom"} and axes == {False, True}
    assert sizes == {2, 3, 4} and counts == {2, 3, 4, 5}
    assert all(values == set(range(10)) for values in role_values)
    assert all(flags[name] for name in ("mixed_widths", "ties", "unit", "sorted", "repeated_color", "touching", "one_sided", "even", "blank_output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # All compatible size/count/axis/panel-side combinations, including the
    # size-four/five-slot combination not used by either test curriculum.
    rng = np.random.default_rng(20261022)
    for n in (2, 3, 4):
        for count in range(2, 6):
            for vertical in (False, True):
                for side in ("left", "right", "top", "bottom"):
                    pair = _make_pair(rng, n, count, vertical, side, (0, 1, 9))
                    d = _decode_input(pair["input"])
                    _check_output(d, pair["output"])
                    assert np.array_equal(solve(pair["input"]), pair["output"])
    # Output corruption and meaningful competing-rule mutations must fail the
    # relational checker. These are authored checks, not independent solving.
    p = generate_puzzle(0, 2, 1)
    decoded = [_decode_input(pair["input"]) for pair in p["train"]]
    for name in _ALTERNATIVES:
        for d, pair in zip(decoded, p["train"]):
            mutation = _alternative(d, name)
            if np.array_equal(mutation, pair["output"]):
                continue
            try:
                _check_output(d, mutation)
            except AssertionError:
                break
            raise AssertionError(f"accepted {name} mutation")
        else:
            raise AssertionError(f"no informative {name} mutation")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "tile_sizes": sorted(sizes), "bar_counts": sorted(counts),
            "features": {k: int(v) for k, v in flags.items()}}
