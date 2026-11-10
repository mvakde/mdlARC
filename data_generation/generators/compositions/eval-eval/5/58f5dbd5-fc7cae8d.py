"""Color-matched, independently registered holes in an ordered opaque stack.

Only standard library and NumPy are used. No artifacts are written by this file.
The decoder consumes one input grid; construction parameters never enter solve.
"""
from functools import lru_cache
import hashlib
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}


def _transform(a, op):
    """0..3: CCW turns; 4..7: left/right reflection followed by turns."""
    return np.rot90(np.fliplr(a) if op >= 4 else a, op % 4).copy()


def _side_maps():
    cross = np.zeros((3, 3), dtype=int)
    for i, side in enumerate(SIDES, 1):
        dr, dc = NORMALS[side]
        cross[1 + dr, 1 + dc] = i
    result = []
    for op in range(8):
        moved = _transform(cross, op)
        result.append({side: next(t for t in SIDES
                      if moved[1 + NORMALS[t][0], 1 + NORMALS[t][1]] == i)
                       for i, side in enumerate(SIDES, 1)})
    return result


MAPS = _side_maps()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _bar(a, bounds, side):
    r, c, size = bounds
    if side == "T":
        return a[r - 1, c:c + size]
    if side == "R":
        return a[r:r + size, c + size]
    if side == "B":
        return a[r + size, c:c + size]
    return a[r:r + size, c - 1]


def _components(a, color):
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(a == color):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not seen[yy, xx] and a[yy, xx] == color):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _straight(cells):
    rr, cc = zip(*cells)
    h, w = max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
    if len(cells) >= 2 and (h == 1 or w == 1) and h * w == len(cells):
        return ("H" if h == 1 else "V", min(rr), min(cc), len(cells))
    return None


def _operation(source, destination):
    choices = [op for op, mapping in enumerate(MAPS)
               if all(destination.get(mapping[s]) == color for s, color in source.items())]
    assert len(choices) == 1, "two adjacent identity matches must fix one symmetry"
    return choices[0]


def _decode(input_grid):
    """Recover cards, roles, stack order and independent frames from the input."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    colors, counts = np.unique(a, return_counts=True)
    singleton = [int(c) for c, n in zip(colors, counts) if c != bg and n == 1]
    assert len(singleton) == 1, "one unique rivet color is required"
    pin = singleton[0]
    candidates = {}
    for color in colors:
        color = int(color)
        if color in (bg, pin):
            continue
        descriptions = [_straight(c) for c in _components(a, color)]
        if len(descriptions) < 4 or any(d is None for d in descriptions):
            continue
        lengths = sorted({d[3] for d in descriptions})
        if len(lengths) == 2 and lengths[1] == lengths[0] + 2:
            candidates[color] = descriptions
    # A disconnected distractor can itself contain several straight strokes.
    # It cannot contain a stroke as long as a plate-side control (k+2).
    assert candidates
    longest = max(d[3] for descriptions in candidates.values() for d in descriptions)
    candidates = {color: descriptions for color, descriptions in candidates.items()
                  if max(d[3] for d in descriptions) == longest}
    assert len(candidates) == 2, "exactly two repeated bar identities are required"
    markers = tuple(sorted(candidates))
    lines = [(color, *d) for color in markers for d in candidates[color]]
    cards, used = [], set()
    for hi, (hc, direction, hr, hx, length) in enumerate(lines):
        if direction != "H":
            continue
        for vi, (vc, vertical, vy, vx, vlength) in enumerate(lines):
            if vertical != "V" or vc == hc or vlength != length:
                continue
            hs = "T" if hr == vy - 1 else "B" if hr == vy + length else None
            vs = "L" if vx == hx - 1 else "R" if vx == hx + length else None
            if hs is None or vs is None:
                continue
            assert hi not in used and vi not in used, "ambiguous guide pairing"
            used.update((hi, vi))
            cards.append({"bounds": (vy, hx, length), "guides": {hs: hc, vs: vc}})
    assert len(used) == len(lines) and len(cards) >= 4
    sizes = sorted({c["bounds"][2] for c in cards})
    assert len(sizes) == 2 and sizes[1] == sizes[0] + 2
    k, s = sizes
    assert 2 <= k <= 6
    claimed = np.zeros(a.shape, dtype=bool)
    plates, library = [], {}
    for card in cards:
        r, c, length = card["bounds"]
        assert r >= 2 and c >= 2 and r + length <= a.shape[0] - 2 and c + length <= a.shape[1] - 2
        box = (slice(r - 1, r + length + 1), slice(c - 1, c + length + 1))
        # A blank cell separates the reserved rectangles of different cards.
        assert not claimed[r - 2:r + length + 2, c - 2:c + length + 2].any()
        claimed[box] = True
        window = a[r:r + length, c:c + length]
        material = set(int(v) for v in np.unique(window)) - {bg, pin, *markers}
        assert len(material) == 1
        color = material.pop()
        card["color"] = color
        expected_box = np.full((length + 2, length + 2), bg, dtype=int)
        expected_box[1:-1, 1:-1] = window
        for side, identity in card["guides"].items():
            _bar(expected_box, (1, 1, length), side)[:] = identity
        assert np.array_equal(a[box], expected_box), "unregistered marks around a card"
        if length == s:
            assert np.all((window == color) | (window == pin))
            card["tagged"] = bool(np.any(window == pin))
            if card["tagged"]:
                rr, cc = np.argwhere(window == pin)[0]
                assert rr in (0, s - 1) or cc in (0, s - 1), "rivet belongs on the perimeter"
            plates.append(card)
        else:
            assert color not in library
            assert np.all((window == bg) | (window == color))
            mask = window == color
            assert mask.sum() >= 2, "stencil identity must not imitate the singleton rivet"
            card["mask"] = mask.copy()
            library[color] = card
    assert np.all(a[~claimed] == bg), "unexplained input pixels"
    assert 2 <= len(plates) <= 4 and len({p["color"] for p in plates}) == len(plates)
    assert set(p["color"] for p in plates) <= set(library)
    assert sum(p["tagged"] for p in plates) == 1
    if len({p["bounds"][0] for p in plates}) == 1:
        axis = 1
    else:
        assert len({p["bounds"][1] for p in plates}) == 1, "plates must lie on a straight line"
        axis = 0
    plates.sort(key=lambda p: p["bounds"][axis])
    assert plates[0]["tagged"] or plates[-1]["tagged"], "front must be an endpoint"
    if plates[-1]["tagged"]:
        plates.reverse()
    for plate in plates:
        source = library[plate["color"]]
        plate["source"] = source
        plate["op"] = _operation(source["guides"], plate["guides"])
        plate["holes"] = _transform(source["mask"], plate["op"])
    return {"background": bg, "pin": pin, "markers": markers, "k": k,
            "plates": plates, "library": library, "axis": axis}


def _view(colors, holes, bg):
    """Front-to-back ray tracing. Perimeters never have holes."""
    k = holes[0].shape[0]
    out = np.full((k + 4, k + 4), bg, dtype=np.int8)
    out[1:-1, 1:-1] = colors[0]
    inner = out[2:-2, 2:-2]
    open_ray = np.ones((k, k), dtype=bool)
    inner[:] = bg
    for color, mask in zip(colors, holes):
        inner[open_ray & ~mask] = color
        open_ray &= mask
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    plates = info["plates"]
    return _view([p["color"] for p in plates], [p["holes"] for p in plates], info["background"])


def _coordinate_holes(plate):
    """Invariant check: use signed axes, without using the array transform."""
    source = plate["source"]
    matrix = np.zeros((2, 2), dtype=int)
    for side, identity in source["guides"].items():
        target = next(s for s, color in plate["guides"].items() if color == identity)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    mask = source["mask"]
    coords = matrix @ np.indices(mask.shape).reshape(2, -1)
    coords -= coords.min(axis=1, keepdims=True)
    moved = np.zeros(mask.shape, dtype=bool)
    moved[coords[0], coords[1]] = mask.ravel()
    return moved


def _shortcut_views(info):
    plates, bg = info["plates"], info["background"]
    colors = [p["color"] for p in plates]
    holes = [p["holes"] for p in plates]
    nearest = []
    for plate in plates:
        pr, pc, ps = plate["bounds"]
        source = min(info["library"].values(), key=lambda card:
                     ((2 * card["bounds"][0] + info["k"] - 2 * pr - ps) ** 2
                      + (2 * card["bounds"][1] + info["k"] - 2 * pc - ps) ** 2,
                      card["bounds"]))
        nearest.append(_transform(source["mask"], _operation(source["guides"], plate["guides"])))
    trimmed = []
    for plate in plates:
        mask = plate["source"]["mask"]
        lo = np.argwhere(mask).min(axis=0)
        hi = np.argwhere(mask).max(axis=0) + 1
        tight = np.zeros_like(mask)
        tight[:hi[0] - lo[0], :hi[1] - lo[1]] = mask[lo[0]:hi[0], lo[1]:hi[1]]
        trimmed.append(_transform(tight, plate["op"]))
    return {
        "ignore_orientation": _view(colors, [p["source"]["mask"] for p in plates], bg),
        "reverse_depth": _view(colors[::-1], holes[::-1], bg),
        "front_only": _view(colors[:1], holes[:1], bg),
        "ink_is_material": _view(colors, [~h for h in holes], bg),
        "ignore_subtraction": _view(colors, [np.zeros_like(h) for h in holes], bg),
        "front_frame_for_all": _view(colors, [_transform(p["source"]["mask"], plates[0]["op"]) for p in plates], bg),
        "nearest_stencil": _view(colors, nearest, bg),
        "trim_stencil": _view(colors, trimmed, bg),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching, defeated = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            before = a.copy()
            info = _decode(a)
            bg, plates, k = info["background"], info["plates"], info["k"]
            assert out.shape == (k + 4, k + 4)
            assert all(np.all(edge == bg) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            rim = out[1:-1, 1:-1].copy()
            assert all(np.all(edge == plates[0]["color"]) for edge in (rim[0], rim[-1], rim[:, 0], rim[:, -1]))
            # Coordinate correspondence, first-opaque ownership, and hole intersection.
            holes = [_coordinate_holes(p) for p in plates]
            assert all(np.array_equal(h, p["holes"]) for h, p in zip(holes, plates))
            for r in range(k):
                for c in range(k):
                    first = next((p["color"] for p, h in zip(plates, holes) if not h[r, c]), bg)
                    assert out[r + 2, c + 2] == first, "wrong visible surface"
            inner = out[2:-2, 2:-2]
            assert np.array_equal(inner == bg, np.logical_and.reduce(holes))
            assert np.array_equal(solve(a), out) and np.array_equal(a, before)
            assert info["pin"] not in out and not (set(info["markers"]) & set(np.unique(out)))
            # Every example visibly uses transparency, more than the front plate,
            # and an orientation that changes the composite answer.
            assert np.any(inner == bg)
            assert any(np.any(inner == p["color"]) for p in plates[1:])
            shortcuts = _shortcut_views(info)
            assert not np.array_equal(shortcuts["ignore_orientation"], out)
            if split == "train":
                teaching.append(info)
                defeated.update(name for name, candidate in shortcuts.items() if not np.array_equal(candidate, out))
    assert defeated == {"ignore_orientation", "reverse_depth", "front_only", "ink_is_material", "ignore_subtraction", "front_frame_for_all", "nearest_stencil", "trim_stencil"}
    # The minimum two demonstrations visibly distinguish matching identities
    # from the occupied-side pair and forward turns from inverse turns.
    assert any(p["op"] in (1, 3) and len({_key(_transform(p["source"]["mask"], op)) for op in range(8)}) == 8
               for t in teaching for p in t["plates"])
    assert any(p["op"] >= 4 for t in teaching for p in t["plates"])
    assert any(set(a["plates"][0]["source"]["guides"]) == set(b["plates"][0]["source"]["guides"])
               and set(a["plates"][0]["guides"]) == set(b["plates"][0]["guides"])
               and a["plates"][0]["op"] != b["plates"][0]["op"]
               for i, a in enumerate(teaching) for b in teaching[i + 1:])


def _mask(rng, k, asymmetric=False):
    for _ in range(256):
        style = int(rng.integers(10))
        m = np.zeros((k, k), dtype=bool)
        if style == 0:
            np.fill_diagonal(m, True)
            if rng.random() < .5:
                m |= np.fliplr(m)
        elif style == 1:
            m[0] = True
            m[:, int(rng.integers(k))] = True
        elif style == 2:
            m[0] = m[-1] = True
        elif style == 3:
            m[:] = True
            m.flat[int(rng.integers(k * k))] = False
        elif style == 7:
            # Explicit guides register a smaller, off-center connected patch.
            h, w = int(rng.integers(1, k + 1)), int(rng.integers(1, k + 1))
            r, c = int(rng.integers(k - h + 1)), int(rng.integers(k - w + 1))
            m[r:r + h, c:c + w] = True
        elif style == 8:
            # Sparse disconnected cuts need not reach any register edge.
            m = rng.random((k, k)) < float(rng.uniform(.15, .4))
        elif style == 9:
            m[:] = True
        else:
            m = rng.random((k, k)) < float(rng.uniform(.35, .85))
            if style == 4:
                m |= np.fliplr(m)
        m = _transform(m, int(rng.integers(8)))
        if m.sum() < 2:
            continue
        if asymmetric and (len({_key(_transform(m, op)) for op in range(8)}) != 8 or 2 * m.sum() == m.size):
            continue
        return m
    # A finite full-span asymmetric fallback for k>=3.
    m = np.zeros((k, k), dtype=bool)
    m[0, 0] = m[0, -1] = m[-1, 0] = True
    if k >= 3:
        m[1, -1] = True
    return m


@lru_cache(maxsize=96)
def _layouts(k, n, m):
    small, large = k + 2, k + 4
    options = []
    for gap in (1, 2, 3):
        line_w = n * large + (n - 1) * gap
        for cols in range(1, m + 2):
            for vacancy in (0, 1):
                rows = (m + cols - 1) // cols + vacancy
                if rows * cols > m + 4:
                    continue
                for gy in (1, 2):
                    for gx in (1, 2):
                        bh, bw = rows * small + (rows - 1) * gy, cols * small + (cols - 1) * gx
                        for sep in (1, 2, 3):
                            h, w = large + sep + bh, max(line_w, bw)
                            if h <= 28 and w <= 28:
                                options.append((gap, rows, cols, gy, gx, sep, h, w, line_w, bw))
    return tuple(options)


def _render(rng, k, colors, patterns, source_guides, destination_guides, bg, markers, pin, n, global_op):
    m, s, large, small = len(colors), k + 2, k + 4, k + 2
    layout = _pick(rng, _layouts(k, n, m))
    gap, rows, cols, gy, gx, sep, h, w, line_w, bank_w = layout
    # Asymmetric extra margins are independent of card geometry.
    H = int(rng.integers(h + 2, 31))
    W = int(rng.integers(w + 2, 31))
    y0 = int(rng.integers(1, H - h))
    x0 = int(rng.integers(1, W - w))
    line_x = x0 + int(rng.integers(w - line_w + 1))
    bank_x = x0 + int(rng.integers(w - bank_w + 1))
    a = np.full((H, W), bg, dtype=np.int8)
    front_right = bool(rng.integers(2))
    spatial_order = list(range(n))[::-1] if front_right else list(range(n))
    for slot, index in enumerate(spatial_order):
        r, c = y0 + 1, line_x + slot * (large + gap) + 1
        a[r:r + s, c:c + s] = colors[index]
        for side, identity in destination_guides[index].items():
            _bar(a, (r, c, s), side)[:] = identity
        if index == 0:
            corner = _pick(rng, ((0, 0), (0, s - 1), (s - 1, 0), (s - 1, s - 1)))
            a[r + corner[0], c + corner[1]] = pin
    slots = [(y0 + large + sep + row * (small + gy) + 1,
              bank_x + col * (small + gx) + 1) for row in range(rows) for col in range(cols)]
    selected = rng.choice(len(slots), m, replace=False)
    for index, slot in enumerate(selected):
        r, c = slots[int(slot)]
        window = a[r:r + k, c:c + k]
        window[patterns[index]] = colors[index]
        for side, identity in source_guides[index].items():
            _bar(a, (r, c, k), side)[:] = identity
    # Construction uses back-to-front overpainting, separate from solve's rays.
    output = np.full((k + 4, k + 4), bg, dtype=np.int8)
    output[1:-1, 1:-1] = colors[0]
    inner = np.full((k, k), bg, dtype=np.int8)
    for index in range(n - 1, -1, -1):
        op = _operation(source_guides[index], destination_guides[index])
        inner[~_transform(patterns[index], op)] = colors[index]
    output[2:-2, 2:-2] = inner
    return {"input": _transform(a, global_op), "output": _transform(output, global_op)}


def _new_pair(rng, k, n, palette, global_op, teaching=False):
    bg, marker_a, marker_b, pin = palette[:4]
    markers = (marker_a, marker_b)
    possible_m = [m for m in range(n, 7) if _layouts(k, n, m)]
    if teaching:
        possible_m = [m for m in possible_m if m > n]
    m = _pick(rng, possible_m)
    colors = list(palette[4:4 + m])
    rng.shuffle(colors)
    sources = [dict(zip((SIDES[c], SIDES[(c + 1) % 4]), markers))
               for c in rng.integers(0, 4, size=m)]
    ops = [int(v) for v in rng.integers(0, 8, size=n)]
    if not any(ops):
        ops[0] = int(rng.integers(1, 8))
    if teaching:
        ops[0] = _pick(rng, (1, 3))
        ops[1] = _pick(rng, [op for op in range(8) if op != ops[0]])
    destinations = [{MAPS[op][s]: color for s, color in source.items()}
                    for source, op in zip(sources, ops)]
    for _ in range(384):
        patterns = [_mask(rng, k, asymmetric=teaching and i == 0) for i in range(m)]
        holes = [_transform(mask, op) for mask, op in zip(patterns, ops)]
        answer = _view(colors[:n], holes, bg)
        inner = answer[2:-2, 2:-2]
        if not np.any(inner == bg) or not any(np.any(inner == c) for c in colors[1:n]):
            continue
        if np.array_equal(answer, _view(colors[:n], patterns[:n], bg)):
            continue
        if teaching:
            if not np.any(np.argwhere(patterns[0]).min(axis=0) > 0):
                continue
            flipped = {side: markers[1] if color == markers[0] else markers[0]
                       for side, color in destinations[0].items()}
            other_op = _operation(sources[0], flipped)
            alternate = _view(colors[:n], [_transform(patterns[0], other_op), *holes[1:]], bg)
            alt_inner = alternate[2:-2, 2:-2]
            if (not np.any(alt_inner == bg) or not np.any(alt_inner == colors[1])
                    or np.array_equal(alternate, _view(colors[:n], patterns[:n], bg))):
                continue
            if np.array_equal(answer, _view(colors[:n], [_transform(mask, ops[0]) for mask in patterns[:n]], bg)):
                continue
        pair = _render(rng, k, colors, patterns, sources, destinations, bg, markers, pin, n, global_op)
        if teaching and np.array_equal(_shortcut_views(_decode(pair["input"]))["nearest_stencil"], pair["output"]):
            continue
        if teaching and np.array_equal(_shortcut_views(_decode(pair["input"]))["trim_stencil"], pair["output"]):
            continue
        return pair
    raise RuntimeError("bounded stencil sampling exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    regime = _pick(rng, ("depth", "axis", "scale"))
    stable = bool(rng.random() < .25)
    palette = [int(v) for v in rng.permutation(10)]
    train_axis = int(rng.integers(2))
    first_op = train_axis + 2 * int(rng.integers(2)) + 4 * int(rng.integers(2))
    first = _new_pair(rng, int(rng.integers(3, 5)), 2, palette, first_op, teaching=True)
    second = {role: a.copy() for role, a in first.items()}
    info = _decode(first["input"])
    front = info["plates"][0]
    for side, color in front["guides"].items():
        _bar(second["input"], front["bounds"], side)[:] = next(m for m in info["markers"] if m != color)
    second["output"] = solve(second["input"])
    puzzle, seen = {"train": [first, second], "test": []}, {_key(first["input"]), _key(second["input"])}
    for index in range(2, nt + ne):
        split = "train" if index < nt else "test"
        configurations = [(k, n) for k in range(2, 7) for n in range(2, 5) if _layouts(k, n, n)]
        if regime == "depth":
            configurations = [(k, n) for k, n in configurations if (n == 2 if split == "train" else n > 2)]
        elif regime == "scale":
            configurations = [(k, n) for k, n in configurations if (k in (3, 4) if split == "train" else k in (2, 5, 6))]
        axis = 1 - train_axis if regime == "axis" and split == "test" else train_axis if regime == "axis" else int(rng.integers(2))
        global_op = axis + 2 * int(rng.integers(2)) + 4 * int(rng.integers(2))
        for _ in range(16):
            k, n = _pick(rng, configurations)
            local_palette = palette if stable else [int(v) for v in rng.permutation(10)]
            pair = _new_pair(rng, k, n, local_palette, global_op)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """42 sequential seeds; bounded grids and hashes only, no retained corpus."""
    sizes, counts, axes, operations, shapes, palettes, fingerprints = set(), set(), set(), set(), set(), set(), set()
    symmetric = asymmetric = connected = disconnected = distractors = registered_gaps = full = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        first, second = puzzle["train"][:2]
        front = _decode(first["input"])["plates"][0]
        expected_changes = np.zeros(first["input"].shape, dtype=bool)
        for side in front["guides"]:
            _bar(expected_changes, front["bounds"], side)[:] = True
        assert np.array_equal(first["input"] != second["input"], expected_changes)
        digest = hashlib.sha256()
        features = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                features[split].append((len(info["plates"]), info["k"], info["axis"]))
                sizes.add(info["k"]); counts.add(len(info["plates"])); axes.add(info["axis"])
                operations.update(p["op"] for p in info["plates"])
                shapes.add(pair["input"].shape)
                palettes.add((info["background"], info["markers"], info["pin"]))
                distractors += len(info["library"]) > len(info["plates"])
                for card in info["library"].values():
                    mask = card["mask"]
                    is_asym = len({_key(_transform(mask, op)) for op in range(8)}) == 8
                    asymmetric += is_asym; symmetric += not is_asym
                    is_connected = len(_components(mask.astype(int), 1)) == 1
                    connected += is_connected; disconnected += not is_connected
                    registered_gaps += not all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))
                    full += bool(mask.all())
                for role in ("input", "output"):
                    a = pair[role]
                    digest.update(bytes(a.shape)); digest.update(a.astype(np.uint8).tobytes())
                # Equivariance tests arbitrary global orientations, not just the
                # construction distribution. Recoloring must preserve role logic.
                if seed < 4:
                    for op in range(8):
                        assert np.array_equal(solve(_transform(pair["input"], op)), _transform(pair["output"], op))
                    permutation = np.random.default_rng(seed + 100).permutation(10)
                    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
                    # Moving the rivet reverses depth, without moving any cards.
                    retagged = pair["input"].copy()
                    retagged[retagged == info["pin"]] = info["plates"][0]["color"]
                    r, c, _ = info["plates"][-1]["bounds"]
                    retagged[r, c] = info["pin"]
                    reversed_plates = info["plates"][::-1]
                    expected = _view([p["color"] for p in reversed_plates],
                                     [p["holes"] for p in reversed_plates], info["background"])
                    assert np.array_equal(solve(retagged), expected)
        fingerprints.add(digest.hexdigest())
        train, test = features["train"], features["test"]
        depth_shift = max(v[0] for v in train) < min(v[0] for v in test)
        axis_shift = not ({v[2] for v in train} & {v[2] for v in test})
        scale_shift = not ({v[1] for v in train} & {v[1] for v in test})
        assert depth_shift or axis_shift or scale_shift, "missing observable train/test shift"
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[role], q[role]) for split in puzzle
                       for p, q in zip(puzzle[split], other[split]) for role in ("input", "output"))
    assert len(fingerprints) == 42 and sizes == set(range(2, 7)) and counts == {2, 3, 4}
    assert axes == {0, 1} and operations == set(range(8))
    assert len(shapes) >= 35 and len(palettes) >= 30
    assert min(symmetric, asymmetric, connected, disconnected, distractors) > 10
    assert registered_gaps > 20 and full > 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123 + nt * 17 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a wrong layer color, a damaged frame correspondence, and crop loss.
    for mutation in ("surface", "bar", "crop"):
        p = generate_puzzle(0, 2, 1)
        pair = p["test"][0]
        if mutation == "surface":
            pair["output"][2, 2] = (int(pair["output"][2, 2]) + 1) % 10
        elif mutation == "bar":
            info = _decode(pair["input"])
            plate = info["plates"][0]
            line = _bar(pair["input"], plate["bounds"], next(iter(plate["guides"])))
            line[len(line) // 2] = info["background"]
        else:
            pair["output"] = pair["output"][1:].copy()
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {mutation} corruption")
