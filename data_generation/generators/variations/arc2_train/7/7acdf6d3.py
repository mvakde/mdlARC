"""Capacity matching puzzles with varied open cups and unary payloads."""

from collections import Counter
from itertools import product

import numpy as np


_SOURCE_INPUTS = (
    ("7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777272777777777",
     "7777727777277727", "7777777777727277", "7777777777772777",
     "7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777977777777", "7777777777777777",
     "7777777777777777"),
    ("7777777777777", "7797777779777", "7777777777777",
     "7777777777777", "7797977777777", "7777777777777",
     "7777777777777", "7777772777277", "7777777272777",
     "7777777727777", "7272777777777", "7727777777777",
     "7777777777777"),
    ("777777777777777", "777777777777777", "777777777777777",
     "777777777777777", "777777777777777", "777777777777777",
     "777777777777777", "777777777777777", "777777779779777",
     "777777779779777", "777777779999777", "779797777777772",
     "779997777777772", "777777777777772", "777777777777772"),
)

_AUTO_COUNTS = ((2, 1), (3, 2), (4, 1), (5, 3), (6, 2),
                (7, 4), (8, 3), (9, 1), (10, 4))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        choices = list(range(test + 1, 11))
        train = choices[seed % len(choices)]
    if test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[seed % len(choices)]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _cup(left_moves, right_moves, top_width, pointed):
    depth = len(left_moves)
    left, right = 0, top_width + 1
    outline, cavity = set(), set()
    for row in range(depth):
        if right - left < 2:
            return None
        outline.update(((row, left), (row, right)))
        cavity.update((row, col) for col in range(left + 1, right))
        left += left_moves[row]
        right -= right_moves[row]
    if pointed:
        if left != right:
            return None
        outline.add((depth, left))
        style = "point"
    else:
        if right - left < 2:
            return None
        outline.update((depth, col) for col in range(left, right + 1))
        style = "slope" if any(left_moves) or any(right_moves) else "u"
    if not 1 <= len(cavity) <= 15:
        return None
    return {"outline": tuple(sorted(outline)), "cavity": tuple(sorted(cavity)),
            "height": depth + 1, "width": top_width + 2,
            "capacity": len(cavity), "style": style}


def _make_bank():
    bank = {}
    seen = set()
    for depth in range(1, 5):
        for moves in product((0, 1), repeat=2 * (depth - 1)):
            left = moves[:depth - 1] + (1,)
            right = moves[depth - 1:] + (1,)
            width = sum(left) + sum(right) - 1
            shape = _cup(left, right, width, True)
            if shape is not None:
                key = shape["outline"]
                if key not in seen:
                    bank.setdefault(shape["capacity"], {}).setdefault("point", []).append(shape)
                    seen.add(key)
        for width in range(1, 8):
            for moves in product((0, 1), repeat=2 * depth):
                shape = _cup(moves[:depth], moves[depth:], width, False)
                if shape is not None:
                    key = shape["outline"]
                    if key not in seen:
                        bank.setdefault(shape["capacity"], {}).setdefault(shape["style"], []).append(shape)
                        seen.add(key)
    return bank


_BANK = _make_bank()
_QUANTITIES = tuple(q for q in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15) if q in _BANK)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _pick_shape(rng, capacity, preferred=None):
    styles = list(_BANK[capacity])
    style = preferred if preferred in styles else _choice(rng, styles)
    options = _BANK[capacity][style]
    return _choice(rng, options)


def _bbox(shape, top, left):
    return (top, left, top + shape["height"] - 1, left + shape["width"] - 1)


def _separated(a, b):
    return (a[2] + 1 < b[0] or b[2] + 1 < a[0] or
            a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _place_cups(rng, height, width, shapes):
    placements, boxes = [], []
    for shape in shapes:
        if shape["height"] > height or shape["width"] > width:
            return None
        candidates = [(r, c) for r in range(height - shape["height"] + 1)
                      for c in range(width - shape["width"] + 1)
                      if all(_separated(_bbox(shape, r, c), b) for b in boxes)]
        if not candidates:
            return None
        top, left = _choice(rng, candidates)
        placements.append((top, left))
        boxes.append(_bbox(shape, top, left))
    return placements


def _pattern(rng, count, mode):
    if mode == "single":
        return [(0, 0)]
    if mode == "horizontal":
        return [(0, col) for col in range(count)]
    if mode == "vertical":
        return [(row, 0) for row in range(count)]
    if mode == "block":
        factors = [(h, count // h) for h in range(2, count + 1)
                   if count % h == 0 and 2 <= count // h <= 4]
        if factors:
            h, w = _choice(rng, factors)
            return [(r, c) for r in range(h) for c in range(w)]
        mode = "bent"
    if mode == "bent":
        for _ in range(30):
            cells = {(0, 0)}
            while len(cells) < count:
                border = {(r + dr, c + dc) for r, c in cells
                          for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if (r + dr, c + dc) not in cells}
                cells.add(_choice(rng, sorted(border)))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            shifted = [(r - min(rows), c - min(cols)) for r, c in cells]
            if len({r for r, _ in shifted}) > 1 and len({c for _, c in shifted}) > 1:
                return shifted
        return shifted
    return None


def _near_outline(cell, outline):
    r, c = cell
    return any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in outline)


def _payload(rng, height, width, outline, boxes, count, mode):
    allowed = {(r, c) for r in range(height) for c in range(width)
               if not any(b[0] <= r <= b[2] and b[1] <= c <= b[3] for b in boxes)
               and not _near_outline((r, c), outline)}
    if mode == "scattered":
        order = list(allowed)
        rng.shuffle(order)
        result = []
        for cell in order:
            if all(max(abs(cell[0] - old[0]), abs(cell[1] - old[1])) > 1
                   for old in result):
                result.append(cell)
                if len(result) == count:
                    return result
        return None
    if mode == "mixed":
        first = _payload(rng, height, width, outline, boxes, count - 1, "horizontal")
        if first is None:
            return None
        candidates = [cell for cell in allowed if all(
            max(abs(cell[0] - old[0]), abs(cell[1] - old[1])) > 1 for old in first)]
        return first + [_choice(rng, candidates)] if candidates else None
    pattern = _pattern(rng, count, mode)
    if pattern is None:
        return None
    ph = max(r for r, _ in pattern) + 1
    pw = max(c for _, c in pattern) + 1
    candidates = []
    for top in range(height - ph + 1):
        for left in range(width - pw + 1):
            cells = [(top + r, left + c) for r, c in pattern]
            if all(cell in allowed for cell in cells):
                candidates.append(cells)
    if not candidates:
        return None
    if rng.random() < 0.4:
        def proximity(cells):
            return min(min(r, c, height - 1 - r, width - 1 - c) for r, c in cells)
        scores = [proximity(cells) for cells in candidates]
        threshold = sorted(scores)[max(0, len(scores) // 5 - 1)]
        candidates = [cells for cells, score in zip(candidates, scores) if score <= threshold]
    return _choice(rng, candidates)


def _palette(rng, base, mode):
    if mode == "same":
        return base
    if mode == "swap":
        return (base[0], base[2], base[1])
    colors = [int(x) for x in rng.choice(10, size=3, replace=False)]
    if colors[0] == base[0]:
        colors[0] = _choice(rng, [c for c in range(10) if c not in colors and c != base[0]])
    return tuple(colors)


def _draw_pair(rng, quantity, other_caps, palette, mode, styles, size_band):
    bg, outline_color, payload_color = palette
    caps = [quantity] + list(other_caps)
    for attempt in range(75):
        shapes = [_pick_shape(rng, cap, style) for cap, style in zip(caps, styles)]
        if rng.random() < 0.4:
            possible_inversions = []
            for i in range(len(caps)):
                for j in range(i + 1, len(caps)):
                    small, large = (i, j) if caps[i] < caps[j] else (j, i)
                    small_options = _BANK[caps[small]][shapes[small]["style"]]
                    large_options = _BANK[caps[large]][shapes[large]["style"]]
                    spacious = max(small_options, key=lambda s: s["height"] * s["width"])
                    compact = min(large_options, key=lambda s: s["height"] * s["width"])
                    if spacious["height"] * spacious["width"] > \
                            compact["height"] * compact["width"]:
                        possible_inversions.append((small, large, spacious, compact))
            if possible_inversions:
                small, large, spacious, compact = _choice(rng, possible_inversions)
                shapes[small], shapes[large] = spacious, compact
        order = rng.permutation(len(shapes))
        shapes = [shapes[int(i)] for i in order]
        selected = [int(i) for i in order].index(0)
        if size_band == "small" and attempt < 35:
            height, width = int(rng.integers(12, 18)), int(rng.integers(12, 18))
        elif size_band == "large" or attempt >= 65:
            height, width = int(rng.integers(24, 31)), int(rng.integers(24, 31))
        else:
            height, width = int(rng.integers(17, 25)), int(rng.integers(17, 25))
        placements = _place_cups(rng, height, width, shapes)
        if placements is None:
            continue
        base = np.full((height, width), bg, dtype=np.int8)
        outline, cavities, boxes = set(), [], []
        for shape, (top, left) in zip(shapes, placements):
            cells = {(top + r, left + c) for r, c in shape["outline"]}
            outline.update(cells)
            for r, c in cells:
                base[r, c] = outline_color
            cavities.append({(top + r, left + c) for r, c in shape["cavity"]})
            boxes.append(_bbox(shape, top, left))
        payload = _payload(rng, height, width, outline, boxes, quantity, mode)
        if payload is None:
            continue
        inp = base.copy()
        for r, c in payload:
            inp[r, c] = payload_color
        out = base.copy()
        for r, c in cavities[selected]:
            out[r, c] = payload_color
        return {"input": inp, "output": out}
    raise RuntimeError("could not fit a clear cup scene")


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = remaining.pop()
        group = {start}
        fringe = [start]
        while fringe:
            r, c = fringe.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    cell = (r + dr, c + dc)
                    if cell in remaining:
                        remaining.remove(cell)
                        group.add(cell)
                        fringe.append(cell)
        result.append(group)
    return result


def _parse_cup(cells):
    rows = sorted({r for r, _ in cells})
    _require(rows == list(range(rows[0], rows[-1] + 1)) and len(rows) >= 2,
             "cup rows must be contiguous")
    boundaries = []
    cavity = set()
    for r in rows[:-1]:
        cols = sorted(c for rr, c in cells if rr == r)
        _require(len(cols) == 2 and cols[1] - cols[0] >= 2,
                 "cup needs two sides around a nonempty cavity")
        boundaries.append((cols[0], cols[1]))
        cavity.update((r, c) for c in range(cols[0] + 1, cols[1]))
    bottom = sorted(c for rr, c in cells if rr == rows[-1])
    _require(len(bottom) == 1 or
             (len(bottom) >= 3 and bottom == list(range(bottom[0], bottom[-1] + 1))),
             "cup bottom must be a point or a flat base")
    left, right = bottom[0], bottom[-1]
    for old, new in zip(boundaries, boundaries[1:] + [(left, right)]):
        _require(new[0] - old[0] in (0, 1) and old[1] - new[1] in (0, 1),
                 "cup sides must move inward in unit steps")
    if len(bottom) == 1:
        _require(boundaries[-1][1] - boundaries[-1][0] == 2,
                 "pointed cup must close below a one-cell cavity")
        style = "point"
    else:
        style = "u" if all(b == boundaries[0] for b in boundaries) and \
            (left, right) == boundaries[0] else "slope"
    r0, r1 = rows[0], rows[-1]
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return {"cavity": cavity, "capacity": len(cavity), "style": style,
            "bbox": (r0, c0, r1, c1), "outline": cells}


def _infer_pair(inp, out):
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "pair must have three colors")
    bg = int(colors[int(np.argmax(counts))])
    _require(int(max(counts)) > int(sorted(counts)[-2]), "background must dominate")
    changes = {(int(inp[r, c]), int(out[r, c])) for r, c in zip(*np.nonzero(inp != out))}
    payloads = {old for old, new in changes if new == bg and old != bg}
    _require(len(payloads) == 1, "detached payload color is unclear")
    payload_color = next(iter(payloads))
    outline_color = next(c for c in map(int, colors) if c not in (bg, payload_color))
    _require(changes == {(payload_color, bg), (bg, payload_color)},
             "only quantity transfer may change cells")
    return bg, outline_color, payload_color


def _check_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "dimensions changed")
    bg, outline_color, payload_color = _infer_pair(inp, out)
    outline = {(int(r), int(c)) for r, c in zip(*np.nonzero(inp == outline_color))}
    components = _components(outline)
    _require(len(components) in (2, 3), "need two or three separate cups")
    cups = [_parse_cup(comp) for comp in components]
    capacities = [cup["capacity"] for cup in cups]
    _require(len(capacities) == len(set(capacities)), "cup capacities must differ")
    payload = {(int(r), int(c)) for r, c in zip(*np.nonzero(inp == payload_color))}
    _require(bool(payload), "payload missing")
    _require(all(inp[r, c] == bg for cup in cups for r, c in cup["cavity"]),
             "cup cavities must start empty")
    _require(all(not any(b[0] <= r <= b[2] and b[1] <= c <= b[3]
                         for b in (cup["bbox"] for cup in cups)) for r, c in payload),
             "payload lies inside a cup")
    _require(all(not _near_outline(cell, outline) for cell in payload),
             "payload touches an outline")
    match = [cup for cup in cups if cup["capacity"] == len(payload)]
    _require(len(match) == 1, "quantity must select exactly one cup")
    expected = inp.copy()
    for r, c in payload:
        expected[r, c] = bg
    for r, c in match[0]["cavity"]:
        expected[r, c] = payload_color
    _require(np.array_equal(out, expected), "output differs from count-and-fill rule")
    return {"quantity": len(payload), "cups": cups, "selected": match[0],
            "palette": (bg, outline_color, payload_color), "payload": payload}


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _is_source_input(grid):
    return any(grid.shape == (len(rows), len(rows[0])) and
               all("".join(map(str, map(int, row))) == source_row
                   for row, source_row in zip(grid, rows)) for rows in _SOURCE_INPUTS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    base_palette = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    style_policy = _choice(rng, ("free", "free", "free", "free", "free",
                                 "free", "uniform", "uniform",
                                 "source_shift", "reverse_shift"))
    uniform_style = _choice(rng, ("point", "u"))
    train_quantities = [1, 4]
    while len(train_quantities) < ntrain:
        train_quantities.append(int(_choice(rng, _QUANTITIES)))
    rng.shuffle(train_quantities)
    available_new = [q for q in _QUANTITIES if q not in train_quantities]
    test_quantities = []
    if ntest >= 2:
        test_quantities.append(1)
        test_quantities.append(int(_choice(rng, available_new or _QUANTITIES)))
    while len(test_quantities) < ntest:
        test_quantities.append(int(_choice(rng, _QUANTITIES)))
    rng.shuffle(test_quantities)
    puzzle = {"train": [], "test": []}
    seen = set()
    total = ntrain + ntest
    for index in range(total):
        train = index < ntrain
        quantity = train_quantities[index] if train else test_quantities[index - ntrain]
        pair_rng = np.random.default_rng([seed, index, int(rng.integers(0, 2**31))])
        if train and quantity == 1:
            other = int(_choice(pair_rng, (2, 3, 4, 5, 6)))
        elif train and quantity == 4:
            other = int(_choice(pair_rng, (1, 2, 3)))
        else:
            other = int(_choice(pair_rng, [q for q in _QUANTITIES if q != quantity]))
        other_caps = [other]
        if pair_rng.random() < (0.16 if train else 0.3):
            options = [q for q in _QUANTITIES if q not in (quantity, other) and q <= 10]
            other_caps.append(int(_choice(pair_rng, options)))
        if index == 0:
            palette_mode = "same"
        elif index == 1:
            palette_mode = _choice(pair_rng, ("swap", "fresh"))
        else:
            palette_mode = _choice(pair_rng, ("same", "swap", "fresh", "fresh"))
        palette = _palette(pair_rng, base_palette, palette_mode)
        style_modes = ["point", "u", "slope"]
        if style_policy == "uniform":
            styles = [uniform_style] * (len(other_caps) + 1)
        elif style_policy in ("source_shift", "reverse_shift"):
            style = ("point" if train else "u") if style_policy == "source_shift" \
                else ("u" if train else "point")
            styles = [style] * (len(other_caps) + 1)
        elif pair_rng.random() < 0.3:
            styles = [_choice(pair_rng, style_modes) for _ in range(len(other_caps) + 1)]
        else:
            style = _choice(pair_rng, style_modes)
            styles = [style] * (len(other_caps) + 1)
        if quantity == 1:
            mode = "single"
        else:
            mode = _choice(pair_rng, ("scattered", "horizontal", "vertical",
                                      "block", "bent", "mixed"))
        size_band = _choice(pair_rng, ("small", "small", "medium", "medium", "large"))
        for _ in range(8):
            pair = _draw_pair(pair_rng, quantity, other_caps, palette, mode, styles, size_band)
            signature = _signature(pair["input"])
            if signature not in seen and not _is_source_input(pair["input"]):
                break
        else:
            raise RuntimeError("could not make unique pair")
        seen.add(signature)
        puzzle["train" if train else "test"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys are invalid")
    _require(all(isinstance(puzzle[key], list) for key in ("train", "test")),
             "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "pair counts are invalid")
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys are invalid")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer), "grid type is invalid")
                _require(all(1 <= n <= 30 for n in grid.shape) and
                         bool(np.all((grid >= 0) & (grid <= 9))), "grid bounds are invalid")
            inp_sig, out_sig = _signature(pair["input"]), _signature(pair["output"])
            _require(inp_sig not in seen_inputs and (inp_sig, out_sig) not in seen_pairs,
                     "duplicate grid or pair")
            _require(not _is_source_input(pair["input"]), "source input copied")
            seen_inputs.add(inp_sig)
            seen_pairs.add((inp_sig, out_sig))
            metadata[split].append(_check_pair(pair))
    train_info = metadata["train"]
    _require({1, 4}.issubset({m["quantity"] for m in train_info}),
             "training must demonstrate one and four cells")
    _require(any(m["quantity"] < max(c["capacity"] for c in m["cups"])
                 for m in train_info), "training must select a smaller cup")
    _require(any(m["quantity"] > min(c["capacity"] for c in m["cups"])
                 for m in train_info), "training must select a larger cup")
    _require(len({m["palette"] for m in train_info}) >= 2,
             "training must demonstrate that literal colors can change")


def _predict_from_input(train_pairs, test_input):
    # Confirm the demonstrations use the count-and-fill rule before predicting.
    _require(len(train_pairs) >= 2, "test prediction needs training demonstrations")
    for pair in train_pairs:
        _check_pair(pair)
    # The test input identifies local colors by cup geometry and a unique match.
    inp = test_input
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    candidates = []
    for outline_color in (int(c) for c in colors if c != bg):
        payload_color = next(int(c) for c in colors if c not in (bg, outline_color))
        try:
            outline = {(int(r), int(c)) for r, c in zip(*np.nonzero(inp == outline_color))}
            cups = [_parse_cup(comp) for comp in _components(outline)]
            if len(cups) not in (2, 3):
                continue
            payload = {(int(r), int(c)) for r, c in zip(*np.nonzero(inp == payload_color))}
            matching = [cup for cup in cups if cup["capacity"] == len(payload)]
            if len(matching) != 1 or any(cell in outline for cell in payload):
                continue
            pred = inp.copy()
            for r, c in payload:
                pred[r, c] = bg
            for r, c in matching[0]["cavity"]:
                pred[r, c] = payload_color
            candidates.append(pred)
        except AssertionError:
            continue
    _require(len(candidates) == 1, "test input does not determine a unique output")
    return candidates[0]


def _payload_category(cells):
    if len(cells) == 1:
        return "single"
    remaining = set(cells)
    groups = []
    while remaining:
        start = remaining.pop()
        group = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    frontier.append(neighbor)
        groups.append(group)
    if len(groups) > 1:
        return "scattered" if all(len(group) == 1 for group in groups) else "mixed"
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        return "horizontal"
    if len(cols) == 1:
        return "vertical"
    return "block" if len(cells) == len(rows) * len(cols) else "bent"


def self_test() -> None:
    a = generate_puzzle(12345, 6, 2)
    b = generate_puzzle(12345, 6, 2)
    _require(all(np.array_equal(x[key], y[key]) for split in ("train", "test")
                 for x, y in zip(a[split], b[split]) for key in ("input", "output")),
             "generation is not deterministic")
    _require(_signature(a["train"][0]["input"]) !=
             _signature(generate_puzzle(12346, 6, 2)["train"][0]["input"]),
             "different seeds produced the same puzzle")
    cases = ((800, 2, 1), (801, 10, 1), (802, 10, 4),
             (803, None, 1), (804, None, 4), (805, 2, None), (806, 10, None))
    for seed, nt, ns in cases:
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        _require(nt is None or len(puzzle["train"]) == nt, "train count ignored")
        _require(ns is None or len(puzzle["test"]) == ns, "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
    train_counts, test_counts, dimensions, structures = set(), set(), set(), set()
    styles_train, styles_test, quantities, palette_count = set(), set(), set(), set()
    arrangements = {"train": set(), "test": set()}
    mouth_widths, depths, capacities = set(), set(), set()
    measurements = Counter()
    mixed_pairs = three_cups = small_grids = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes = set()
        metadata = {"train": [], "test": []}
        first_palette = _check_pair(puzzle["train"][0])["palette"]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                meta = _check_pair(pair)
                metadata[split].append(meta)
                palette_count.add(meta["palette"])
                palettes.add(meta["palette"])
                dimensions.add(pair["input"].shape)
                structures.add(_signature(pair["input"]))
                quantities.add(meta["quantity"])
                arrangement = _payload_category(meta["payload"])
                arrangements[split].add(arrangement)
                pair_styles = {cup["style"] for cup in meta["cups"]}
                (styles_train if split == "train" else styles_test).update(pair_styles)
                mixed_pairs += len(pair_styles) > 1
                three_cups += len(meta["cups"]) == 3
                small_grids += max(pair["input"].shape) <= 17
                measurements["large_grids"] += min(pair["input"].shape) >= 24
                measurements["one_cell_tests"] += split == "test" and meta["quantity"] == 1
                measurements["selected_middle"] += (
                    len(meta["cups"]) == 3 and
                    min(c["capacity"] for c in meta["cups"]) < meta["quantity"] <
                    max(c["capacity"] for c in meta["cups"]))
                if meta["palette"] == first_palette:
                    palette_mode = "same"
                elif meta["palette"] == (first_palette[0], first_palette[2], first_palette[1]):
                    palette_mode = "swap"
                else:
                    palette_mode = "fresh"
                measurements[f"palette_{palette_mode}"] += 1
                measurements["combined"] += (len(pair_styles) > 1 and
                    arrangement in {"horizontal", "vertical", "block", "bent", "mixed"}
                    and palette_mode == "fresh")
                height, width = pair["input"].shape
                measurements["cup_edge"] += any(
                    cup["bbox"][0] == 0 or cup["bbox"][1] == 0 or
                    cup["bbox"][2] == height - 1 or cup["bbox"][3] == width - 1
                    for cup in meta["cups"])
                measurements["payload_edge"] += min(
                    min(r, c, height - 1 - r, width - 1 - c)
                    for r, c in meta["payload"]) <= 1
                outline = set().union(*(cup["outline"] for cup in meta["cups"]))
                measurements["payload_near_cup"] += min(
                    max(abs(r - rr), abs(c - cc))
                    for r, c in meta["payload"] for rr, cc in outline) == 2
                for cup in meta["cups"]:
                    capacities.add(cup["capacity"])
                    r0, c0, r1, c1 = cup["bbox"]
                    depths.add(r1 - r0)
                    top = [c for r, c in cup["outline"] if r == r0]
                    mouth_widths.add(max(top) - min(top) - 1)
                    bounds = []
                    for row in range(r0, r1):
                        columns = [c for r, c in cup["outline"] if r == row]
                        bounds.append((min(columns), max(columns)))
                    measurements["asymmetric"] += any(
                        b[0] - bounds[0][0] != bounds[0][1] - b[1]
                        for b in bounds)
                    measurements["sloping_flat"] += (cup["style"] == "slope" and
                        len({right - left - 1 for left, right in bounds}) > 1)
                for i, first_cup in enumerate(meta["cups"]):
                    for second_cup in meta["cups"][i + 1:]:
                        smaller, larger = sorted((first_cup, second_cup),
                                                 key=lambda cup: cup["capacity"])
                        small_box, large_box = smaller["bbox"], larger["bbox"]
                        small_area = (small_box[2] - small_box[0] + 1) * \
                            (small_box[3] - small_box[1] + 1)
                        large_area = (large_box[2] - large_box[0] + 1) * \
                            (large_box[3] - large_box[1] + 1)
                        measurements["footprint_inversions"] += small_area > large_area
                        measurements["outline_inversions"] += \
                            len(smaller["outline"]) > len(larger["outline"])
                if split == "test":
                    predicted = _predict_from_input(puzzle["train"], pair["input"].copy())
                    _require(np.array_equal(predicted, pair["output"]),
                             "training plus test input did not recover output")
        _require(len(palettes) >= 2, "within-puzzle palette variation missing")
        train_styles = {cup["style"] for meta in metadata["train"] for cup in meta["cups"]}
        test_styles = {cup["style"] for meta in metadata["test"] for cup in meta["cups"]}
        measurements["uniform_puzzles"] += len(train_styles | test_styles) == 1
        measurements["style_varied_puzzles"] += len(train_styles | test_styles) > 1
        measurements["source_shift"] += train_styles == {"point"} and test_styles == {"u"}
        measurements["reverse_shift"] += train_styles == {"u"} and test_styles == {"point"}
        measurements["test_style_seen"] += test_styles <= train_styles
        measurements["background_varied"] += len({m["palette"][0]
            for split in metadata.values() for m in split}) > 1
        measurements["one_first"] += metadata["train"][0]["quantity"] == 1
        demonstrated = {m["quantity"] for m in metadata["train"]}
        measurements["novel_tests"] += sum(m["quantity"] not in demonstrated
                                               for m in metadata["test"])
        measurements["multi_test_distinct"] += len(metadata["test"]) > 1 and \
            len({m["quantity"] for m in metadata["test"]}) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack diversity")
    _require(len(dimensions) >= 45 and len(structures) >= 180,
             "grid structures lack diversity")
    _require(styles_train == {"point", "u", "slope"} and
             styles_test == {"point", "u", "slope"}, "cup styles are split-locked")
    _require(len(quantities) >= 9 and len(palette_count) >= 25,
             "quantities or color roles lack diversity")
    _require(mixed_pairs >= 10 and three_cups >= 10 and small_grids >= 25,
             "mixed cups, extra distractors, or compact grids are rare")
    _require(all(arrangements[split] == {"single", "scattered", "horizontal",
              "vertical", "block", "bent", "mixed"} for split in ("train", "test")),
             "payload topology is split-locked or too narrow")
    _require(len(capacities) >= 10 and len(mouth_widths) >= 5 and len(depths) >= 3,
             "cup capacities, mouth widths, or depths lack variety")
    for metric, minimum in (("uniform_puzzles", 1), ("style_varied_puzzles", 20),
                            ("source_shift", 1), ("reverse_shift", 1),
                            ("test_style_seen", 10), ("background_varied", 30),
                            ("one_first", 1), ("novel_tests", 10),
                            ("one_cell_tests", 5), ("multi_test_distinct", 10),
                            ("large_grids", 15), ("cup_edge", 20),
                            ("payload_edge", 20), ("payload_near_cup", 10),
                            ("selected_middle", 1), ("asymmetric", 20),
                            ("sloping_flat", 10), ("footprint_inversions", 1),
                            ("outline_inversions", 1), ("combined", 2),
                            ("palette_same", 10), ("palette_swap", 10),
                            ("palette_fresh", 10)):
        _require(measurements[metric] >= minimum,
                 f"review axis {metric} was not adequately exercised")
