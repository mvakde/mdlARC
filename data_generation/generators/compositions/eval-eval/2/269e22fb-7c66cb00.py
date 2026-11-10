"""Atlas-addressed cargo: exact crop registration controls component docking.

All output-affecting information is visible in the current input. No parent
generator, stored parent bitmap, seed, or training label is used by solve().
"""
from collections import Counter
from functools import lru_cache
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _transform(a, index):
    """Four rotations, followed by four rotations of a horizontal reflection."""
    return (np.rot90(a, index) if index < 4 else
            np.rot90(np.fliplr(a), index - 4)).copy()


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _canonical(a):
    return min(_key(b) for i in range(8)
               for b in (_transform(a, i), 1 - _transform(a, i)))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(a, points=None, diagonal=False):
    unseen = set(np.ndindex(a.shape)) if points is None else set(points)
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen and a[q] == a[start]:
                    unseen.remove(q)
                    stack.append(q)
        result.append((int(a[start]), tuple(sorted(cells))))
    return result


def _independence_witness(patch):
    independent, grouped = set(), set()
    for color, cells in _components(patch):
        bottom = max(r for r, c in cells)
        independent.update((color, r - bottom, c) for r, c in cells)
    for color in (0, 1):
        cells = [tuple(p) for p in np.argwhere(patch == color)]
        bottom = max(r for r, c in cells)
        grouped.update((color, r - bottom, c) for r, c in cells)
    return independent != grouped


@lru_cache(maxsize=4)
def _crop_counts(reference_key):
    """Pure generation-time acceleration; solve does not consult this cache."""
    shape, data = reference_key
    reference = np.frombuffer(data, dtype=np.int8).reshape(shape)
    H, W = reference.shape
    counts = Counter()
    shapes = [(h, w) for h in range(2, min(7, max(H, W)) + 1)
              for w in range(2, min(7, max(H, W)) + 1) if min(h, w) <= 5]
    for i in range(8):
        full = _transform(reference, i)
        for h, w in shapes:
            for r in range(full.shape[0] - h + 1):
                for c in range(full.shape[1] - w + 1):
                    p = full[r:r+h, c:c+w]
                    counts[_key(p)] += 1
                    counts[_key(1-p)] += 1
    return counts


def _crop_bank(reference):
    """Count all matches; omit undemonstrated diagonal grouping ambiguity."""
    H, W = reference.shape
    counts = _crop_counts(_canonical(reference))
    bank = []
    for h in range(2, min(5, H) + 1):
        for w in range(2, min(7, W) + 1):
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    p = reference[r:r+h, c:c+w]
                    if (counts[_key(p)] == 1 and len(np.unique(p)) == 2
                            and len(_components(p)) == len(_components(p, diagonal=True))):
                        bank.append((r, c, h, w, _independence_witness(p)))
    return bank


def _random_reference(rng):
    H, W = int(rng.integers(6, 9)), int(rng.integers(6, 10))
    a = np.zeros((H, W), dtype=np.int8)
    for _ in range(int(rng.integers(3, 7))):
        r0, r1 = sorted(map(int, rng.choice(H, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(W, 2, replace=False)))
        kind = int(rng.integers(5))
        if kind == 0:  # bend
            a[r0:r1+1, c0] = 1
            a[r1, c0:c1+1] = 1
        elif kind == 1:  # loop
            a[r0:r1+1, [c0, c1]] = 1
            a[[r0, r1], c0:c1+1] = 1
        elif kind == 2:  # branching comb
            a[r0:r1+1, c0] = 1
            for r in range(r0, r1+1, 2):
                a[r, c0:c1+1] = 1
        elif kind == 3:  # orthogonally connected stepped path
            r, c = r0, c0
            a[r, c] = 1
            while r < r1 or c < c1:
                if r < r1 and (c == c1 or rng.random() < .5):
                    r += 1
                else:
                    c += 1
                a[r, c] = 1
        else:  # solid region or a cut through existing strokes
            a[r0:r1+1, c0:c1+1] = int(rng.random() < .6)
    return a


def _eligible_banks(a):
    # Reflections preserve bank sizes. Transposition exchanges shape budgets.
    banks = [_crop_bank(a), _crop_bank(a.T)]
    return (any(e[4] and e[1] > 0 and e[2]*e[3] >= 20 for e in banks[0])
            and all(sum(3 <= e[2] <= 4 and 3 <= e[3] <= 5
                        and e[2]*e[3] >= 14 for e in b) >= 4 for b in banks)
            and all(sum(e[2] <= 3 and e[2]*e[3] <= 13 for e in b) >= 4
                    for b in banks))


def _reference(rng):
    for _ in range(60):
        a = _random_reference(rng)
        if .22 < a.mean() < .75 and _eligible_banks(a):
            return a
    # Self-authored finite reserve, not parent content. Ordinary sampling is
    # procedural; self_test records normalized reference diversity directly.
    return np.array([[int(c) for c in row] for row in
                     ("00010000", "00010000", "11111000", "11111111",
                      "11111100", "11111000", "11111000", "11000000")], dtype=np.int8)


def _ring(box, thickness):
    top, left, h, w = box
    return {(top+r, left+c) for r in range(h) for c in range(w)
            if r < thickness or c < thickness or
            r >= h-thickness or c >= w-thickness}


def _frame_decompositions(stage, background):
    """Find a shared frame color: one double rim, one or more single rims."""
    for color in map(int, np.unique(stage)):
        if color == background:
            continue
        points = [tuple(p) for p in np.argwhere(stage == color)]
        frames, valid = [], True
        for _, cells in _components(stage, points):
            r0 = min(r for r, c in cells)
            c0 = min(c for r, c in cells)
            h = 1 + max(r for r, c in cells) - r0
            w = 1 + max(c for r, c in cells) - c0
            box = (r0, c0, h, w)
            thickness = next((t for t in (1, 2) if h >= 2*t+2 and w >= 2*t+2
                              and set(cells) == _ring(box, t)), None)
            if thickness is None:
                valid = False
                break
            frames.append((box, thickness))
        if not valid or sum(t == 2 for b, t in frames) != 1 or not any(t == 1 for b, t in frames):
            continue
        coverage = set()
        interiors = []
        for box, t in frames:
            r, c, h, w = box
            bbox = {(r+y, c+x) for y in range(h) for x in range(w)}
            if bbox & coverage:
                valid = False
                break
            coverage.update(bbox)
            interior = stage[r+t:r+h-t, c+t:c+w-t]
            if len(np.unique(interior)) != 2 or np.any((interior == background) | (interior == color)):
                valid = False
                break
            interiors.append(dict(box=box, thickness=t, top=r+t, left=c+t,
                                  grid=interior.copy()))
        if valid and all(int(stage[p]) == background for p in np.ndindex(stage.shape) if p not in coverage):
            atlas = next(f for f in interiors if f["thickness"] == 2)
            cards = sorted((f for f in interiors if f["thickness"] == 1), key=lambda f: f["box"])
            yield color, atlas, cards


def _band_decompositions(a, background):
    rows = []
    for row in a:
        nz = np.flatnonzero(row != background)
        if not len(nz):
            rows.append(None)
        else:
            left, right = int(nz[0]), int(nz[-1])
            info = False
            if right-left >= 3 and row[left] == row[right] and row[left] != row[left+1] and np.all(row[left+1:right] == row[left+1]):
                info = (left, right, int(row[left+1]), int(row[left]))
            rows.append(info)
    for first in range(2, a.shape[0]):
        if not rows[first] or rows[first-1] is not None or any(x is False for x in rows[first:]):
            continue
        bands, r = [], first
        while r < a.shape[0]:
            if rows[r] is None:
                r += 1
                continue
            info, top = rows[r], r
            while r+1 < a.shape[0] and rows[r+1] == info:
                r += 1
            left, right, route, rail = info
            bands.append(dict(top=top, bottom=r, left=left, right=right,
                              route=route, rail=rail))
            r += 1
        if len(bands) == 2 and bands[1]["top"] > bands[0]["bottom"]+1 and len({b["route"] for b in bands}) == 2:
            yield first, bands


def _matches(atlas, card):
    roles = sorted(map(int, np.unique(atlas)))
    colors = sorted(map(int, np.unique(card)))
    if len(roles) != 2 or len(colors) != 2:
        return []
    binary = (atlas == roles[1]).astype(np.int8)
    patch = (card == colors[1]).astype(np.int8)
    h, w = patch.shape
    matches = []
    for index in range(8):
        full = _transform(binary, index)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        for swap in (0, 1):
            for r, c in np.argwhere(np.all(windows == (patch ^ swap), axis=(-2, -1))):
                matches.append((index, swap, int(r), int(c)))
    return matches


def _register(atlas, cards):
    H, W = atlas["grid"].shape
    indices = np.arange(H*W).reshape(H, W)
    registered = []
    for card in cards:
        matches = _matches(atlas["grid"], card["grid"])
        _require(len(matches) == 1, "a crop must have exactly one geometric/color/offset match")
        index, swap, r0, c0 = matches[0]
        chart = _transform(indices, index)
        comps = []
        for color, cells in _components(card["grid"]):
            mapped = tuple(sorted((int(chart[r0+r, c0+c]) // W,
                                   int(chart[r0+r, c0+c]) % W) for r, c in cells))
            routes = {int(atlas["grid"][p]) for p in mapped}
            _require(len(routes) == 1, "role registration must preserve monochromatic components")
            comps.append(dict(literal=color, route=routes.pop(), cells=mapped,
                              source_cells=cells))
        card = dict(card, match=matches[0], components=comps,
                    crop_left=min(c for x in comps for r, c in x["cells"]),
                    crop_top=min(r for x in comps for r, c in x["cells"]))
        registered.append(card)
    return registered


def _render(analysis, mode="independent"):
    a = analysis["canonical"]
    background, atlas = analysis["background"], analysis["atlas"]
    bands = {b["route"]: b for b in analysis["bands"]}
    out = a.copy()
    for card in analysis["cards"]:
        r, c = card["top"], card["left"]
        h, w = card["grid"].shape
        out[r:r+h, c:c+w] = background
    grouped_bottoms = {route: max(r for card in analysis["cards"] for comp in card["components"]
                                 if comp["route"] == route for r, c in comp["cells"])
                       for route in bands}
    for card in analysis["cards"]:
        for comp in card["components"]:
            band = bands[comp["route"]]
            bottom = (grouped_bottoms[comp["route"]] if mode == "grouped" else
                      max(r for r, c in comp["cells"]))
            offset = card["crop_left"] if mode == "unregistered" else 0
            for r, c in comp["cells"]:
                target = band["bottom"] - bottom + r, atlas["left"] + c - offset
                _require(band["top"] <= target[0] <= band["bottom"] and
                         band["left"] < target[1] < band["right"], "cargo crosses a rail or far boundary")
                out[target] = band["rail"]  # same-route overlaps are a union
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    corners = [int(a[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _require(len(set(corners)) == 1, "outer corners must reveal the background")
    background, candidates = corners[0], []
    for rotation in range(4):
        canonical = np.rot90(a, rotation).copy()
        for first, bands in _band_decompositions(canonical, background):
            for marker, atlas, cards in _frame_decompositions(canonical[:first], background):
                if set(map(int, np.unique(atlas["grid"]))) != {b["route"] for b in bands}:
                    continue
                if any(b["rail"] in (marker, background, b["route"]) for b in bands):
                    continue
                try:
                    cards = _register(atlas, cards)
                    analysis = dict(canonical=canonical, background=background, marker=marker,
                                    atlas=atlas, cards=cards, bands=bands, rotation=rotation,
                                    first_top=first)
                    expected = _render(analysis)
                except AssertionError:
                    continue
                analysis["output"] = np.rot90(expected, -rotation).copy()
                candidates.append(analysis)
    _require(len(candidates) == 1, "input must have one complete frame/band/crop interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse the visible atlas, locate each card, and dock its separate masks."""
    return _analyze(input_grid)["output"]


def _pack(rng, shapes, stage, left, right):
    """Bounded rectangle packing with a one-cell gap between closed rims."""
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0]*shapes[i][1])
    placed = {}
    trials = [0]

    def visit(depth):
        if depth == len(order):
            return True
        i = order[depth]
        h, w = shapes[i]
        positions = [(r, c) for r in range(1, stage-h)
                     for c in range(left, right-w+1)]
        for j in rng.permutation(len(positions)):
            trials[0] += 1
            if trials[0] > 2500:
                return False
            r, c = positions[int(j)]
            if any(not (r+h < y or y+hh < r or c+w < x or x+ww < c)
                   for y, x, hh, ww in placed.values()):
                continue
            placed[i] = (r, c, h, w)
            if visit(depth+1):
                return True
            del placed[i]
        return False

    return [placed[i] for i in range(len(shapes))] if visit(0) else None


def _layout_cards(rng, atlas, cards, depths):
    ah, aw = atlas.shape[0]+4, atlas.shape[1]+4
    shapes = [(p.shape[0]+2, p.shape[1]+2) for p in cards]
    minimum_width = aw + max(w for h, w in shapes) + 3
    _require(minimum_width <= 30, "frame widths exceed the canvas")
    gap0, gap1 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    max_stage = 30 - sum(depths) - gap0 - gap1 - 1
    min_stage = max(ah+2, max(h for h, w in shapes)+2)
    if max_stage < min_stage:
        return None
    widths = list(map(int, rng.permutation(np.arange(minimum_width, 31))))
    stages = list(map(int, rng.permutation(np.arange(min_stage, max_stage+1))))
    # Retry with the most spacious budget before discarding the sampled cards.
    configurations = [(widths[0], stages[0]), (30, max_stage)]
    for width, stage in configurations:
        side = int(rng.integers(2))
        atlas_left = 1 if side == 0 else width-aw-1
        lo, hi = (aw+2, width-1) if side == 0 else (1, width-aw-2)
        boxes = _pack(rng, shapes, stage, lo, hi)
        if boxes is not None:
            atlas_box = (int(rng.integers(1, stage-ah)), atlas_left, ah, aw)
            return width, stage, gap0, gap1, atlas_box, boxes
    return None


def _make_pair(rng, reference, index, split, turn):
    atlas_index = 0 if split == "train" and index == 0 else int(rng.integers(8))
    atlas = _transform(reference, atlas_index)
    bank = _crop_bank(atlas)
    if split == "test":
        eligible = [e for e in bank if e[2] <= 3 and e[2]*e[3] <= 13]
        count = 3 if index == 0 else int(rng.integers(1, 4))
    elif index == 0:
        eligible = [e for e in bank if e[4] and e[1] > 0 and e[2]*e[3] >= 20]
        count = 1
    elif index == 1:
        eligible = [e for e in bank if 3 <= e[2] <= 4 and 3 <= e[3] <= 5 and e[2]*e[3] >= 14]
        count = 2
    else:
        eligible = [e for e in bank if e[2]*e[3] >= 14]
        count = int(rng.integers(1, 3))
    _require(eligible, "reference lacks the requested distinctive crops")
    for attempt in range(60):
        selection = [eligible[int(j)] for j in rng.choice(len(eligible), count, replace=False)]
        transforms = ([0] if split == "train" and index == 0 else [1, 4]
                      if split == "train" and index == 1 else list(map(int, rng.integers(8, size=count))))
        patches = []
        min_depths = [1, 1]
        for (r, c, h, w, witness), trans in zip(selection, transforms):
            raw = atlas[r:r+h, c:c+w]
            patches.append(_transform(raw, trans))
            for role, cells in _components(raw):
                min_depths[role] = max(min_depths[role], 1+max(y for y, x in cells)-min(y for y, x in cells))
        depths = [d+int(rng.integers(2)) for d in min_depths] if attempt < 30 else min_depths
        layout = _layout_cards(rng, atlas, patches, depths)
        if layout is not None:
            break
    else:
        raise AssertionError("bounded card packing exhausted")
    width, stage, gap0, gap1, atlas_box, boxes = layout
    palette = list(map(int, rng.permutation(10)))
    background, marker = palette[:2]
    routes = palette[2:4]
    available = palette[2:]
    rails = [int(rng.choice([c for c in available if c != route])) for route in routes]
    height = stage + sum(depths) + gap0 + gap1 + 1
    inp = np.full((height, width), background, dtype=np.int8)
    ar, ac, ah, aw = atlas_box
    inp[ar:ar+ah, ac:ac+aw] = marker
    colored_atlas = np.where(atlas == 0, routes[0], routes[1]).astype(np.int8)
    inp[ar+2:ar+ah-2, ac+2:ac+aw-2] = colored_atlas
    construction_cards = []
    for j, (box, patch, crop, trans) in enumerate(zip(boxes, patches, selection, transforms)):
        r, c, h, w = box
        inp[r:r+h, c:c+w] = marker
        if split == "train" and index == 0:
            colors = routes
        elif split == "train" and index == 1 and j == 0:
            colors = routes[::-1]  # literal routing would select the wrong band
        elif split == "train" and index == 1 and j == 1:
            colors = list(map(int, rng.choice(palette[4:], 2, replace=False)))
        else:
            colors = list(map(int, rng.choice(available, 2, replace=False)))
        inp[r+1:r+h-1, c+1:c+w-1] = np.where(patch == 0, colors[0], colors[1])
        construction_cards.append((box, crop))
    # Receptacle order is independent of role number, crop shape and area.
    order = list(map(int, rng.permutation(2)))
    bands, cursor = {}, stage+gap0
    for j, role in enumerate(order):
        left = int(rng.integers(0, ac+2))
        right = int(rng.integers(ac+2+atlas.shape[1], width))
        depth, route, rail = depths[role], routes[role], rails[role]
        inp[cursor:cursor+depth, left:right+1] = route
        inp[cursor:cursor+depth, [left, right]] = rail
        bands[role] = (cursor, cursor+depth-1, rail)
        cursor += depth + (gap1 if j == 0 else 0)
    # Labels use the sampled canonical windows. solve instead rediscovers their
    # transforms, role maps and offsets by exhaustive input-only matching.
    out = inp.copy()
    for (r, c, h, w), (cr, cc, ch, cw, witness) in construction_cards:
        out[r+1:r+h-1, c+1:c+w-1] = background
        raw = atlas[cr:cr+ch, cc:cc+cw]
        for role, cells in _components(raw):
            bottom, rail = bands[role][1:]
            extent = max(y for y, x in cells)
            for y, x in cells:
                out[bottom-extent+y, ac+2+cc+x] = rail
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    reference = _reference(rng)
    turns = list(map(int, rng.permutation(4)))
    regime = int(rng.integers(3))  # fixed, mixed, held-out scene direction
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if regime == 0:
                turn = turns[0]
            elif regime == 2:
                turn = turns[index % 2] if split == "train" else turns[2+index % 2]
            else:
                turn = turns[index % 4] if split == "train" else int(rng.integers(4))
            for _ in range(12):
                pair = _make_pair(rng, reference, index, split, turn)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct-input reserve exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _safe_ablation(a, mode):
    try:
        return _render(a, mode)
    except AssertionError:
        return None  # an out-of-bounds alternative also fails


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal example counts")
    seen, refs, analyses = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "scene dimensions must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(a["output"], out), "incorrect docking output")
            _require(not np.array_equal(inp, out), "cargo must move")
            atlas = a["atlas"]["grid"]
            roles = sorted(map(int, np.unique(atlas)))
            refs.add(_canonical((atlas == roles[1]).astype(np.int8)))
            canonical_out = np.rot90(out, a["rotation"])
            changed = a["canonical"] != canonical_out
            allowed = np.zeros(a["canonical"].shape, dtype=bool)
            for card in a["cards"]:
                r, c = card["top"], card["left"]
                h, w = card["grid"].shape
                allowed[r:r+h, c:c+w] = True
                _require(np.all(canonical_out[r:r+h, c:c+w] == a["background"]), "card was not erased")
                area = h*w
                _require(area < atlas.size, "cards must be proper crops")
                _require(len(_components(card["grid"])) == len(_components(card["grid"], diagonal=True)),
                         "sampled cards must avoid diagonal-only same-color grouping ambiguity")
                _require(area >= 14 if split == "train" else area <= 13,
                         "crop-area teaching/shift constraint")
            for band in a["bands"]:
                allowed[band["top"]:band["bottom"]+1, band["left"]+1:band["right"]] = True
            _require(not np.any(changed & ~allowed), "atlas, rims, rails or separators were edited")
            analyses[split].append(a)
    _require(len(refs) == 1, "all examples must share one reference modulo D4/color exchange")
    first, second = analyses["train"][:2]
    _require(len(first["cards"]) == 1 and len(second["cards"]) == 2,
             "the first two lessons need one and two cargo cards")
    _require(len(analyses["test"][0]["cards"]) == 3, "first test must recombine three cards")
    for mode, message in (("grouped", "independent component alignment was not distinguished"),
                          ("unregistered", "hidden crop location was not distinguished")):
        correct = np.rot90(first["output"], first["rotation"])
        _require(not np.array_equal(correct, _safe_ablation(first, mode)), message)
    _require({c["match"][0] % 4 for c in second["cards"]} & {1, 3},
             "second lesson needs an axis-exchanging crop")
    _require(any(c["match"][0] >= 4 for c in second["cards"]), "reflection lesson missing")
    _require(any(comp["literal"] != comp["route"] for c in second["cards"] for comp in c["components"]),
             "literal-color routing shortcut was not contradicted")
    train_rotations = {a["rotation"] for a in analyses["train"]}
    if {a["rotation"] for a in analyses["test"]} - train_rotations:
        _require(len(train_rotations) >= 2, "direction holdout needs varying demonstrations")


def _signature(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed coverage plus count, corruption and ablation checks."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    _require(_signature(corpus[0]) == _signature(generate_puzzle(0)), "nondeterminism")
    refs, dimensions, shapes, masks, locations, transforms, backgrounds = (set() for _ in range(7))
    cards_seen, overlaps, rail_reuse, shared_rails, inset_bands = 0, 0, 0, 0, 0
    counts, directions = set(), set()
    for puzzle in corpus:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                atlas = a["atlas"]["grid"]
                roles = sorted(map(int, np.unique(atlas)))
                refs.add(_canonical((atlas == roles[1]).astype(np.int8)))
                dimensions.add(tuple(sorted(atlas.shape)))
                shapes.add(pair["input"].shape)
                backgrounds.add(a["background"])
                counts.add(len(a["cards"]))
                directions.add(a["rotation"])
                targets = set()
                lookup = {b["route"]: b for b in a["bands"]}
                shared_rails += len({b["rail"] for b in a["bands"]}) == 1
                rail_reuse += any(b["rail"] in lookup for b in a["bands"])
                inset_bands += any(b["left"] > 0 or b["right"] < a["canonical"].shape[1]-1 for b in a["bands"])
                for card in a["cards"]:
                    cards_seen += 1
                    transforms.add(card["match"][0])
                    locations.add((card["crop_top"], card["crop_left"]))
                    for comp in card["components"]:
                        rr = min(r for r, c in comp["cells"])
                        cc = min(c for r, c in comp["cells"])
                        masks.add(tuple((r-rr, c-cc) for r, c in comp["cells"]))
                        bottom = max(r for r, c in comp["cells"])
                        dest = {(lookup[comp["route"]]["bottom"]-bottom+r, a["atlas"]["left"]+c)
                                for r, c in comp["cells"]}
                        overlaps += bool(dest & targets)
                        targets.update(dest)
    _require(len(refs) >= 40, "references are insufficiently diverse after D4/color normalization")
    _require(len(dimensions) >= 7 and len(shapes) >= 30, "insufficient geometry variation")
    _require(len(masks) >= 80 and len(locations) >= 20, "insufficient masks or crop locations")
    _require(transforms == set(range(8)) and directions == set(range(4)), "missing transform or direction")
    _require(backgrounds == set(range(10)) and counts == {1, 2, 3}, "missing palette/count coverage")
    _require(overlaps and shared_rails and rail_reuse and inset_bands, "compatible structural axes were not exercised")
    # The host verifier exercises every legal count combination. Keep this
    # required self-test bounded by adding just the three capacity boundaries.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(700+nt*10+nq, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "requested counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in corpus[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Removing information must not invoke a default geometry or location.
    # A two-color 2x2 patch has unresolved D4 symmetry; one-color data is invalid.
    atlas = _analyze(corpus[0]["train"][0]["input"])["atlas"]["grid"]
    _require(not _matches(atlas, np.full((1, 1), 7, dtype=np.int8)), "one-color crop accepted")
    _require(any(len(_matches(atlas, atlas[r:r+2, c:c+2])) > 1
                 for r in range(atlas.shape[0]-1) for c in range(atlas.shape[1]-1)),
             "ambiguous evidence was accepted")
    return dict(seeds=42, references=len(refs), reference_dimensions=len(dimensions),
                canvas_shapes=len(shapes), component_masks=len(masks), crop_locations=len(locations),
                cards=cards_seen, overlapping_component_stamps=overlaps,
                shared_rail_examples=shared_rails, rail_reuse_examples=rail_reuse,
                inset_band_examples=inset_bands)
