"""Ranked recess closing: jigsaw geometry creates implicit capped bars.

No parent code, fixtures, or generation state are consulted by solve().
"""
from __future__ import annotations

import itertools
import math
import numpy as np

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts: 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid grid colors")
    return grid.astype(np.int8, copy=False)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRECTIONS:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _crop(mask):
    coords = np.argwhere(mask)
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy()


def _mask_bits(mask, side, row=0, col=0):
    return sum(1 << ((int(r) + row) * side + int(c) + col) for r, c in np.argwhere(mask))


def _tilings(shapes, side, quarter_turns=range(4)):
    """Enumerate distinct corner ownership maps, collapsing shape symmetries."""
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    if not shapes[0][0, 0] or any(n > side for n in shapes[0].shape):
        return []
    anchor = _mask_bits(shapes[0], side)
    if sum(bool(anchor & (1 << p)) for p in corners) != 1:
        return []
    table = {}
    for piece in range(1, 4):
        for corner in range(1, 4):
            candidates = {}
            for turn in quarter_turns:
                mask = np.rot90(shapes[piece], turn)
                h, w = mask.shape
                if max(h, w) > side:
                    continue
                r, c = (0 if corner < 2 else side - h), (0 if corner % 2 == 0 else side - w)
                bits = _mask_bits(mask, side, r, c)
                if sum(bool(bits & (1 << p)) for p in corners) == 1 and bits & (1 << corners[corner]):
                    candidates.setdefault(bits, turn)
            table[piece, corner] = list(candidates.items())
    full = (1 << (side * side)) - 1
    found = {}

    def visit(corner, remaining, occupied, masks, assignments, turns):
        if len(found) >= 2:
            return
        if corner == 4:
            if occupied == full:
                found.setdefault(tuple(masks), (tuple(assignments), tuple(turns)))
            return
        for piece in remaining:
            for bits, turn in table[piece, corner]:
                if not bits & occupied:
                    visit(corner + 1, tuple(p for p in remaining if p != piece), occupied | bits,
                          masks + [bits], assignments + [piece], turns + [turn])

    visit(1, (1, 2, 3), anchor, [anchor], [0], [0])
    result = []
    for masks, (assignment, turns) in found.items():
        owner = np.empty((side, side), dtype=np.int8)
        for corner, bits in enumerate(masks):
            for cell in range(side * side):
                if bits & (1 << cell):
                    owner.flat[cell] = corner
        result.append((owner, assignment, turns))
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) / 2, "background must dominate border")
    components = _components(grid != bg)
    mixed = [p for p in components if len(np.unique(grid[p[:, 0], p[:, 1]])) > 1]
    _require(len(mixed) == 1, "one keyed anchor required")
    anchor = mixed[0]
    colors, counts = np.unique(grid[anchor[:, 0], anchor[:, 1]], return_counts=True)
    _require(len(colors) == 5 and sum(counts == 1) == 4 and sum(counts > 1) == 1,
             "anchor needs four singleton key colors and one carrier")
    key_colors = colors[counts == 1]
    points = anchor[np.isin(grid[anchor[:, 0], anchor[:, 1]], key_colors)]
    r, c = points.min(axis=0)
    _require(set(map(tuple, points)) == {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)},
             "corner legend must be 2 by 2")
    key = grid[r:r + 2, c:c + 2].copy().reshape(-1)
    pieces, ranks, tokens = [anchor], {}, {}
    for comp in components:
        if np.array_equal(comp, anchor):
            continue
        value = int(grid[comp[0, 0], comp[0, 1]])
        _require(np.all(grid[comp[:, 0], comp[:, 1]] == value), "plain components must be uniform")
        if value in key:
            _require(value not in ranks, "one unary token per active family")
            low, high = comp.min(axis=0), comp.max(axis=0)
            _require((low[0] == high[0] or low[1] == high[1])
                     and len(comp) == int(np.sum(high - low)) + 1, "unary token must be a straight run")
            ranks[value], tokens[value] = len(comp), comp
        else:
            pieces.append(comp)
    _require(len(pieces) == 4 and 1 <= len(ranks) <= 4, "four tiles and at least one instruction required")
    for a, b in itertools.combinations(components, 2):
        _require(np.max(np.abs(a[:, None] - b[None, :]), axis=2).min() >= 2,
                 "components need one background cell of separation")
    shapes = []
    for coords in pieces:
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        shapes.append(_crop(mask))
    total = sum(int(mask.sum()) for mask in shapes)
    side = math.isqrt(total)
    _require(side * side == total, "tile area must be square")
    tilings = _tilings(shapes, side)
    _require(len(tilings) == 1, "a unique corner tiling is required")
    owner, assignment, turns = tilings[0]
    return {"grid": grid, "background": bg, "key": key, "ranks": ranks, "tokens": tokens,
            "pieces": pieces, "shapes": shapes, "owner": owner, "assignment": assignment, "turns": turns}


def _recesses(board):
    """Maximal uniform horizontal payload runs with equal immediate caps."""
    bars = {}
    for row, values in enumerate(board):
        boundaries = [0] + (np.flatnonzero(values[1:] != values[:-1]) + 1).tolist() + [len(values)]
        for a, b in zip(boundaries, boundaries[1:]):
            if a > 0 and b < len(values) and values[a - 1] == values[b]:
                cap = int(values[a - 1])
                bars.setdefault(cap, []).append((row, a, b, b - a + 2))
    return bars


def _close(board, ranks):
    output = board.copy()
    bars = _recesses(board)
    selected = {}
    for cap, rank in ranks.items():
        _require(cap in bars, "active color has no enclosed recess")
        lengths = sorted({bar[3] for bar in bars[cap]}, reverse=True)
        _require(1 <= rank <= len(lengths), "requested distinct-length tier does not exist")
        selected[cap] = [bar for bar in bars[cap] if bar[3] == lengths[rank - 1]]
        for row, a, b, _ in selected[cap]:
            output[row, a:b] = cap
    return output, bars, selected


def solve(input_grid: np.ndarray) -> np.ndarray:
    analysis = _parse(input_grid)
    board = analysis["key"][analysis["owner"]]
    return np.ascontiguousarray(_close(board, analysis["ranks"])[0], dtype=np.int8)


def _pocket_partition(rng, side, tiers, repeated_ranks):
    """Two opposing pocket pieces, surrounded by connected return arms.

    This constructs geometry, not outputs from either parent generator.
    Optional cell exchanges perturb the central join without touching pockets.
    """
    split = int(rng.integers(7, side - 6))
    owner = np.empty((side, side), dtype=np.int8)
    owner[:split] = 0
    owner[split:] = 3
    for band, height in enumerate((split, side - split)):
        available = np.arange(2, side - 5)
        count = min(tiers, height - 3, len(available))
        widths = sorted(map(int, rng.choice(available, count, replace=False)), reverse=True)
        values = widths.copy()
        while len(values) < height - 3:
            chosen = repeated_ranks[band] - 1 if len(values) == count else int(rng.integers(count))
            values.append(widths[min(chosen, count - 1)])
        rng.shuffle(values)
        root = max(widths) + int(rng.integers(1, 3))
        # Work in a local top-right-rooted frame; the lower pocket is a half turn.
        local = np.zeros((height, side), dtype=bool)
        local[:2, side - root:] = True
        for row, width in enumerate(values, 2):
            end = side - 1 - int(rng.integers(1, 3))
            local[row, end - width + 1:end + 1] = True
        if band == 0:
            owner[:split][local] = 1
        else:
            owner[split:][np.rot90(local, 2)] = 2
    # Small tab exchanges vary how the two enclosing pieces interlock.
    # Boundary-attached tabs introduce no new enclosed length tier. Thus the
    # competitor-count test shift is visible rather than merely an RNG intent.
    if rng.integers(2):
        owner[split - 1, :int(rng.integers(1, 4))] = 3
    if rng.integers(2):
        owner[split, -int(rng.integers(1, 4)):] = 0
    if rng.integers(2):
        owner = np.fliplr(owner)
    if rng.integers(2):
        owner = np.flipud(owner)
    # Reassign labels by the actual output corners, never by construction order.
    corner_values = [int(owner[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    mapping = np.empty(4, dtype=np.int8)
    for corner, value in enumerate(corner_values):
        mapping[value] = corner
    return mapping[owner]


def _key_locations(mask):
    return [(r, c) for r in range(mask.shape[0] - 1) for c in range(mask.shape[1] - 1)
            if mask[r:r + 2, c:c + 2].all()]


def _scatter(rng, tiles, token_grid, bg):
    """Any ordered shelf grouping, with independent tile order and unary-key side."""
    layouts = []
    gap = int(rng.integers(1, 3))
    for transpose in (False, True):
        dimensions = [t.shape[::-1] if transpose else t.shape for t in tiles]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, piece in enumerate(order):
                    group.append(piece)
                    if j == 3 or cuts & (1 << j):
                        groups.append(group)
                        group = []
                heights = [max(dimensions[i][0] for i in g) for g in groups]
                widths = [sum(dimensions[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
                h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
                h, w = (w, h) if transpose else (h, w)
                kh, kw = token_grid.shape
                for key_side in range(4):
                    size = (h + kh + gap, max(w, kw)) if key_side < 2 else (max(h, kh), w + kw + gap)
                    if max(size) <= 28:
                        layouts.append((transpose, groups, dimensions, heights, widths, (h, w), key_side, size))
    if not layouts:
        return None
    layout = layouts[int(rng.integers(len(layouts)))]
    transpose, groups, dimensions, heights, widths, (h, w), key_side, size = layout
    offsets, row = {}, 0
    axis_width = h if transpose else w
    for group, gh, gw in zip(groups, heights, widths):
        col = int(rng.integers(axis_width - gw + 1))
        for i in group:
            rr = row + int(rng.integers(gh - dimensions[i][0] + 1))
            offsets[i] = (col, rr) if transpose else (rr, col)
            col += dimensions[i][1] + gap
        row += gh + gap
    kh, kw = token_grid.shape
    if key_side == 0:
        tile_shift, key_offset = (kh + gap, 0), (0, int(rng.integers(size[1] - kw + 1)))
    elif key_side == 1:
        tile_shift, key_offset = (0, 0), (h + gap, int(rng.integers(size[1] - kw + 1)))
    elif key_side == 2:
        tile_shift, key_offset = (0, kw + gap), (int(rng.integers(size[0] - kh + 1)), 0)
    else:
        tile_shift, key_offset = (0, 0), (int(rng.integers(size[0] - kh + 1)), w + gap)
    grid = np.full(size, bg, dtype=np.int8)
    for i, tile in enumerate(tiles):
        r, c = np.add(offsets[i], tile_shift)
        view = grid[r:r + tile.shape[0], c:c + tile.shape[1]]
        view[tile != bg] = tile[tile != bg]
    r, c = key_offset
    grid[r:r + kh, c:c + kw] = token_grid
    pads = []
    for length in grid.shape:
        room = 30 - length
        before = int(rng.integers(1, min(3, room)))
        after = int(rng.integers(1, min(3, room - before) + 1))
        pads.append((before, after))
    return np.pad(grid, pads, constant_values=bg)


def _make_scene(rng, side, tiers, teaching=False):
    for _ in range(160):
        owner = _pocket_partition(rng, side, tiers, (1, 2))
        shapes = [_crop(owner == i) for i in range(4)]
        if any(len(_components(owner == i)) != 1 for i in range(4)) or not _key_locations(shapes[0]):
            continue
        answers = _tilings(shapes, side)
        if len(answers) != 1 or not np.array_equal(answers[0][0], owner):
            continue
        bars = _recesses(owner)
        active = sorted(c for c, bs in bars.items() if len({b[3] for b in bs}) >= 3)
        if len(active) < 2:
            continue
        # Select the two enclosing families; exchange-created extra gaps are also
        # ranked if they belong to one of these families.
        active = sorted(active, key=lambda c: -len(bars[c]))[:2]
        if any(len({b[3] for b in bars[c]}) < tiers for c in active):
            continue
        key = rng.choice(10, 4, replace=False).astype(np.int8)
        remaining = [c for c in range(10) if c not in key]
        bg = int(rng.choice(remaining))
        carrier_pool = [c for c in remaining if c != bg]
        rng.shuffle(carrier_pool)
        carrier_count = int(rng.integers(1, 5))
        carriers = [carrier_pool[i % carrier_count] for i in rng.permutation(4)]
        turns = [0] + list(map(int, rng.permutation([1, 2, 3]) if teaching else rng.integers(0, 4, 3)))
        tiles = []
        key_r, key_c = _key_locations(shapes[0])[int(rng.integers(len(_key_locations(shapes[0]))))]
        for i, mask in enumerate(shapes):
            tile = np.full(mask.shape, bg, dtype=np.int8)
            tile[mask] = carriers[i]
            if i == 0:
                tile[key_r:key_r + 2, key_c:key_c + 2] = key.reshape(2, 2)
            tiles.append(np.rot90(tile, turns[i]).copy())
        if teaching and _tilings([tile != bg for tile in tiles], side, (0,)):
            continue
        # A provisional token grid reserves enough room for any rank up to four.
        reserve = np.full((4, 3), bg, dtype=np.int8)
        grid = _scatter(rng, tiles, reserve, bg)
        if grid is None:
            continue
        # Find a compact token site in the returned empty background, separated
        # from tiles; this frees tokens from any absolute corner convention.
        sites = []
        occupied = grid != bg
        for r in range(1, grid.shape[0] - 4):
            for c in range(1, grid.shape[1] - 3):
                if not occupied[r - 1:r + 5, c - 1:c + 4].any():
                    sites.append((r, c))
        if not sites:
            continue
        site = sites[int(rng.integers(len(sites)))]
        return {"grid": grid, "owner": owner, "key": key, "background": bg,
                "active": active, "site": site}
    raise RuntimeError("bounded geometry/scattering search exhausted")


def _pair_from_scene(scene, ranks):
    grid = scene["grid"].copy()
    r, c = scene["site"]
    for i, (corner, rank) in enumerate(ranks.items()):
        grid[r:r + rank, c + 2 * i] = scene["key"][corner]
    # Labels originate from the constructed partition, whereas validation packs
    # actual scattered pixels afresh. Selection is simultaneous on this board.
    board = scene["key"][scene["owner"]]
    color_ranks = {int(scene["key"][corner]): rank for corner, rank in ranks.items()}
    output = _close(board, color_ranks)[0]
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # new ordinal, more competitors, or larger square
    side = int(rng.integers(14, 17))
    contrast = _make_scene(rng, side, 3, teaching=True)
    a, b = contrast["active"]
    teaching_pairs = [_pair_from_scene(contrast, {a: 1, b: 2}),
                      _pair_from_scene(contrast, {a: 2, b: 1})]
    puzzle = {"train": teaching_pairs, "test": []}
    seen = {_signature(p["input"]) for p in teaching_pairs}
    seen_cuts = {_signature(contrast["owner"])}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for attempt in range(40):
                target = split == "test"
                n = int(rng.integers(17, 19)) if target and shift == 2 else int(rng.integers(14, 17))
                tiers = 4 if target and shift == 1 else 3
                # Four tiers require at least four enclosed rows per band.
                scene = _make_scene(rng, n, tiers)
                cut_signature = _signature(scene["owner"])
                if cut_signature in seen_cuts:
                    continue
                active = scene["active"].copy()
                if rng.random() < 0.35:
                    active = [active[int(rng.integers(2))]]
                ranks = {c: int(rng.integers(1, 3)) for c in active}
                if target and shift == 0:
                    ranks[active[0]] = 3
                pair = _pair_from_scene(scene, ranks)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    seen_cuts.add(cut_signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input search exhausted")
    # Vary teaching contrast positions while preserving its controlled evidence.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    a = _parse(pair["input"])
    board = a["key"][a["owner"]]
    expected, bars, selected = _close(board, a["ranks"])
    _require(np.array_equal(expected, pair["output"]), "output disagrees with input geometry and rank")
    return {"side": len(board), "shape": list(a["grid"].shape), "background": a["background"],
            "areas": [int(np.sum(a["owner"] == c)) for c in range(4)],
            "active_corners": [int(np.flatnonzero(a["key"] == c)[0]) for c in a["ranks"]],
            "ranks": list(a["ranks"].values()),
            "populations": [len(bars[c]) for c in a["ranks"]],
            "tiers": [len({b[3] for b in bars[c]}) for c in a["ranks"]],
            "ties_selected": [len(selected[c]) for c in a["ranks"]],
            "turns": list(a["turns"]), "changed": int(np.count_nonzero(board != expected)),
            "cut": a["owner"].tobytes().hex()}


def _edge_shortcut(board, ranks):
    """Counterfactual: admit edge-open strips with only one cap as bars."""
    bars = _recesses(board)
    for row, values in enumerate(board):
        boundaries = [0] + (np.flatnonzero(values[1:] != values[:-1]) + 1).tolist() + [len(values)]
        if len(boundaries) > 2:
            a, b = boundaries[0], boundaries[1]
            bars.setdefault(int(values[b]), []).append((row, a, b, b - a + 2))
            a, b = boundaries[-2], boundaries[-1]
            bars.setdefault(int(values[a - 1]), []).append((row, a, b, b - a + 2))
    alternative = board.copy()
    for color, rank in ranks.items():
        lengths = sorted({b[3] for b in bars[color]}, reverse=True)
        for row, a, b, length in bars[color]:
            if length == lengths[rank - 1]:
                alternative[row, a:b] = color
    return alternative


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            m = measure_pair({"input": inp, "output": out})
            _require(m["changed"] > 0, "rank instruction must change the plain assembly")
            _require(len(np.unique(out)) == 4, "all four corner colors must remain visible")
            if split == "train":
                training.append(m)
    demonstrated = {r for m in training for r in m["ranks"]}
    _require({1, 2} <= demonstrated, "training needs ranks one and two")
    _require(any(1 < r < t for m in training for r, t in zip(m["ranks"], m["tiers"])),
             "training needs a nonextreme selection")
    _require(any(n > 1 for m in training for n in m["ties_selected"]), "training must show tied bars")
    # These are training-evidence requirements, not additional scene rules.
    _require(any(not np.array_equal(_edge_shortcut(a["key"][a["owner"]], a["ranks"]), p["output"])
                 for p in puzzle["train"] for a in [_parse(p["input"])]),
             "training must distinguish shape boundaries from one-capped raw spans")


def self_test():
    """Bounded 42-seed structural, geometric, ablation, and metamorphic checks."""
    measurements, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        measurements.extend(measure_pair(p) for s in ("train", "test") for p in puzzle[s])
    _require(len(signatures) == 42, "cross-seed puzzles must differ")
    _require(len({m["cut"] for m in measurements}) >= 100, "partition geometry lacks variation")
    _require(len({tuple(m["shape"]) for m in measurements}) >= 25, "scatter dimensions lack variation")
    _require({14, 15, 16, 17, 18} <= {m["side"] for m in measurements}, "scale coverage missing")
    _require({0, 1, 2, 3} == {c for m in measurements for c in m["active_corners"]}, "corner roles fixed")
    _require({1, 2, 3} <= {r for m in measurements for r in m["ranks"]}, "ordinal progression missing")
    _require({0, 1, 2, 3} == {t for m in measurements for t in m["turns"]}, "rotation coverage missing")
    _require(len({tuple(m["areas"]) for m in measurements}) >= 80, "area profiles lack variation")
    _require(any(t >= 4 for m in measurements for t in m["tiers"]), "competitor-tier shift missing")
    # Recoloring and translation operate on input alone and commute with solve.
    puzzle = generate_puzzle(0, 2, 1)
    repeat = generate_puzzle(0, 2, 1)
    for split in puzzle:
        for p, q in zip(puzzle[split], repeat[split]):
            _require(all(np.array_equal(p[k], q[k]) for k in p), "determinism failed")
    for pair in puzzle["train"]:
        a = _parse(pair["input"])
        plain = a["key"][a["owner"]]
        _require(not np.array_equal(plain, pair["output"]), "omitting ordinal selection must change answer")
        _require(not np.array_equal(_edge_shortcut(plain, a["ranks"]), pair["output"]),
                 "omitting assembled two-sided enclosure must change the teaching answer")
        _require(not _tilings(a["shapes"], len(plain), (0,)),
                 "omitting B's quarter-turn assembly must prevent the teaching reconstruction")
        permutation = np.array([8, 3, 5, 0, 9, 7, 2, 6, 4, 1], dtype=np.int8)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "literal colors must not control solving")
        inp = pair["input"]
        shifted = np.pad(inp[1:-1, 1:-1], ((0, 2), (2, 0)), constant_values=a["background"])
        _require(np.array_equal(solve(shifted), pair["output"]), "scatter translation changes answer")
        # The constructor uses vertical tokens; the rule and parser also allow
        # horizontal tokens. Reorient only these instructions, not the anchor.
        horizontal = inp.copy()
        r = min(int(p[:, 0].min()) for p in a["tokens"].values())
        c = min(int(p[:, 1].min()) for p in a["tokens"].values())
        for p in a["tokens"].values():
            horizontal[p[:, 0], p[:, 1]] = a["background"]
        for i, (color, rank) in enumerate(a["ranks"].items()):
            horizontal[r + 2 * i, c:c + rank] = color
        _require(np.array_equal(solve(horizontal), pair["output"]), "horizontal unary instructions failed")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            _counts(train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
