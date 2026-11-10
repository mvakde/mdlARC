"""Contour loom: binary microtiles interleave adjacent macro-contour phases.

All output controls are visible in three freely positioned, framed cards.
No parent module, example, seed, or generation metadata is used by solve.
"""
from collections import deque
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ARMS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        part = np.zeros(mask.shape, bool)
        queue = [(sr, sc)]
        seen[sr, sc] = True
        for r, c in queue:
            part[r, c] = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w
                        and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(part)
    return result


def _depth(mask):
    """Solver's eight-neighbor boundary BFS; outside the board is exterior."""
    h, w = mask.shape
    d = np.full(mask.shape, -1, int)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in DIR8):
            d[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and d[nr, nc] < 0):
                d[nr, nc] = d[r, c] + 1
                queue.append((nr, nc))
    return d


def _erosion(mask, diagonal=True):
    """Construction/validation use iterative erosion rather than solver BFS."""
    cur = mask.copy()
    d = np.full(mask.shape, -1, int)
    h, w = mask.shape
    level = 0
    while cur.any():
        pad = np.pad(cur, 1)
        inside = cur.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            inside &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        d[cur & ~inside] = level
        cur = inside
        level += 1
    return d


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]


def _paths(mask, depth):
    paths = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                        or not all(_inside(mask, rr, cc) for rr, cc in
                                   ((r + dr, c), (r, c + dc), (r + dr, c + dc)))):
                    continue
                length = 0
                while (_inside(mask, r + length * dr, c + length * dc)
                       and depth[r + length * dr, c + length * dc] == length):
                    length += 1
                paths.append((r, c, dr, dc, length))
    return paths


def _cards(grid):
    """Infer framing from its geometry, with no predetermined card coordinates."""
    background = int(grid[0, 0])
    assert (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)), "missing exterior margin"
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        parts = _components(grid == color)
        if len(parts) != 3:
            continue
        boxes = []
        for part in parts:
            rr, cc = np.nonzero(part)
            r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
            if r1 - r0 < 3 or c1 - c0 < 3:
                break
            border = np.zeros(part.shape, bool)
            border[r0:r1 + 1, (c0, c1)] = True
            border[(r0, r1), c0:c1 + 1] = True
            if not np.array_equal(part, border):
                break
            boxes.append((r0, r1, c0, c1))
        if len(boxes) == 3:
            candidates.append((color, boxes))
    assert len(candidates) == 1, "three disjoint rectangular frames must identify one frame color"
    frame, boxes = candidates[0]
    covered = np.zeros(grid.shape, bool)
    cards = []
    for r0, r1, c0, c1 in boxes:
        assert not covered[max(0, r0 - 1):r1 + 2, max(0, c0 - 1):c1 + 2].any(), "cards need a blank gap"
        covered[r0:r1 + 1, c0:c1 + 1] = True
        cards.append(grid[r0 + 1:r1, c0 + 1:c1].copy())
    assert np.all(grid[~covered] == background), "unframed marks"
    return background, frame, cards, boxes


def _decode(input_grid, depth_function=_depth):
    grid = _grid(input_grid)
    background, frame, cards, boxes = _cards(grid)
    controls, tiles = [], []
    for i, card in enumerate(cards):
        if card.shape == (3, 3):
            pts = set(map(tuple, np.argwhere(card != background)))
            vals = np.unique(card[card != background])
            if len(vals) == 1:
                for turn, arm in enumerate(ARMS):
                    if pts == {(1, 1), arm}:
                        controls.append((i, turn))
        if (card.shape[0] == card.shape[1] and 2 <= len(card) <= 5
                and background not in card and len(np.unique(card)) == 2):
            tiles.append(i)
    assert len(controls) == len(tiles) == 1, "one arrow card and one complete binary tile required"
    ci, turn = controls[0]
    ti = tiles[0]
    assert ci != ti
    si = next(i for i in range(3) if i not in (ci, ti))
    tile, stencil = cards[ti], cards[si]
    occupied = stencil != background
    assert occupied.any() and all((occupied[0].any(), occupied[-1].any(), occupied[:, 0].any(), occupied[:, -1].any())), "stencil support must be tight"
    colors, counts = np.unique(stencil[occupied], return_counts=True)
    assert len(colors) >= 3
    carrier = int(colors[np.argmax(counts)])
    assert np.count_nonzero(counts == counts.max()) == 1, "carrier color needs a strict plurality"
    assert carrier not in tile and frame not in stencil
    regions = []
    for mask in _components(occupied):
        depth = depth_function(mask)
        cues = np.argwhere(mask & (stencil != carrier))
        assert len(cues) >= 2, "each component needs a diagonal key"
        length = int(max(depth[r, c] for r, c in cues)) + 1
        assert 2 <= length <= 4
        readings = []
        for r, c, dr, dc, capacity in _paths(mask, depth):
            if capacity < length:
                continue
            if all((int(rr), int(cc)) == (r + int(depth[rr, cc]) * dr,
                                          c + int(depth[rr, cc]) * dc) for rr, cc in cues):
                p = tuple(int(stencil[r + j * dr, c + j * dc]) for j in range(length))
                readings.append((p, (dr, dc)))
        assert readings and len({p for p, _ in readings}) == 1, "ambiguous diagonal period"
        period = readings[0][0]
        assert set(period[:2]) == set(map(int, np.unique(tile)))
        assert period[-1] != carrier and background not in period and frame not in period
        regions.append(dict(mask=mask, depth=depth, period=period,
                            directions=tuple(sorted({drdc for _, drdc in readings}))))
    assert 1 <= len(regions) <= 4
    n = len(tile)
    shape = stencil.shape[::-1] if turn % 2 else stencil.shape
    assert max(n * x for x in shape) <= 30
    return dict(background=background, frame=frame, carrier=carrier, tile=tile,
                stencil=stencil, occupied=occupied, turn=turn, regions=regions,
                boxes=boxes, stencil_shape=stencil.shape)


def _render(info, behavior="normal", turn_override=None):
    """Forward solver; variants measure specific alternative-rule witnesses."""
    tile, regions = info["tile"], info["regions"]
    n = len(tile)
    turn = info["turn"] if turn_override is None else turn_override
    shape = info["stencil_shape"][::-1] if turn % 2 else info["stencil_shape"]
    out = np.full(tuple(n * x for x in shape), info["background"], int)
    for region in regions:
        mask, depth = region["mask"], region["depth"]
        if behavior == "flat_depth":
            depth = np.zeros(mask.shape, int)
        elif behavior == "manhattan":
            depth = _erosion(mask, False)
        elif behavior == "bounding_box":
            rr, cc = np.nonzero(mask)
            rows, cols = np.indices(mask.shape)
            depth = np.minimum.reduce((rows - rr.min(), rr.max() - rows,
                                       cols - cc.min(), cc.max() - cols))
        p = np.asarray(region["period"], int)
        if behavior == "skip_carrier":
            p = p[p != info["carrier"]]
        # The key fixes the order of the tile's two roles, locally per region.
        phase = (tile == p[0]).astype(int)  # exchange: role 0 becomes phase 1
        if behavior == "no_exchange":
            phase = 1 - phase
        elif behavior == "uniform_blocks":
            phase = np.zeros(tile.shape, int)
        elif behavior == "rotate_tile":
            phase = np.rot90(phase, -turn)
        rotated_mask = np.rot90(mask, -turn)
        rotated_depth = np.rot90(depth, -turn)
        if behavior == "fine_depth":
            expanded = np.repeat(np.repeat(rotated_mask, n, axis=0), n, axis=1)
            fine = _erosion(expanded)
            for y, x in np.argwhere(expanded):
                out[y, x] = p[(int(fine[y, x]) + int(phase[y % n, x % n])) % len(p)]
            continue
        for r, c in np.argwhere(rotated_mask):
            d = int(rotated_depth[r, c])
            block = p[(d + phase) % len(p)]
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible cards and weave each component's adjacent phases."""
    return _render(_decode(input_grid))


def _inverse_output(info):
    """Scalar inverse coordinates, separately checking the solver's block paint."""
    n, k = len(info["tile"]), info["turn"]
    h, w = info["stencil_shape"]
    rh, rw = (w, h) if k % 2 else (h, w)
    labels = np.full((h, w), -1, int)
    independent_depths = []
    for i, region in enumerate(info["regions"]):
        labels[region["mask"]] = i
        independent_depths.append(_erosion(region["mask"]))
    out = np.full((rh * n, rw * n), info["background"], int)
    for y in range(len(out)):
        for x in range(out.shape[1]):
            mr, mc = y // n, x // n
            if k == 0:
                sr, sc = mr, mc
            elif k == 1:
                sr, sc = h - 1 - mc, mr
            elif k == 2:
                sr, sc = h - 1 - mr, w - 1 - mc
            else:
                sr, sc = mc, w - 1 - mr
            label = labels[sr, sc]
            if label < 0:
                continue
            period = info["regions"][label]["period"]
            original_color = int(info["tile"][y % n, x % n])
            offset = 1 if original_color == period[0] else 0
            d = int(independent_depths[label][sr, sc])
            out[y, x] = period[(d + offset) % len(period)]
    return out


def _tile(rng, n):
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.22, .78)
        if bits.any() and not bits.all() and all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            return bits
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return bits


def _shape(rng, h, w, length, witness=False):
    """Carve orthogonal notches/holes from rectangles; retain readable depth keys."""
    for attempt in range(60):
        mask = np.ones((h, w), bool)
        style = int(rng.integers(5))
        if witness or style in (1, 2, 3):
            work = mask.T if rng.integers(2) else mask
            hh, ww = work.shape
            if hh >= 5 and ww >= 5:
                start = int(rng.integers(1, hh - 2))
                end = min(hh - 1, start + int(rng.integers(1, min(5, hh - start))))
                cut = int(rng.integers(1, min(3, ww - 2)))
                work[start:end, -cut:] = False
        if style == 2 and min(h, w) >= 5:
            # Near-corner holes retain deep diagonals elsewhere in the region.
            rr = int(rng.choice([1, h - 2]))
            cc = int(rng.choice([1, w - 2]))
            mask[rr, cc] = False
        if not witness and style == 3 and min(h, w) >= 5:
            ah, aw = int(rng.integers(1, h // 2)), int(rng.integers(1, w // 2))
            mask[:ah, :aw] = False
        if not witness and style == 4 and min(h, w) >= 5:
            # Offset opposing notches form a thick bending corridor.
            mask[1:max(2, h // 2), -1] = False
            mask[max(2, h // 2):-1, 0] = False
        if rng.integers(2):
            mask = mask[::-1].copy()
        if rng.integers(2):
            mask = mask[:, ::-1].copy()
        if len(_components(mask)) != 1:
            continue
        depth = _erosion(mask)
        paths = [p for p in _paths(mask, depth) if p[4] >= length]
        if not paths or (witness and depth.max() < length):
            continue
        if witness:
            rr, cc = np.nonzero(mask)
            box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
            if (not np.any(depth[mask] != box)
                    or not np.any(depth[mask] != _erosion(mask, False)[mask])):
                continue
        return mask
    # This notch retains an untouched corner diagonal and a deep interior.
    mask = np.ones((h, w), bool)
    if witness:
        mask[1:2, -1] = False
    assert max(p[4] for p in _paths(mask, _erosion(mask))) >= length
    assert not witness or _erosion(mask).max() >= length
    return mask


def _rectangles(rng, h, w, count):
    """Recursive rectangular allocation; gaps may be one or two macro cells."""
    rooms = [(0, h, 0, w)]
    while len(rooms) < count:
        options = []
        for i, (a, b, c, d) in enumerate(rooms):
            for axis, extent in ((0, b - a), (1, d - c)):
                if extent >= 7:
                    options.append((i, axis))
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        extent = b - a if axis == 0 else d - c
        gap = int(rng.integers(1, min(2, extent - 6) + 1))
        first = int(rng.integers(3, extent - gap - 2))
        if axis == 0:
            rooms.extend(((a, a + first, c, d), (a + first + gap, b, c, d)))
        else:
            rooms.extend(((a, b, c, c + first), (a, b, c + first + gap, d)))
    return rooms


def _stencil(rng, n, index, background, carrier, a, b, extra):
    """Generate occupied geometry and keys, returning construction data as well."""
    if index == 0:
        # Teach period 4 with a carrier-valued gap, cycle repetition, and period 2.
        mh = int(rng.integers(11, 16))
        mw = int(rng.integers(9, 12))
        sw = int(rng.integers(3, min(5, 15 - mw)))
        sh = int(rng.integers(3, 6))
        h, w = mh, mw + 1 + sw
        sr = int(rng.integers(h - sh + 1))
        rooms = [(0, mh, 0, mw), (sr, sr + sh, mw + 1, w)]
        lengths = [4, 2]
    elif index == 1:
        # Different tile scale; period 3 on another informative concavity.
        h, w = int(rng.integers(9, 11)), int(rng.integers(9, 11))
        rooms, lengths = [(0, h, 0, w)], [3]
    else:
        limit = 30 // n
        h, w = int(rng.integers(5, limit + 1)), int(rng.integers(5, limit + 1))
        rooms = _rectangles(rng, h, w, int(rng.integers(1, 5)))
        lengths = [int(rng.integers(2, min(4, (min(bb - aa, dd - cc) + 1) // 2) + 1))
                   for aa, bb, cc, dd in rooms]
    board = np.full((h, w), background, int)
    specs = []
    for i, ((r0, r1, c0, c1), length) in enumerate(zip(rooms, lengths)):
        witness = index in (0, 1) and i == 0
        local = _shape(rng, r1 - r0, c1 - c0, length, witness)
        mask = np.zeros((h, w), bool)
        mask[r0:r1, c0:c1] = local
        depth = _erosion(mask)
        available = [p for p in _paths(mask, depth) if p[4] >= length]
        path = available[int(rng.integers(len(available)))]
        order = (a, b) if rng.integers(2) else (b, a)
        palette = list(order)
        if length >= 3:
            palette.append(int(rng.choice(extra)))
        if length == 4:
            palette.append(int(rng.choice([c for c in extra if c != palette[2]])))
            if index == 0 or rng.random() < .45:
                palette[2] = carrier
        period = tuple(palette)
        board[mask] = carrier
        rr, cc, dr, dc, _ = path
        for d, color in enumerate(period):
            board[rr + d * dr, cc + d * dc] = color
        specs.append(dict(mask=mask, depth=depth, period=period))
    # A global reflection/transposition moves both geometry and its embedded keys.
    transpose = bool(rng.integers(2))
    if transpose:
        board = board.T.copy()
        for spec in specs:
            spec["mask"] = spec["mask"].T.copy()
            spec["depth"] = spec["depth"].T.copy()
    return board, specs


def _frame(card, color):
    result = np.full(tuple(x + 2 for x in card.shape), color, int)
    result[1:-1, 1:-1] = card
    return result


def _place_cards(rng, cards, background):
    """Random rectangle packing, with a small constructive fallback."""
    large = max(range(3), key=lambda i: cards[i].size)
    small = [i for i in range(3) if i != large]
    lh, lw = cards[large].shape
    max_sh = max(cards[i].shape[0] for i in small)
    max_sw = max(cards[i].shape[1] for i in small)
    for _ in range(20):
        vertical = bool(rng.integers(2))
        min_h = lh + max_sh + 3 if vertical else max(lh + 2, sum(cards[i].shape[0] for i in small) + 3)
        min_w = max(lw + 2, sum(cards[i].shape[1] for i in small) + 3) if vertical else lw + max_sw + 3
        if min_h > 30 or min_w > 30:
            continue
        h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        taken = np.zeros((h, w), bool)
        grid = np.full((h, w), background, int)
        success = True
        for i in [large] + [small[j] for j in rng.permutation(2)]:
            ch, cw = cards[i].shape
            positions = [(r, c) for r in range(1, h - ch)
                         for c in range(1, w - cw)
                         if not taken[r - 1:r + ch + 1, c - 1:c + cw + 1].any()]
            if not positions:
                success = False
                break
            r, c = positions[int(rng.integers(len(positions)))]
            taken[r:r + ch, c:c + cw] = True
            grid[r:r + ch, c:c + cw] = cards[i]
        if success:
            return grid
    h = max(lh, sum(cards[i].shape[0] for i in small) + 1) + 2
    w = lw + max_sw + 3
    assert h <= 30 and w <= 30
    grid = np.full((h, w), background, int)
    grid[1:1 + lh, 1:1 + lw] = cards[large]
    r = 1
    for i in small:
        ch, cw = cards[i].shape
        grid[r:r + ch, lw + 2:lw + 2 + cw] = cards[i]
        r += ch + 1
    return grid


def _make_pair(rng, n, turn, index):
    colors = list(map(int, rng.permutation(10)))
    background, frame, carrier, a, b = colors[:5]
    tile = np.where(_tile(rng, n), a, b)
    board, specs = _stencil(rng, n, index, background, carrier, a, b, colors[5:])
    arrow = np.full((3, 3), background, int)
    marker = int(rng.choice(colors[2:]))
    arrow[1, 1] = arrow[ARMS[turn]] = marker
    inp = _place_cards(rng, [_frame(board, frame), _frame(tile, frame), _frame(arrow, frame)], background)
    # Construct outputs from sampled masks, not by calling solve or its decoder.
    shape = board.shape[::-1] if turn % 2 else board.shape
    out = np.full(tuple(n * x for x in shape), background, int)
    for spec in specs:
        period = spec["period"]
        for r, c in np.argwhere(np.rot90(spec["mask"], -turn)):
            d = int(np.rot90(spec["depth"], -turn)[r, c])
            stamp = np.empty(tile.shape, int)
            for tr in range(n):
                for tc in range(n):
                    exchanged = 1 if tile[tr, tc] == period[0] else 0
                    stamp[tr, tc] = period[(d + exchanged) % len(period)]
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = stamp
    return {"input": inp, "output": out}


def _legal_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _legal_count(num_train, 2, 10, "num_train")
    ne = _legal_count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    quarter = int(rng.choice([1, 3]))
    even = int(rng.choice([0, 2]))
    witnesses = [(2, quarter), (3, even)]
    combinations = [(n, k) for n in range(2, 6) for k in range(4)]
    candidates = [p for p in combinations if p not in witnesses]
    held = [candidates[int(i)] for i in rng.choice(len(candidates), ne, replace=False)]
    train_choices = [p for p in combinations if p not in held]
    controls = witnesses + [train_choices[int(rng.integers(len(train_choices)))] for _ in range(nt - 2)] + held
    puzzle = {"train": [], "test": []}
    seen = set()
    for i, (n, k) in enumerate(controls):
        for _ in range(30):
            pair = _make_pair(rng, n, k, i if i < 2 else 2)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("Could not construct distinct input")
        seen.add(_key(pair["input"]))
        puzzle["train" if i < nt else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(info, output, behavior):
    return not np.array_equal(_render(info, behavior), output)


def validate_generated_puzzle(puzzle) -> None:
    """Check returned pixels, semantic invariants, curriculum, and ablations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    infos = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _inverse_output(info)), "output violates contour phase substitution"
            assert np.array_equal(out, solve(inp))
            expected_mask = np.repeat(np.repeat(np.rot90(info["occupied"], -info["turn"]), len(info["tile"]), 0), len(info["tile"]), 1)
            assert np.all(out[~expected_mask] == info["background"]), "transparent stencil cells were filled"
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert info["frame"] not in out, "instruction frames must be discarded"
            infos[split].append((info, out))
    training = infos["train"]
    for alternative in ("flat_depth", "uniform_blocks", "no_exchange", "rotate_tile",
                        "manhattan", "bounding_box", "fine_depth", "skip_carrier"):
        assert any(_different(info, out, alternative) for info, out in training), f"no training witness against {alternative}"
    lengths = {len(r["period"]) for info, _ in training for r in info["regions"]}
    assert {2, 3, 4} <= lengths
    assert any(info["carrier"] in r["period"][2:-1] for info, _ in training for r in info["regions"])
    for length in (3, 4):
        assert any(len(r["period"]) == length and r["depth"].max() >= length
                   for info, _ in training for r in info["regions"]), "cycle must visibly repeat"
    assert any(len(info["regions"]) > 1 for info, _ in training)
    assert len({len(info["tile"]) for info, _ in training}) >= 2
    assert any(info["turn"] % 2 for info, _ in training)
    # Check the four-direction convention in a finite, explicit hypothesis class.
    survivors = []
    for sign in (-1, 1):
        for offset in range(4):
            if all(np.array_equal(out, _render(info, turn_override=(sign * info["turn"] + offset) % 4))
                   for info, out in training):
                survivors.append((sign, offset))
    assert survivors == [(1, 0)], "rotation chirality/offset is not established"
    train_controls = {(len(info["tile"]), info["turn"]) for info, _ in training}
    assert all((len(info["tile"]), info["turn"]) not in train_controls for info, _ in infos["test"]), "test combination must be held out"


def self_test():
    """Bounded 42-seed audit, determinism, legal counts, and corruptions."""
    shapes, geometries, tile_patterns, palettes, directions = (set() for _ in range(5))
    sizes, periods, counts, turns, holes = (set() for _ in range(5))
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(_grid(pair[role])) for split in ("train", "test")
                          for pair in puzzle[split] for role in ("input", "output"))
        fingerprints.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                geometries.add(_key(info["occupied"].astype(int)))
                tile_patterns.add(_key((info["tile"] == info["tile"][0, 0]).astype(int)))
                palettes.add((info["background"], info["frame"], info["carrier"]))
                sizes.add(len(info["tile"]))
                counts.add(len(info["regions"]))
                turns.add(info["turn"])
                for r in info["regions"]:
                    assert np.array_equal(r["depth"], _erosion(r["mask"]))
                    periods.add(len(r["period"]))
                    directions.update(r["directions"])
                holes.add(len(_components(np.pad(~info["occupied"], 1, constant_values=True))) - 1)
    assert len(fingerprints) == 42
    assert len(shapes) >= 70 and len(geometries) >= 90 and len(tile_patterns) >= 60
    assert len(palettes) >= 70
    assert sizes == {2, 3, 4, 5} and periods == {2, 3, 4}
    assert counts == {1, 2, 3, 4} and turns == {0, 1, 2, 3}
    assert directions == {(r, c) for r in (-1, 1) for c in (-1, 1)} and max(holes) >= 1
    first, second = generate_puzzle(8128), generate_puzzle(8128)
    assert all(np.array_equal(p[r], q[r]) for s in first for p, q in zip(first[s], second[s]) for r in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(431, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (10, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{r: a.copy() for r, a in pair.items()} for pair in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (broken["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shape_pairs": len(shapes),
            "stencil_geometries": len(geometries), "binary_tiles": len(tile_patterns),
            "role_palettes": len(palettes), "tile_sizes": sorted(sizes),
            "region_counts": sorted(counts), "period_lengths": sorted(periods),
            "rotations": sorted(turns), "maximum_holes": max(holes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
