"""Union-selected, provenance-colored microtile mosaics.

The public solver reads only input pixels. No parent module, saved example,
random state, training output, or external file is used by it.
"""
from __future__ import annotations

import hashlib
import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_ARMS = ((0, 1), (1, 2), (2, 1), (1, 0))
_STYLES = ("frame", "bars", "branch", "path", "islands", "open")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer rectangles of size 1..30 with colors 0..9")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _relations(masks):
    found = []
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            if not (np.any(masks[a] & ~masks[b]) and np.any(masks[b] & ~masks[a])):
                continue
            union = masks[a] | masks[b]
            for c in range(len(masks)):
                if c not in (a, b) and np.array_equal(union, masks[c]):
                    found.append((a, b, c))
    return found


def _runs(bits):
    padded = np.r_[False, bits, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _decode_bank(bank, divider):
    h, w = bank.shape
    assert 9 <= w <= 13
    assert np.all(bank[:, 0] == divider) and np.all(bank[:, -1] == divider)
    assert np.all(bank[0] == divider) and np.all(bank[-1] == divider)
    bands = _runs(np.any(bank != divider, axis=1))
    assert 5 <= len(bands) <= 7
    cards, key_band = bands[:-1], bands[-1]
    kr, ke = key_band
    assert ke - kr == 3
    fields = [bank[kr:ke, c:c + 3] for c in (1, 5)]
    vals, counts = np.unique(np.concatenate([x.ravel() for x in fields]), return_counts=True)
    ground = int(vals[np.argmax(counts)])
    assert ground != divider
    control = {}
    for field in fields:
        points = set(map(tuple, np.argwhere(field != ground)))
        ink = np.unique(field[field != ground])
        assert len(ink) == 1 and int(ink[0]) != divider
        if len(points) == 1:
            pos = next(iter(points))
            options = [(2 * r, 2 * c) for r, c in _CORNERS]
            assert pos in options and "anchor" not in control
            control.update(anchor=options.index(pos), result_color=int(ink[0]))
        else:
            assert len(points) == 2 and (1, 1) in points and "turn" not in control
            arm = next(iter(points - {(1, 1)}))
            assert arm in _ARMS
            control.update(turn=_ARMS.index(arm), operand_color=int(ink[0]))
    assert {"anchor", "turn", "result_color", "operand_color"} == set(control)
    assert control["operand_color"] != control["result_color"]
    mh = cards[0][1] - cards[0][0]
    first = cards[0][0]
    boundaries = np.flatnonzero(bank[first, 1:] == divider)
    assert len(boundaries)
    mw = int(boundaries[0])
    n = w - mw - 3
    assert 3 <= mh <= 5 and 3 <= mw <= 6 and 2 <= n <= 4 and n <= mh
    rebuilt = np.full(bank.shape, divider, int)
    for c in (1, 5):
        rebuilt[kr:ke, c:c + 3] = bank[kr:ke, c:c + 3]
    masks, tiles, mask_boxes, tile_boxes = [], [], [], []
    all_ink = set()
    for r, end in cards:
        assert end - r == mh
        mask_box = (slice(r, end), slice(1, mw + 1))
        field = bank[mask_box]
        assert not np.any(field == divider)
        all_ink.update(map(int, np.unique(field[field != ground])))
        mask = field != ground
        assert mask.any() and not mask.all()
        tile_band = bank[r:end, mw + 2:-1]
        rows = np.flatnonzero(np.any(tile_band != divider, axis=1))
        assert len(rows) == n and np.array_equal(rows, np.arange(rows[0], rows[0] + n))
        tr = r + int(rows[0])
        tile_box = (slice(tr, tr + n), slice(mw + 2, w - 1))
        tile = bank[tile_box].copy()
        assert tile.shape == (n, n) and len(np.unique(tile)) == 2
        assert not np.any(tile == divider)
        rebuilt[mask_box] = field
        rebuilt[tile_box] = tile
        masks.append(mask)
        tiles.append(tile)
        mask_boxes.append(mask_box)
        tile_boxes.append(tile_box)
    assert len(all_ink) == 1
    foreground = next(iter(all_ink))
    assert len({ground, divider, foreground, control["operand_color"], control["result_color"]}) == 5
    assert np.array_equal(rebuilt, bank), "unexplained panel pixels"
    assert len({m.tobytes() for m in masks}) == len(masks), "duplicate stencils are outside this family"
    relations = _relations(masks)
    assert len(relations) == 1, "expected exactly one nonredundant union equation"
    a, b, result = relations[0]
    union = masks[result]
    assert np.any(masks[a] & masks[b]), "overlap is required to expose precedence"
    assert union[0].any() and union[-1].any() and union[:, 0].any() and union[:, -1].any()
    assert not np.array_equal(tiles[a], tiles[b])
    control.update(ground=ground, divider=divider, foreground=foreground,
                   masks=masks, tiles=tiles, mask_boxes=mask_boxes,
                   tile_boxes=tile_boxes, relation=(a, b, result), size=n,
                   bank=bank.copy())
    return control


def _decode(input_grid):
    g = _grid(input_grid)
    candidates = []
    for side in ("left", "right"):
        divider = int(g[0, 0 if side == "left" else -1])
        cols = np.flatnonzero(np.all(g == divider, axis=0))
        if not len(cols):
            continue
        if side == "left":
            width = int(cols[-1]) + 1
            bank, canvas, start, cs = g[:, :width], g[:, width:], 0, slice(width, None)
        else:
            start = int(cols[0])
            width = g.shape[1] - start
            bank, canvas, cs = g[:, start:], g[:, :start], slice(0, start)
        if not 9 <= width <= 13 or canvas.shape[1] < 1 or np.any(canvas == divider):
            continue
        try:
            info = _decode_bank(bank, divider)
            n, mask = info["size"], info["masks"][info["relation"][2]]
            rh, rw = mask.shape[::-1] if info["turn"] % 2 else mask.shape
            assert n * rh < canvas.shape[0] and n * rw < canvas.shape[1]
        except (AssertionError, IndexError, ValueError):
            continue
        info.update(canvas=canvas.copy(), bank_start=start, canvas_slice=cs,
                    side=side, input=g.copy())
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input does not contain one unambiguous bounded instruction bank")
    return candidates[0]


def _exchanged(tile):
    colors = np.unique(tile)
    return colors[0] + colors[1] - tile


def _assemble(info, bank, canvas):
    return np.concatenate((bank, canvas) if info["side"] == "left" else (canvas, bank), axis=1)


def _forward(info):
    """Label the derivation and copy blocks from the two selected tiles."""
    a, b, result = info["relation"]
    bank, canvas = info["bank"].copy(), info["canvas"].copy()
    tiles = {i: _exchanged(info["tiles"][i]) for i in (a, b)}
    for i in (a, b, result):
        field = bank[info["mask_boxes"][i]]
        field[info["masks"][i]] = info["result_color"] if i == result else info["operand_color"]
    for i in (a, b):
        bank[info["tile_boxes"][i]] = tiles[i]
    union = info["masks"][result]
    # Codes 1/2 carry provenance; zero is transparent. The result itself
    # supplies support. The first operand owns their shared coordinates.
    owner = np.where(union, np.where(info["masks"][a], 1, 2), 0)
    owner = np.rot90(owner, -info["turn"])
    n = info["size"]
    ar, ac = _CORNERS[info["anchor"]]
    top = ar * (canvas.shape[0] - n * owner.shape[0])
    left = ac * (canvas.shape[1] - n * owner.shape[1])
    for r, c in np.argwhere(owner):
        index = a if owner[r, c] == 1 else b
        canvas[top + r * n:top + (r + 1) * n,
               left + c * n:left + (c + 1) * n] = tiles[index]
    return _assemble(info, bank, canvas)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible fields, relation, keys and tiles, without generation."""
    return _forward(_decode(input_grid))


def _corner_transform(anchor, transform):
    r, c = _CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _inverse_canvas(info, behavior="normal", convention=(1, 0, 0)):
    """Cell-coordinate validator and explicit competing interpretations.

    This does not rotate a mask or call the forward block copier.
    """
    a, b, result = info["relation"]
    ma, mb, support = (info["masks"][i] for i in (a, b, result))
    if behavior == "none":
        return info["canvas"].copy()
    if behavior == "largest":
        support = max(info["masks"], key=lambda m: int(m.sum()))
    elif behavior == "intersection":
        support = ma & mb
    elif behavior == "xor":
        support = ma ^ mb
    elif behavior == "first_mask":
        support = ma
    elif behavior == "second_mask":
        support = mb
    elif behavior == "rectangle":
        support = np.ones(support.shape, bool)
    sign, offset, transform = convention
    turn = (sign * info["turn"] + offset) % 4
    h, w = support.shape
    rh, rw = (w, h) if turn % 2 else (h, w)
    n = info["size"]
    canvas = info["canvas"].copy()
    if n * rh >= len(canvas) or n * rw >= canvas.shape[1]:
        return None
    ar, ac = _corner_transform(info["anchor"], transform)
    rr, cc = np.indices(canvas.shape)
    y, x = rr - ar * (len(canvas) - n * rh), cc - ac * (canvas.shape[1] - n * rw)
    inside = (y >= 0) & (x >= 0) & (y < n * rh) & (x < n * rw)
    mr, mc = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
    if turn == 0:
        sy, sx = mr, mc
    elif turn == 1:
        sy, sx = h - 1 - mc, mr
    elif turn == 2:
        sy, sx = h - 1 - mr, w - 1 - mc
    else:
        sy, sx = mc, w - 1 - mr
    active = inside & support[sy, sx]
    use_a = ma[sy, sx] | ~mb[sy, sx]
    if behavior == "reverse_overlap":
        use_a = ~mb[sy, sx]
    elif behavior == "first_tile":
        use_a = np.ones(canvas.shape, bool)
    elif behavior == "second_tile":
        use_a = np.zeros(canvas.shape, bool)
    ty, tx = y % n, x % n
    if behavior == "rotate_tiles":
        for _ in range(turn):
            ty, tx = n - 1 - tx, ty
    va, vb = info["tiles"][a][ty, tx], info["tiles"][b][ty, tx]
    if behavior != "no_exchange":
        ca, cb = np.unique(info["tiles"][a]), np.unique(info["tiles"][b])
        va, vb = int(ca.sum()) - va, int(cb.sum()) - vb
    values = np.where(use_a, va, vb)
    canvas[active] = values[active]
    return canvas


_ALTERNATIVES = ("none", "largest", "intersection", "xor", "first_mask",
                 "second_mask", "rectangle", "first_tile", "second_tile",
                 "reverse_overlap", "no_exchange", "rotate_tiles")


def _tight(mask):
    return mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()


def _union_mask(rng, h, w, style):
    rr, cc = np.indices((h, w))
    for attempt in range(100):
        if style == "frame":
            mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
            mask |= rng.random((h, w)) < .12
        elif style == "bars":
            mask = (rr % 2 == 0) | (cc == int(rng.integers(w)))
        elif style == "branch":
            mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
            mask |= rng.random((h, w)) < .2
        elif style == "islands":
            mask = (rr % 2 == 0) & (cc % 2 == 0)
            mask |= rng.random((h, w)) < .22
        elif style == "path":
            mask = np.zeros((h, w), bool)
            c = int(rng.integers(w))
            for r in range(h):
                dest = int(rng.integers(w))
                mask[r, min(c, dest):max(c, dest) + 1] = True
                c = dest
        else:
            mask = rng.random((h, w)) < rng.uniform(.35, .72)
        # Local edits break symmetry and give all three decoy types room.
        flip = rng.choice(h * w, int(rng.integers(1, 4)), replace=False)
        mask.flat[flip] = ~mask.flat[flip]
        if (3 <= mask.sum() <= mask.size - 3 and _tight(mask)
                and not np.array_equal(mask, np.rot90(mask, 2))):
            return mask
    # A deterministic, tight asymmetric support with ample holes.
    mask = np.zeros((h, w), bool)
    mask[0, -1] = mask[-1, 0] = mask[1, 0] = True
    return mask


def _masks(rng, count, h, w, style):
    union = _union_mask(rng, h, w, style)
    cells = rng.permutation(np.flatnonzero(union))
    # Each operand has unique cells, and their intersection is nonempty.
    max_unique = min(4, (len(cells) - 1) // 2)
    ua = int(rng.integers(1, max_unique + 1))
    ub = int(rng.integers(1, min(4, len(cells) - ua - 1) + 1))
    a, b = union.copy(), union.copy()
    b.flat[cells[:ua]] = False
    a.flat[cells[ua:ua + ub]] = False
    masks = [a, b, union]
    holes = rng.permutation(np.flatnonzero(~union))
    # Close supersets are false results, including a mask larger than U.
    for j in range(1 if count == 4 else 2):
        decoy = union.copy()
        decoy.flat[holes[j]] = True
        masks.append(decoy)
    if count == 6:
        # A small independent distractor defeats foreground-count ranking.
        for _ in range(30):
            decoy = np.zeros((h, w), bool)
            decoy.flat[holes[2]] = True
            maximum = max(1, min(int(a.sum()), int(b.sum())) - 1)
            sites = rng.choice(h * w, int(rng.integers(1, maximum + 1)), replace=False)
            decoy.flat[sites] = True
            if decoy.sum() < min(a.sum(), b.sum()) and _relations(masks + [decoy]) == [(0, 1, 2)]:
                break
        else:
            decoy = np.zeros((h, w), bool)
            decoy.flat[holes[2]] = True
        masks.append(decoy)
    assert _relations(masks) == [(0, 1, 2)]
    order = rng.permutation(count)
    return [masks[int(i)] for i in order]


def _tile(rng, n, palette):
    for _ in range(50):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if bits.any() and not bits.all() and not np.array_equal(bits, np.rot90(bits, 2)):
            colors = rng.choice(palette, 2, replace=False)
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, palette[0], palette[1]).astype(int)


def _geometry(rng, count, turn, forbidden_size=None):
    choices = []
    for mh in range(3, (30 - 5) // count):
        for mw in range(3, 7):
            for n in range(2, min(4, mh) + 1):
                bw = mw + n + 3
                rh, rw = (mw, mh) if turn % 2 else (mh, mw)
                if (9 <= bw <= 13 and n != forbidden_size and
                        n * rw + 2 <= 30 - bw and n * rh < count * (mh + 1) + 5):
                    choices.append((mh, mw, n, bw, rw))
    assert choices
    return choices[int(rng.integers(len(choices)))]


def _canvas(rng, h, w, colors):
    rr, cc = np.indices((h, w))
    style = int(rng.integers(5))
    if style == 0:
        ix = np.zeros((h, w), int)
    elif style == 1:
        ix = rr // int(rng.integers(2, 7))
    elif style == 2:
        ix = cc // int(rng.integers(2, 7))
    elif style == 3:
        ix = rr // int(rng.integers(3, 8)) + cc // int(rng.integers(3, 8))
    else:
        centers = rng.random((3, 2)) * [h, w]
        ix = np.argmin([(rr - r) ** 2 + (cc - c) ** 2 for r, c in centers], axis=0)
    return np.asarray(colors)[ix % len(colors)].astype(int)


def _make_input(rng, count, turn, anchor, side, style, forbidden_size=None):
    mh, mw, n, bw, rw = _geometry(rng, count, turn, forbidden_size)
    minimum_h = count * (mh + 1) + 5
    h = int(rng.integers(minimum_h, 31))
    cw = int(rng.integers(max(8, n * rw + 2), 31 - bw))
    colors = list(map(int, rng.permutation(10)))
    ground, divider, foreground, op, res = colors[:5]
    available = [c for c in range(10) if c != divider]
    pool = list(map(int, rng.choice(available, int(rng.integers(2, 4)), replace=False)))
    canvas_colors = [c for c in available if c not in pool]
    canvas_colors = rng.choice(canvas_colors, int(rng.integers(1, 4)), replace=False)
    canvas = _canvas(rng, h, cw, canvas_colors)
    masks = _masks(rng, count, mh, mw, style)
    a, b, _ = _relations(masks)[0]
    tiles = [_tile(rng, n, pool) for _ in masks]
    if np.array_equal(tiles[a], tiles[b]):
        tiles[b] = _exchanged(tiles[a])
    bank = np.full((h, bw), divider, int)
    extra = h - minimum_h
    top = int(rng.integers(extra + 1))
    gap = int(rng.integers(extra - top + 1))
    for i, (mask, tile) in enumerate(zip(masks, tiles)):
        r = top + i * (mh + 1) + 1
        bank[r:r + mh, 1:mw + 1] = np.where(mask, foreground, ground)
        tr = r + int(rng.integers(mh - n + 1))
        bank[tr:tr + n, mw + 2:-1] = tile
    kr = top + count * (mh + 1) + gap + 1
    columns = [1, 5]
    rng.shuffle(columns)
    for c in columns:
        bank[kr:kr + 3, c:c + 3] = ground
    ar, ac = _CORNERS[anchor]
    bank[kr + 2 * ar, columns[0] + 2 * ac] = res
    bank[kr + 1, columns[1] + 1] = op
    dr, dc = _ARMS[turn]
    bank[kr + dr, columns[1] + dc] = op
    return np.concatenate((bank, canvas) if side == "left" else (canvas, bank), axis=1)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    quarter, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    adjacent = (anchor + int(rng.choice([1, 3]))) % 4
    held = (quarter, adjacent) if rng.integers(2) else (even, anchor)
    side0 = str(rng.choice(["left", "right"]))
    style_offset = int(rng.integers(len(_STYLES)))
    puzzle, seen, first_size, previous_result = {"train": [], "test": []}, set(), None, None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index < 2:
                turn, corner = (quarter, anchor) if index == 0 else (even, adjacent)
                cards = 4 if index == 0 else 6
                side = side0 if index == 0 else ("right" if side0 == "left" else "left")
            else:
                turn, corner = held if split == "test" and index == 0 else (int(rng.integers(4)), int(rng.integers(4)))
                if split == "train" and (turn, corner) == held:
                    turn = (turn + 1) % 4
                cards = int(rng.integers(4, 7))
                side = str(rng.choice(["left", "right"]))
            style = _STYLES[(style_offset + index + (2 if split == "test" else 0)) % len(_STYLES)]
            for _ in range(40):
                inp = _make_input(rng, cards, turn, corner, side, style,
                                  first_size if split == "train" and index == 1 else None)
                info = _decode(inp)
                result = info["relation"][2]
                if _key(inp) in seen or (split == "train" and index == 1 and result == previous_result):
                    continue
                pair = {"input": inp, "output": _forward(info)}
                seen.add(_key(inp))
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_size, previous_result = info["size"], result
                break
            else:
                raise RuntimeError("bounded distinct-example construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate schemas, visible semantics, ablations, and demonstration coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            a, b, result = info["relation"]
            masks, tiles = info["masks"], info["tiles"]
            assert np.array_equal(masks[a] | masks[b], masks[result])
            assert any(m.sum() > masks[result].sum() for m in masks), "need a denser false result"
            assert not set(map(int, np.unique(info["canvas"]))) & set(map(int, np.unique(np.stack(tiles))))
            expected_canvas = _inverse_canvas(info)
            assert np.array_equal(out[:, info["canvas_slice"]], expected_canvas)
            expected_bank = info["bank"].copy()
            for i in range(len(masks)):
                color = info["result_color"] if i == result else info["operand_color"] if i in (a, b) else info["foreground"]
                expected_bank[info["mask_boxes"][i]] = np.where(masks[i], color, info["ground"])
                tile = tiles[i]
                if i in (a, b):
                    unique = np.unique(tile)
                    tile = np.where(tile == unique[0], unique[1], unique[0])
                expected_bank[info["tile_boxes"][i]] = tile
            assert np.array_equal(out[:, info["bank_start"]:info["bank_start"] + expected_bank.shape[1]], expected_bank)
            assert not np.array_equal(expected_canvas, info["canvas"])
            assert np.array_equal(solve(inp), out)
            (training if split == "train" else testing).append(info)
    # Two adjacent corner demonstrations plus odd/even rotations disambiguate
    # the 64 dihedral corner / angular direction conventions tested here.
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                conv = (sign, offset, transform)
                if all(np.array_equal(_inverse_canvas(i, convention=conv), _inverse_canvas(i)) for i in training[:2]):
                    survivors.append(conv)
    assert survivors == [(1, 0, 0)], "insufficient control witnesses"
    for alternative in _ALTERNATIVES:
        assert any(not np.array_equal(_inverse_canvas(i, alternative), _inverse_canvas(i)) for i in training[:2]), alternative
    assert {len(i["masks"]) for i in training} >= {4, 6}
    assert len({i["size"] for i in training[:2]}) == 2
    assert len({i["relation"][2] for i in training[:2]}) == 2
    assert {i["side"] for i in training} == {"left", "right"}
    assert (testing[0]["turn"], testing[0]["anchor"]) not in {(i["turn"], i["anchor"]) for i in training}


def self_test():
    """42 streaming seeds, bounded diagnostics, no saved sample archive."""
    signatures, shapes, stencil_shapes, sizes, sides, controls = set(), set(), set(), set(), set(), set()
    categories, normalized, palettes, counts = set(), set(), set(), set()
    first_puzzle = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        if seed == 0:
            first_puzzle = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes())
                info = _decode(inp)
                a, b, result = info["relation"]
                ma, mb, u = (info["masks"][i] for i in (a, b, result))
                categories.add((int((ma & ~mb).sum()), int((mb & ~ma).sum()), int((ma & mb).sum())))
                shapes.add(inp.shape)
                stencil_shapes.add(u.shape)
                sizes.add(info["size"])
                sides.add(info["side"])
                controls.add((info["turn"], info["anchor"]))
                counts.add(len(info["masks"]))
                palettes.add((info["ground"], info["divider"], info["foreground"], info["operand_color"], info["result_color"]))
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.ravel())
                normalized.add((inp.shape, hashlib.sha256(norm).digest()))
                untouched = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, untouched)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(normalized) >= 200
    assert sizes == {2, 3, 4} and sides == {"left", "right"} and counts == {4, 5, 6}
    assert controls == {(k, a) for k in range(4) for a in range(4)}
    assert len(shapes) >= 20 and len(stencil_shapes) >= 8 and len(categories) >= 30 and len(palettes) >= 100
    again = generate_puzzle(0)
    assert all(np.array_equal(p[r], q[r]) for s in ("train", "test")
               for p, q in zip(first_puzzle[s], again[s]) for r in ("input", "output"))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(393094351, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{r: np.asarray(g).copy() for r, g in pair.items()} for pair in first_puzzle[s]]
               for s in ("train", "test")}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output was accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "input_shapes": len(shapes),
            "stencil_shapes": len(stencil_shapes), "membership_counts": len(categories),
            "palettes": len(palettes), "tile_sizes": sorted(sizes),
            "card_counts": sorted(counts), "control_combinations": len(controls)}
