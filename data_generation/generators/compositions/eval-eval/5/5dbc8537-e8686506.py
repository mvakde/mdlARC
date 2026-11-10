"""Laced rectangles: register patterned cards on a perforated fixed scaffold.

Only standard-library modules and NumPy are used. No generation state, source
files, examples, or seed is consulted by solve(). Experiments and verification
reports are external to this module; execution traces are not saved.
"""
from functools import lru_cache
import hashlib

import numpy as np


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    yy, xx = zip(*cells)
    return min(yy), min(xx), max(yy) + 1, max(xx) + 1


def _parse(inp):
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(counts.max() > inp.size // 2, "Background must be the strict majority")
    candidates = []
    components = _components(inp != background)
    for comp in components:
        y, x, bottom, right = _bbox(comp)
        board = inp[y:bottom, x:right]
        if min(board.shape) < 5 or not np.any(board == background):
            continue
        frame = np.concatenate((board[0], board[-1], board[:, 0], board[:, -1]))
        scaffold = int(frame[0])
        if scaffold == background or not np.all(frame == scaffold):
            continue
        if set(map(int, np.unique(board))) != {background, scaffold}:
            continue
        if not (np.array_equal(board, board[:, ::-1]) or
                np.array_equal(board, board[::-1])):
            continue
        outside = inp != background
        outside[y:bottom, x:right] = False
        cards, boxes = [], []
        for cells in _components(outside):
            a, b, c, d = _bbox(cells)
            card = inp[a:c, b:d].copy()
            if card.size != len(cells) or np.any(card == background):
                break
            if not np.any(card == scaffold) or not np.any(card != scaffold):
                break
            cards.append(card)
            boxes.append((a, b, c, d))
        else:
            if cards:
                candidates.append((board.copy(), background, scaffold,
                                   (y, x, bottom, right), cards, boxes))
    _require(len(candidates) == 1, "Expected one framed bilateral board and solid cards")
    return candidates[0]


def _placements(board, background, scaffold, card):
    """A card's entire scaffold/payload mask must register with the board."""
    h, w = card.shape
    backing = card == scaffold
    holes = board == background
    for y in range(board.shape[0] - h + 1):
        for x in range(board.shape[1] - w + 1):
            if np.array_equal(~backing, holes[y:y + h, x:x + w]):
                yield y, x


def _assemble(board, background, scaffold, cards, node_limit=24000, placement_uniqueness=False):
    """Enumerate up to two rendered answers with exact finite inventory.

    Matching scaffold cells can be covered repeatedly; only vacancies enter
    exact cover. Equal full matrices are grouped with a physical-copy count.
    The bounded memo stores suffix color encodings, never trace snapshots.
    """
    cells = list(map(tuple, np.argwhere(board == background).tolist()))
    ids = {p: i for i, p in enumerate(cells)}
    if sum(int(np.count_nonzero(t != scaffold)) for t in cards) != len(cells):
        return [], {}, []
    groups, inventory, lookup = [], [], {}
    for card in cards:
        k = _key(card)
        if k not in lookup:
            lookup[k] = len(groups)
            groups.append(card)
            inventory.append(0)
        inventory[lookup[k]] += 1
    moves, by_cell = [], [[] for _ in cells]
    for t, card in enumerate(groups):
        payload = np.argwhere(card != scaffold)
        for y, x in _placements(board, background, scaffold, card):
            mask = paint = 0
            for dy, dx in payload:
                i = ids[y + int(dy), x + int(dx)]
                mask |= 1 << i
                paint |= int(card[dy, dx]) << (4 * i)
            move = (t, mask, paint, y, x)
            m = len(moves)
            moves.append(move)
            for dy, dx in payload:
                by_cell[ids[y + int(dy), x + int(dx)]].append(m)
    nodes = 0

    @lru_cache(maxsize=24000)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        _require(nodes <= node_limit, "Exact-cover proof exceeded its node budget")
        if not remaining:
            return ((0, ()),) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            options = [m for m in by_cell[i] if counts[moves[m][0]] and
                       moves[m][1] & remaining == moves[m][1]]
            if not options:
                return ()
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        answers = {}
        for m in best:
            t, mask, paint, _, _ = moves[m]
            rest = list(counts)
            rest[t] -= 1
            for suffix, path in search(remaining ^ mask, tuple(rest)):
                code = ((1 << m) if placement_uniqueness else paint) | suffix
                answers.setdefault(code, (m,) + path)
                if len(answers) == 2:
                    return tuple(answers.items())
        return tuple(answers.items())

    encoded = search((1 << len(cells)) - 1, tuple(inventory))
    answers, paths = [], []
    for code, path in encoded:
        answer = board.copy()
        if placement_uniqueness:
            for m in path:
                t, _, _, y, x = moves[m]
                card = groups[t]
                answer[y:y + card.shape[0], x:x + card.shape[1]] = card
        else:
            for p, i in ids.items():
                answer[p] = (code >> (4 * i)) & 15
        answers.append(answer)
        paths.append([(groups[moves[m][0]], moves[m][3], moves[m][4]) for m in path])
    return answers, {"nodes": nodes, "local_placements": len(moves)}, paths


def _infer(inp):
    board, bg, sc, bbox, cards, boxes = _parse(inp)
    answers, stats, paths = _assemble(board, bg, sc, cards)
    _require(len(answers) == 1, "Inventory must have one rendered completion")
    return dict(board=board, background=bg, scaffold=sc, bbox=bbox, cards=cards,
                card_boxes=boxes, output=answers[0], path=paths[0], **stats)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the crop, role colors, and unique registered assembly from input."""
    return _infer(_grid(input_grid))["output"].copy()


def _orbit_rect(rect, width):
    y, x, h, w = rect
    return list(dict.fromkeys((rect, (y, width - x - w, h, w))))


def _layout(rng, testing):
    """Sample overlapping rectangle orbits, then perforate their unique layers."""
    h = int(rng.integers(7, 12 if testing else 11))
    w = int(rng.integers(7, 13 if testing else 12))
    first = (int(rng.integers(h - 2)), int(rng.integers(w - 2)),
             int(rng.integers(2, min(6, h) + 1)), int(rng.integers(2, min(7, w) + 1)))
    y, x, sh, sw = first
    first = (min(y, h - sh), min(x, w - sw), sh, sw)
    rects = _orbit_rect(first, w)
    # A second orbit is usually edge-adjacent, so one gap can contain multiple cards.
    sh, sw = int(rng.integers(2, min(6, h) + 1)), int(rng.integers(2, min(7, w) + 1))
    ry, rx, rh, rw = first
    side = int(rng.integers(4))
    if side < 2:
        y = ry - sh if side == 0 else ry + rh
        x = rx + int(rng.integers(1 - sw, rw))
    else:
        x = rx - sw if side == 2 else rx + rw
        y = ry + int(rng.integers(1 - sh, rh))
    if not (0 <= y <= h - sh and 0 <= x <= w - sw):
        y, x = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
    rects += _orbit_rect((y, x, sh, sw), w)
    extra = int(rng.integers(1, 3)) if testing else int(rng.integers(0, 2))
    for _ in range(extra):
        sh, sw = int(rng.integers(2, min(6, h) + 1)), int(rng.integers(2, min(7, w) + 1))
        y, x = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
        rects += _orbit_rect((y, x, sh, sw), w)
    rects = list(dict.fromkeys(rects))
    if not (3 <= len(rects) <= 8) or sum(a * b for _, _, a, b in rects) > 180:
        return None
    coverage = np.zeros((h, w), dtype=np.int8)
    for y, x, sh, sw in rects:
        coverage[y:y + sh, x:x + sw] += 1
    depth = int(coverage.max())
    if (testing and depth < 3) or (not testing and depth != 2):
        return None
    holes = coverage == 1
    cuts = rng.random((h, (w + 1) // 2)) < float(rng.uniform(.08, .24))
    for x in range((w + 1) // 2):
        holes[:, x] &= ~cuts[:, x]
        holes[:, w - 1 - x] &= ~cuts[:, x]
    # A short stitch through a sampled carrier binds disconnected payload lobes.
    y, x, sh, sw = rects[int(rng.integers(len(rects)))]
    if sh >= 3:
        row = y + int(rng.integers(1, sh - 1))
        holes[row, x:x + sw] = False
        holes[row, w - x - sw:w - x] = False
    masks = [holes[y:y + sh, x:x + sw] for y, x, sh, sw in rects]
    if any(m.sum() < 3 or m.all() for m in masks):
        return None
    if not any(len(_components(m)) > 1 for m in masks):
        return None
    return holes, rects


def _fallback_layout(rng, testing):
    """A bounded constructive rescue; normal sampling is measured separately."""
    w = int(rng.choice([9, 11]))
    band = int(rng.choice([3, 4]))
    gap = 2 if testing else 0
    h = 2 * band + gap
    a, b = (w + 3) // 2, (w + 1) // 2
    rects = _orbit_rect((0, 0, band, a), w)
    rects += _orbit_rect((band + gap, 0, band, b), w)
    if testing:
        rects += [(band - 1, (w - 5) // 2, gap + 2, 5)]
    coverage = np.zeros((h, w), dtype=np.int8)
    for y, x, sh, sw in rects:
        coverage[y:y + sh, x:x + sw] += 1
    holes = coverage == 1
    holes[1] = False
    holes[-2] = False
    return holes, rects


def _paint(rng, holes, rects, background, scaffold, palette):
    board = np.full((holes.shape[0] + 2, holes.shape[1] + 2), scaffold, dtype=np.int8)
    board[1:-1, 1:-1][holes] = background
    out = board.copy()
    cards, patterns = [], {}
    for y, x, h, w in rects:
        mask = holes[y:y + h, x:x + w]
        sig = (mask.shape, mask.tobytes())
        if sig not in patterns:
            card = np.full((h, w), scaffold, dtype=np.int8)
            card[mask] = rng.choice(palette, int(mask.sum()))
            # At least two colors in every carrier: fragments cannot be collapsed
            # to a single geometry-to-color lookup.
            pp = np.argwhere(mask)
            card[tuple(pp[0])] = palette[0]
            card[tuple(pp[-1])] = palette[1]
            patterns[sig] = card
        card = patterns[sig].copy()
        cards.append(card)
        region = out[y + 1:y + h + 1, x + 1:x + w + 1]
        region[mask] = card[mask]
    return board, out, cards


def _scatter(rng, board, cards, background):
    for attempt in range(10):
        ch = int(rng.integers(max(23, board.shape[0] + 3), 31)) if attempt < 7 else 30
        cw = int(rng.integers(max(23, board.shape[1] + 3), 31)) if attempt < 7 else 30
        by = int(rng.integers(ch - board.shape[0] + 1))
        bx = int(rng.integers(cw - board.shape[1] + 1))
        canvas = np.full((ch, cw), background, dtype=np.int8)
        canvas[by:by + board.shape[0], bx:bx + board.shape[1]] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, by - 1):min(ch, by + board.shape[0] + 1),
                max(0, bx - 1):min(cw, bx + board.shape[1] + 1)] = True
        for i in sorted(range(len(cards)), key=lambda j: (-cards[j].size, j)):
            card = cards[i]
            h, w = card.shape
            candidates = [(y, x) for y in range(ch - h + 1) for x in range(cw - w + 1)
                          if not blocked[y:y + h, x:x + w].any()]
            if not candidates:
                break
            y, x = candidates[int(rng.integers(len(candidates)))]
            canvas[y:y + h, x:x + w] = card
            blocked[max(0, y - 1):min(ch, y + h + 1),
                    max(0, x - 1):min(cw, x + w + 1)] = True
        else:
            if np.count_nonzero(canvas == background) > canvas.size // 2:
                return canvas
    return None


def _features(a):
    sc, bg, board = a["scaffold"], a["background"], a["board"]
    cover = np.zeros(board.shape, dtype=int)
    owner = np.full(board.shape, -1, dtype=int)
    fragmented = nonrectangular = 0
    backing = payload = 0
    for i, (card, y, x) in enumerate(a["path"]):
        h, w = card.shape
        mask = card != sc
        cover[y:y + h, x:x + w] += 1
        owner[y:y + h, x:x + w][mask] = i
        fragmented += len(_components(mask)) > 1
        yy, xx = np.where(mask)
        nonrectangular += int(int(mask.sum()) != int((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1)))
        backing += int((~mask).sum())
        payload += int(mask.sum())
    merged = sum(len({int(owner[p]) for p in comp}) > 1
                 for comp in _components(board == bg))
    return dict(cards=len(a["cards"]), depth=int(cover.max()), fragmented=fragmented,
                merged_gaps=merged, nonrectangular=nonrectangular,
                holes=int((board == bg).sum()), payload=payload, backing=backing,
                repeated_backing=int(np.maximum(cover - 1, 0).sum()),
                board_shape=board.shape, canvas_bbox=a["bbox"],
                symmetry="both" if np.array_equal(board, board[::-1]) and
                    np.array_equal(board, board[:, ::-1]) else
                    "lr" if np.array_equal(board, board[:, ::-1]) else "tb")


def _unbound_alternative(a):
    """Exhibit a wrong answer allowed by discarding the rectangular carriers.

    Exchange two congruent, differently printed payload fragments. This keeps
    all loose fragment shapes, orientations, colors and multiplicities, but
    violates their binding inside the source cards. No second solver is used.
    """
    seen = {}
    for card, y, x in a["path"]:
        for comp in _components(card != a["scaffold"]):
            top, left, _, _ = _bbox(comp)
            comp = sorted(comp)
            shape = tuple((r - top, c - left) for r, c in comp)
            values = tuple(int(card[p]) for p in comp)
            cells = [(y + r, x + c) for r, c in comp]
            if shape in seen and seen[shape][0] != values:
                other_values, other_cells = seen[shape]
                out = a["output"].copy()
                for p, value in zip(cells, other_values):
                    out[p] = value
                for p, value in zip(other_cells, values):
                    out[p] = value
                return out
            seen.setdefault(shape, (values, cells))
    return None


def _build_pair(rng, testing, rotation, require_merged=False):
    roles = list(map(int, rng.permutation(10)))
    bg, sc = roles[:2]
    palette = roles[2:2 + int(rng.integers(2, 6))]
    for attempt in range(360):
        layout = _layout(rng, testing) if attempt < 320 else _fallback_layout(rng, testing)
        if layout is None:
            continue
        board, out, cards = _paint(rng, *layout, bg, sc, palette)
        try:
            answers, _, _ = _assemble(board, bg, sc, cards)
            if len(answers) != 1 or not np.array_equal(answers[0], out):
                continue
            # The overlap-depth experiment also needs an unambiguous footprint
            # arrangement. Identical-copy exchanges have the same move set.
            arrangements, _, _ = _assemble(board, bg, sc, cards, placement_uniqueness=True)
            if len(arrangements) != 1:
                continue
        except ValueError:
            # A search budget exhaustion is an unresolved proposal, not a
            # uniqueness certificate; discard it within the bounded loop.
            continue
        canvas = _scatter(rng, board, cards, bg)
        if canvas is None:
            continue
        inp, out = np.rot90(canvas, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            a = _infer(inp)
        except ValueError:
            continue
        f = _features(a)
        # The input reconstruction, including rescue layouts, must establish
        # every sampling contract. A construction-time rectangle arrangement
        # alone cannot certify the rendered solution's interaction features.
        if not (f["fragmented"] and f["nonrectangular"] and f["repeated_backing"]):
            continue
        if (testing and f["depth"] < 3) or (not testing and f["depth"] != 2):
            continue
        if require_merged and (not f["merged_gaps"] or _unbound_alternative(a) is None):
            continue
        if require_merged and (np.array_equal(out, out[::-1]) or np.array_equal(out, out[:, ::-1])):
            continue
        if np.array_equal(a["output"], out):
            return {"input": inp, "output": out}
    raise RuntimeError("Bounded laced-card construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"Illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "Seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    phase = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            rotation = (phase + i % 2) % 4 if split == "train" else int(rng.integers(4))
            for _ in range(8):
                pair = _build_pair(rng, split == "test", rotation,
                                   require_merged=split == "train" and i == 0)
                k = _key(pair["input"])
                if k not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input budget exhausted")
            seen.add(k)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        _require(_key(inp) not in seen, "Duplicate input")
        seen.add(_key(inp))
        a = _infer(inp)
        _require(np.array_equal(a["output"], out), "Output disagrees with unique assembly")
        board, sc, bg = a["board"], a["scaffold"], a["background"]
        _require(len(_assemble(board, bg, sc, a["cards"], placement_uniqueness=True)[0]) == 1,
                 "Overlap-depth witness has ambiguous footprint arrangements")
        _require(np.array_equal(out[board == sc], board[board == sc]), "Fixed scaffold changed")
        _require(not np.any(out == bg), "Vacancy left unfilled")
        f = _features(a)
        supply = np.zeros(10, dtype=int)
        for card in a["cards"]:
            supply += np.bincount(card[card != sc], minlength=10)
        _require(np.array_equal(supply, np.bincount(out[board == bg], minlength=10)),
                 "Payload colors were copied, discarded, or recolored")
        _require(f["payload"] == f["holes"], "Payload inventory not conserved")
        _require(f["fragmented"] and f["nonrectangular"] and f["repeated_backing"],
                 "Missing compound-carrier and shared-scaffold witnesses")
        coverage = np.zeros(board.shape, dtype=int)
        for card, y, x in a["path"]:
            coverage[y:y + card.shape[0], x:x + card.shape[1]] += 1
            _require(np.array_equal(out[y:y + card.shape[0], x:x + card.shape[1]], card),
                     "A translated full card pattern changed")
        _require(np.all(coverage[board == bg] == 1), "Vacancy was not covered exactly once")
        _require(np.all(board[coverage > 1] == sc), "Cards overlap on payload")
        _require(f["backing"] == int(np.count_nonzero((coverage > 0) & (board == sc)))
                 + f["repeated_backing"], "Shared backing accounting failed")
        features.append(f)
    _require(any(f["merged_gaps"] for f in features[:nt]), "Training lacks a multi-card gap")
    _require(any(_unbound_alternative(_infer(_grid(pair["input"]))) is not None
                 for pair in puzzle["train"]), "Training fails to distinguish unbound fragments")
    _require(all(f["depth"] == 2 for f in features[:nt]), "Training overlap-depth contract violated")
    _require(all(f["depth"] >= 3 for f in features[nt:]), "Test lacks the higher-order overlap shift")
    _require(any(not np.array_equal(np.asarray(p["output"]), np.asarray(p["output"])[:, ::-1]) and
                 not np.array_equal(np.asarray(p["output"]), np.asarray(p["output"])[::-1])
                 for p in puzzle["train"]), "Training does not refute symmetric color copying")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed semantic, metamorphic, and diversity checks."""
    signatures, layouts, shapes, masks, backgrounds, symmetries = set(), set(), set(), set(), set(), set()
    totals = dict(pairs=0, merged=0, fragmented=0, max_depth=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _infer(pair["input"])
                f = _features(a)
                layouts.add((a["board"].shape, (a["board"] == a["background"]).tobytes()))
                backgrounds.add(a["background"])
                shapes.add(a["board"].shape)
                symmetries.add(f["symmetry"])
                for card in a["cards"]:
                    masks.add((card.shape, (card == a["scaffold"]).tobytes()))
                totals["pairs"] += 1
                totals["merged"] += bool(f["merged_gaps"])
                totals["fragmented"] += bool(f["fragmented"])
                totals["max_depth"] = max(totals["max_depth"], f["depth"])
    _require(len(signatures) == 42 and len(layouts) >= 100, "Insufficient structural diversity")
    _require(len(shapes) >= 25 and len(masks) >= 150, "Insufficient geometry variation")
    _require(backgrounds == set(range(10)) and {"lr", "tb"} <= symmetries, "Role/orientation diversity missing")
    first = generate_puzzle(314159, 2, 1)
    _require(_signature(first) == _signature(generate_puzzle(314159, 2, 1)), "Nondeterministic generation")
    pair = first["train"][0]
    a = _infer(pair["input"])
    # Whole-scene rotation and color permutation must commute with the solution.
    for turn in (1, 2, 3):
        _require(np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn)),
                 "Rotation changed the rule")
    palette = np.array([8, 3, 0, 6, 1, 9, 4, 2, 7, 5], dtype=np.int8)
    _require(np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]]), "Literal-color shortcut")
    # Ordinary A-style rectangle packing cannot place even one of these carriers
    # entirely on holes, although its registered placement fills them correctly.
    holes = a["board"] == a["background"]
    _require(all(not holes[y:y + card.shape[0], x:x + card.shape[1]].all()
                 for card, y, x in a["path"]), "Ordinary rectangular packing shortcut survived")
    alternative = _unbound_alternative(a)
    _require(alternative is not None and not np.array_equal(alternative, a["output"]),
             "Ignoring rectangular binding did not change the answer")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    fixed = tuple(np.argwhere(a["board"] == a["scaffold"])[0])
    bad["train"][0]["output"][fixed] = a["background"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Validator accepted a changed scaffold cell")
    # Exact physical inventory: deleting one card's payload cannot fill the board.
    _require(not _assemble(a["board"], a["background"], a["scaffold"], a["cards"][1:])[0],
             "Missing card was copied")
    # Visually identical physical copies are interchangeable. Differently
    # printed carriers with the same mask can instead give genuine ambiguity.
    board = np.full((5, 7), 8, dtype=np.int8)
    board[1, 1:3] = board[1, 4:6] = 0
    card = np.array([[1, 2], [8, 8]], dtype=np.int8)
    other = np.array([[3, 4], [8, 8]], dtype=np.int8)
    _require(len(_assemble(board, 0, 8, [card, card])[0]) == 1,
             "Identical-copy exchanges counted as ambiguity")
    _require(len(_assemble(board, 0, 8, [card, other])[0]) == 2,
             "Differently printed carriers were silently exchanged")
    vertical_board = np.full((6, 5), 8, dtype=np.int8)
    vertical_board[1:4, 1] = 0
    vertical_card = np.array([[1, 8], [2, 8], [3, 8]], dtype=np.int8)
    _require(len(_assemble(vertical_board, 0, 8, [vertical_card])[0]) == 1 and
             not _assemble(vertical_board, 0, 8, [np.rot90(vertical_card)])[0],
             "An independently rotated card was accepted")
    # Source location/order carries no matching information.
    rescattered = _scatter(np.random.default_rng(932), a["board"], list(reversed(a["cards"])), a["background"])
    _require(rescattered is not None and np.array_equal(solve(rescattered), a["output"]),
             "Source tray positions affected the answer")
    return dict(seeds=42, distinct_puzzles=len(signatures), layouts=len(layouts),
                board_shapes=len(shapes), carrier_masks=len(masks), **totals)
