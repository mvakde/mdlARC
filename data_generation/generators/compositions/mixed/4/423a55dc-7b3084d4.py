"""Four-piece anchored square assembly followed by a clipped left row shear.

Only NumPy and the standard library are used. No evidence files are loaded.
The solver first proves uniqueness of the complete square, before any clipping.
"""
from __future__ import annotations

from math import isqrt
import hashlib
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grids must have two dimensions of size 1..30.")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid cells must be integer colors 0..9.")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, neighbors=EIGHT):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _connected(mask, neighbors=EIGHT):
    return len(_components(mask, neighbors)) == 1


def _tight(mask, colors):
    rr, cc = np.nonzero(mask)
    view = colors[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    occupied = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return np.where(occupied, view, -1).astype(np.int8)


def _orientations(piece, turns=(0, 1, 2, 3)):
    seen, result = set(), []
    for k in turns:
        p = np.rot90(piece, k).copy()
        if _key(p) not in seen:
            result.append(p)
            seen.add(_key(p))
    return result


def _read_input(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "Input must have a uniform empty outer border.")
    colors, counts = np.unique(a, return_counts=True)
    singletons = [int(c) for c, n in zip(colors, counts) if c != bg and n == 1]
    _require(len(colors) == 6 and len(singletons) == 1,
             "Require background, four carriers, and one singleton marker.")
    marker = singletons[0]
    area = int(np.count_nonzero(a != bg))
    side = isqrt(area)
    _require(side >= 4 and side * side == area, "Total piece area must be a square, side >= 4.")
    groups = _components(a != bg)
    _require(len(groups) == 4, "Require four separated eight-connected objects.")
    pieces, carriers = [], []
    for cells in groups:
        mask = np.zeros(a.shape, dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        p = _tight(mask, a)
        values = set(map(int, p[p >= 0])) - {marker}
        _require(len(cells) >= 3 and len(values) == 1,
                 "Each piece needs >= 3 cells and one carrier color.")
        pieces.append(p)
        carriers.append(next(iter(values)))
    _require(len(set(carriers)) == 4, "Carrier colors must be distinct.")
    return a, bg, marker, side, pieces, carriers


def _assemblies(pieces, side, marker, turns=(0, 1, 2, 3)):
    """Enumerate up to two distinct complete colored, rotation-only packings.

    The unique marker must occupy (0,0). Because each piece is tightly cropped,
    a legal orientation of its marked piece has the marker at its own (0,0).
    First-empty-cell exact cover avoids piece-order permutations. The last
    piece is looked up by the exact remaining mask. No search budget can turn
    an unfinished uniqueness check into an accepted answer.
    """
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    if len(marked) != 1 or sum(int(np.sum(p >= 0)) for p in pieces) != side * side:
        return []
    anchor = marked[0]
    candidates = []
    for i, piece in enumerate(pieces):
        options = {}
        for p in _orientations(piece, turns):
            h, w = p.shape
            if h > side or w > side or (i == anchor and p[0, 0] != marker):
                continue
            cells = tuple((int(r) * side + int(c), int(p[r, c]))
                          for r, c in np.argwhere(p >= 0))
            base = sum(1 << pos for pos, _ in cells)
            origins = [(0, 0)] if i == anchor else (
                (r, c) for r in range(side - h + 1) for c in range(side - w + 1))
            for r, c in origins:
                offset = r * side + c
                mask = base << offset
                # A carrier is monochrome; the only exceptional color is
                # already fixed at (0,0), so equal masks have equal colors.
                options.setdefault(mask, (mask, offset, cells))
        if not options:
            return []
        candidates.append(options)
    full = (1 << (side * side)) - 1
    placed, answers, signatures = [None] * 4, [], set()

    def record():
        signature = tuple(p[0] for p in placed)
        if signature in signatures:
            return
        signatures.add(signature)
        answer = np.empty((side, side), dtype=np.int8)
        for _, offset, cells in placed:
            for pos, color in cells:
                answer.flat[pos + offset] = color
        answers.append(answer)

    def search(occupied, remaining):
        if len(answers) >= 2:
            return
        if len(remaining) == 1:
            i = remaining[0]
            candidate = candidates[i].get(full ^ occupied)
            if candidate is not None:
                placed[i] = candidate
                record()
                placed[i] = None
            return
        empty = full ^ occupied
        first = empty & -empty
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in candidates[i].values():
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(answers) >= 2:
                        return
            placed[i] = None

    rest = tuple(i for i in range(4) if i != anchor)
    for p in candidates[anchor].values():
        placed[anchor] = p
        search(p[0], rest)
        if len(answers) >= 2:
            break
    return answers


def _shear(square, background):
    """A's cellwise affine map on B's cropped square, with left clipping."""
    n = square.shape[0]
    out = np.full(square.shape, background, dtype=np.int8)
    rr, cc = np.indices(square.shape)
    dest = cc - (n - 1 - rr)
    keep = dest >= 0
    out[rr[keep], dest[keep]] = square[rr[keep], cc[keep]]
    return out


def _row_suffix_answer(square, background):
    # Construction/validation formulation, separate from the cellwise solver.
    n = square.shape[0]
    out = np.full((n, n), background, dtype=np.int8)
    for r in range(n):
        out[r, :r + 1] = square[r, n - 1 - r:]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique assembled square and its sheared result from input."""
    _, bg, marker, n, pieces, _ = _read_input(input_grid)
    answers = _assemblies(pieces, n, marker)
    _require(len(answers) == 1, "Input must have exactly one complete anchored colored square.")
    return _shear(answers[0], bg)


def _grow(rng, n, diagonal, balanced):
    neighbors = EIGHT if diagonal else EDGE
    labels = np.full((n, n), -1, dtype=np.int8)
    if balanced:
        seeds = [0, n - 1, n * (n - 1), n * n - 1]
        weights = rng.uniform(.8, 1.3, 4)
    else:
        seeds = [0] + list(map(int, rng.choice(np.arange(1, n * n), 3, replace=False)))
        weights = np.exp(rng.uniform(-1.1, 1.1, 4))
    adjacency = [[(r + dr) * n + c + dc for dr, dc in neighbors
                  if 0 <= r + dr < n and 0 <= c + dc < n]
                 for r in range(n) for c in range(n)]
    frontiers, sizes = [set() for _ in range(4)], [1] * 4
    for label, pos in enumerate(seeds):
        labels.flat[pos] = label
    for label, pos in enumerate(seeds):
        frontiers[label].update(p for p in adjacency[pos] if labels.flat[p] < 0)
    for _ in range(n * n - 4):
        eligible = [i for i in range(4) if frontiers[i]]
        i = min(eligible, key=lambda j: sizes[j] / weights[j] + float(rng.random()) * .35)
        choices = sorted(frontiers[i])
        if diagonal and rng.random() < .82:
            edge_choices = [p for p in choices if any(
                0 <= (r := p // n + dr) < n and 0 <= (c := p % n + dc) < n
                and labels[r, c] == i for dr, dc in EDGE)]
            if edge_choices:
                choices = edge_choices
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for frontier in frontiers:
            frontier.discard(pos)
        frontiers[i].update(p for p in adjacency[pos] if labels.flat[p] < 0)
    return labels


def _cuts(rng, n, diagonal):
    rectangles = [(0, 0, n, n)]
    while len(rectangles) < 4:
        options = []
        for i, (_, _, h, w) in enumerate(rectangles):
            options += [(i, 0, k) for k in range(1, h) if min(k, h - k) * w >= 3]
            options += [(i, 1, k) for k in range(1, w) if min(k, w - k) * h >= 3]
        if not options:
            return None
        i, axis, cut = options[int(rng.integers(len(options)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, cut, w), (r + cut, c, h - cut, w)] if axis == 0 else
                          [(r, c, h, cut), (r, c + cut, h, w - cut)])
    labels = np.empty((n, n), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r + h, c:c + w] = i
    for _ in range(5 * n):
        r, c = map(int, rng.integers(n, size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < n and 0 <= cc < n):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.sum(labels == a) <= 3:
            continue
        labels[r, c] = b
        if not _connected(labels == a, EIGHT if diagonal else EDGE):
            labels[r, c] = a
    return labels


def _inserts(rng, n):
    labels = np.zeros((n, n), dtype=np.int8)
    for i in range(1, 4):
        for _ in range(30):
            h, w = map(int, rng.integers(1, max(3, n // 2 + 1), size=2))
            if h * w < 3:
                continue
            r, c = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
            if np.any(labels[r:r + h, c:c + w]) or (r, c) == (0, 0):
                continue
            mask = np.ones((h, w), dtype=bool)
            if min(h, w) >= 2 and h * w > 4 and rng.random() < .65:
                mask[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
            if np.sum(mask) < 3:
                continue
            proposal = labels.copy()
            proposal[r:r + h, c:c + w][mask] = i
            if _connected(proposal == 0, EDGE):
                labels = proposal
                break
        else:
            return None
    return labels


def _colored(labels, colors, marker):
    anchor = int(labels[0, 0])
    order = [anchor] + [i for i in range(4) if i != anchor]
    square = np.empty(labels.shape, dtype=np.int8)
    for label, color in zip(order, colors):
        square[labels == label] = color
    square[0, 0] = marker
    return square, [_tight(labels == label, square) for label in order]


def _fallback_labels():
    # Solid enclosing piece with three isolated, different-area cavities.
    # Its full 8x8 box and corner marker fix orientation; cavity areas 3,4,5
    # force the three inserts, including one asymmetric L triomino.
    labels = np.zeros((8, 8), dtype=np.int8)
    labels[1, 5:7] = 1
    labels[2, 6] = 1
    labels[4, 3:7] = 2
    labels[6, 1:6] = 3
    return labels


def _frame(rng, canvas, background):
    rr, cc = np.nonzero(canvas >= 0)
    crop = canvas[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    h, w = crop.shape
    top = int(rng.integers(1, min(3, 29 - h) + 1))
    bottom = int(rng.integers(1, min(3, 30 - h - top) + 1))
    left = int(rng.integers(1, min(3, 29 - w) + 1))
    right = int(rng.integers(1, min(3, 30 - w - left) + 1))
    return np.pad(np.where(crop >= 0, crop, background),
                  ((top, bottom), (left, right)), constant_values=background).astype(np.int8)


def _scatter(rng, oriented, background, compact):
    # Greedy mask placements allow tight bounding boxes to interleave.
    if compact:
        for attempt in range(5):
            occupied = np.zeros((28, 28), dtype=bool)
            blocked = np.zeros_like(occupied)
            canvas = np.full((28, 28), -1, dtype=np.int8)
            order = sorted(range(4), key=lambda i: -oriented[i].size)
            maxr = maxc = 0
            success = True
            aspect = float(rng.uniform(.7, 1.4))
            for i in order:
                p = oriented[i]
                h, w = p.shape
                mask = p >= 0
                choices = []
                for r in range(29 - h):
                    for c in range(29 - w):
                        if np.any(blocked[r:r + h, c:c + w] & mask):
                            continue
                        hr, wc = max(maxr, r + h), max(maxc, c + w)
                        score = hr * wc + 2 * abs(hr - aspect * wc) + float(rng.random()) * 14
                        choices.append((score, r, c))
                if not choices:
                    success = False
                    break
                _, r, c = min(choices)
                view = canvas[r:r + h, c:c + w]
                view[mask] = p[mask]
                occupied[r:r + h, c:c + w] |= mask
                padded = np.pad(occupied, 1)
                blocked = np.logical_or.reduce([padded[1 + dr:29 + dr, 1 + dc:29 + dc]
                                                for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
                maxr, maxc = max(maxr, r + h), max(maxc, c + w)
            if success:
                return _frame(rng, canvas, background)
    # Tight-box rows, with shuffled membership and independent alignment.
    order = list(map(int, rng.permutation(4)))
    gap = int(rng.integers(1, 4))
    rows = [order[:2], order[2:]]
    heights = [max(oriented[i].shape[0] for i in group) for group in rows]
    widths = [sum(oriented[i].shape[1] for i in group) + gap for group in rows]
    canvas = np.full((sum(heights) + gap, max(widths)), -1, dtype=np.int8)
    r0 = 0
    for group, height, width in zip(rows, heights, widths):
        c0 = int(rng.integers(canvas.shape[1] - width + 1))
        for i in group:
            p = oriented[i]
            h, w = p.shape
            r = r0 + int(rng.integers(height - h + 1))
            view = canvas[r:r + h, c0:c0 + w]
            view[p >= 0] = p[p >= 0]
            c0 += w + gap
        r0 += height + gap
    return _frame(rng, canvas, background)


def _triangle_crop(square, bg):
    return np.where(np.indices(square.shape)[1] <= np.indices(square.shape)[0], square, bg)


def _local_anchor_differs(square, bg, marker, anchor_carrier):
    """Counterfactual: shear each assembled piece about its own lowest row."""
    out = np.full(square.shape, bg, dtype=np.int8)
    used = np.zeros(square.shape, dtype=bool)
    for color in np.unique(square):
        if color == marker:
            continue
        mask = square == color
        if color == anchor_carrier:
            mask |= square == marker
        rr, cc = np.nonzero(mask)
        dest = cc - (int(rr.max()) - rr)
        keep = dest >= 0
        if np.any(used[rr[keep], dest[keep]]):
            return True  # The counterfactual collides, unlike the intended map.
        used[rr[keep], dest[keep]] = True
        out[rr[keep], dest[keep]] = square[rr[keep], cc[keep]]
    return not np.array_equal(out, _row_suffix_answer(square, bg))


def _make_pair(rng, n, colors, bg, marker, rotation, style, diagonal, witness, anchor_2d):
    for attempt in range(241):
        if attempt == 240:
            labels = _fallback_labels()
        elif style == "inserts" and attempt < 100:
            labels = _inserts(rng, n)
        elif style == "cuts" and attempt < 100:
            labels = _cuts(rng, n, diagonal)
        else:
            labels = _grow(rng, n, diagonal, style == "balanced" and attempt < 100)
        if labels is None or min(int(np.sum(labels == i)) for i in range(4)) < 3:
            continue
        square, pieces = _colored(labels, colors, marker)
        if anchor_2d and min(pieces[0].shape) == 1:
            continue
        has_diagonal = any(not _connected(p >= 0, EDGE) for p in pieces)
        if any(len(_components(p >= 0, EDGE)) > 3 for p in pieces):
            continue
        # Only the first training demonstration is required to show a genuine
        # diagonal attachment; retry growth for it rather than demand it later.
        if witness and diagonal and not has_diagonal and attempt != 240:
            continue
        out = _row_suffix_answer(square, bg)
        surviving = set(map(int, np.unique(out))) - {bg}
        if len(surviving) < (4 if witness else 3):
            continue
        if np.array_equal(out, _triangle_crop(square, bg)):
            continue
        if witness and not _local_anchor_differs(square, bg, marker, colors[0]):
            continue
        if any(np.array_equal(out, _shear(np.rot90(square, k), bg)) for k in (1, 2, 3)):
            continue
        turns = [rotation] + list(map(int, rng.integers(4, size=3)))
        asym = [i for i in range(1, 4) if len(_orientations(pieces[i])) == 4]
        if witness:
            if not asym:
                continue
            turns[asym[0]] = int(rng.integers(1, 4))
        oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, turns)]
        answers = _assemblies(oriented, len(square), marker)
        if len(answers) != 1 or not np.array_equal(answers[0], square):
            continue
        inp = _scatter(rng, oriented, bg, bool(rng.random() < .65))
        return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a unique assembly and informative shear.")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}.")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    seed = int(seed)
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test.")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    marker, base_bg = palette[:2]
    mode = int(rng.integers(3))  # Stable, changing carriers, or changing backgrounds too.
    scale_shift = bool(rng.integers(2))
    diagonal = bool(rng.random() < .4)
    held = int(rng.integers(4))
    train_n = [int(rng.integers(4, 7)), int(rng.integers(7, 9))]
    train_n += list(map(int, rng.integers(4, 9, size=a - 2)))
    rng.shuffle(train_n)
    train_turns = [int(k) for _ in range((a + 2) // 3)
                   for k in rng.permutation([v for v in range(4) if v != held])][:a]
    if scale_shift:
        test_n = list(map(int, rng.integers(9, 12, size=b)))
        test_turns = list(map(int, rng.integers(4, size=b)))
    else:
        test_n = [train_n[int(rng.integers(a))] for _ in range(b)]
        test_turns = [held] * b
    styles = [str(k) for _ in range((a + b + 3) // 4)
              for k in rng.permutation(["growth", "balanced", "cuts", "inserts"])]
    puzzle, seen = {"train": [], "test": []}, set()
    diagonal_seen = False
    previous_bg = base_bg
    for i, (n, turn) in enumerate(zip(train_n + test_n, train_turns + test_turns)):
        bg = (int(rng.choice([c for c in range(10) if c not in (marker, previous_bg)]))
              if mode == 2 and i else base_bg)
        previous_bg = bg
        available = [c for c in range(10) if c not in (bg, marker)]
        colors = palette[2:6] if mode == 0 else list(map(int, rng.choice(available, 4, replace=False)))
        allow_diagonal = diagonal and (i < a or diagonal_seen)
        style = "growth" if i == 0 and allow_diagonal else styles[i]
        for _ in range(8):
            pair = _make_pair(rng, n, colors, bg, marker, turn, style, allow_diagonal,
                              i == 0, not scale_shift)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("Could not scatter a distinct input.")
        seen.add(key)
        if i < a:
            pieces = _read_input(pair["input"])[4]
            diagonal_seen |= any(not _connected(p >= 0, EDGE) for p in pieces)
        puzzle["train" if i < a else "test"].append(pair)
    return puzzle


def _facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema.")
    inp, bg, marker, n, pieces, carriers = _read_input(pair["input"])
    out = _grid(pair["output"])
    answers = _assemblies(pieces, n, marker)
    _require(len(answers) == 1, "Complete square must be unique before clipping.")
    square = answers[0]
    _require(out.shape == (n, n) and np.array_equal(out, _row_suffix_answer(square, bg)),
             "Output must be the clipped bottom-anchored left shear of the assembled square.")
    _require(np.count_nonzero(out != bg) == n * (n + 1) // 2, "Incorrect clipping area.")
    _require(np.array_equal(out[-1], square[-1]), "Bottom row must be fixed.")
    _require(not np.any(out == marker), "Top-left marker must be clipped away.")
    _require(not np.array_equal(out, square), "Omitting A must change the answer.")
    _require(not np.array_equal(out, _triangle_crop(square, bg)), "A triangle mask is an invalid shortcut.")
    _require(all(not np.array_equal(out, _shear(np.rot90(square, k), bg)) for k in (1, 2, 3)),
             "The marker's choice of global orientation must affect visible colors.")
    # Omitting B means shearing the scattered canvas using its own bottom.
    raw = np.full(inp.shape, bg, dtype=np.int8)
    rr, cc = np.nonzero(inp != bg)
    dest = cc - (int(rr.max()) - rr)
    keep = dest >= 0
    raw[rr[keep], dest[keep]] = inp[rr[keep], cc[keep]]
    _require(not np.array_equal(raw, out), "Omitting B must change the answer.")
    visible = set(map(int, np.unique(out))) - {bg}
    _require(len(visible) >= 3, "At least three carriers must remain visible.")
    diagonal = any(not _connected(p >= 0, EDGE) for p in pieces)
    _require(all(len(_components(p >= 0, EDGE)) <= 3 for p in pieces),
             "Generated diagonal pieces must remain visually coherent.")
    asymmetric_rotation = False
    for p, carrier in zip(pieces, carriers):
        mask = square == carrier
        if np.any(p == marker):
            mask |= square == marker
        target = _tight(mask, square)
        if not np.array_equal(p, target) and len(_orientations(p)) == 4:
            asymmetric_rotation = True
    marked = next(p for p in pieces if np.any(p == marker))
    anchor_carrier = next(int(c) for c in np.unique(marked) if c >= 0 and c != marker)
    mr, mc = map(int, np.argwhere(marked == marker)[0])
    corner = (int(mr != 0), int(mc != 0))
    normalized, mapping = [], {}
    for v in square.flat:
        v = int(v)
        mapping.setdefault(v, len(mapping))
        normalized.append(mapping[v])
    structure = (n, bytes(normalized))
    return dict(bg=bg, marker=marker, n=n, corner=corner, diagonal=diagonal,
                rotation=asymmetric_rotation, visible=len(visible), square=square,
                local_differs=_local_anchor_differs(square, bg, marker, anchor_carrier),
                structure=structure, areas=tuple(int(np.sum(p >= 0)) for p in pieces))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts.")
    seen, facts = set(), {}
    for split in ("train", "test"):
        facts[split] = []
        for pair in puzzle[split]:
            f = _facts(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "Inputs must be distinct within a puzzle.")
            seen.add(key)
            facts[split].append(f)
    train, test = facts["train"], facts["test"]
    _require(len({f["marker"] for f in train + test}) == 1, "Marker color must be fixed per puzzle.")
    _require(any(f["rotation"] and f["visible"] == 4 for f in train),
             "Training must show asymmetric rigid rotation and all four surviving carriers.")
    _require(any(f["local_differs"] for f in train), "Training must distinguish global from per-piece anchors.")
    if any(f["diagonal"] for f in test):
        _require(any(f["diagonal"] for f in train), "Demonstrate diagonal attachment before testing it.")
    if len({f["bg"] for f in train}) == 1:
        _require(all(f["bg"] == train[0]["bg"] for f in test), "Show background variation in training.")
    for pair in puzzle["test"]:
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "Input-only solve disagrees.")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(pa[k], pb[k]) for pa, pb in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded 42-seed checks; no files, logs, traces, or persistent caches."""
    structures, shapes, sides, backgrounds, markers, corners = set(), set(), set(), set(), set(), set()
    area_budgets, puzzle_hashes = set(), set()
    diagonal_pairs = rotations = four_visible = scale = holdout = pairs = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        train = [_facts(p) for p in puzzle["train"]]
        test = [_facts(p) for p in puzzle["test"]]
        scale += min(f["n"] for f in test) > max(f["n"] for f in train)
        holdout += not ({f["corner"] for f in train} & {f["corner"] for f in test})
        for pair, f in zip(puzzle["train"] + puzzle["test"], train + test):
            pairs += 1
            structures.add(f["structure"])
            shapes.add(np.asarray(pair["input"]).shape)
            sides.add(f["n"])
            backgrounds.add(f["bg"])
            markers.add(f["marker"])
            corners.add(f["corner"])
            area_budgets.add(tuple(sorted(f["areas"])))
            diagonal_pairs += f["diagonal"]
            rotations += f["rotation"]
            four_visible += f["visible"] == 4
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape) + g.tobytes())
            # Rotation without the marker produces different colored results.
            assert all(not np.array_equal(pair["output"], _shear(np.rot90(f["square"], k), f["bg"]))
                       for k in (1, 2, 3))
        puzzle_hashes.add(digest.hexdigest())
        if seed == 0:
            first = puzzle
    assert len(puzzle_hashes) == 42 and pairs == 210
    assert len(structures) >= 170 and len(shapes) >= 35 and len(area_budgets) >= 60
    assert sides == set(range(4, 12)) and len(backgrounds) >= 8 and len(markers) >= 8
    assert len(corners) == 4 and diagonal_pairs >= 10 and rotations >= 100 and four_visible >= 60
    assert scale >= 8 and holdout >= 8
    assert _same(first, generate_puzzle(0)) and not _same(first, generate_puzzle(43))
    # Recoloring and rescattering must not expose any hidden palette or layout
    # parameter. The input-only solver must also handle the safety geometry.
    pair = first["test"][0]
    before = pair["input"].copy()
    recolor = np.arange(9, -1, -1, dtype=np.int8)
    assert np.array_equal(solve(recolor[before]), recolor[pair["output"]])
    _, bg, _, _, pieces, _ = _read_input(before)
    rng = np.random.default_rng(710)
    rearranged = _scatter(rng, [np.rot90(p, int(rng.integers(4))) for p in pieces], bg, True)
    assert np.array_equal(solve(rearranged), pair["output"])
    assert np.array_equal(pair["input"], before)
    square, pieces = _colored(_fallback_labels(), [2, 3, 4, 5], 1)
    for k in range(4):
        inp = _scatter(np.random.default_rng(k), [np.rot90(p, k) for p in pieces], 0, False)
        assert np.array_equal(solve(inp), _row_suffix_answer(square, 0))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(991, a, b)
        validate_generated_puzzle(p)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(991, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts.")
    # A negative semantic check must reject a plausible unsheared output.
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"] = _facts(first["test"][0])["square"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted output that omits shear.")
