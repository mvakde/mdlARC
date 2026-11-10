"""Count-locked jigsaws: glyph multiplicities constrain piece boundary exposure.

All interpretation in solve() comes from the supplied grid. Construction uses
partitions; solving uses rigid, colored exact cover with per-piece border quotas.
"""
from __future__ import annotations

from math import isqrt
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
BLOCK = ((0, 0), (0, 1), (1, 0), (1, 1))
SINGLETON = ((0, 0),)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, steps=NEIGHBORS):
    remaining = set((int(r), int(c)) for r, c in np.argwhere(mask))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(cells)
    return groups


def _tight(grid, cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    r1, c1 = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    result = np.full((r1 - r0, c1 - c0), -1, dtype=np.int8)
    for r, c in cells:
        result[r - r0, c - c0] = grid[r, c]
    return result


def _rotations(piece):
    seen, result = set(), []
    for k in range(4):
        a = np.rot90(piece, k).copy()
        if _key(a) not in seen:
            seen.add(_key(a))
            result.append(a)
    return result


def _read_input(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "The entire input border must be background")
    groups = _components(grid != bg)
    pieces, tokens = [], []
    for cells in groups:
        values = {int(grid[r, c]) for r, c in cells}
        if len(values) == 1:
            tokens.append((next(iter(values)), cells))
        else:
            pieces.append(_tight(grid, cells))
    _require(3 <= len(pieces) <= 5, "There must be three to five tagged rigid pieces")
    color_sets = [set(map(int, p[p >= 0])) for p in pieces]
    common = set.intersection(*color_sets)
    _require(len(common) == 1, "Pieces must share exactly one rivet color")
    rivet = next(iter(common))
    _require(all(np.sum(p == rivet) == 1 for p in pieces), "Each piece needs exactly one rivet")
    marked = [i for i, colors in enumerate(color_sets) if len(colors) == 3]
    _require(len(marked) == 1 and all(len(c) in (2, 3) for c in color_sets),
             "Exactly one piece must have an additional anchor marker")
    anchor = marked[0]
    extras = color_sets[anchor] - {rivet}
    marker_candidates = [c for c in extras if np.sum(grid == c) == 1]
    _require(len(marker_candidates) == 1, "The marked piece needs one globally unique marker")
    marker = marker_candidates[0]
    carriers = []
    for p, colors in zip(pieces, color_sets):
        values = colors - {rivet, marker}
        _require(len(values) == 1, "Every piece must have one category carrier")
        color = next(iter(values))
        _require(np.sum(p == color) >= 2, "Every carrier must occupy at least two cells")
        carriers.append(color)
    _require(len(set(carriers)) == len(pieces), "Carrier categories must be distinct")
    _require(not any(c in (rivet, marker) for c, cells in tokens), "Rivets and markers are not loose glyphs")
    counts = {c: sum(color == c for color, cells in tokens) for c in carriers}
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = isqrt(area)
    _require(4 <= side <= 8 and side * side == area, "Piece area must be a square of side 4..8")
    _require(sum(counts.values()) == 4 * side - 4, "Quota sum must equal the square's boundary cell count")
    return dict(grid=grid, background=bg, rivet=rivet, marker=marker, anchor=anchor,
                pieces=pieces, carriers=carriers, counts=counts, side=side, tokens=tokens)


def _packings(pieces, side, marker, quotas=None, *, limit=2, allow_rotations=True):
    """Enumerate distinct colored answers; ties of placements are harmless.

    quotas[i] is the number of cells owned by piece i on the target boundary.
    Symmetric masks are deduplicated only when their colored patterns agree.
    There is no seed, template, generation metadata, or answer in this search.
    """
    full = (1 << (side * side)) - 1
    border = sum(1 << (r * side + c) for r in range(side) for c in range(side)
                 if r in (0, side - 1) or c in (0, side - 1))
    candidates = []
    for i, piece in enumerate(pieces):
        options = {}
        oriented = _rotations(piece) if allow_rotations else [piece]
        for p in oriented:
            h, w = p.shape
            if h > side or w > side:
                continue
            cells = tuple((int(r) * side + int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0))
            base = sum(1 << pos for pos, color in cells)
            if np.any(piece == marker):
                offsets = [0] if p[0, 0] == marker else []
            else:
                offsets = [r * side + c for r in range(side - h + 1) for c in range(side - w + 1)]
            for offset in offsets:
                mask = base << offset
                if quotas is not None and (mask & border).bit_count() != quotas[i]:
                    continue
                values = tuple((pos + offset, color) for pos, color in cells)
                options[(mask, values)] = (mask, values)
        candidates.append(list(options.values()))
    answers, chosen, dead = {}, [None] * len(pieces), set()

    def search(used, remaining):
        if len(answers) >= limit:
            return True
        state = (used, remaining)
        if state in dead:
            return False
        if not remaining:
            if used != full:
                return False
            answer = np.empty((side, side), dtype=np.int8)
            for mask, cells in chosen:
                for pos, color in cells:
                    answer.flat[pos] = color
            answers[answer.tobytes()] = answer
            return True
        fitting = [(i, [x for x in candidates[i] if not x[0] & used]) for i in remaining]
        if any(not options for i, options in fitting):
            dead.add(state)
            return False
        # Choose an uncovered cell with the fewest legal covering placements.
        empty, best = full ^ used, None
        while empty:
            bit = empty & -empty
            empty -= bit
            ways = [(i, x) for i, options in fitting for x in options if x[0] & bit]
            if not ways:
                dead.add(state)
                return False
            if best is None or len(ways) < len(best):
                best = ways
            if len(best) == 1:
                break
        success = False
        for i, candidate in best:
            chosen[i] = candidate
            success |= search(used | candidate[0], tuple(j for j in remaining if j != i))
            if len(answers) >= limit:
                return True
        if not success:
            dead.add(state)
        return success

    search(0, tuple(range(len(pieces))))
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    quotas = [facts['counts'][c] for c in facts['carriers']]
    answers = _packings(facts['pieces'], facts['side'], facts['marker'], quotas)
    _require(len(answers) == 1, "Input must admit exactly one colored count-locked assembly")
    return answers[0].copy()


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _grown_shape(rng, height, width, area, diagonal=False):
    occupied = {(int(rng.integers(height)), int(rng.integers(width)))}
    steps = NEIGHBORS if diagonal else EDGE
    while len(occupied) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in steps
                           if 0 <= r + dr < height and 0 <= c + dc < width} - occupied)
        occupied.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize(occupied)


def _shape_mask(shape):
    mask = np.zeros((1 + max(r for r, c in shape), 1 + max(c for r, c in shape)), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _grow_partition(rng, side, count, diagonal):
    labels = np.full((side, side), -1, dtype=np.int8)
    points = rng.choice(side * side, count, replace=False)
    for i, pos in enumerate(points):
        labels.flat[pos] = i
    weights = np.exp(rng.uniform(-1.0, 1.0, count))
    while np.any(labels < 0):
        options = []
        for i in range(count):
            steps = NEIGHBORS if diagonal and rng.random() < .22 else EDGE
            frontier = sorted({(int(r) + dr, int(c) + dc) for r, c in np.argwhere(labels == i)
                               for dr, dc in steps if 0 <= r + dr < side and 0 <= c + dc < side
                               and labels[r + dr, c + dc] < 0})
            if frontier:
                options.append((float(np.sum(labels == i)) / weights[i] + float(rng.random()), i, frontier))
        _, i, frontier = min(options)
        r, c = frontier[int(rng.integers(len(frontier)))]
        labels[r, c] = i
    return labels


def _stepped_cuts(rng, side, count, diagonal):
    boxes = [(0, 0, side, side)]
    while len(boxes) < count:
        options = [(i, axis, t) for i, (r, c, h, w) in enumerate(boxes)
                   for axis, length, breadth in ((0, h, w), (1, w, h))
                   for t in range(1, length) if min(t, length - t) * breadth >= 3]
        if not options:
            return None
        i, axis, t = options[int(rng.integers(len(options)))]
        r, c, h, w = boxes.pop(i)
        boxes.extend([(r, c, t, w), (r + t, c, h - t, w)] if axis == 0 else
                     [(r, c, h, t), (r, c + t, h, w - t)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(boxes):
        labels[r:r + h, c:c + w] = i
    steps = NEIGHBORS if diagonal else EDGE
    for _ in range(side * 5):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < side and 0 <= cc < side):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.sum(labels == a) <= 3:
            continue
        labels[r, c] = b
        if len(_components(labels == a, steps)) != 1:
            labels[r, c] = a
    return labels


def _cavity_partition(rng, side, count, diagonal, require_diagonal):
    """Congruent asymmetric inserts at different depths make counts necessary."""
    for _ in range(35):
        shape = _grown_shape(rng, 3, 3, int(rng.integers(3, 7)), diagonal)
        mask = _shape_mask(shape)
        if len({_key(a) for a in _rotations(mask.astype(np.int8))}) < 4:
            continue
        if require_diagonal and len(_components(mask, EDGE)) == 1:
            continue
        labels = np.zeros((side, side), dtype=np.int8)
        success = True
        for i in range(1, count):
            p = mask if i <= 2 else _shape_mask(_grown_shape(rng, 3, 3, int(rng.integers(3, 7)), diagonal))
            p = np.rot90(p, int(rng.integers(4)))
            h, w = p.shape
            options = []
            for r in range(side - h + 1):
                for c in range(side - w + 1):
                    proposal = labels.copy()
                    view = proposal[r:r + h, c:c + w]
                    if np.any(view[p] != 0):
                        continue
                    view[p] = i
                    steps = NEIGHBORS if diagonal else EDGE
                    if proposal[0, 0] != 0 or len(_components(proposal == 0, steps)) != 1:
                        continue
                    boundary = int(np.sum(proposal[0] == i) + np.sum(proposal[-1] == i)
                                   + np.sum(proposal[1:-1, 0] == i) + np.sum(proposal[1:-1, -1] == i))
                    if (i == 1 and boundary == 0) or (i == 2 and boundary != 0):
                        continue
                    options.append(proposal)
            if not options:
                success = False
                break
            labels = options[int(rng.integers(len(options)))]
        if success:
            return labels
    return None


def _pieces_from_partition(rng, labels, colors, rivet, marker, witness):
    side = labels.shape[0]
    anchor = int(labels[0, 0])
    order = [anchor] + [i for i in range(len(colors)) if i != anchor]
    border = np.zeros(labels.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    output = np.empty(labels.shape, dtype=np.int8)
    quotas, pieces = [], []
    for i, label in enumerate(order):
        cells = [(int(r), int(c)) for r, c in np.argwhere(labels == label)]
        if len(cells) < (4 if i == 0 else 3):
            return None
        output[labels == label] = colors[i]
        choices = [(r, c) for r, c in cells if (r, c) != (0, 0)]
        # A border rivet demonstrates that quotas count owned cells, not
        # merely literal carrier-colored cells. The anchor marker also does.
        edge_choices = [(r, c) for r, c in choices if border[r, c]]
        if witness and edge_choices:
            choices = edge_choices
        r, c = choices[int(rng.integers(len(choices)))]
        output[r, c] = rivet
        quotas.append(int(np.sum((labels == label) & border)))
    output[0, 0] = marker
    for label in order:
        pieces.append(_tight(output, [(int(r), int(c)) for r, c in np.argwhere(labels == label)]))
    return output, pieces, quotas


def _twin_partition(rng, side, count, diagonal, require_diagonal):
    """Carve two congruent inserts, then partition their surrounding material.

    Swapping the inserts witnesses geometric ambiguity even when the rest
    becomes several interlocking regions rather than a single enclosing frame.
    """
    labels = _cavity_partition(rng, side, 3, diagonal, require_diagonal)
    if labels is None or count == 3:
        return labels
    available = labels == 0
    points = [(0, 0)] + [tuple(map(int, p)) for p in rng.permutation(np.argwhere(available &
              (np.indices(labels.shape).sum(axis=0) > 0)))[:count - 3]]
    regions = [0] + list(range(3, count))
    labels[available] = -1
    for label, point in zip(regions, points):
        labels[point] = label
    weights = np.exp(rng.uniform(-.7, .7, len(regions)))
    steps = NEIGHBORS if diagonal else EDGE
    while np.any(labels < 0):
        options = []
        for i, label in enumerate(regions):
            frontier = sorted({(int(r) + dr, int(c) + dc) for r, c in np.argwhere(labels == label)
                               for dr, dc in steps if 0 <= r + dr < side and 0 <= c + dc < side
                               and labels[r + dr, c + dc] < 0})
            if frontier:
                options.append((float(np.sum(labels == label)) / weights[i] + float(rng.random()), label, frontier))
        _, label, frontier = min(options)
        point = frontier[int(rng.integers(len(frontier)))]
        labels[point] = label
    return labels


def _token_shape(rng):
    mode = int(rng.integers(7))
    if mode == 0:
        return DIAMOND
    if mode == 1:
        return BLOCK
    if mode == 2:
        return SINGLETON
    if mode == 3:
        return tuple((0, c) for c in range(int(rng.integers(2, 5))))
    if mode == 4:
        return ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    return _grown_shape(rng, 3, 3, int(rng.integers(2, 8)), diagonal=True)


def _scatter(rng, arrays, background):
    """Pack actual occupied cells with a Chebyshev halo, then add blank margins."""
    order = sorted(range(len(arrays)), key=lambda i: (-arrays[i].size, -int(np.sum(arrays[i] >= 0))))
    for attempt in range(20):
        height = 28 if attempt >= 8 else int(rng.integers(22, 29))
        width = 28 if attempt >= 8 else int(rng.integers(22, 29))
        grid = np.full((height, width), background, dtype=np.int8)
        blocked = np.zeros((height, width), dtype=bool)
        okay = True
        for i in order:
            p = arrays[i]
            h, w = p.shape
            nh, nw = height - h + 1, width - w + 1
            if min(nh, nw) <= 0:
                okay = False
                break
            allowed = np.ones((nh, nw), dtype=bool)
            cells = np.argwhere(p >= 0)
            for r, c in cells:
                allowed &= ~blocked[r:r + nh, c:c + nw]
            locations = np.argwhere(allowed)
            if not len(locations):
                okay = False
                break
            r0, c0 = locations[int(rng.integers(len(locations)))]
            for r, c in cells:
                rr, cc = int(r0 + r), int(c0 + c)
                grid[rr, cc] = p[r, c]
                blocked[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2] = True
        if not okay:
            continue
        coords = np.argwhere(grid != background)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        cropped = grid[r0:r1, c0:c1]
        h, w = cropped.shape
        # Blank outer border is a visible role cue, not a dominant-color guess.
        H = h + int(rng.integers(2, min(5, 30 - h) + 1))
        W = w + int(rng.integers(2, min(5, 30 - w) + 1))
        top, left = int(rng.integers(1, H - h)), int(rng.integers(1, W - w))
        result = np.full((H, W), background, dtype=np.int8)
        result[top:top + h, left:left + w] = cropped
        return result
    return None


def _make_pair(rng, spec):
    colors, bg, rivet, marker = spec['colors'], spec['background'], spec['rivet'], spec['marker']
    count, side = len(colors), spec['side']
    for attempt in range(260):
        preferred_attempts = 32 if spec['count_required'] and not spec['witness'] else 160
        style = spec['style'] if attempt < preferred_attempts else 'twins' if spec['count_required'] else 'growth'
        if style == 'twins':
            labels = _twin_partition(rng, side, count, spec['diagonal'], spec['require_diagonal'])
        elif style == 'cavities':
            labels = _cavity_partition(rng, side, count, spec['diagonal'], spec['require_diagonal'])
        elif style == 'cuts':
            labels = _stepped_cuts(rng, side, count, spec['diagonal'])
        else:
            labels = _grow_partition(rng, side, count, spec['diagonal'])
        if labels is None:
            continue
        if any(len(_components(labels == i)) != 1 for i in range(count)):
            continue
        if any(len(_components(labels == i, EDGE)) > 3 for i in range(count)):
            continue
        if spec['require_diagonal'] and not any(len(_components(labels == i, EDGE)) > 1 for i in range(count)):
            continue
        built = _pieces_from_partition(rng, labels, colors, rivet, marker, spec['witness'])
        if built is None:
            continue
        output, pieces, quotas = built
        if spec['anchor_2d'] and min(pieces[0].shape) < 2:
            continue
        if spec['witness'] and (0 not in quotas or max(quotas) < 3):
            continue
        answers = _packings(pieces, side, marker, quotas)
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        if spec['count_required'] and len(_packings(pieces, side, marker)) < 2:
            continue
        rotations = [spec['anchor_rotation']] + list(map(int, rng.integers(4, size=count - 1)))
        arrays = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
        # Glyphs of a color can have different areas and shapes. Equal-area
        # diamond and block plus singleton rule out counting colored pixels.
        witness_color = int(np.argmax(quotas))
        for i, (color, quota) in enumerate(zip(colors, quotas)):
            for j in range(quota):
                shape = ((DIAMOND, BLOCK, SINGLETON)[j] if spec['witness'] and i == witness_color and j < 3
                         else _token_shape(rng))
                mask = _shape_mask(shape)
                arrays.append(np.where(mask, color, -1).astype(np.int8))
        for color, multiplicity in spec['distractors']:
            for _ in range(multiplicity):
                mask = _shape_mask(_token_shape(rng))
                arrays.append(np.where(mask, color, -1).astype(np.int8))
        grid = _scatter(rng, arrays, bg)
        if grid is None:
            continue
        facts = _read_input(grid)
        if not np.array_equal(solve(grid), output):
            raise AssertionError("Construction and input-derived packing disagree")
        return {'input': grid, 'output': output}
    raise RuntimeError("Bounded construction could not find a unique count-locked jigsaw")


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # larger assembly, more pieces, held-out anchor corner
    diagonal = bool(rng.integers(2))
    stable_palette = rng.random() < .25
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_side = int(rng.integers(5, 7))
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            training = split == 'train'
            witness = training and index == 0
            if regime == 0:
                side = first_side if witness else int(rng.integers(4, 7)) if training else int(rng.integers(7, 9))
                count = int(rng.integers(3, 5)) if witness else int(rng.integers(3, 6))
            elif regime == 1:
                side = first_side if witness else int(rng.integers(5, 8))
                count = 3 + index % 2 if training else 5
            else:
                side = first_side if witness else int(rng.integers(5, 8))
                count = int(rng.integers(3, 5)) if witness else int(rng.integers(3, 6))
            # Five tagged pieces need at least sixteen cells, with nearly no
            # geometric freedom on a 4x4 board. Keep that scale for 3/4 pieces.
            if count == 5:
                side = max(side, 5)
            if training and index == 1 and side == first_side:
                side = 5 if first_side == 6 else 6
            if stable_palette:
                local = palette
            else:
                local = list(map(int, rng.permutation(10)))
            bg, rivet, marker = local[:3]
            colors = local[3:3 + count]
            available = local[3 + count:]
            nd = int(rng.integers(0, min(2, len(available)) + 1))
            if training and index == 1:
                nd = max(1, nd)
            distractors = [(c, int(rng.integers(1, 4))) for c in available[:nd]]
            anchor_rotation = (int(rng.choice((1, 3))) if witness else
                               int(rng.choice((0, 1, 3))) if training and regime == 2 else
                               2 if not training and regime == 2 else int(rng.integers(4)))
            spec = dict(side=side, colors=colors, background=bg, rivet=rivet, marker=marker,
                        distractors=distractors, witness=witness, diagonal=diagonal,
                        count_required=witness or not training,
                        anchor_2d=regime == 2,
                        require_diagonal=witness and diagonal, anchor_rotation=anchor_rotation,
                        style='cavities' if witness else ('growth', 'cuts', 'cavities')[index % 3])
            for retry in range(10):
                pair = _make_pair(rng, spec)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def _features(pair):
    facts = _read_input(pair['input'])
    quotas = [facts['counts'][c] for c in facts['carriers']]
    token_areas = {c: [len(cells) for color, cells in facts['tokens'] if color == c] for c in facts['carriers']}
    token_shapes = {c: {_normalize(cells) for color, cells in facts['tokens'] if color == c}
                    for c in facts['carriers']}
    anchor_piece = facts['pieces'][facts['anchor']]
    r, c = map(int, np.argwhere(anchor_piece == facts['marker'])[0])
    h, w = anchor_piece.shape
    corners = tuple(name for rr, cc, name in ((0, 0, 'TL'), (0, w - 1, 'TR'),
                    (h - 1, 0, 'BL'), (h - 1, w - 1, 'BR')) if (r, c) == (rr, cc))
    facts.update(quotas=quotas,
                 marker_corners=corners,
                 diagonal_tokens=any(len(_components(_shape_mask(_normalize(cells)), EDGE)) > 1
                                     for c, cells in facts['tokens'] if c in facts['carriers']),
                 mixed_areas=any(len(set(v)) > 1 for v in token_areas.values()),
                 counting_witness=any({DIAMOND, BLOCK, SINGLETON} <= shapes for shapes in token_shapes.values()),
                 diagonal_pieces=any(len(_components(p >= 0, EDGE)) > 1 for p in facts['pieces']),
                 areas=tuple(sorted(int(np.sum(p >= 0)) for p in facts['pieces'])),
                 distractor_count=sum(c not in facts['carriers'] for c, cells in facts['tokens']),
                 count_essential=len(_packings(facts['pieces'], facts['side'], facts['marker'])) > 1,
                 rotation_essential=not _packings(facts['pieces'], facts['side'], facts['marker'], quotas,
                                                allow_rotations=False))
    return facts


def validate_generated_puzzle(puzzle):
    """Check visible semantics, conservation, uniqueness, and training witnesses."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        _check_counts(len(puzzle['train']), len(puzzle['test']))
        seen, records = set(), {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                inp, out = _grid(pair['input']), _grid(pair['output'])
                assert _key(inp) not in seen, "Duplicate input"
                seen.add(_key(inp))
                f = _features(pair)
                assert np.array_equal(solve(inp), out), "Wrong count-locked assembly"
                assert out.shape == (f['side'], f['side']) and out[0, 0] == f['marker']
                assert not np.any(out == f['background'])
                # Preserve piece cells only; category glyphs and distractors vanish.
                for color in range(10):
                    expected = sum(int(np.sum(p == color)) for p in f['pieces'])
                    assert int(np.sum(out == color)) == expected, "Piece color conservation failed"
                assert int(np.sum(out == f['rivet'])) == len(f['pieces'])
                assert int(np.sum(out == f['marker'])) == 1
                border = np.zeros(out.shape, dtype=bool)
                border[[0, -1], :] = True
                border[:, [0, -1]] = True
                for p, color, quota in zip(f['pieces'], f['carriers'], f['quotas']):
                    # The unique colored placement reconstructs the owner mask.
                    placements = []
                    for oriented in _rotations(p):
                        h, w = oriented.shape
                        for r in range(f['side'] - h + 1):
                            for c in range(f['side'] - w + 1):
                                mask = oriented >= 0
                                if np.array_equal(out[r:r + h, c:c + w][mask], oriented[mask]):
                                    own = np.zeros(out.shape, dtype=bool)
                                    own[r:r + h, c:c + w] = mask
                                    placements.append(own)
                    # An isolated pattern match need not extend to a full
                    # packing. Shared rivets can also make two owner masks
                    # render the same colors; the exact-cover check handles
                    # their joint legality. Require a quota-compatible match.
                    assert any(int(np.sum(m & border)) == quota for m in placements), "Boundary quota failed"
                records[split].append(f)
        training = records['train']
        assert any(f['count_essential'] and 0 in f['quotas'] for f in training), "Missing count/depth witness"
        assert any(f['counting_witness'] for f in training), "Missing same-color diamond/block/singleton witness"
        assert any(f['rotation_essential'] for f in training), "Missing rigid-rotation witness"
        assert any(f['distractor_count'] for f in training), "Missing irrelevant-color witness"
        assert len({f['side'] for f in training}) >= 2, "Training must vary the assembly area"
        assert all(f['count_essential'] for f in records['test']), "Every test must require the counts"
        testing = records['test']
        scale_shift = min(f['side'] for f in testing) > max(f['side'] for f in training)
        count_shift = min(len(f['pieces']) for f in testing) > max(len(f['pieces']) for f in training)
        train_corners = {c for f in training for c in f['marker_corners']}
        test_corners = {c for f in testing for c in f['marker_corners']}
        corner_shift = test_corners == {'BR'} and 'BR' not in train_corners
        assert scale_shift or count_shift or corner_shift, "Missing structural train/test shift"
        if any(f['diagonal_pieces'] for f in records['test']):
            assert any(f['diagonal_pieces'] for f in training), "Demonstrate diagonal pieces before testing them"
    except AssertionError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise AssertionError(f"Malformed puzzle: {error}") from error


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed semantic and structural audit; no external resources."""
    fingerprints, dimensions, partitions, sides, piece_counts = set(), set(), set(), set(), set()
    role_colors = {role: set() for role in ('background', 'rivet', 'marker')}
    shapes, quota_profiles, zero_cases, count_cases, diagonal_cases = set(), set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                dimensions.add(tuple(pair['input'].shape))
                sides.add(f['side'])
                piece_counts.add(len(f['pieces']))
                quota_profiles.add(tuple(sorted(f['quotas'])))
                zero_cases += 0 in f['quotas']
                count_cases += f['count_essential']
                diagonal_cases += f['diagonal_pieces']
                for role in role_colors:
                    role_colors[role].add(f[role])
                for p in f['pieces']:
                    shapes.add(_key((p >= 0).astype(np.int8)))
                # Normalize carrier names while keeping the shared rivet role.
                mapping = {f['rivet']: 0, f['marker']: 1}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair['output'].flat)
                partitions.add((pair['output'].shape, normalized))
    assert len(fingerprints) == 42
    assert len(dimensions) >= 20 and len(shapes) >= 80 and len(partitions) >= 150
    assert sides == set(range(4, 9)) and piece_counts == {3, 4, 5}
    assert len(quota_profiles) >= 35 and zero_cases >= 42 and count_cases >= 84 and diagonal_cases >= 10
    assert all(len(colors) >= 8 for colors in role_colors.values())
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(901 + a + b, a, b)
        validate_generated_puzzle(p)
        assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
