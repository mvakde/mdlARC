"""Broken blueprints: jointly assemble cavity fragments and match scaled keys.

Only Python and NumPy are used. No files, parent generators, stored puzzles,
randomness, or training labels are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
from math import isqrt
import numpy as np

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEAR = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
BACKGROUND, CAVITY, ANCHOR = 0, 5, 9
COLORS = (1, 2, 3, 4, 6, 7, 8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)), "integer colors must be 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.tobytes()


def _components(mask, neighbors=EDGE):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _tight(g, cells):
    rs, cs = zip(*cells)
    r0, c0 = min(rs), min(cs)
    p = np.full((max(rs) - r0 + 1, max(cs) - c0 + 1), -1, dtype=np.int8)
    for r, c in cells:
        p[r-r0, c-c0] = g[r, c]
    return p


def _orientations(p, reflect=False):
    seen, result = set(), []
    for base in (p, np.fliplr(p)) if reflect else (p,):
        for k in range(4):
            q = np.rot90(base, k).copy()
            if _key(q) not in seen:
                seen.add(_key(q))
                result.append(q)
    return result


def _shape(cells):
    rs, cs = zip(*cells)
    p = np.zeros((max(rs)-min(rs)+1, max(cs)-min(cs)+1), dtype=np.int8)
    for r, c in cells:
        p[r-min(rs), c-min(cs)] = 1
    return p


def _canonical(mask, reflect=True):
    return min(_key(p) for p in _orientations(mask.astype(np.int8), reflect))


def _decode(inp):
    fragments, keys = [], []
    for cells in _components(inp != BACKGROUND, NEAR):
        p = _tight(inp, cells)
        colors = set(map(int, p[p >= 0]))
        if CAVITY in colors or ANCHOR in colors:
            carriers = colors - {CAVITY, ANCHOR}
            _require(len(carriers) == 1 and CAVITY in colors, "every fragment needs a carrier and cavity ink")
            _require(np.sum(p == next(iter(carriers))) >= 2, "fragment needs at least two carrier cells")
            fragments.append(p)
        else:
            _require(len(colors) == 1, "each enlarged key must be monochrome")
            keys.append(p)
    _require(3 <= len(fragments) <= 4 and 2 <= len(keys) <= 3, "need 3..4 fragments and 2..3 keys")
    _require(np.count_nonzero(inp == ANCHOR) == 1, "one anchor cell is required")
    carriers = [next(int(c) for c in np.unique(p) if c >= 0 and c not in (CAVITY, ANCHOR)) for p in fragments]
    key_colors = [int(p[p >= 0][0]) for p in keys]
    _require(len(set(carriers)) == len(carriers), "fragment carriers must be distinct")
    _require(not set(carriers) & set(key_colors), "key and fragment colors must be disjoint")
    area = sum(int(np.sum(p >= 0)) for p in fragments)
    side = isqrt(area)
    _require(side * side == area and 6 <= side <= 9, "fragment area must form a side-6..9 square")
    ink = sum(int(np.sum(p == CAVITY)) for p in fragments)
    large_area = sum(int(np.sum(p >= 0)) for p in keys)
    ratio, remainder = divmod(large_area, ink)
    scale = isqrt(ratio)
    _require(not remainder and scale * scale == ratio and scale in (2, 3), "keys must define a common scale 2 or 3")
    groups = {}
    logical_keys = []
    for p, color in zip(keys, key_colors):
        h, w = p.shape
        _require(h % scale == 0 and w % scale == 0, "key bounds disagree with scale")
        mask = p >= 0
        small = mask[::scale, ::scale]
        _require(np.array_equal(mask, np.repeat(np.repeat(small, scale, 0), scale, 1)), "broken macrocell")
        _require(len(_components(small)) == 1, "key must be edge-connected")
        cls = _canonical(small)
        groups.setdefault(cls, []).append(color)
        logical_keys.append((color, small.astype(np.int8)))
    _require(all(len(set(colors)) == 1 for colors in groups.values()), "congruent keys with different colors are ambiguous")
    return dict(fragments=fragments, side=side, scale=scale, groups=groups,
                logical_keys=logical_keys, carriers=carriers)


def _assemblies(fragments, side, with_owners=False):
    """Enumerate complete colored tilings, preserving ink patterns under rotation.

    Placement deduplication includes the complete colored pattern: two rotations
    of a rectangle can share a silhouette but have different cavity fragments.
    """
    options, anchor_index = [], None
    for i, p in enumerate(fragments):
        marked = np.any(p == ANCHOR)
        if marked:
            anchor_index = i
        choices = []
        for q in _orientations(p):
            h, w = q.shape
            if h > side or w > side:
                continue
            if marked and q[0, 0] != ANCHOR:
                continue
            cells = [(int(r)*side+int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            base = sum(1 << pos for pos, _ in cells)
            origins = [(0, 0)] if marked else [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
            for r, c in origins:
                shift = r*side+c
                choices.append((base << shift, tuple((pos+shift, color) for pos, color in cells)))
        options.append(choices)
    if anchor_index is None:
        return
    full = (1 << (side*side))-1
    placed = [None] * len(fragments)
    seen = set()

    def search(occupied, remaining):
        if not remaining:
            if occupied == full:
                board = np.empty((side, side), dtype=np.int8)
                for choice in placed:
                    for pos, color in choice[1]:
                        board.flat[pos] = color
                sig = board.tobytes()
                if sig not in seen:
                    seen.add(sig)
                    if with_owners:
                        owners = np.empty(board.shape, dtype=np.int8)
                        for i, choice in enumerate(placed):
                            carrier = next(int(c) for c in np.unique(fragments[i])
                                           if c >= 0 and c not in (CAVITY, ANCHOR))
                            for pos, _ in choice[1]:
                                owners.flat[pos] = carrier
                        yield board, owners
                    else:
                        yield board
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            for candidate in options[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    yield from search(occupied | candidate[0], tuple(j for j in remaining if j != i))
            placed[i] = None

    for choice in options[anchor_index]:
        placed[anchor_index] = choice
        yield from search(choice[0], tuple(i for i in range(len(fragments)) if i != anchor_index))


def _fill(board, groups):
    if not _closed_border(board):
        return None
    holes = _components(board == CAVITY)
    classes = [_canonical(_shape(h)) for h in holes]
    if Counter(classes) != Counter({key: len(colors) for key, colors in groups.items()}):
        return None
    out = board.copy()
    for cells, cls in zip(holes, classes):
        for r, c in cells:
            out[r, c] = groups[cls][0]
    return out


def _closed_border(board):
    return not any(np.any(edge == CAVITY) for edge in (board[0], board[-1], board[:, 0], board[:, -1]))


def _recover(input_grid, diagnostics=False):
    inp = _grid(input_grid)
    facts = _decode(inp)
    answers, geometric, boundary, area_only, accepted, solved_board = {}, 0, 0, 0, 0, None
    key_areas = sorted(int(p.sum()) for _, p in facts['logical_keys'])
    owners = None
    for board, candidate_owners in _assemblies(facts['fragments'], facts['side'], True):
        geometric += 1
        if _closed_border(board):
            boundary += 1
            area_only += sorted(map(len, _components(board == CAVITY))) == key_areas
        logical = _fill(board, facts['groups'])
        if logical is None:
            continue
        accepted += 1
        sig = logical.tobytes()
        answers[sig] = logical
        solved_board = board
        owners = candidate_owners
        _require(len(answers) <= 1, "more than one filled colored assembly")
    _require(len(answers) == 1, "no compatible filled assembly")
    logical = next(iter(answers.values()))
    k = facts['scale']
    output = np.repeat(np.repeat(logical, k, 0), k, 1)
    if not diagnostics:
        return output
    holes = _components(solved_board == CAVITY)
    # Owners come from the actual complete exact cover, including its cavity
    # cells. A local patch match alone could falsely overlap another fragment.
    spans = [len({int(owners[r, c]) for r, c in cells}) for cells in holes]
    reflected = 0
    for cells in holes:
        target = _shape(cells)
        cls = _canonical(target)
        sources = [p for _, p in facts['logical_keys'] if _canonical(p) == cls]
        reflected += all(_canonical(p, False) != _canonical(target, False) for p in sources)
    partial = [_canonical(_shape(cells)) for p in facts['fragments']
               for cells in _components(p == CAVITY)]
    facts.update(geometric_assemblies=geometric, boundary_assemblies=boundary,
                 area_only_assemblies=area_only, compatible_assemblies=accepted,
                 unjoined_match_fails=Counter(partial) != Counter({k: len(v) for k, v in facts['groups'].items()}),
                 spans=spans, reflection_required=reflected,
                 board=solved_board, owners=owners, logical=logical)
    return output, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique answer entirely from this input."""
    return _recover(input_grid)


def _partition(rng, side, count, style):
    if style == 0:
        rectangles = [(0, 0, side, side)]
        while len(rectangles) < count:
            choices = [(i, axis, cut) for i, (_, _, h, w) in enumerate(rectangles)
                       for axis, length, other in ((0, h, w), (1, w, h))
                       for cut in range(1, length) if min(cut, length-cut)*other >= 5]
            if not choices:
                return None
            i, axis, cut = choices[int(rng.integers(len(choices)))]
            r, c, h, w = rectangles.pop(i)
            rectangles.extend([(r, c, cut, w), (r+cut, c, h-cut, w)] if axis == 0 else
                              [(r, c, h, cut), (r, c+cut, h, w-cut)])
        labels = np.empty((side, side), dtype=np.int8)
        for i, (r, c, h, w) in enumerate(rectangles):
            labels[r:r+h, c:c+w] = i
        return labels
    # Preserve one rotatable rectangular insert, then procedurally grow the
    # remaining regions around it. This supplies irregular outlines while
    # retaining genuine ambiguity for the cavity clues to resolve.
    labels = np.full((side, side), -1, dtype=np.int8)
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    r, c = int(rng.integers(1, side-h)), int(rng.integers(1, side-w))
    labels[r:r+h, c:c+w] = count-1
    points = [(0, 0)]
    free = [(r, c) for r in range(side) for c in range(side) if labels[r, c] < 0 and (r, c) != (0, 0)]
    points += [free[int(i)] for i in rng.choice(len(free), count-2, replace=False)]
    for i, p in enumerate(points):
        labels[p] = i
    weights = rng.uniform(.5, 2.0, count-1)
    while np.any(labels < 0):
        frontiers = []
        for i in range(count-1):
            frontiers.append(sorted({(int(r)+dr, int(c)+dc) for r, c in np.argwhere(labels == i)
                                     for dr, dc in EDGE if 0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] < 0}))
        eligible = [i for i, f in enumerate(frontiers) if f]
        if not eligible:
            return None
        i = min(eligible, key=lambda j: float(np.sum(labels == j))/weights[j] + float(rng.random()))
        choices = frontiers[i]
        labels[choices[int(rng.integers(len(choices)))]] = i
    if min(int(np.sum(labels == i)) for i in range(count)) < 5:
        return None
    return labels


def _carve(rng, labels, hole_count, triple):
    side = len(labels)
    allowed = {(r, c) for r in range(1, side-1) for c in range(1, side-1)}
    occupied, holes, classes = set(), [], set()
    for index in range(hole_count):
        success = False
        for _ in range(50):
            free = allowed - occupied - {(r+dr, c+dc) for r, c in occupied for dr, dc in EDGE}
            seams = sorted((p, (p[0]+dr, p[1]+dc)) for p in free for dr, dc in EDGE
                           if (p[0]+dr, p[1]+dc) in free and labels[p] != labels[p[0]+dr, p[1]+dc])
            if not seams:
                break
            cells = set(seams[int(rng.integers(len(seams)))])
            target = int(rng.integers(3, 9))
            for _ in range(target-2):
                choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in EDGE} & free - cells)
                choices = [p for p in choices if max(r for r, _ in cells | {p})-min(r for r, _ in cells | {p}) < 4
                           and max(c for _, c in cells | {p})-min(c for _, c in cells | {p}) < 4
                           and (triple or len({int(labels[q]) for q in cells | {p}}) <= 2)]
                if not choices:
                    break
                cells.add(choices[int(rng.integers(len(choices)))])
            span = len({int(labels[p]) for p in cells})
            cls = _canonical(_shape(cells))
            if len(cells) < 3 or cls in classes or (index == 0 and triple and span < 3):
                continue
            # Never erase the visible identity of a fragment.
            union = occupied | cells
            if any(np.sum(labels == label) - sum(labels[p] == label for p in union) < 2 for label in np.unique(labels)):
                continue
            classes.add(cls)
            occupied.update(cells)
            holes.append(sorted(cells))
            success = True
            break
        if not success:
            return None
    if {int(labels[p]) for p in occupied} != set(map(int, np.unique(labels))):
        return None
    return holes


def _scatter(rng, objects):
    for attempt in range(24):
        H, W = (int(rng.integers(20, 31)), int(rng.integers(20, 31))) if attempt < 16 else (30, 30)
        canvas = np.zeros((H, W), dtype=np.int8)
        blocked = np.zeros((H, W), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: -objects[i].size + float(rng.random())*12)
        for i in order:
            p = objects[i]
            h, w = p.shape
            if h > H or w > W:
                break
            valid = np.ones((H-h+1, W-w+1), dtype=bool)
            for r, c in np.argwhere(p >= 0):
                valid &= ~blocked[r:r+H-h+1, c:c+W-w+1]
            origins = np.argwhere(valid)
            if not len(origins):
                break
            r0, c0 = map(int, origins[int(rng.integers(len(origins)))])
            view = canvas[r0:r0+h, c0:c0+w]
            view[p >= 0] = p[p >= 0]
            for r, c in np.argwhere(p >= 0):
                rr, cc = int(r)+r0, int(c)+c0
                blocked[max(0, rr-1):min(H, rr+2), max(0, cc-1):min(W, cc+2)] = True
        else:
            rs, cs = np.nonzero(canvas)
            top, left = max(0, int(rs.min())-int(rng.integers(4))), max(0, int(cs.min())-int(rng.integers(4)))
            bottom, right = min(H, int(rs.max())+1+int(rng.integers(4))), min(W, int(cs.max())+1+int(rng.integers(4)))
            return canvas[top:bottom, left:right].copy()
    return None


def _make_pair(rng, scale, triple, reflection_witness, area_witness, style):
    for _ in range(600):
        side = int(rng.integers(6, 10))
        count = int(rng.integers(3, 5))
        hole_count = int(rng.integers(2, 4))
        labels = _partition(rng, side, count, style)
        if labels is None:
            continue
        holes = _carve(rng, labels, hole_count, triple)
        if holes is None:
            continue
        palette = list(map(int, rng.permutation(COLORS)))
        board = np.array(palette[:count], dtype=np.int8)[labels]
        for cells in holes:
            for r, c in cells:
                board[r, c] = CAVITY
        board[0, 0] = ANCHOR
        logical = board.copy()
        keys, groups, reflected = [], {}, False
        for i, cells in enumerate(holes):
            color, target = palette[count+i], _shape(cells)
            possibilities = _orientations(target, True)
            mirror_only = [p for p in possibilities if _canonical(p, False) != _canonical(target, False)]
            if reflection_witness and not reflected and mirror_only:
                small = mirror_only[int(rng.integers(len(mirror_only)))]
                reflected = True
            else:
                small = possibilities[int(rng.integers(len(possibilities)))]
            key = np.where(small, color, -1).astype(np.int8)
            keys.append(np.repeat(np.repeat(key, scale, 0), scale, 1))
            groups[_canonical(target)] = [color]
            for r, c in cells:
                logical[r, c] = color
        if reflection_witness and not reflected:
            continue
        pieces = [_tight(board, list(zip(*np.nonzero(labels == i)))) for i in range(count)]
        answers, boundary, area_only = set(), 0, 0
        key_areas = sorted(map(len, holes))
        for candidate in _assemblies(pieces, side):
            if _closed_border(candidate):
                boundary += 1
                area_only += sorted(map(len, _components(candidate == CAVITY))) == key_areas
            filled = _fill(candidate, groups)
            if filled is not None:
                answers.add(filled.tobytes())
                if len(answers) > 1:
                    break
        if answers != {logical.tobytes()} or boundary < 2 or (area_witness and area_only < 2):
            continue
        oriented = [np.rot90(p, int(rng.integers(4))).copy() for p in pieces]
        # At least one visibly rotated patterned fragment in every pair.
        if all(np.array_equal(a, b) for a, b in zip(oriented, pieces)):
            oriented[0] = np.rot90(pieces[0]).copy()
        inp = _scatter(rng, oriented + keys)
        if inp is None:
            continue
        expected = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        recovered, facts = _recover(inp, True)
        if np.array_equal(recovered, expected) and facts['boundary_assemblies'] >= 2:
            return {'input': inp, 'output': expected}
    raise RuntimeError("bounded construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and low <= value <= high,
                 f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    first_scale = int(rng.integers(2, 4))
    styles = [int(v) for _ in range((a+b+1)//2) for v in rng.permutation(2)]
    for i in range(a+b):
        split = 'train' if i < a else 'test'
        scale = first_scale if i % 2 == 0 else 5-first_scale
        for _ in range(8):
            pair = _make_pair(rng, scale, split == 'test', i == 1, i == 0, styles[i])
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("distinct-pair budget exhausted")
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            predicted, f = _recover(inp, True)
            _require(np.array_equal(predicted, out), "incorrect output")
            _require(f['boundary_assemblies'] >= 2, "shape matching must resolve ambiguity even with a closed border")
            _require(f['unjoined_match_fails'], "unjoined cavity fragments must not match the keys")
            _require(min(f['spans']) >= 2, "every cavity must cross a fragment seam")
            _require(not np.any((out == BACKGROUND) | (out == CAVITY)), "background and cavity ink must disappear")
            k = f['scale']
            _require(np.all(out[:k, :k] == ANCHOR) and np.sum(out == ANCHOR) == k*k, "anchor must enlarge into the upper-left block")
            for color in f['carriers']:
                _require(np.sum(out == color) == np.sum(inp == color)*k*k, "fragment carrier conservation failed")
            for color, _ in f['logical_keys']:
                _require(np.sum(out == color) == np.sum(inp == color), "key cells must be conserved")
            facts[split].append(f)
    _require({f['scale'] for f in facts['train']} == {2, 3}, "training must demonstrate both scales")
    _require(any(f['reflection_required'] for f in facts['train']), "training must demonstrate reflection")
    _require(any(f['area_only_assemblies'] >= 2 for f in facts['train']), "training must distinguish shape matching from area alone")
    _require(all(max(f['spans']) == 2 for f in facts['train']), "training uses two-fragment cavities")
    _require(all(max(f['spans']) >= 3 for f in facts['test']), "tests must recombine at least three fragments in one cavity")
    return True


def self_test():
    """42 seeds, bounded streaming measurements; no saved grids or traces."""
    shapes, partitions, classes, counts, spans, scales, signatures = set(), set(), set(), set(), set(), set(), set()
    irregular = reflected = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, f = _recover(pair['input'], True)
                pairs += 1
                shapes.add(pair['input'].shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in f['owners'].flat)
                partitions.add((f['owners'].shape, norm))
                classes.update(f['groups'])
                counts.add((len(f['fragments']), len(f['logical_keys'])))
                spans.update(f['spans'])
                scales.add(f['scale'])
                irregular += any(np.any(p < 0) for p in f['fragments'])
                reflected += f['reflection_required'] > 0
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 20 and len(partitions) >= 100 and len(classes) >= 25
    assert {2, 3} == scales and {2, 3} <= spans
    assert len(counts) == 4 and irregular >= 25 and reflected >= 42
    p = generate_puzzle(707, 2, 1)
    q = generate_puzzle(707, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong label accepted')
    # Input-only metamorphic checks exercise representations not supplied by
    # generate_puzzle: global rotations, transposition, and a new local palette.
    for split in ('train', 'test'):
        inp, out = p[split][0]['input'], p[split][0]['output']
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(inp, turns)), out)
        assert np.array_equal(solve(inp.T), out.T)
        palette = np.arange(10, dtype=np.int8)
        for old, new in zip(COLORS, COLORS[1:] + COLORS[:1]):
            palette[old] = new
        assert np.array_equal(solve(palette[inp]), palette[out])
        f = _decode(inp)
        broken = inp.copy()
        color = f['logical_keys'][0][0]
        r, c = np.argwhere(broken == color)[0]
        broken[r, c] = BACKGROUND
        try:
            solve(broken)
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete macrocell accepted')
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), normalized_fragment_partitions=len(partitions),
                key_shape_classes=len(classes), count_combinations=sorted(counts),
                seam_spans=sorted(spans), irregular_pairs=irregular, reflection_pairs=reflected)
