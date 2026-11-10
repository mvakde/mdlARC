"""Difference jigsaw: the changes to repeated stencils are rigid square pieces.

Only the standard library and NumPy are used. No files, seeds, demonstration
labels, or construction metadata are consulted by solve(). All searches and
retries stay in memory. See task_spec.md for the color and difference rules.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import isqrt

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grids must have two dimensions, each 1..30.")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid colors must be integers 0..9.")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    pending = [cells.pop()]
    while pending:
        r, c = pending.pop()
        for dr, dc in EDGE:
            p = (r + dr, c + dc)
            if p in cells:
                cells.remove(p)
                pending.append(p)
    return not cells


def _tight(a):
    points = np.argwhere(a >= 0)
    _require(len(points) > 0, "An empty difference is not a piece.")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    return a[r0:r1, c0:c1].copy()


def _orientations(piece, allow_rotation=True):
    seen = set()
    for k in range(4 if allow_rotation else 1):
        p = np.rot90(piece, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            yield p


def _assemblies(pieces, side, marker, allow_rotation=True, limit=2):
    """Enumerate distinct colored exact covers, stopping after `limit` answers.

    Placements with equal masks but different colors are retained. Equivalent
    symmetries/permutations count once only if the complete colored answer is
    identical. The search has depth equal to the number of supplied pieces.
    """
    if sum(int(np.sum(p >= 0)) for p in pieces) != side * side:
        return []
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    if len(marked) != 1:
        return []
    choices = []
    for i, piece in enumerate(pieces):
        placements = {}
        for p in _orientations(piece, allow_rotation):
            h, w = p.shape
            if h > side or w > side:
                continue
            if i == marked[0]:
                if p[0, 0] != marker:
                    continue
                offsets = [(0, 0)]
            else:
                offsets = [(r, c) for r in range(side - h + 1)
                           for c in range(side - w + 1)]
            cells = [(int(r), int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0)]
            for r0, c0 in offsets:
                colored = tuple(((r + r0) * side + c + c0, v) for r, c, v in cells)
                mask = sum(1 << pos for pos, _ in colored)
                placements[(mask, colored)] = (mask, colored)
        if not placements:
            return []
        choices.append(list(placements.values()))

    full = (1 << (side * side)) - 1
    placed = [None] * len(pieces)
    answers, seen_answers = [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            answer = np.full((side, side), -1, dtype=np.int8)
            for candidate in placed:
                for pos, color in candidate[1]:
                    answer.flat[pos] = color
            key = _key(answer)
            if key not in seen_answers:
                seen_answers.add(key)
                answers.append(answer)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in choices[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    anchor = marked[0]
    for candidate in choices[anchor]:
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(pieces)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def _read(input_grid):
    """Infer the lattice, modal stencils, differences, colors and square area."""
    g = _grid(input_grid)
    lattice = int(g[0, 0])
    rows = np.flatnonzero(np.all(g == lattice, axis=1)).tolist()
    cols = np.flatnonzero(np.all(g == lattice, axis=0)).tolist()
    _require(len(rows) >= 2 and len(cols) >= 4, "Missing framed panel lattice.")
    _require(rows[0] == cols[0] == 0 and rows[-1] == g.shape[0] - 1
             and cols[-1] == g.shape[1] - 1, "Lattice needs a complete outer border.")
    _require(len(set(np.diff(rows))) == len(set(np.diff(cols))) == 1,
             "Panel spacing must be regular on each axis.")
    ph, pw = rows[1] - 1, cols[1] - 1
    count, copies = len(rows) - 1, len(cols) - 1
    _require(ph > 0 and pw > 0 and 3 <= count <= 4 and 3 <= copies <= 5,
             "Require three or four bands and three to five copies.")
    stencils, variants, differences, positions, row_colors = [], [], [], [], []
    for r0, r1 in zip(rows, rows[1:]):
        panels = [g[r0 + 1:r1, c0 + 1:c1] for c0, c1 in zip(cols, cols[1:])]
        _require(all(np.all(p != lattice) for p in panels), "Lattice color inside a panel.")
        counts = Counter(_key(p) for p in panels)
        modal = [key for key, n in counts.items() if n == copies - 1]
        _require(len(counts) == 2 and len(modal) == 1,
                 "Each band needs identical references and one visible exception.")
        master = next(p for p in panels if _key(p) == modal[0])
        column = next(i for i, p in enumerate(panels) if _key(p) != modal[0])
        variant = panels[column]
        colors = set(map(int, np.unique(master)))
        _require(len(colors) == 2, "Each stencil uses carrier plus band ink.")
        stencils.append(master.copy())
        variants.append(variant.copy())
        differences.append(master != variant)
        positions.append(column)
        row_colors.append(colors)
    common = set.intersection(*row_colors)
    _require(len(common) == 1, "Bands must share exactly one carrier color.")
    carrier = common.pop()
    inks = [next(iter(colors - {carrier})) for colors in row_colors]
    _require(len(set(inks)) == count, "Every band has a distinct ink color.")
    extras = set(map(int, np.unique(g))) - {lattice, carrier, *inks}
    _require(len(extras) == 1, "Require one extra marker color.")
    marker = extras.pop()
    _require(np.sum(g == marker) == 1, "The marker must occur exactly once.")
    pieces, additions, erasures, balanced = [], [], [], []
    for master, variant, mask, ink in zip(stencils, variants, differences, inks):
        _require(set(map(int, np.unique(variant))) <= {carrier, ink, marker},
                 "A band contains another band's ink.")
        _require(np.sum(mask) >= 3 and _connected(mask),
                 "Each difference must be an edge-connected piece of at least three cells.")
        p = np.full((ph, pw), -1, dtype=np.int8)
        p[mask] = ink
        p[variant == marker] = marker
        pieces.append(_tight(p))
        additions.append(int(np.sum((master == carrier) & (variant == ink))))
        erasures.append(int(np.sum((master == ink) & (variant == carrier))))
        balanced.append(bool(not np.any(variant == marker)
                             and np.sum(master == ink) == np.sum(variant == ink)))
    area = sum(int(np.sum(mask)) for mask in differences)
    side = isqrt(area)
    _require(side * side == area and 4 <= side <= 7,
             "Total difference area must be a square of side 4..7.")
    return dict(grid=g, lattice=lattice, carrier=carrier, marker=marker,
                inks=inks, ph=ph, pw=pw, count=count, copies=copies,
                pieces=pieces, stencils=stencils, variants=variants,
                differences=differences, columns=positions, side=side,
                additions=additions, erasures=erasures, balanced=balanced)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Materialize the changes and return their unique anchored rigid assembly."""
    facts = _read(input_grid)
    answers = _assemblies(facts['pieces'], facts['side'], facts['marker'])
    _require(len(answers) == 1, "The input must have one distinct colored assembly.")
    return answers[0].copy()


def _partition(rng, side, count, style):
    """Procedural shapes: varied region growth or cuts with connected transfers."""
    if style == 2:
        # Large wrapping region with two or three edge/interior inserts.
        labels = np.zeros((side, side), dtype=np.int8)
        for label in range(1, count):
            okay = False
            for _ in range(24):
                h, w = map(int, rng.integers(1, max(3, side - 1), size=2))
                if not 3 <= h * w <= side * side // 3:
                    continue
                r, c = int(rng.integers(side - h + 1)), int(rng.integers(side - w + 1))
                if (r, c) == (0, 0) or np.any(labels[r:r+h, c:c+w]):
                    continue
                candidate = labels.copy()
                candidate[r:r+h, c:c+w] = label
                if h > 1 and w > 1 and h*w > 4 and rng.random() < .65:
                    candidate[r, c] = 0
                if _connected(candidate == 0) and np.sum(candidate == label) >= 3:
                    labels, okay = candidate, True
                    break
            if not okay:
                return None
        return labels
    if style == 1:
        boxes = [(0, 0, side, side)]
        while len(boxes) < count:
            options = [(i, axis, k) for i, (_, _, h, w) in enumerate(boxes)
                       for axis, length, other in ((0, h, w), (1, w, h))
                       for k in range(1, length) if min(k, length-k)*other >= 3]
            if not options:
                return None
            i, axis, k = options[int(rng.integers(len(options)))]
            r, c, h, w = boxes.pop(i)
            boxes.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                         [(r, c, h, k), (r, c+k, h, w-k)])
        labels = np.empty((side, side), dtype=np.int8)
        for i, (r, c, h, w) in enumerate(boxes):
            labels[r:r+h, c:c+w] = i
        for _ in range(side * side):
            r, c = map(int, rng.integers(side, size=2))
            dr, dc = EDGE[int(rng.integers(4))]
            nr, nc = r+dr, c+dc
            if not (0 <= nr < side and 0 <= nc < side):
                continue
            a, b = int(labels[r, c]), int(labels[nr, nc])
            if a != b and np.sum(labels == a) > 3:
                labels[r, c] = b
                if not _connected(labels == a):
                    labels[r, c] = a
        return labels
    labels = np.full((side, side), -1, dtype=np.int8)
    if style == 3 and count == 4:
        seeds = [0, side - 1, side*(side-1), side*side-1]
    else:
        seeds = list(map(int, rng.choice(side * side, count, replace=False)))
    for label, pos in enumerate(seeds):
        labels.flat[pos] = label
    weights = np.exp(rng.uniform(-.9, .9, count))
    for _ in range(side * side - count):
        frontiers = []
        for label in range(count):
            frontier = set()
            for r, c in np.argwhere(labels == label):
                for dr, dc in EDGE:
                    nr, nc = int(r)+dr, int(c)+dc
                    if 0 <= nr < side and 0 <= nc < side and labels[nr, nc] < 0:
                        frontier.add(nr * side + nc)
            frontiers.append(sorted(frontier))
        label = min((i for i in range(count) if frontiers[i]),
                    key=lambda i: np.sum(labels == i)/weights[i] + float(rng.random())*.7)
        choices = frontiers[label]
        labels.flat[choices[int(rng.integers(len(choices)))]] = label
    return labels


def _fallback(side, count):
    """Guaranteed unique frame with differently sized, separated cavities."""
    if (side, count) == (4, 4):
        return np.array([[0,0,1,1],[2,0,1,3],[2,2,2,3],[2,2,3,3]], dtype=np.int8)
    labels = np.zeros((side, side), dtype=np.int8)
    labels[1, 1:3] = 1
    labels[2, 1] = 1
    if count == 3:
        # Even area supplies a count-preserving unmarked stencil edit at odd
        # square sizes too. The frame's exact cavity fixes the bar's position.
        labels[side-1, :side-side % 2] = 2
    else:
        # The two inserts have distinct areas 3 and 4; the bottom bar has side
        # cells. Side four uses the small interlocking construction above.
        labels[1:3, side-2:side] = 2
        labels[side-1, :side] = 3
    return labels


def _choose_partition(rng, side, count, inks, marker):
    for attempt in range(240):
        labels = _partition(rng, side, count, int(rng.integers(4)))
        if labels is None:
            continue
        anchor = int(labels[0, 0])
        masks = [labels == i for i in range(count)]
        if min(int(m.sum()) for m in masks) < 3:
            continue
        if min(_tight(np.where(masks[anchor], 1, -1)).shape) < 2:
            continue
        if not any(i != anchor and m.sum() % 2 == 0 and m.sum() >= 4
                   for i, m in enumerate(masks)):
            continue
        out = np.empty((side, side), dtype=np.int8)
        for i, mask in enumerate(masks):
            out[mask] = inks[i]
        out[0, 0] = marker
        pieces = [_tight(np.where(mask, out, -1)) for mask in masks]
        answers = _assemblies(pieces, side, marker)
        if len(answers) == 1 and np.array_equal(answers[0], out):
            return out, pieces, anchor
    # A finite deterministic safety net, independent of future random draws.
    labels = _fallback(side, count)
    anchor = int(labels[0, 0])
    out = np.empty((side, side), dtype=np.int8)
    for i, ink in enumerate(inks):
        out[labels == i] = ink
    out[0, 0] = marker
    pieces = [_tight(np.where(labels == i, out, -1)) for i in range(count)]
    answers = _assemblies(pieces, side, marker)
    _require(len(answers) == 1 and np.array_equal(answers[0], out),
             "Internal fallback must have one colored assembly.")
    _require(any(i != anchor and np.sum(p >= 0) % 2 == 0 for i, p in enumerate(pieces)),
             "Internal fallback needs an even unmarked difference.")
    return out, pieces, anchor


def _stencil(rng, h, w):
    rr, cc = np.indices((h, w))
    style = int(rng.integers(5))
    if style == 0:
        return rng.random((h, w)) < float(rng.uniform(.25, .75))
    if style == 1:
        return (rr + int(rng.integers(2))*cc) % int(rng.integers(2, 4)) == 0
    if style == 2:
        r0, c0 = int(rng.integers(max(1, h//2))), int(rng.integers(max(1, w//2)))
        return ((rr == r0) | (rr == h-1-r0) | (cc == c0) | (cc == w-1-c0))
    if style == 3:
        return (rr == h//2) | (cc == w//2) | ((rr + cc) % 4 == 0)
    return (rr % 2 == 0) ^ (cc % 3 == 0)


def _make_pair(rng, side, count, copies, anchor_rotation, pure_mode, palette, witness):
    lattice, carrier, marker = palette[:3]
    inks = palette[3:3+count]
    out, pieces, anchor = _choose_partition(rng, side, count, inks, marker)
    even = [i for i, p in enumerate(pieces)
            if i != anchor and np.sum(p >= 0) % 2 == 0]
    balanced_id = even[int(rng.integers(len(even)))]
    pure_ids = [i for i in range(count) if i not in (anchor, balanced_id)]
    pure_id = pure_ids[int(rng.integers(len(pure_ids)))]
    rotations = list(map(int, rng.integers(4, size=count)))
    rotations[anchor] = anchor_rotation
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    maxh, maxw = max(p.shape[0] for p in oriented), max(p.shape[1] for p in oriented)
    h_limit, w_limit = 29 // count - 1, 29 // copies - 1
    _require(maxh <= h_limit and maxw <= w_limit, "Internal panels do not fit.")
    ph = int(rng.integers(maxh, min(h_limit, maxh+2) + 1))
    pw = int(rng.integers(maxw, min(w_limit, maxw+2) + 1))
    if any(p.shape == (ph, pw) and np.all(p >= 0) for p in oriented):
        # A solid piece still needs some stencil context outside its support.
        # There is room on at least one axis in every scheduled geometry.
        if ph < h_limit:
            ph += 1
        else:
            _require(pw < w_limit, 'No space for reference context.')
            pw += 1
    order = list(map(int, rng.permutation(count)))
    columns = list(map(int, rng.integers(copies, size=count)))
    # The first two demonstrations rebut a fixed-column rule; other pairs can
    # have aligned exceptions, adjacent repeats or arbitrary unused columns.
    if not witness and rng.random() < .18:
        columns = [columns[0]] * count
    if witness and len(set(columns)) == 1:
        columns[-1] = (columns[-1] + 1) % copies
    grid = np.full((count*(ph+1)+1, copies*(pw+1)+1), lattice, dtype=np.int8)
    for band, i in enumerate(order):
        p = oriented[i]
        r0, c0 = int(rng.integers(ph-p.shape[0]+1)), int(rng.integers(pw-p.shape[1]+1))
        embedded = np.full((ph, pw), -1, dtype=np.int8)
        embedded[r0:r0+p.shape[0], c0:c0+p.shape[1]] = p
        mask = embedded >= 0
        base = _stencil(rng, ph, pw)
        cells = np.flatnonzero(mask)
        if i == balanced_id:
            base.flat[cells] = False
            base.flat[rng.choice(cells, len(cells)//2, replace=False)] = True
        elif i == pure_id:
            base[mask] = pure_mode == 'erase'
        # Keep the two stencil roles visibly present, even for a pure edit.
        outside = np.flatnonzero(~mask)
        _require(len(outside) > 0, "A piece must leave reference context in its panel.")
        if not np.any(base):
            base.flat[outside[0]] = True
        if np.all(base):
            base.flat[outside[-1]] = False
        master = np.where(base, inks[i], carrier).astype(np.int8)
        variant = np.where(base ^ mask, inks[i], carrier).astype(np.int8)
        variant[embedded == marker] = marker
        r = 1 + band*(ph+1)
        for col in range(copies):
            c = 1 + col*(pw+1)
            grid[r:r+ph, c:c+pw] = variant if col == columns[band] else master
    return {'input': grid, 'output': out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer.')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be {low}..{high}.')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible family member; the test shift is larger pieces."""
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _require(nt > ne, 'num_train must be greater than num_test.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    palette = list(map(int, rng.permutation(10)))
    train_sizes = [4, 5] + list(map(int, rng.integers(4, 6, size=nt-2)))
    train_counts = list(map(int, rng.permutation([3, 4])))
    train_counts += list(map(int, rng.integers(3, 5, size=nt-2)))
    first_corner = int(rng.integers(4))
    train_rotations = [first_corner, (first_corner+1) % 4]
    train_rotations += list(map(int, rng.integers(4, size=nt-2)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        if split == 'train':
            side, count = train_sizes[index], train_counts[index]
            copies = 3 if index == 0 else 4 if index == 1 else int(rng.integers(3, 5))
            rotation = train_rotations[index]
        else:
            side = int(rng.integers(6, 8))
            count = 3 if side == 7 else int(rng.integers(3, 5))
            copies = 3
            rotation = int(rng.integers(4))
        if not stable:
            palette = list(map(int, rng.permutation(10)))
        for _ in range(16):
            pair = _make_pair(rng, side, count, copies, rotation,
                              'erase' if index % 2 == 0 else 'add', palette, index < 2)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('Could not construct a distinct input within the retry bound.')
    return puzzle


def _pair_facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys.')
    facts = _read(pair['input'])
    out = _grid(pair['output'])
    _require(out.shape == (facts['side'], facts['side']), 'Wrong output area.')
    _require(out[0, 0] == facts['marker'] and np.sum(out == facts['marker']) == 1,
             'Marker must anchor the upper-left exactly once.')
    _require(set(map(int, np.unique(out))) == {facts['marker'], *facts['inks']},
             'Output uses only the materialized band inks and marker.')
    for ink in facts['inks']:
        expected = sum(int(np.sum(p == ink)) for p in facts['pieces'])
        _require(np.sum(out == ink) == expected, 'Difference-cell color/area conservation failed.')
    answers = _assemblies(facts['pieces'], facts['side'], facts['marker'])
    _require(len(answers) == 1 and np.array_equal(answers[0], out),
             'Output must be the unique rotation-only colored assembly.')
    rotation_required, asymmetric_rotation = False, False
    for ink, p in zip(facts['inks'], facts['pieces']):
        marked = bool(np.any(p == facts['marker']))
        mask = (out == ink) | ((out == facts['marker']) if marked else False)
        target = _tight(np.where(mask, out, -1))
        changed = not np.array_equal(target, p)
        rotation_required |= changed
        asymmetric_rotation |= changed and len(list(_orientations(np.where(p >= 0, 1, -1)))) == 4
    facts['rotation_required'] = rotation_required
    facts['asymmetric_rotation'] = asymmetric_rotation
    return facts


def validate_generated_puzzle(puzzle):
    """Check visible semantics, exact packing, witnesses, schema and uniqueness."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Illegal pair counts.')
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _pair_facts(pair)
            key = _key(f['grid'])
            _require(key not in seen, 'Duplicate input within a puzzle.')
            seen.add(key)
            _require(any(f['balanced']), 'Each pair must include an unmarked count-preserving edit.')
            facts[split].append(f)
    train = facts['train']
    _require(any(a == 0 and e >= 3 for f in train for a, e in zip(f['additions'], f['erasures'])),
             'Training must show a pure erased piece.')
    _require(any(e == 0 and a >= 3 for f in train for a, e in zip(f['additions'], f['erasures'])),
             'Training must show a pure added piece.')
    _require(any(f['rotation_required'] for f in train), 'Training must demonstrate rigid rotation.')
    _require(any(len(set(f['columns'])) > 1 for f in train),
             'Training must demonstrate changing exception columns.')
    _require({f['count'] for f in train} == {3, 4}, 'Demonstrate changing piece count.')
    _require(len({f['copies'] for f in train}) >= 2, 'Demonstrate changing copy count.')
    _require(len({f['side'] for f in train}) >= 2, 'Demonstrate area-based output size.')
    _require(min(f['side'] for f in facts['test']) > max(f['side'] for f in train),
             'Generated tests must make the documented scale shift.')
    return None


def self_test():
    """Bounded 42-seed audit; retain summaries, never a corpus or search traces."""
    signatures, shapes, partitions, stencils = set(), set(), set(), set()
    counts, sides, palettes, corners = set(), set(), set(), set()
    pure_add = pure_erase = balanced = rotated = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _pair_facts(pair)
                predicted = solve(pair['input'])
                _require(np.array_equal(predicted, pair['output']), 'Input-only solve disagrees.')
                digest.update(bytes(pair['input'].shape) + pair['input'].tobytes())
                shapes.add(pair['input'].shape)
                counts.add(f['count'])
                sides.add(f['side'])
                palettes.add((f['lattice'], f['carrier'], f['marker']))
                for stencil in f['stencils']:
                    normalized = stencil != f['carrier']
                    stencils.add(sha256(bytes(normalized.shape) + normalized.tobytes()).digest())
                normalized, role_map = [], {}
                for v in pair['output'].flat:
                    normalized.append(role_map.setdefault(int(v), len(role_map)))
                partitions.add((pair['output'].shape, bytes(normalized)))
                p = next(p for p in f['pieces'] if np.any(p == f['marker']))
                r, c = map(int, np.argwhere(p == f['marker'])[0])
                corners.add((int(r > 0), int(c > 0)))
                pure_add += sum(a > 0 and e == 0 for a, e in zip(f['additions'], f['erasures']))
                pure_erase += sum(a == 0 and e > 0 for a, e in zip(f['additions'], f['erasures']))
                balanced += sum(f['balanced'])
                rotated += f['rotation_required']
                pairs += 1
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and pairs == 210, '42 distinct five-pair puzzles required.')
    _require(counts == {3, 4} and sides == {4, 5, 6, 7}, 'Missing count or scale variation.')
    _require(len(shapes) >= 15 and len(partitions) >= 150 and len(stencils) >= 300,
             'Insufficient structural variation across 42 seeds.')
    _require(len(palettes) >= 30 and len(corners) == 4, 'Missing role/orientation variation.')
    _require(pure_add >= 42 and pure_erase >= 42 and balanced >= 210 and rotated >= 100,
             'Missing family-specific semantic coverage.')
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(20261022+nt, nt, ne)
        validate_generated_puzzle(p)
        _require((len(p['train']), len(p['test'])) == (nt, ne), 'Counts were ignored.')
    # Check every safety-net geometry directly, rather than assuming its
    # packing, connectivity or even-area witness is guaranteed by construction.
    for side, count in ((4,3), (4,4), (5,3), (5,4), (6,3), (6,4), (7,3)):
        labels = _fallback(side, count)
        out = labels + 1
        out[0,0] = 9
        pieces = [_tight(np.where(labels == i, out, -1)) for i in range(count)]
        _require(all(_connected(p >= 0) and np.sum(p >= 0) >= 3 for p in pieces),
                 'Fallback pieces must remain connected and nontrivial.')
        _require(any(np.sum(p >= 0) % 2 == 0 for p in pieces[1:]),
                 'Fallback needs an unmarked balanced-edit witness.')
        for k in range(4):
            answers = _assemblies([np.rot90(p,k) for p in pieces], side, 9)
            _require(len(answers) == 1 and np.array_equal(answers[0], out),
                     'Fallback packing must be unique for every input rotation.')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    first, second = generate_puzzle(17), generate_puzzle(17)
    _require(all(np.array_equal(a[k], b[k]) for s in first
                 for a, b in zip(first[s], second[s]) for k in ('input', 'output')),
             'Seed determinism failed.')
    # Counterfactuals: ink counts cannot identify a balanced exception; added
    # ink alone loses erasures; leaving pieces unrotated must fail on a witness.
    p = generate_puzzle(0)
    fs = [_read(pair['input']) for pair in p['train']]
    _require(any(not _assemblies(f['pieces'], f['side'], f['marker'], False) for f in fs),
             'Need a witness where omitting quarter-turns changes solvability.')
    for f in fs:
        i = f['balanced'].index(True)
        master, variant = f['stencils'][i], f['variants'][i]
        _require(Counter(master.flat) == Counter(variant.flat)
                 and not np.array_equal(master, variant), 'Count shortcut not rebutted.')
        _require(sum(f['additions']) < f['side']**2, 'Added-only shortcut must lose cells.')
    # Solving must cover family inputs beyond the sampler's layouts: five
    # copies, reversed bands and aligned exceptions retain the same answer.
    f = fs[0]
    copies = min(5, 29 // (f['pw'] + 1))
    expanded = np.full((f['grid'].shape[0], copies*(f['pw']+1)+1),
                       f['lattice'], dtype=np.int8)
    for band, source in enumerate(reversed(range(f['count']))):
        r0 = 1 + band*(f['ph']+1)
        for col in range(copies):
            c0 = 1 + col*(f['pw']+1)
            expanded[r0:r0+f['ph'], c0:c0+f['pw']] = (
                f['variants'][source] if col == 0 else f['stencils'][source])
    _require(copies == 5 and np.array_equal(solve(expanded), p['train'][0]['output']),
             'Solver must support five references/exception copies and arbitrary band order.')
    lookup = np.array([7,0,9,4,2,8,1,6,3,5], dtype=np.int8)
    _require(np.array_equal(solve(lookup[expanded]), lookup[p['train'][0]['output']]),
             'Literal colors must not affect solving.')
    for defect in ('answer', 'absent_exception', 'border'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][0]
        if defect == 'answer':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif defect == 'border':
            pair['input'][0, 0] = (int(pair['input'][0, 0]) + 1) % 10
        else:
            f = _read(pair['input'])
            c0 = 1 + f['columns'][0]*(f['pw']+1)
            pair['input'][1:1+f['ph'], c0:c0+f['pw']] = f['stencils'][0]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('Validator accepted ' + defect)
    return {'seeds': 42, 'pairs': pairs, 'input_shapes': len(shapes),
            'normalized_outputs': len(partitions), 'reference_stencils': len(stencils),
            'role_palettes': len(palettes), 'pure_additions': pure_add,
            'pure_erasures': pure_erase, 'balanced_bands': balanced,
            'pairs_requiring_rotation': rotated}
