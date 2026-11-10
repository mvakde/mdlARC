"""Seam-activated counters: a rigid jigsaw whose contact graph selects addends.

Only NumPy and the standard library are used. No files are read or written.
"""
from itertools import product
import numpy as np

_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'Grid dimensions must be 1..30.')
    _require(np.issubdtype(a.dtype, np.integer), 'Integer cells required.')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'Colors must be 0..9.')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        queue, cells = [start], []
        while queue:
            r, c = queue.pop()
            cells.append((r, c))
            for dr, dc in _EDGE:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    queue.append(p)
        result.append(cells)
    return result


def _tight(a, mask):
    rr, cc = np.nonzero(mask)
    b = np.where(mask, a, -1)
    return b[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, 'Background must be uniquely most frequent.')
    bg = int(colors[np.argmax(counts)])
    markers = [int(c) for c, n in zip(colors, counts) if c != bg and n == 1]
    _require(len(markers) == 1, 'Exactly one singleton anchor color required.')
    marker = markers[0]
    comps = _components(a != bg)
    _require(3 <= len(comps) <= 5, 'Three to five separated panels required.')
    pieces, palettes = [], []
    for cells in comps:
        mask = np.zeros(a.shape, bool)
        for r, c in cells:
            mask[r, c] = True
        p = _tight(a, mask)
        pieces.append(p)
        palettes.append(set(map(int, p[p >= 0])))
    common = set.intersection(*palettes)
    _require(len(common) == 1, 'One counter-ink color must occur on every panel.')
    ink = common.pop()
    _require(ink not in (bg, marker), 'Roles must be distinct.')
    carriers = []
    for p, palette in zip(pieces, palettes):
        cs = palette - {ink, marker}
        _require(len(cs) == 1, 'Each panel needs its own carrier color.')
        carrier = cs.pop()
        _require(np.sum(p == carrier) >= 3, 'Carrier needs at least three cells.')
        carriers.append(carrier)
        coords = np.argwhere(p == ink)
        _require(1 <= len(coords) <= 3, 'Input counters have one to three pips.')
        # In the scattered orientation a stack can be vertical or horizontal.
        if len(coords) > 1:
            vertical = len(set(coords[:, 1])) == 1 and np.all(np.diff(coords[:, 0]) == 2)
            horizontal = len(set(coords[:, 0])) == 1 and np.all(np.diff(coords[:, 1]) == 2)
            _require(vertical or horizontal, 'Pips must be a rigid pitch-two stack.')
            if vertical:
                shaft = p[coords[0, 0]:coords[-1, 0] + 1, coords[0, 1]]
            else:
                shaft = p[coords[0, 0], coords[0, 1]:coords[-1, 1] + 1]
            _require(bool(np.all((shaft == carrier) | (shaft == ink))),
                     'A counter must lie on a continuous shaft of its own panel.')
    _require(len(set(carriers)) == len(carriers), 'Distinct carrier identities required.')
    # Even diagonal contact between scattered panels is excluded.
    for i, cells in enumerate(comps):
        own = set(cells)
        for r, c in cells:
            for dr, dc in _DIAG:
                rr, cc = r + dr, c + dc
                _require(not (0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
                              and a[rr, cc] != bg and (rr, cc) not in own),
                         'Scattered panels must be separated in all eight directions.')
    area = int(np.sum(a != bg))
    side = int(round(area ** .5))
    _require(side * side == area, 'Panel area must be a square.')
    return dict(grid=a, background=bg, marker=marker, ink=ink, pieces=pieces,
                carriers=carriers, side=side)


def _assemblies(facts, limit=2):
    """Enumerate distinct colored covers, retaining content-changing symmetries."""
    pieces, side = facts['pieces'], facts['side']
    marker, ink = facts['marker'], facts['ink']
    candidates = []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    for i, p in enumerate(pieces):
        seen, options = set(), []
        for k in range(4):
            q = np.rot90(p, k).copy()
            signature = _key(q)
            if signature in seen:
                continue
            seen.add(signature)
            h, w = q.shape
            if h > side or w > side:
                continue
            pips = np.argwhere(q == ink)
            if len(set(pips[:, 1])) != 1:
                continue  # Completed counters are vertical, never horizontal.
            if i == anchor and q[0, 0] != marker:
                continue
            cells = [(int(r) * side + int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            base = sum(1 << pos for pos, _ in cells)
            offsets = [0] if i == anchor else [r * side + c for r in range(side - h + 1)
                                                               for c in range(side - w + 1)]
            for offset in offsets:
                options.append((base << offset, offset, cells))
        if not options:
            return []
        candidates.append(options)
    full = (1 << (side * side)) - 1
    placed = [None] * len(pieces)
    answers, signatures = [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            a = np.full((side, side), -1, np.int8)
            for _, offset, cells in placed:
                for pos, color in cells:
                    a.flat[pos + offset] = color
            sig = a.tobytes()
            if sig not in signatures:
                signatures.add(sig)
                answers.append(a)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in candidates[i]:
                mask = candidate[0]
                if mask & first and not mask & occupied:
                    placed[i] = candidate
                    search(occupied | mask, rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for candidate in candidates[anchor]:
        placed[anchor] = candidate
        search(candidate[0], rest)
        if len(answers) >= limit:
            break
    return answers


def _contact(owners, anchor):
    edge, diagonal = set(), set()
    h, w = owners.shape
    for r, c in np.argwhere(owners == anchor):
        for moves, found in ((_EDGE, edge), (_DIAG, diagonal)):
            for dr, dc in moves:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and owners[rr, cc] != anchor:
                    found.add(int(owners[rr, cc]))
    return edge, diagonal - edge


def _compute(a, facts):
    """Apply transfer to a recovered assembly; return inspectable pixel facts."""
    marker, ink = facts['marker'], facts['ink']
    carriers = facts['carriers']
    owner = np.empty(a.shape, np.int8)
    slots = {}
    anchor = None
    for i, p in enumerate(facts['pieces']):
        carrier = carriers[i]
        if np.any(p == marker):
            anchor = carrier
    # Recover ownership of ink using rigid placements, which also handles a pip
    # exactly on a seam or a thin arm. Pips cannot be grouped by neighboring ink.
    for i, p in enumerate(facts['pieces']):
        carrier = carriers[i]
        matches = []
        carrier_cells = np.argwhere(a == carrier)
        for k in range(4):
            q = np.rot90(p, k)
            rc = np.argwhere(q == carrier)
            offset = carrier_cells.min(axis=0) - rc.min(axis=0)
            rr, cc = np.argwhere(q >= 0).T
            yy, xx = rr + offset[0], cc + offset[1]
            if (yy.min() >= 0 and xx.min() >= 0 and yy.max() < len(a) and xx.max() < len(a)
                    and np.array_equal(a[yy, xx], q[rr, cc])):
                matches.append(tuple(sorted(zip(yy.tolist(), xx.tolist()))))
        unique = set(matches)
        _require(len(unique) == 1, 'Panel ownership must be recoverable even at counter pips.')
        cells = next(iter(unique))
        pips = []
        for r, c in cells:
            owner[r, c] = carrier
            if a[r, c] == ink:
                pips.append((r, c))
        pips.sort()
        _require(len({c for r, c in pips}) == 1, 'Completed counter must be vertical.')
        slots[carrier] = pips
    for carrier, pips in slots.items():
        top, x = pips[0]
        bottom, _ = pips[-1]
        _require(bool(np.all(owner[top:bottom + 1, x] == carrier)),
                 'Counter spacing cannot cross a seam.')
    sources, corner_only = _contact(owner, anchor)
    unmarked = set(carriers) - sources - {anchor}
    _require(sources and unmarked, 'Both contributing and ignored panels are required.')
    total = sum(len(slots[c]) for c in sources)
    dst = slots[anchor]
    bottom, x = dst[-1]
    initial = len(dst)
    final_top = bottom - 2 * (initial + total - 1)
    _require(final_top >= 0 and bool(np.all(owner[final_top:bottom + 1, x] == anchor))
             and bool(np.all((a[final_top:bottom + 1, x] == anchor) |
                             (a[final_top:bottom + 1, x] == ink))),
             'The complete extended counter must stay on a continuous receiver shaft.')
    out = a.copy()
    for carrier in sources:
        for r, c in slots[carrier]:
            out[r, c] = marker
    added = []
    for k in range(initial, initial + total):
        r = bottom - 2 * k
        _require(r >= 0 and owner[r, x] == anchor and a[r, x] == anchor,
                 'The receiver must have room for the full sum on its own panel.')
        out[r, x] = ink
        added.append((r, x))
    return out, dict(anchor=anchor, owner=owner, slots=slots, sources=sources,
                     unmarked=unmarked, corner_only=corner_only, total=total,
                     initial=initial, final=initial + total, added=added)


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    covers = _assemblies(facts)
    _require(len(covers) == 1, 'Require exactly one upright, anchored colored assembly.')
    return _compute(covers[0], facts)[0]


def _grow_partition(rng, side, count):
    labels = np.full((side, side), -1, np.int8)
    spine = int(rng.integers(2, 4))
    labels[:, :spine] = 0
    shafts = [(side - 2, int(rng.integers(1, spine)))]
    for i in range(1, count):
        for _ in range(60):
            x = int(rng.integers(spine + 1, side - 1))
            h = int(rng.integers(5, min(8, side - 1)))
            r = int(rng.integers(1, side - h))
            if np.all(labels[r:r + h, x] < 0):
                labels[r:r + h, x] = i
                shafts.append((r + h - 1, x))
                break
        else:
            return None
    adjacency = []
    for r in range(side):
        for c in range(side):
            adjacency.append([rr * side + cc for dr, dc in _EDGE
                              if 0 <= (rr := r + dr) < side and 0 <= (cc := c + dc) < side])
    frontier = [set() for _ in range(count)]
    sizes = [int(np.sum(labels == i)) for i in range(count)]
    weights = np.exp(rng.uniform(-.9, .9, count))
    for i in range(count):
        for pos in np.flatnonzero(labels == i):
            frontier[i].update(q for q in adjacency[pos] if labels.flat[q] < 0)
    for _ in range(int(np.sum(labels < 0))):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j] / weights[j] + float(rng.random()) * 3)
        choices = sorted(frontier[i])
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(pos)
        frontier[i].update(q for q in adjacency[pos] if labels.flat[q] < 0)
    return labels, shafts


def _terraced_partition(rng, side):
    """A four-way seam junction with an explicitly corner-only fourth panel."""
    mid = int(rng.integers(5, side - 4))
    left = int(rng.integers(3, side // 2 + 1))
    a = np.empty((side, side), np.int8)
    for r in range(mid):
        end = left + 1 if r == mid - 1 else int(rng.integers(left - 1, left + 2))
        a[r, :end], a[r, end:] = 0, 1
    for r in range(mid, side):
        end = left if r == mid else int(rng.integers(left - 1, left + 1))
        end2 = left + 1 if r == mid else int(rng.integers(left + 1, min(side - 2, left + 4)))
        a[r, :end], a[r, end:end2], a[r, end2:] = 0, 2, 3
    return a, [(side - 2, int(rng.integers(1, left - 1))),
               (mid - 1, side - 2), (side - 1, left), (side - 1, side - 2)]


def _pieces(a, labels, count):
    return [_tight(a, labels == i) for i in range(count)]


def _scatter(rng, pieces, bg, anchor_rotation):
    rotations = [anchor_rotation] + list(map(int, rng.integers(4, size=len(pieces) - 1)))
    # Always demonstrate a nontrivial rotation on at least one unmarked panel.
    if all(k == 0 for k in rotations):
        rotations[1] = int(rng.integers(1, 4))
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    options = []
    for p in oriented:
        h, w = p.shape
        base = sum(1 << (int(r) * 30 + int(c)) for r, c in np.argwhere(p >= 0))
        opts = [(base << (r * 30 + c), r, c) for r in range(31 - h) for c in range(31 - w)]
        options.append(opts)
    for attempt in range(12):
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt > 2:
            order = list(map(int, rng.permutation(order)))
        placed, blocked, maxr, maxc = {}, 0, 0, 0
        aspect = float(rng.uniform(.75, 1.4))
        for i in order:
            h, w = oriented[i].shape
            candidates = [(max(maxr, r + h) * max(maxc, c + w)
                           + abs(max(maxr, r + h) - aspect * max(maxc, c + w)) * 3
                           + float(rng.random()) * (15 + 15 * attempt), r, c)
                          for mask, r, c in options[i] if not mask & blocked]
            if not candidates:
                break
            _, r, c = min(candidates)
            placed[i] = (r, c)
            maxr, maxc = max(maxr, r + h), max(maxc, c + w)
            for pr, pc in np.argwhere(oriented[i] >= 0):
                rr, cc = r + int(pr), c + int(pc)
                for nr in range(max(0, rr - 1), min(30, rr + 2)):
                    for nc in range(max(0, cc - 1), min(30, cc + 2)):
                        blocked |= 1 << (nr * 30 + nc)
        else:
            canvas = np.full((30, 30), bg, np.int8)
            for i, (r, c) in placed.items():
                p = oriented[i]
                canvas[r:r + p.shape[0], c:c + p.shape[1]][p >= 0] = p[p >= 0]
            rr, cc = np.nonzero(canvas != bg)
            cropped = canvas[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            h, w = cropped.shape
            H = h + int(rng.integers(min(5, 30 - h) + 1))
            W = w + int(rng.integers(min(5, 30 - w) + 1))
            area = int(np.sum(cropped != bg))
            largest = max(int(np.sum(cropped == v)) for v in np.unique(cropped) if v != bg)
            while H * W - area <= largest:
                if H < 30 and (H <= W or W == 30):
                    H += 1
                elif W < 30:
                    W += 1
                else:
                    break
            if H * W - area <= largest:
                continue
            a = np.full((H, W), bg, np.int8)
            top, left = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
            a[top:top + h, left:left + w] = cropped
            return a
    return None


def _operand_options(nsource, capacity):
    return [(tuple(src), d) for src in product(range(1, 4), repeat=nsource)
            for d in range(1, 4) if sum(src) + d <= capacity
            and (nsource == 1 or len(set(src)) > 1)]


def _informative(first, second):
    operands = [first, second]
    s1, d1 = sum(first[0]), first[1]
    s2, d2 = sum(second[0]), second[1]
    if s1 == s2 or d1 == d2 or s1 * d2 == s2 * d1:
        return False
    rivals = (lambda s, d: 2 * d, lambda s, d: 2 * sum(s),
              lambda s, d: sum(s) * d, lambda s, d: max(sum(s), d),
              lambda s, d: sum(s), lambda s, d: d + len(s),
              lambda s, d: d + max(s))
    return all(any(rival(s, d) != sum(s) + d for s, d in operands) for rival in rivals)


def _make_pair(rng, side, count, nsource, operands, roles, anchor_rotation, terraced=False):
    bg, marker, ink = roles
    pool = [c for c in range(10) if c not in roles]
    carriers = list(map(int, rng.choice(pool, count, replace=False)))
    for _ in range(180):
        result = _terraced_partition(rng, side) if terraced else _grow_partition(rng, side, count)
        if result is None:
            continue
        labels, shafts = result
        sources, corners = _contact(labels, 0)
        if len(sources) != nsource or len(sources) >= count - 1:
            continue
        if any(len(_components(labels == i)) != 1 for i in range(count)):
            continue
        source_order = list(map(int, rng.permutation(sorted(sources))))
        heights = list(map(int, rng.integers(1, 4, size=count)))
        src_heights, destination = operands
        heights[0] = destination
        for i, h in zip(source_order, src_heights):
            heights[i] = h
        a = np.empty(labels.shape, np.int8)
        for i, c in enumerate(carriers):
            a[labels == i] = c
        a[0, 0] = marker
        valid = True
        for i, (bottom, x) in enumerate(shafts):
            for k in range(heights[i]):
                r = bottom - 2 * k
                if r < 0 or labels[r, x] != i or a[r, x] == marker:
                    valid = False
                    break
                a[r, x] = ink
        if not valid or any(np.sum(a == c) < 3 for c in carriers):
            continue
        # Ground truth is drawn directly from construction annotations. The
        # input-derived solver below separately recovers ownership and counts.
        out = a.copy()
        for i in sources:
            bottom, x = shafts[i]
            for k in range(heights[i]):
                out[bottom - 2 * k, x] = marker
        bottom, x = shafts[0]
        for k in range(heights[0], heights[0] + sum(heights[i] for i in sources)):
            r = bottom - 2 * k
            if r < 0 or labels[r, x] != 0 or a[r, x] != carriers[0]:
                valid = False
                break
            out[r, x] = ink
        if not valid:
            continue
        pieces = _pieces(a, labels, count)
        facts = dict(pieces=pieces, side=side, marker=marker, ink=ink, carriers=carriers)
        covers = _assemblies(facts)
        if len(covers) != 1 or not np.array_equal(covers[0], a):
            continue
        try:
            recovered, _ = _compute(a, facts)
        except AssertionError:
            continue
        _require(np.array_equal(out, recovered), 'Construction annotations and recovered transfer disagree.')
        inp = _scatter(rng, pieces, bg, anchor_rotation)
        if inp is not None:
            return dict(input=inp, output=out)
    raise RuntimeError('Bounded panel construction exhausted; no ambiguous pair is emitted.')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}.')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    mode = str(rng.choice(['more_panels', 'larger_square']))
    sizes = list(map(int, rng.integers(11, 15, size=num_train)))
    test_sizes = list(map(int, rng.integers(11 if mode == 'more_panels' else 15, 17, size=num_test)))
    first_multi = bool(rng.integers(2))
    source_counts = [2, 1] if first_multi else [1, 2]
    counts = [4 if n == 2 else 3 for n in source_counts]
    compatible = [(a, b) for a in _operand_options(source_counts[0], (sizes[0] - 2) // 2 + 1)
                  for b in _operand_options(source_counts[1], (sizes[1] - 2) // 2 + 1)
                  if _informative(a, b)]
    first_operands = compatible[int(rng.integers(len(compatible)))]
    plans = [(sizes[i], counts[i], source_counts[i], first_operands[i], source_counts[i] == 2)
             for i in range(2)]
    for i in range(2, num_train + num_test):
        training = i < num_train
        side = sizes[i] if training else test_sizes[i - num_train]
        n = int(rng.choice([3, 4])) if training or mode == 'larger_square' else 5
        ns = int(rng.integers(1, n - 1))
        choices = _operand_options(ns, (side - 2) // 2 + 1)
        operand = choices[int(rng.integers(len(choices)))]
        plans.append((side, n, ns, operand, n == 4 and ns == 2 and bool(rng.random() < .35)))
    held = int(rng.integers(4))
    holdout = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (side, count, ns, operand, terraced) in enumerate(plans):
        split = 'train' if i < num_train else 'test'
        rotation = (held if split == 'test' else int(rng.choice([k for k in range(4) if k != held]))) if holdout else int(rng.integers(4))
        for _ in range(8):
            pair = _make_pair(rng, side, count, ns, operand, roles, rotation, terraced)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('Bounded distinct-input construction exhausted.')
        seen.add(key)
        puzzle[split].append(pair)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair schema error.')
    facts = _read_input(pair['input'])
    output = _grid(pair['output'])
    covers = _assemblies(facts)
    _require(len(covers) == 1, 'Assembly must be uniquely recoverable.')
    assembled = covers[0]
    expected, info = _compute(assembled, facts)
    _require(np.array_equal(expected, output), 'Output violates seam-selected transfer.')
    _require(output.shape == (facts['side'], facts['side']), 'Output must be the assembled square.')
    _require(output[0, 0] == facts['marker'], 'Anchor must be at upper left.')
    for c in facts['carriers']:
        mask = info['owner'] == c
        if c in info['unmarked']:
            _require(np.array_equal(output[mask], assembled[mask]), 'Unmarked panel changed.')
        if c in info['sources']:
            _require(all(output[r, x] == facts['marker'] for r, x in info['slots'][c]), 'Source pips must highlight.')
    # Arithmetic and conservation, checked separately from equality with solve.
    _require(np.sum(output == facts['ink']) ==
             sum(len(info['slots'][c]) for c in info['unmarked']) + info['final'], 'Ink count mismatch.')
    _require(np.sum(output == facts['marker']) == 1 + info['total'], 'Processed-source count mismatch.')
    _require(len(info['added']) == info['total'], 'One new destination pip per source pip.')
    _require(not np.array_equal(assembled, output), 'Omitting the transfer must change the answer.')
    all_other_total = sum(len(info['slots'][c]) for c in facts['carriers'] if c != info['anchor'])
    _require(all_other_total > info['total'], 'Ignoring the seam selection must change the answer.')
    _require(np.sum(output == info['anchor']) == np.sum(assembled == info['anchor']) - info['total'],
             'Only the receiver carrier is consumed by extension.')
    for carrier in facts['carriers']:
        if carrier != info['anchor']:
            _require(np.sum(output == carrier) == np.sum(assembled == carrier), 'Other carrier colors must be preserved.')
    info.update(facts=facts, assembled=assembled)
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle schema error.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Illegal counts.')
    seen, train, conventions = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _analyze_pair(pair)
            k = _key(f['facts']['grid'])
            _require(k not in seen, 'Duplicate input.')
            seen.add(k)
            conventions.add(tuple(f['facts'][r] for r in ('background', 'marker', 'ink')))
            if split == 'train':
                train.append(f)
    _require(len(conventions) == 1, 'Generated puzzles share background, marker and ink roles.')
    _require(any(f['corner_only'] for f in train), 'Training must distinguish edge from corner contact.')
    _require(any(len(f['sources']) >= 2 and len({len(f['slots'][c]) for c in f['sources']}) > 1
                 for f in train), 'Training must show unequal multiple-source addition.')
    operands = [(tuple(len(f['slots'][c]) for c in sorted(f['sources'])), f['initial']) for f in train]
    _require(any(_informative(a, b) for a in operands for b in operands), 'Training arithmetic is underdetermined by simple rivals.')
    _require(any(f['unmarked'] for f in train), 'Training must show ignored panels.')


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[r], q[r])
               for p, q in zip(a[s], b[s]) for r in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """42 seeds, finite extra count cases and targeted semantic mutations; no I/O."""
    shapes, sides, structures, contacts, palettes, areas = set(), set(), set(), set(), set(), set()
    counts, input_keys = set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _analyze_pair(pair)
                facts = f['facts']
                shapes.add(pair['input'].shape)
                sides.add(facts['side'])
                counts.add(len(facts['pieces']))
                contacts.add(len(f['sources']))
                palettes.add((facts['background'], facts['marker'], facts['ink']))
                areas.add(tuple(sorted(int(np.sum(f['owner'] == c)) for c in facts['carriers'])))
                normalized, roles = [], {}
                for x in f['owner'].flat:
                    normalized.append(roles.setdefault(int(x), len(roles)))
                structures.add((f['owner'].shape, tuple(normalized)))
                input_keys.add(_key(pair['input']))
                _require(np.array_equal(solve(pair['input']), pair['output']), 'Input-only solver mismatch.')
    _require(counts == {3, 4, 5} and contacts == {1, 2, 3}, 'Panel and contact counts must vary.')
    _require(len(shapes) >= 25 and len(sides) >= 5 and len(palettes) >= 25, 'Insufficient geometric/palette variation.')
    _require(len(structures) >= 180 and len(areas) >= 150 and len(input_keys) == 210, 'Insufficient structural variation.')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + 10 * a + b, a, b)
            validate_generated_puzzle(p)
            _require(len(p['train']) == a and len(p['test']) == b, 'Explicit counts ignored.')
    _require(_same(generate_puzzle(17), generate_puzzle(17)), 'Seed reproducibility failed.')
    _require(not _same(generate_puzzle(17), generate_puzzle(18)), 'Different seeds produced the same puzzle.')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted.')
    original = generate_puzzle(7, 2, 1)
    f = _analyze_pair(original['train'][0])
    for defect in ('addition', 'highlight', 'ignored', 'anchor'):
        bad = {s: [{r: np.array(g, copy=True) for r, g in pair.items()} for pair in original[s]] for s in original}
        out = bad['train'][0]['output']
        if defect == 'addition':
            r, c = f['added'][0]
            out[r, c] = f['anchor']
        elif defect == 'highlight':
            carrier = min(f['sources'])
            r, c = f['slots'][carrier][0]
            out[r, c] = f['facts']['ink']
        elif defect == 'ignored':
            carrier = min(f['unmarked'])
            r, c = f['slots'][carrier][0]
            out[r, c] = f['facts']['marker']
        else:
            out[0, 0] = f['anchor']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Semantic mutation was accepted: ' + defect)
