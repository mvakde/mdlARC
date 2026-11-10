"""Rivet-ranked seam welding of a uniquely assembled rigid-piece square.

Only standard-library code and NumPy are used. No parent code or data is loaded.
The solver recovers roles, rigid pieces, assembly, commands, and contacts from
the input. Sampling settings never enter the solver.
"""
from collections import Counter
import math

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grid must be two dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "Grid must have integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(a, background):
    """Eight-neighbor segmentation; actual family pieces are edge-connected."""
    remaining = set(map(tuple, np.argwhere(a != background).tolist()))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                pos = r + dr, c + dc
                if pos in remaining:
                    remaining.remove(pos)
                    stack.append(pos)
                    cells.append(pos)
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        h = max(p[0] for p in cells) - r0 + 1
        w = max(p[1] for p in cells) - c0 + 1
        p = np.full((h, w), -1, dtype=np.int8)
        for r, c in cells:
            p[r-r0, c-c0] = a[r, c]
        objects.append(p)
    return objects


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    if not cells:
        return False
    stack = [min(cells)]
    cells.remove(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in EDGE:
            q = r+dr, c+dc
            if q in cells:
                cells.remove(q)
                stack.append(q)
    return not cells


def _orientations(p):
    results, seen = [], set()
    for k in range(4):
        q = np.ascontiguousarray(np.rot90(p, k))
        key = _signature(q)
        if key not in seen:
            seen.add(key)
            results.append(q)
    return results


def _parse(input_grid):
    a = _grid(input_grid)
    freq = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(freq))
    _require(np.sum(freq == freq[bg]) == 1, "Background must be uniquely modal")
    objects = _components(a, bg)
    _require(3 <= len(objects) <= 5, "Expected three to five separated pieces")
    area = sum(int(np.sum(p >= 0)) for p in objects)
    side = math.isqrt(area)
    _require(5 <= side <= 10 and side*side == area, "Piece area must be a square of side 5..10")
    singletons = [c for c in range(10) if c != bg and freq[c] == 1]
    _require(len(singletons) == 1, "Exactly one singleton anchor marker is required")
    marker = singletons[0]
    color_objects = {c: [i for i, p in enumerate(objects) if np.any(p == c)]
                     for c in range(10) if c != bg and freq[c] > 0}
    shared = [c for c, ids in color_objects.items() if len(ids) > 1]
    _require(len(shared) == 1 and len(color_objects[shared[0]]) == len(objects),
             "One common rivet color must occur in every piece")
    rivet = shared[0]
    carriers, ranks = [], {}
    anchor = None
    for i, p in enumerate(objects):
        _require(_connected(p >= 0), "Each rigid piece must be edge-connected")
        colors = set(map(int, p[p >= 0])) - {marker, rivet}
        _require(len(colors) == 1 and np.sum(p >= 0) >= 6, "Each piece needs one carrier and at least six cells")
        carrier = next(iter(colors))
        _require(np.sum(p == carrier) >= 3, "Carrier must remain visible")
        pts = np.argwhere(p == rivet)
        rank = len(pts)
        _require(1 <= rank <= 3, "Rivet unary commands have lengths 1..3")
        _require((len(set(pts[:, 0])) == 1 and np.ptp(pts[:, 1]) == rank-1) or
                 (len(set(pts[:, 1])) == 1 and np.ptp(pts[:, 0]) == rank-1),
                 "Rivets must form a single straight contiguous unary run")
        carriers.append(carrier)
        ranks[carrier] = rank
        if np.any(p == marker):
            _require(anchor is None, "Marker must belong to just one piece")
            anchor = i
    _require(len(set(carriers)) == len(objects) and anchor is not None,
             "Carrier identities must be distinct")
    return dict(grid=a, background=bg, marker=marker, rivet=rivet, side=side,
                pieces=objects, carriers=carriers, ranks=ranks, anchor=anchor)


def _assemblies(facts, limit=2, rotations=True, with_owners=False):
    """Exhaustive exact cover. Stop once two distinct full assemblies exist.

    Deduplicate complete colored placements, NOT just silhouette masks: a
    symmetric silhouette can carry an asymmetric rivet command. Distinguish
    owner assignments even if exchanging same-colored rivets hides the seam
    in the colored grid; later ranking needs that seam to be unambiguous.
    """
    n, marker = facts['side'], facts['marker']
    pieces, anchor = facts['pieces'], facts['anchor']
    candidates = []
    for i, p in enumerate(pieces):
        choices, seen = [], set()
        for q in (_orientations(p) if rotations else [p]):
            h, w = q.shape
            if h > n or w > n:
                continue
            cells = [(int(r)*n+int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            if i == anchor:
                if q[0, 0] != marker:
                    continue
                origins = [(0, 0)]
            else:
                origins = [(r, c) for r in range(n-h+1) for c in range(n-w+1)]
            for r, c in origins:
                offset = r*n+c
                colored = tuple((pos+offset, v) for pos, v in cells)
                if colored in seen:
                    continue
                seen.add(colored)
                bits = sum(1 << pos for pos, _ in colored)
                choices.append((bits, colored))
        if not choices:
            return []
        candidates.append(choices)
    full = (1 << (n*n))-1
    placed, results, seen_answers = [None]*len(pieces), [], set()

    def search(occupied, remaining):
        if len(results) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            answer = np.empty((n, n), dtype=np.int8)
            owners = np.empty((n, n), dtype=np.int8)
            for i, placement in enumerate(placed):
                for pos, v in placement[1]:
                    answer.flat[pos] = v
                    owners.flat[pos] = facts['carriers'][i]
            key = _signature(answer), _signature(owners)
            if key not in seen_answers:
                seen_answers.add(key)
                results.append((answer, owners) if with_owners else answer)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        eligible = []
        for i in remaining:
            opts = [x for x in candidates[i] if x[0] & first and not x[0] & occupied]
            if opts:
                eligible.append((i, opts))
        eligible.sort(key=lambda x: len(x[1]))
        for i, opts in eligible:
            rest = tuple(j for j in remaining if j != i)
            for placement in opts:
                placed[i] = placement
                search(occupied | placement[0], rest)
                if len(results) >= limit:
                    return

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for placement in candidates[anchor]:
        placed[anchor] = placement
        search(placement[0], rest)
        if len(results) >= limit:
            break
    return results


def _interfaces(owners):
    """A length is the number of shared unit edges, even on a bent boundary."""
    seams = {}
    n = owners.shape[0]
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r+dr, c+dc
                if rr < n and cc < n and owners[r, c] != owners[rr, cc]:
                    key = tuple(sorted((int(owners[r, c]), int(owners[rr, cc]))))
                    seams.setdefault(key, []).append(((r, c), (rr, cc)))
    return seams


def _weld(assembled, owners, facts, override=None):
    seams = _interfaces(owners)
    selected, paint = set(), np.zeros(owners.shape, dtype=bool)
    for carrier in facts['carriers']:
        incident = [key for key in seams if carrier in key]
        _require(incident, "Every piece must have a neighbor")
        lengths = [len(seams[key]) for key in incident]
        _require(len(set(lengths)) == len(lengths), "Incident interface lengths must be distinct")
        rank = facts['ranks'][carrier] if override is None else override
        _require(rank <= len(incident), "Requested neighbor rank must exist")
        key = sorted(incident, key=lambda k: -len(seams[k]))[rank-1]
        selected.add(key)
    for key in selected:
        for p, q in seams[key]:
            paint[p] = paint[q] = True
    paint[assembled == facts['marker']] = False
    output = assembled.copy()
    output[paint] = facts['rivet']
    return output, seams, selected, paint


def _analyze(input_grid):
    facts = _parse(input_grid)
    answers = _assemblies(facts, with_owners=True)
    _require(len(answers) == 1, "Pieces must have exactly one anchored colored and owned square assembly")
    assembled, owners = answers[0]
    output, seams, selected, paint = _weld(assembled, owners, facts)
    return facts, assembled, owners, output, seams, selected, paint


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the supplied input pixels."""
    return _analyze(input_grid)[3]


def _contacts(labels):
    seams = _interfaces(labels)
    neighbors = {int(i): {} for i in np.unique(labels)}
    for (a, b), cells in seams.items():
        neighbors[a][b] = neighbors[b][a] = len(cells)
    return neighbors


def _grow_partition(rng, side, count, style):
    labels = np.full((side, side), -1, dtype=np.int8)
    if style == 'corners':
        seeds = [0, side-1, side*(side-1), side*side-1]
        if count == 3:
            seeds = seeds[:3]
        elif count == 5:
            seeds.append((side//2)*side+side//2)
    else:
        seeds = list(map(int, rng.choice(side*side, count, replace=False)))
    for i, pos in enumerate(seeds):
        labels.flat[pos] = i
    weights = np.exp(rng.uniform(-.7, .7, count)) if style == 'unequal' else np.ones(count)
    frontiers = [set() for _ in range(count)]
    sizes = [1]*count

    def adjacent(pos):
        r, c = divmod(pos, side)
        return [rr*side+cc for dr, dc in EDGE if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]

    for i, pos in enumerate(seeds):
        frontiers[i].update(p for p in adjacent(pos) if labels.flat[p] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if frontiers[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*2)
        candidates = sorted(frontiers[i])
        pos = candidates[int(rng.integers(len(candidates)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontiers:
            f.discard(pos)
        frontiers[i].update(p for p in adjacent(pos) if labels.flat[p] < 0)
    return labels


def _cut_partition(rng, side, count):
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        eligible = [i for i, (r, c, h, w) in enumerate(rectangles) if h*w >= 12]
        if not eligible:
            return None
        i = int(rng.choice(eligible))
        r, c, h, w = rectangles[i]
        cuts = [(0, k) for k in range(1, h) if min(k, h-k)*w >= 6]
        cuts += [(1, k) for k in range(1, w) if min(k, w-k)*h >= 6]
        if not cuts:
            return None
        axis, k = cuts[int(rng.integers(len(cuts)))]
        rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    for _ in range(side*5):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if not (0 <= rr < side and 0 <= cc < side):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.sum(labels == a) <= 6:
            continue
        labels[r, c] = b
        if not _connected(labels == a):
            labels[r, c] = a
    return labels


def _runs(mask, length, excluded):
    choices = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            for dr, dc in ((0, 1), (1, 0)):
                cells = [(r+dr*k, c+dc*k) for k in range(length)]
                if all(rr < h and cc < w and mask[rr, cc] and (rr, cc) != excluded for rr, cc in cells):
                    choices.append(cells)
    return choices


def _extract_pieces(assembled, labels):
    pieces = []
    for i in range(int(labels.max())+1):
        coords = np.argwhere(labels == i)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)+1
        p = np.full((r1-r0, c1-c0), -1, dtype=np.int8)
        view = assembled[r0:r1, c0:c1]
        mask = labels[r0:r1, c0:c1] == i
        p[mask] = view[mask]
        pieces.append(p)
    return pieces


def _decorate(rng, labels, carriers, marker, rivet, ranks):
    assembled = np.asarray(carriers, dtype=np.int8)[labels]
    for i, rank in enumerate(ranks):
        choices = _runs(labels == i, rank, (0, 0))
        if not choices:
            return None
        # Prefer a visible embedded mark, but edge marks are also permitted.
        cells = choices[int(rng.integers(len(choices)))]
        for r, c in cells:
            assembled[r, c] = rivet
    assembled[0, 0] = marker
    if any(np.sum((labels == i) & (assembled == color)) < 3 for i, color in enumerate(carriers)):
        return None
    return assembled


def _synthetic_facts(assembled, labels, carriers, marker, rivet, ranks):
    return dict(side=labels.shape[0], marker=marker, rivet=rivet,
                pieces=_extract_pieces(assembled, labels), carriers=carriers,
                anchor=int(labels[0, 0]), ranks=dict(zip(carriers, ranks)))


def _scatter(rng, pieces, background, marker, loose=False):
    """Place actual masks, leaving a one-cell moat even diagonally."""
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    asymmetric = [i for i, p in enumerate(pieces) if len(_orientations((p >= 0).astype(np.int8))) == 4]
    if asymmetric:
        rotations[asymmetric[0]] = int(rng.integers(1, 4))
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
    for attempt in range(12):
        canvas = np.full((30, 30), background, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        maxr = maxc = 0
        if attempt:
            rng.shuffle(order)
        success = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            coords = np.argwhere(p >= 0)
            options = []
            for r in range(31-h):
                for c in range(31-w):
                    if np.any(blocked[r+coords[:, 0], c+coords[:, 1]]):
                        continue
                    hh, ww = max(maxr, r+h), max(maxc, c+w)
                    score = hh*ww + abs(hh-ww)*2 + float(rng.random())*(60 if loose else 5)
                    options.append((score, r, c))
            if not options:
                success = False
                break
            _, r, c = min(options)
            for rr, cc in coords:
                y, x = r+int(rr), c+int(cc)
                canvas[y, x] = p[rr, cc]
                blocked[max(0, y-1):min(30, y+2), max(0, x-1):min(30, x+2)] = True
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not success:
            continue
        foreground = np.argwhere(canvas != background)
        r0, c0 = foreground.min(axis=0)
        r1, c1 = foreground.max(axis=0)+1
        core = canvas[r0:r1, c0:c1]
        h, w = core.shape
        H = h+int(rng.integers(min(4, 30-h)+1))
        W = w+int(rng.integers(min(4, 30-w)+1))
        freq = np.bincount(core.ravel(), minlength=10)
        while H*W-h*w+freq[background] <= max(freq[c] for c in range(10) if c != background):
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                break
        top = int(rng.integers(H-h+1))
        left = int(rng.integers(W-w+1))
        result = np.full((H, W), background, dtype=np.int8)
        result[top:top+h, left:left+w] = core
        return result
    return None


def _make_geometry(rng, side, count, carriers, marker, rivet, teaching=False, style=None):
    """Rejection checks feasibility and unique colored assembly, not a template."""
    for attempt in range(1500):
        mode = style or str(rng.choice(['corners', 'free', 'unequal', 'cuts']))
        labels = _cut_partition(rng, side, count) if mode == 'cuts' else _grow_partition(rng, side, count, mode)
        if labels is None or min(np.sum(labels == i) for i in range(count)) < 6:
            continue
        contacts = _contacts(labels)
        if any(len(set(x.values())) != len(x) or not x for x in contacts.values()):
            continue
        eligible = [i for i in range(count) if len(contacts[i]) >= 3]
        if teaching and not eligible:
            continue
        ranks = [int(rng.integers(1, min(3, len(contacts[i]))+1)) for i in range(count)]
        contrast = int(rng.choice(eligible)) if teaching else None
        if teaching:
            ranks[contrast] = 1
        assembled = _decorate(rng, labels, carriers, marker, rivet, ranks)
        if assembled is None:
            continue
        facts = _synthetic_facts(assembled, labels, carriers, marker, rivet, ranks)
        answers = _assemblies(facts)
        if len(answers) != 1 or not np.array_equal(answers[0], assembled):
            continue
        output, seams, selected, paint = _weld(assembled, np.asarray(carriers, dtype=np.int8)[labels], facts)
        if np.sum(output != assembled) < 4:
            continue
        if teaching:
            # Keep every other command and every original rivet fixed; extend
            # just one singleton into a domino to contrast ranks one and two.
            pos = tuple(map(int, np.argwhere((labels == contrast) & (assembled == rivet))[0]))
            extensions = [(pos[0]+dr, pos[1]+dc) for dr, dc in EDGE
                          if 0 <= pos[0]+dr < side and 0 <= pos[1]+dc < side
                          and labels[pos[0]+dr, pos[1]+dc] == contrast
                          and (pos[0]+dr, pos[1]+dc) != (0, 0)]
            if not extensions:
                continue
            rng.shuffle(extensions)
            second = None
            for extension in extensions:
                alternate = assembled.copy()
                alternate[extension] = rivet
                if np.sum((labels == contrast) & (alternate == carriers[contrast])) < 3:
                    continue
                alt_ranks = ranks.copy()
                alt_ranks[contrast] = 2
                alt_facts = _synthetic_facts(alternate, labels, carriers, marker, rivet, alt_ranks)
                alt_answers = _assemblies(alt_facts)
                if len(alt_answers) != 1 or not np.array_equal(alt_answers[0], alternate):
                    continue
                alt_out, _, alt_selected, alt_paint = _weld(alternate, np.asarray(carriers, dtype=np.int8)[labels], alt_facts)
                # Reciprocal choices can make one changed command redundant.
                # Require a visible change to the welded interfaces themselves.
                if selected == alt_selected or not np.any((paint != alt_paint) &
                        (assembled != rivet) & (alternate != rivet) & (assembled != marker)):
                    continue
                second = (alternate, alt_ranks, alt_out)
                break
            if second is None:
                continue
            # Rank two must disagree with the simple always-longest shortcut.
            longest = _weld(second[0], np.asarray(carriers, dtype=np.int8)[labels], alt_facts, override=1)[0]
            if np.array_equal(second[2], longest):
                continue
            return labels, [(assembled, ranks, output), second]
        return labels, [(assembled, ranks, output)]
    raise RuntimeError("Could not construct a unique seam-ranking geometry")


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One rule with a matched command contrast and a geometric test shift."""
    a = _checked_count(num_train, 'num_train', 2, 10)
    b = _checked_count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    marker, rivet, base_bg = palette[:3]
    pool = palette[3:]
    count = int(rng.integers(4, 6))
    side = int(rng.integers(max(6, count+1), 9))
    shift = str(rng.choice(['scale', 'topology', 'recombine']))
    mixed_palette = bool(rng.integers(2))
    labels, teaching = _make_geometry(rng, side, count, pool[:count], marker, rivet, teaching=True)
    puzzle, seen = {'train': [], 'test': []}, set()

    def append(split, labels, assembled, ranks, expected, carriers, bg):
        pieces = _extract_pieces(assembled, labels)
        for _ in range(30):
            inp = _scatter(rng, pieces, bg, marker, loose=bool(rng.integers(2)))
            if inp is None or _signature(inp) in seen:
                continue
            # Enforce actual rotation evidence from pixels, even if a
            # symmetrical colored piece could be copied without rotation.
            facts = _parse(inp)
            if split == 'train' and len(puzzle['train']) < 2 and _assemblies(facts, limit=1, rotations=False):
                continue
            out = solve(inp)
            _require(np.array_equal(out, expected), "Input decode disagrees with construction")
            seen.add(_signature(inp))
            puzzle[split].append({'input': inp, 'output': out})
            return
        raise RuntimeError('Could not scatter a distinct rotated input')

    # Matched topology and carriers make the unary intervention identifiable.
    # Input placements/quarter turns are independently resampled.
    for assembled, ranks, expected in teaching:
        append('train', labels, assembled, ranks, expected, pool[:count], base_bg)
    for index in range(2, a+b):
        split = 'train' if index < a else 'test'
        target = split == 'test'
        n = int(rng.integers(3, 6)) if a >= 3 else count
        s = int(rng.integers(max(5, n+1), 9))
        if target and shift == 'scale':
            s = int(rng.integers(9, 11))
        elif target:
            s = side if shift == 'topology' else int(rng.integers(max(6, n+1), 9))
        bg = base_bg
        carriers = pool[:n]
        if mixed_palette and index >= 2:
            colors = [c for c in range(10) if c not in (marker, rivet)]
            rng.shuffle(colors)
            bg, carriers = colors[0], colors[1:n+1]
        mode = 'unequal' if target and shift == 'topology' else None
        lab, records = _make_geometry(rng, s, n, carriers, marker, rivet, style=mode)
        assembled, ranks, expected = records[0]
        append(split, lab, assembled, ranks, expected, carriers, bg)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts")
    seen, roles, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_signature(inp) not in seen, "Duplicate input")
            seen.add(_signature(inp))
            f, assembled, owners, expected, seams, selected, paint = _analyze(inp)
            _require(np.array_equal(expected, out), "Incorrect assembly or ranked boundary weld")
            _require(out.shape == (f['side'], f['side']), "Output size must follow piece area")
            _require(out[0, 0] == f['marker'] and np.sum(out == f['marker']) == 1, "Preserve anchored marker")
            _require(not np.any(out == f['background']), "Discard background")
            _require(np.all((out == assembled) | (out == f['rivet'])), "Only rivet-colored welding is allowed")
            _require(np.all(out[~paint] == assembled[~paint]), "Nonselected cells must be preserved")
            _require(np.sum(out != assembled) >= 1, "Every example must actually weld")
            # Rivet and marker membership remains rigid even when individual
            # carrier counts decrease after welding.
            _require(np.sum(out == f['rivet']) >= sum(f['ranks'].values()), "Retain every command rivet")
            roles.add((f['marker'], f['rivet']))
            if split == 'train':
                training.append((f, assembled, owners, out, seams, selected, paint))
    _require(len(roles) == 1, "Marker and rivet colors must be consistent within a puzzle")
    ranks = {n for t in training for n in t[0]['ranks'].values()}
    _require({1, 2} <= ranks, "Demonstrations must expose unary ordinal progression")
    _require(any(not np.array_equal(t[3], _weld(t[1], t[2], t[0], override=1)[0]) for t in training),
             "Training must refute always choosing the longest boundary")
    _require(any(not _assemblies(t[0], limit=1, rotations=False) for t in training),
             "Training must require rigid quarter-turn assembly")
    return None


def self_test():
    """Bounded semantic, ablation, count, determinism, and 42-seed diversity checks."""
    records, signatures = [], set()
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple(_signature(q['input']) for s in ('train', 'test') for q in p[s]))
        for split in ('train', 'test'):
            for q in p[split]:
                f, assembled, owners, out, seams, selected, paint = _analyze(q['input'])
                degree = Counter(c for key in seams for c in key)
                records.append(dict(seed=seed, split=split, shape=q['input'].shape, side=f['side'],
                                    count=len(f['pieces']), ranks=tuple(f['ranks'].values()),
                                    lengths=tuple(sorted(len(x) for x in seams.values())),
                                    degree=tuple(sorted(degree.values())), bg=f['background'],
                                    structure=_signature(owners), painted=int(np.sum(paint))))
    _require(len(signatures) == 42, "Different seeds must produce distinct puzzles")
    _require({3, 4, 5} <= {r['count'] for r in records}, "Vary piece counts")
    _require({1, 2, 3} <= {x for r in records for x in r['ranks']}, "Exercise all supported ordinals")
    _require(len({r['shape'] for r in records}) >= 30, "Vary scatter geometry")
    _require(len({r['side'] for r in records}) >= 5, "Vary square scale")
    _require(len({r['degree'] for r in records}) >= 8, "Vary adjacency topology")
    _require(len({r['lengths'] for r in records}) >= 100, "Vary interface geometry")
    _require(len({r['bg'] for r in records}) >= 8, "Vary palette roles")
    # Explicit-count requests include the smallest teaching budget and largest
    # return size. The external verifier exercises every legal combination.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(250+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    repeat = generate_puzzle(0)
    for s in ('train', 'test'):
        for x, y in zip(repeat[s], puzzles[0][s]):
            _require(all(np.array_equal(x[k], y[k]) for k in x), "Seed reproducibility")
    # Input recoloring is equivariant, with roles read anew; translating or
    # globally rotating a scatter leaves the anchored output unchanged.
    pair = puzzles[0]['test'][0]
    f = _parse(pair['input'])
    mapping = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    _require(np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']]), "Color-role invariance")
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(pair['input'], k)), pair['output']), "Scatter rotation invariance")
    inp = pair['input']
    if max(inp.shape) <= 28:
        translated = np.pad(inp, ((1, 1), (1, 1)), constant_values=f['background'])
        _require(np.array_equal(solve(translated), pair['output']), "Scatter translation invariance")
    bad = {s: [{'input': q['input'].copy(), 'output': q['output'].copy()} for q in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = f['rivet']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output was accepted')
    return dict(seeds=42, pairs=len(records), shapes=len({r['shape'] for r in records}),
                topologies=len({r['degree'] for r in records}),
                interface_profiles=len({r['lengths'] for r in records}))


if __name__ == '__main__':
    print(self_test())
