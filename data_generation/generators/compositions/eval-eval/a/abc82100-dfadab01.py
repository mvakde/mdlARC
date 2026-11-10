"""Registered stencil interference: object membership followed by parity printing.

No parent code, saved puzzle, seed, or learned dictionary is used by solve().
The two-cell labels and their neutral masks are a complete local instruction set.
"""
from collections import deque
from numbers import Integral

import numpy as np


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_HOST_SEEDS = (0, 859416435, 971534237, 905199, 153650486, 247139231,
               965383268, 814694696, 810791573, 840699421, 287525121,
               689910765, 889712641, 401892209, 84761770, 649900645,
               111156678, 287865481, 739955770, 164221860, 679071808,
               65100644, 365304656, 541354114, 87157089, 833687687,
               105026783, 171227191, 989215787, 890664080, 448405992,
               316364704, 656790090, 778130349, 916498791, 741576238,
               841204568, 411326494, 23283495, 83291652, 694052550, 475743373)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(n < 1 or n > 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=True)


def _key(g):
    return tuple(g.shape), g.tobytes()


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        part.add(q)
                        stack.append(q)
        result.append(part)
    return result


def _instructions(g, bg, neutral):
    """Read each exact mask, including holes, from a unique adjacent label ray."""
    parts = _components(map(tuple, np.argwhere(g == neutral)))
    if not parts:
        raise ValueError("no stencil")
    records = []
    H, W = g.shape
    for part in parts:
        options = []
        for q in sorted(_halo(part)):
            r, c = q
            if not (0 <= r < H and 0 <= c < W) or int(g[q]) not in (bg, neutral):
                continue
            for dr, dc in _DIRS:
                s, t = (r - 2 * dr, c - 2 * dc), (r - dr, c - dc)
                if not (0 <= s[0] < H and 0 <= s[1] < W):
                    continue
                source, ink = int(g[s]), int(g[t])
                if (source in (bg, neutral) or ink in (bg, neutral, source)):
                    continue
                options.append(dict(source=source, ink=ink, q=q, direction=(dr, dc),
                                    labels={s, t}, mask=part,
                                    offsets=tuple(sorted((a - r, b - c) for a, b in part))))
        if len(options) != 1:
            raise ValueError("ambiguous stencil ray")
        records.append(options[0])
    sources = {x['source'] for x in records}
    inks = {x['ink'] for x in records}
    labels = [p for x in records for p in x['labels']]
    if (len(sources) != len(records) or len(inks) != 1
            or sources & inks or len(set(labels)) != len(labels)):
        raise ValueError("inconsistent local roles")
    return records


def _decode(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= g.size // 2:
        raise ValueError("background must occupy a strict majority")
    readings = []
    for neutral in sorted(set(map(int, np.unique(g))) - {bg}):
        try:
            records = _instructions(g, bg, neutral)
            codes = {x['source'] for x in records}
            ink = records[0]['ink']
            removed = set().union(*(x['mask'] | x['labels'] for x in records))
            material = {tuple(map(int, p)) for p in np.argwhere(g != bg)} - removed
            if any(int(g[p]) not in codes | {ink} for p in material):
                continue
            live, rejected, bare, painted, objects = [], [], [], [], []
            for part in _components(material):
                points = [(p, int(g[p])) for p in sorted(part) if int(g[p]) in codes]
                origin = (min(p[0] for p in part), min(p[1] for p in part))
                if len(points) == 1 and points[0][0] == origin:
                    live.append(points[0])
                    if len(part) > 1:
                        painted.append(points[0])
                    kind = 'live'
                elif points:
                    rejected.extend(points)
                    kind = 'attached'
                else:
                    bare.append(part)
                    kind = 'bare'
                objects.append((part, points, kind))
            if {code for p, code in live} != codes:
                continue
            readings.append(dict(grid=g, bg=bg, neutral=neutral, ink=ink, records=records,
                                 live=live, rejected=rejected, bare=bare, painted=painted,
                                 objects=objects))
        except ValueError:
            continue
    if len(readings) != 1:
        raise ValueError("input must identify one complete stencil instruction set")
    return readings[0]


def _footprint(offsets, anchor, shape):
    H, W = shape
    r, c = anchor
    return {(r + dr, c + dc) for dr, dc in offsets
            if 0 <= r + dr < H and 0 <= c + dc < W}


def _render(shape, bg, ink, records, live, *, union=False, bounding=False):
    book = {x['source']: x['offsets'] for x in records}
    coverage = np.zeros(shape, dtype=np.int16)
    for anchor, code in live:
        offsets = book[code]
        if bounding:
            r0, c0 = min(r for r, c in offsets), min(c for r, c in offsets)
            offsets = tuple((r - r0, c - c0) for r, c in offsets)
        for p in _footprint(offsets, anchor, shape):
            coverage[p] += 1
    out = np.full(shape, bg, dtype=np.int8)
    out[coverage > 0 if union else coverage % 2 == 1] = ink
    return out, coverage


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct a blank field, printing the odd-coverage cells of live objects."""
    d = _decode(input_grid)
    return _render(d['grid'].shape, d['bg'], d['ink'], d['records'], d['live'])[0]


def _rotate(p, turns):
    r, c = p
    for _ in range(turns):
        r, c = c, -r
    return r, c


def _connect(cells, h, w, forbidden):
    """Join procedural pieces with shortest eight-neighbor paths, never hidden pixels."""
    cells = set(cells) - forbidden
    while len(parts := _components(cells)) > 1:
        start = parts[0]
        goal = set().union(*parts[1:])
        queue = deque(sorted(start))
        previous = {p: None for p in start}
        end = None
        while queue:
            p = queue.popleft()
            if p in goal:
                end = p
                break
            for q in sorted(_halo({p}) - {p}):
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and q not in forbidden and q not in previous):
                    previous[q] = p
                    queue.append(q)
        if end is None:
            raise ValueError("disconnected procedural mask")
        while end is not None:
            cells.add(end)
            end = previous[end]
    return cells


def _mask(rng, hole, small=False, triple=False):
    h = int(rng.integers(4 if hole else 3, 5 if small else 8))
    w = int(rng.integers(5 if triple else 3, 6 if small and triple else 5 if small else 8))
    family = int(rng.integers(6))
    qr = int(rng.integers(1, h - 1))
    q = (qr, 0)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    if family == 0:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif family == 1:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif family == 2:
        spine = int(rng.integers(1, w - 1))
        cells = {(r, spine) for r in range(h)}
        for r in rng.choice(np.arange(1, h - 1), int(rng.integers(1, h - 1)), replace=False):
            cells.update((int(r), c) for c in range(w))
    elif family == 3:
        cells = {(r, round(r * (w - 1) / (h - 1))) for r in range(h)}
        cells |= {(r, w - 1 - c) for r, c in cells}
    elif family == 4:
        cells = {(r, w - 2) for r in range(h)}
        cells |= {(int(rng.integers(h)), c) for c in range(w)}
    else:
        cells = {(h // 2, w // 2)}
        for _ in range(int(rng.integers(h * w // 2, h * w * 2))):
            p = sorted(cells)[int(rng.integers(len(cells)))]
            q2 = sorted(_halo({p}))[int(rng.integers(9))]
            if 0 <= q2[0] < h and 0 <= q2[1] < w:
                cells.add(q2)
    # Tight bounds and blank corners make every orientation support marked material.
    cells |= {(0, 1), (1, 0), (0, w - 2), (1, w - 1),
              (h - 2, 0), (h - 1, 1), (h - 2, w - 1), (h - 1, w - 2), (qr, 1)}
    if not hole:
        cells.add(q)
    if triple:
        cells.update((qr, c) for c in range(w))
    forbidden = corners | ({q} if hole else set())
    cells = _connect(cells, h, w, forbidden)
    turns = int(rng.integers(4))
    offsets = tuple(sorted(_rotate((r - qr, c), turns) for r, c in cells))
    return offsets, _rotate((0, 1), turns)


def _normal(offsets):
    r0, c0 = min(r for r, c in offsets), min(c for r, c in offsets)
    return {(r - r0, c - c0) for r, c in offsets}


class _PackingError(Exception):
    pass


def _make_pair(rng, bg, neutral, codes, ink, features, extra, teach, attempt):
    n = len(codes)
    small = attempt >= 15
    if small:
        H = W = 30
    else:
        low = 18 if n == 1 else 23 if n == 3 else 21
        H, W = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    g = np.full((H, W), bg, dtype=np.int8)
    records, reserved = [], set()
    cells = [(r, c) for r in range(H) for c in range(W)]

    def shuffled(items):
        items = list(items)
        return [items[int(i)] for i in rng.permutation(len(items))]

    for i, code in enumerate(codes):
        hole = (teach == 1 if i == 0 and teach is not None else bool(rng.integers(2)))
        offsets, direction = _mask(rng, hole, small, triple=teach == 0 and i == 0)
        dr, dc = direction
        relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        for qr, qc in shuffled(cells):
            obj = {(qr + r, qc + c) for r, c in relative}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in obj) or obj & reserved:
                continue
            mask = {(qr + r, qc + c) for r, c in offsets}
            s, t = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
            candidate = g.copy()
            for p in mask:
                candidate[p] = neutral
            candidate[s], candidate[t] = code, ink
            try:
                parsed = _instructions(candidate, bg, neutral)
            except ValueError:
                continue
            if len(parsed) != i + 1:
                continue
            g = candidate
            records.append(dict(source=code, ink=ink, offsets=offsets, mask=mask,
                                labels={s, t}, q=(qr, qc), direction=direction))
            reserved |= _halo(obj)
            break
        else:
            raise _PackingError("instructions")

    live, protected = [], set()
    book = {x['source']: x for x in records}

    def available(p, code, *, full=False, clip=False, material=False):
        if p in reserved:
            return False
        offsets = book[code]['offsets']
        fp = _footprint(offsets, p, g.shape)
        if not fp or fp & (protected | instruction_cells):
            return False
        if full and len(fp) != len(offsets) or clip and len(fp) == len(offsets):
            return False
        if material:
            obj = {(p[0] + r, p[1] + c) for r, c in _normal(offsets)} | {p}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in obj) or obj & reserved:
                return False
        return True

    instruction_cells = set().union(*(x['mask'] | x['labels'] | {x['q']} for x in records))

    def add(p, code, *, material=False):
        g[p] = code
        obj = {p}
        if material:
            obj |= {(p[0] + r, p[1] + c) for r, c in _normal(book[code]['offsets'])}
            for q in obj - {p}:
                g[q] = ink
        reserved.update(_halo(obj))
        live.append((p, code))

    # A separate complete copy for every code exposes its registration without interference.
    for code in codes:
        for p in shuffled(cells):
            if available(p, code, full=True):
                add(p, code)
                protected |= _footprint(book[code]['offsets'], p, g.shape)
                break
        else:
            raise _PackingError("uncontested full copies")

    # Find a real overlap, never infer an answer from a desired cancellation pattern.
    c1, c2 = codes[0], codes[-1]
    found = False
    for p in shuffled(cells):
        if not available(p, c1, full=True):
            continue
        f1 = _footprint(book[c1]['offsets'], p, g.shape)
        if teach == 0:
            dr, dc = book[c1]['direction']
            group = [(p[0] + j * dr, p[1] + j * dc) for j in (0, 2, 4)]
            if not all(0 <= q[0] < H and 0 <= q[1] < W and available(q, c1, full=True)
                       for q in group):
                continue
            common = set.intersection(*(_footprint(book[c1]['offsets'], q, g.shape) for q in group))
            if not common:
                continue
            for q in group:
                add(q, c1)
            protected.add(min(common))  # Keep an observable three-way restoration.
            found = True
        else:
            nearby = [(p[0] + dr, p[1] + dc) for dr in range(-7, 8)
                      for dc in range(-7, 8) if max(abs(dr), abs(dc)) > 1]
            for q in shuffled(nearby):
                if not (0 <= q[0] < H and 0 <= q[1] < W) or not available(q, c2, full=True):
                    continue
                f2 = _footprint(book[c2]['offsets'], q, g.shape)
                if f1 & f2 and f1 != f2:
                    add(p, c1)
                    add(q, c2)
                    found = True
                    break
        if found:
            break
    if not found:
        raise _PackingError("overlapping footprints")

    for feature in ('painted', 'clip'):
        if feature not in features:
            continue
        code = int(rng.choice(codes))
        for p in shuffled(cells):
            if available(p, code, full=feature == 'painted', clip=feature == 'clip',
                         material=feature == 'painted'):
                add(p, code, material=feature == 'painted')
                break
        else:
            raise _PackingError(feature)

    # Later prints may overlap one another but cannot obscure the protected demonstrations.
    for _ in range(extra):
        code = int(rng.choice(codes))
        for p in shuffled(cells):
            if available(p, code):
                add(p, code)
                break
        else:
            break
    out, coverage = _render(g.shape, bg, ink, records, live)
    if not np.any((coverage > 0) & (coverage % 2 == 0)):
        raise _PackingError("overlap lost after extra prints")

    # Bare and lower-right attached fragments are complete copies of a visible template.
    # Keep their original ink outside actual print footprints to expose their erasure.
    for feature in ('bare', 'attached'):
        if feature not in features:
            continue
        code = int(rng.choice(codes))
        norm = _normal(book[code]['offsets'])
        last = max(norm)
        for r, c in shuffled(cells):
            fragment = {(r + dr, c + dc) for dr, dc in norm}
            attached = (r + last[0] + 1, c + last[1] + 1)
            obj = fragment | ({attached} if feature == 'attached' else set())
            if (any(not (0 <= a < H and 0 <= b < W) for a, b in obj)
                    or obj & reserved or any(coverage[p] for p in obj)):
                continue
            if feature == 'attached':
                fp = _footprint(book[code]['offsets'], attached, g.shape)
                if len(fp) != len(book[code]['offsets']):
                    continue  # Interior counterexample to all-code / edge-only filtering.
            for p in fragment:
                g[p] = ink
            if feature == 'attached':
                g[attached] = code
            reserved.update(_halo(obj))
            break
        else:
            raise _PackingError(feature + " fragment")
    if np.count_nonzero(g == bg) <= g.size // 2:
        raise _PackingError("background")
    try:
        d = _decode(g)
    except ValueError as error:
        raise _PackingError(str(error)) from error
    if (d['neutral'] != neutral or sorted(d['live']) != sorted(live)
            or not np.array_equal(solve(g), out)):
        raise _PackingError("visible reading disagrees with construction")
    return {'input': g, 'output': out}


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    backgrounds, neutral = palette[:2], palette[2]
    vocabulary = palette[3:3 + int(rng.integers(1, 4))]
    inks = palette[6:8]
    shift = int(rng.integers(3))  # More prints, fewer prints, or overlapping ordinary ranges.
    puzzle, seen, training_geometry = {'train': [], 'test': []}, set(), set()
    for index in range(nt + nq):
        training = index < nt
        split = 'train' if training else 'test'
        teach = index if index < 2 else None
        if index == 0:
            codes, features = vocabulary[:1], {'bare', 'attached'}
        elif index == 1:
            codes, features = vocabulary[:], {'painted', 'clip'}
        else:
            codes = list(map(int, rng.choice(vocabulary, int(rng.integers(1, len(vocabulary) + 1)), replace=False)))
            features = {f for f in ('bare', 'attached', 'painted', 'clip') if rng.random() < .45}
        bg = backgrounds[index] if index < 2 else int(rng.choice(backgrounds))
        ink = inks[index] if index < 2 else int(rng.choice(inks))
        dense = (not training if shift == 0 else training if shift == 1 else None)
        extra = int(rng.integers(5, 10) if dense is True else rng.integers(0, 3)
                    if dense is False else rng.integers(0, 7))
        for attempt in range(48):
            try:
                pair = _make_pair(rng, bg, neutral, codes, ink, features, extra, teach, attempt)
            except _PackingError:
                continue
            key = _key(pair['input'])
            d = _decode(pair['input'])
            geometry = {r['offsets'] for r in d['records']}
            if key in seen or not training and geometry <= training_geometry:
                continue
            seen.add(key)
            if training:
                training_geometry |= geometry
            puzzle[split].append(pair)
            break
        else:
            raise RuntimeError("bounded stencil packing exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    d = _decode(pair['input'])
    _, coverage = _render(d['grid'].shape, d['bg'], d['ink'], d['records'], d['live'])
    edges = set()
    book = {r['source']: r['offsets'] for r in d['records']}
    full_codes = set()
    for p, code in d['live']:
        offsets = book[code]
        fp = _footprint(offsets, p, d['grid'].shape)
        if len(fp) == len(offsets) and all(coverage[q] == 1 for q in fp):
            full_codes.add(code)
        for dr, dc in offsets:
            r, c = p[0] + dr, p[1] + dc
            if r < 0: edges.add('top')
            if r >= d['grid'].shape[0]: edges.add('bottom')
            if c < 0: edges.add('left')
            if c >= d['grid'].shape[1]: edges.add('right')
    return d, coverage, edges, full_codes


def _check_fragments(d):
    """Check the restricted, demonstrable material vocabulary from the input alone."""
    g, ink = d['grid'], d['ink']
    book = {r['source']: _normal(r['offsets']) for r in d['records']}
    for part, points, kind in d['objects']:
        if len(part) == 1 and kind == 'live':
            continue
        origin = (min(p[0] for p in part), min(p[1] for p in part))
        material = {p for p in part if int(g[p]) == ink}
        if kind == 'bare':
            _require(any(material == {(origin[0] + r, origin[1] + c) for r, c in mask}
                         for mask in book.values()), "unexplained bare fragment")
        else:
            _require(len(points) == 1, "unexpected multi-code object")
            p, code = points[0]
            mask = book[code]
            expected = {(origin[0] + r, origin[1] + c) for r, c in mask}
            _require(material == expected, "fragment does not match its local stencil")
            if kind == 'live':
                _require(p == origin and (0, 0) not in mask, "bad painted origin")
            else:
                last = max(mask)
                _require(p == (origin[0] + last[0] + 1, origin[1] + last[1] + 1),
                         "bad attached origin")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "bad splits")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad counts")
    seen, neutrals, evidence, tested = set(), set(), set(), set()
    targets, train_geometry = {}, set()
    witnesses = {'A': False, 'B': False, 'parity': False, 'odd_overlap': False}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d, coverage, edges, full_codes = _features(pair)
            _require(np.array_equal(solve(inp), out), "incorrect parity reconstruction")
            _require(not np.array_equal(inp, out), "identity example")
            _require(set(map(int, np.unique(out))) <= {d['bg'], d['ink']}, "unclean output")
            neutrals.add(d['neutral'])
            _check_fragments(d)
            _require(full_codes == {r['source'] for r in d['records']}, "missing uncontested full copy")
            _require(np.any((coverage > 0) & (coverage % 2 == 0)), "no cancellation evidence")
            categories = ({'bare'} if d['bare'] else set()) | ({'attached'} if d['rejected'] else set())
            categories |= ({'painted'} if d['painted'] else set()) | ({'clip'} if edges else set())
            geometry = {r['offsets'] for r in d['records']}
            if split == 'train':
                evidence |= categories
                train_geometry |= geometry
                for r in d['records']:
                    targets.setdefault(r['source'], set()).add(d['ink'])
                witnesses['A'] |= not np.array_equal(out, _render(inp.shape, d['bg'], d['ink'],
                                                                  d['records'], d['live'], bounding=True)[0])
                witnesses['B'] |= not np.array_equal(out, _render(inp.shape, d['bg'], d['ink'],
                                                                  d['records'], d['live'] + d['rejected'])[0])
                witnesses['parity'] |= not np.array_equal(out, _render(inp.shape, d['bg'], d['ink'],
                                                                       d['records'], d['live'], union=True)[0])
                witnesses['odd_overlap'] |= bool(np.any((coverage >= 3) & (coverage % 2 == 1)))
            else:
                tested |= categories
                _require(bool(geometry - train_geometry), "test replays all training geometry")
    _require(len(neutrals) == 1, "shared stencil convention changed")
    _require({'bare', 'attached', 'painted', 'clip'} <= evidence, "missing membership/boundary teaching")
    _require(tested <= evidence, "untaught category")
    _require(any(len(v) > 1 for v in targets.values()), "local target changes not taught")
    _require(all(witnesses.values()), "a parent or parity shortcut explains all training")
    _require(any((0, 0) not in r['offsets'] for p in puzzle['train']
                 for r in _decode(p['input'])['records']), "hole registration not taught")


def self_test():
    """Bounded 42-seed audit, explicit counts, hand regressions, and corruption rejection."""
    # Tiny hand calculation: two 3x3 rounded stencils at distance two cancel
    # their common pixel. The labeled registration is left-middle, not top-left.
    g = np.zeros((12, 16), dtype=np.int8)
    g[2, 0], g[2, 1] = 1, 2
    mask = {(-1, 1), (0, 0), (0, 2), (1, 1)}
    for r, c in mask:
        g[2 + r, 2 + c] = 8
    g[7, 6] = g[7, 8] = 1
    expected = np.zeros_like(g)
    for p in {(6, 7), (8, 7), (7, 6), (6, 9), (8, 9), (7, 10)}:
        expected[p] = 2
    _require(np.array_equal(solve(g), expected), "manual cancellation/registration regression")
    # Three length-five lines overlap at column eight: odd coverage restores it.
    g = np.zeros((12, 16), dtype=np.int8)
    g[2, :2] = (1, 2)
    g[2, 2:7] = 8
    g[7, [4, 6, 8]] = 1
    expected2 = np.zeros_like(g)
    expected2[7, [4, 5, 8, 11, 12]] = 2
    _require(np.array_equal(solve(g), expected2), "third print toggle regression")

    puzzles, shapes, masks, directions, edges, counts, palettes = [], set(), set(), set(), set(), set(), set()
    clean = triple = cross = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple((_key(_grid(p['input'])), _key(_grid(p['output'])))
                             for s in ('train', 'test') for p in puzzle[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                d, coverage, clipping, _ = _features(pair)
                shapes.add(d['grid'].shape)
                masks.update(r['offsets'] for r in d['records'])
                directions.update(r['direction'] for r in d['records'])
                edges.update(clipping)
                counts.add(len(d['live']))
                palettes.add((d['bg'], d['neutral'], d['ink']))
                clean += not (d['bare'] or d['rejected'] or d['painted'])
                triple += bool(np.any(coverage >= 3))
                by_code = []
                for r in d['records']:
                    pts = [(p, c) for p, c in d['live'] if c == r['source']]
                    by_code.append(_render(d['grid'].shape, d['bg'], d['ink'], d['records'], pts)[1] > 0)
                cross += any(np.any(a & b) for i, a in enumerate(by_code) for b in by_code[i + 1:])
    _require(len(set(puzzles)) == 42, "insufficient whole-puzzle variation")
    _require(len(shapes) >= 35 and len(masks) >= 70 and len(counts) >= 8, "structural diversity")
    _require(directions == set(_DIRS) and len(edges) == 4, "orientation/boundary diversity")
    _require(len(palettes) >= 70 and clean > 0 and triple > 0 and cross > 0, "interaction diversity")
    a, b = generate_puzzle(73, 2, 1), generate_puzzle(73, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), "nondeterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(19, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(44, 2, 1)
    d = _decode(broken['test'][0]['input'])
    p = tuple(np.argwhere(broken['test'][0]['output'] == d['ink'])[0])
    broken['test'][0]['output'][p] = d['bg']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("deleted output ink accepted")
    return dict(seeds=42, distinct_puzzles=len(set(puzzles)), shapes=len(shapes),
                registered_masks=len(masks), live_counts=sorted(counts), clean_pairs=clean,
                triple_coverage_pairs=triple, cross_code_overlap_pairs=cross)


if __name__ == '__main__':
    print(self_test())
