"""Ranked mirror mosaics: geometry addresses distributed reflection orbits.

Only the standard library and NumPy are used. Solver state is entirely local to
the input. Labels come from a constructed complete scene, checked afterward
against input-only inference; neither seeds nor construction metadata reach solve.
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int16, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _badge(kind, color, background):
    a = np.full((3, 3), color, dtype=np.int16)
    if kind == 0:
        a[1, 1] = background
    return a


def _components(a, bg):
    todo = a != bg
    result = []
    while np.any(todo):
        r, c = map(int, np.argwhere(todo)[0])
        todo[r, c] = False
        stack, points = [(r, c)], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= x < a.shape[0] and 0 <= y < a.shape[1] and todo[x, y]:
                    todo[x, y] = False
                    stack.append((x, y))
        rows, cols = zip(*points)
        r, c = min(rows), min(cols)
        b, d = max(rows)+1, max(cols)+1
        result.append((r, c, a[r:b, c:d].copy(), len(points)))
    return result


def _queues(cards, horizontal=False):
    def order(i):
        o = cards[i]
        return (o['left'], o['top']) if horizontal else (o['top'], o['left'])
    return [sorted((i for i, o in enumerate(cards) if o['kind'] == k), key=order)
            for k in (0, 1)]


def _parse(a):
    bg = int(a[0, 0])
    _need(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
          "background border is not uniform")
    badges, payloads = [], {}
    for r, c, patch, count in _components(a, bg):
        if patch.shape == (3, 3):
            color = int(patch[0, 0])
            kinds = [k for k in (0, 1) if np.array_equal(patch, _badge(k, color, bg))]
            _need(color != bg and len(kinds) == 1, "malformed badge")
            badges.append((r, c, color, kinds[0]))
        else:
            _need(patch.shape[0] == patch.shape[1] and patch.shape[0] >= 4,
                  "malformed mosaic")
            _need(count == patch.size, "mosaic contains background holes")
            payloads[r, c] = patch
    _need(len(badges) >= 2 and len(badges) == len(payloads), "card association")
    colors = {b[2] for b in badges}
    sizes = {p.shape[0] for p in payloads.values()}
    _need(len(colors) == len(sizes) == 1, "badge color or scale disagreement")
    tag, n = colors.pop(), sizes.pop()
    cards = []
    for r, c, color, kind in sorted(badges):
        _need((r, c+4) in payloads, "missing adjacent mosaic")
        p = payloads.pop((r, c+4))
        _need(not np.any(p == tag), "badge color in mosaic")
        _need(r >= 1 and c >= 1 and r+n < a.shape[0] and c+4+n < a.shape[1],
              "card crosses background border")
        cards.append(dict(top=r, left=c, kind=kind, patch=p))
    _need(not payloads, "unclaimed mosaic")
    for i, o in enumerate(cards):
        for q in cards[:i]:
            _need(_separate((o['top'], o['left']), (q['top'], q['left']), n),
                  "card footprints lack a background gap")
    queues = _queues(cards)
    _need(all(queues), "both badge classes are required")
    return dict(grid=a, background=bg, badge_color=tag, size=n,
                cards=cards, queues=queues)


def _separate(p, q, n):
    r, c = p
    s, d = q
    return r+n < s or s+n < r or c+n+4 < d or d+n+4 < c


def _pairing(a, mode='rank'):
    h, s = a['queues']
    if mode == 'horizontal':
        h, s = _queues(a['cards'], horizontal=True)
    elif mode == 'reverse':
        s = s[::-1]
    elif mode == 'swap':
        s = s.copy()
        if len(s) >= 2:
            s[0], s[1] = s[1], s[0]
    elif mode == 'local':
        return []
    elif mode == 'nearest':
        available, pairs = set(s), []
        for i in h:
            if not available:
                break
            o = a['cards'][i]
            def distance(j):
                q = a['cards'][j]
                return (abs(o['top']-q['top'])+abs(o['left']-q['left']),
                        q['top'], q['left'])
            j = min(available, key=distance)
            available.remove(j)
            pairs.append((i, j))
        return pairs
    elif mode == 'greedy':
        # Global scanning cannot go back to fill a hole in an earlier row.
        pairs, slots = [], {}
        for i in sorted(range(len(a['cards'])),
                        key=lambda j: (a['cards'][j]['top'], a['cards'][j]['left'])):
            k = a['cards'][i]['kind']
            if k in slots:
                if len(slots) == 2:
                    pairs.append((slots[0], slots[1]))
                slots = {}
            slots[k] = i
        if len(slots) == 2:
            pairs.append((slots[0], slots[1]))
        return pairs
    return list(zip(h, s))


def _ids(a, pairing):
    """Canonical coordinates: fold rows for H and columns for S, then transpose S.

    Joining canonical identifiers implements the whole reflection orbit, rather
    than choosing one source patch or a fixed sequence of repair operations.
    """
    n, cards = a['size'], a['cards']
    groups, next_group = {}, 0
    for h, s in pairing:
        _need(h not in groups and s not in groups, "non-bijective pairing")
        groups[h] = groups[s] = next_group
        next_group += 1
    for i in range(len(cards)):
        if i not in groups:
            groups[i] = next_group
            next_group += 1
    rr, cc = np.indices((n, n))
    result = []
    for i, o in enumerate(cards):
        if o['kind'] == 0:
            x, y = np.minimum(rr, n-1-rr), cc
        else:
            x, y = np.minimum(cc, n-1-cc), rr
        result.append(groups[i]*n*n + x*n + y)
    return np.stack(result)


def _rectangle(p, color):
    sites = np.argwhere(p == color)
    if len(sites) == 0:
        return None
    top, left = map(int, sites.min(axis=0))
    bottom, right = map(int, sites.max(axis=0)+1)
    if (bottom-top)*(right-left) != len(sites):
        return False
    return top, bottom, left, right


def _complete(a, color, pairing=None, check_rectangles=True):
    patches = np.stack([o['patch'] for o in a['cards']])
    hidden = patches == color
    if not np.any(hidden):
        return None
    if check_rectangles and any(_rectangle(p, color) is False for p in patches):
        return None
    ids = _ids(a, _pairing(a) if pairing is None else pairing)
    flat = ids.ravel()
    vis = ~hidden.ravel()
    values = patches.ravel()
    length = int(flat.max())+1
    lo = np.full(length, 10, dtype=np.int16)
    hi = np.full(length, -1, dtype=np.int16)
    np.minimum.at(lo, flat[vis], values[vis])
    np.maximum.at(hi, flat[vis], values[vis])
    used = np.unique(flat)
    if np.any(hi[used] < 0) or np.any(lo[used] != hi[used]):
        return None
    return hi[ids]


def _analyze(value):
    a = _parse(_grid(value))
    answers = []
    colors = set(int(c) for o in a['cards'] for c in np.unique(o['patch']))
    for color in sorted(colors):
        restored = _complete(a, color)
        if restored is not None:
            answers.append((color, restored))
    _need(len(answers) == 1, "erasure color has no unique witnessed completion")
    a['mask_color'], a['restored'] = answers[0]
    return a


def _render(a, restored):
    out = a['grid'].copy()
    n = a['size']
    for o, p in zip(a['cards'], restored):
        r, c = o['top'], o['left']+4
        out[r:r+n, c:c+n] = p
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore an input using only its geometry, colors and visible orbit values."""
    a = _analyze(input_grid)
    return _render(a, a['restored'])


def _raw_copy(a):
    patches = np.stack([o['patch'] for o in a['cards']])
    alt = patches.copy()
    partners = {}
    for h, s in _pairing(a):
        partners[h], partners[s] = s, h
    m, n = a['mask_color'], a['size']
    for i, p in enumerate(patches):
        for r, c in np.argwhere(p == m):
            j = partners.get(i)
            v = a['background'] if j is None else int(patches[j, r, c])
            if v == m:
                v = a['background']
            alt[i, r, c] = v
    return alt


def _evidence(a):
    m, restored, cards = a['mask_color'], a['restored'], a['cards']
    alternatives = {}
    for mode in ('local', 'horizontal', 'greedy', 'nearest', 'reverse', 'swap'):
        b = _complete(a, m, _pairing(a, mode))
        alternatives[mode] = b is None or not np.array_equal(b, restored)
        if mode == 'swap':
            alternatives['ambiguous_twins'] = b is not None and not np.array_equal(b, restored)
    alternatives['raw'] = not np.array_equal(_raw_copy(a), restored)
    alternatives['twins'] = any(
        cards[i]['kind'] == cards[j]['kind'] and
        np.array_equal(cards[i]['patch'], cards[j]['patch']) and
        not np.array_equal(restored[i], restored[j])
        for i in range(len(cards)) for j in range(i))
    alternatives['ties'] = any(len({cards[i]['top'] for i in q}) < len(q) for q in a['queues'])
    n, blocked = a['size'], 0
    for h, s in _pairing(a):
        for i, j in ((h, s), (s, h)):
            p, q = cards[i]['patch'], cards[j]['patch']
            for r, c in np.argwhere(p == m):
                mate = p[n-1-r, c] if cards[i]['kind'] == 0 else p[r, n-1-c]
                if mate == m and q[c, r] == m:
                    blocked += 1
    alternatives['composed'] = blocked > 0
    alternatives['blocked_cells'] = blocked
    return alternatives


def _capacity(n):
    return (29//(n+1))*(29//(n+5))


def _positions(rng, n, total, witness=False):
    """Feasible initialization followed by legal free relocation of footprints."""
    if witness:
        # A tied pair of twin targets is anchored during relocation. The three
        # remaining cards move freely; the tie is a demonstrated convention.
        minimum_h, minimum_w = 3*n+4, 2*n+11
        h = int(rng.integers(minimum_h, 31))
        w = int(rng.integers(minimum_w, 31))
        positions = [(1, 1), (1, n+6), (n+2, n+6), (2*n+3, 1), (2*n+3, n+6)]
    else:
        cols_available, rows_available = 29//(n+5), 29//(n+1)
        choices = [c for c in range(1, cols_available+1)
                   if (total+c-1)//c <= rows_available]
        _need(bool(choices), "infeasible count and scale")
        cols = int(rng.choice(choices))
        rows = (total+cols-1)//cols
        minimum_h, minimum_w = rows*(n+1)+1, cols*(n+5)+1
        h = int(rng.integers(minimum_h, 31))
        w = int(rng.integers(minimum_w, 31))
        vg = rng.multinomial(h-minimum_h, np.full(rows+1, 1/(rows+1)))
        hg = rng.multinomial(w-minimum_w, np.full(cols+1, 1/(cols+1)))
        tops, lefts = [], []
        r, c = 1+int(vg[0]), 1+int(hg[0])
        for i in range(rows):
            tops.append(r)
            r += n+1+int(vg[i+1])
        for j in range(cols):
            lefts.append(c)
            c += n+5+int(hg[j+1])
        sites = [(r, c) for r in tops for c in lefts]
        positions = [sites[int(i)] for i in rng.permutation(len(sites))[:total]]
    rr, cc = np.indices((h-n-1, w-n-5))
    rr, cc = rr+1, cc+1
    for _ in range(3):
        moving = range(2, total) if witness else range(total)
        for idx in rng.permutation(list(moving)):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != idx:
                    legal &= ((rr+n < r) | (r+n < rr) |
                              (cc+n+4 < c) | (c+n+4 < cc))
            if witness and idx == 4:
                legal &= rr > positions[0][0]
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(idx)] = int(r+1), int(c+1)
    if witness:
        # Translate the tied target pair jointly to a legal site, keeping it
        # before the unmatched target; unequal gaps and absolute offsets vary.
        options = []
        for r in range(1, positions[4][0]):
            for c in range(1, w-2*(n+4)-1):
                max_gap = w-1-(c+2*(n+4))
                for gap in range(1, max_gap+1):
                    p, q = (r, c), (r, c+n+4+gap)
                    if all(_separate(t, other, n) for t in (p, q) for other in positions[2:]):
                        options.append((p, q))
        if options:
            positions[0], positions[1] = options[int(rng.integers(len(options)))]
    return (h, w), positions


def _colors(rng):
    perm = rng.permutation(10)
    bg, tag, mask = map(int, perm[:3])
    count = int(rng.integers(3, 8))
    return bg, tag, mask, tuple(map(int, perm[3:3+count]))


def _pattern(rng, n, palette, style):
    half_h = (n+1)//2
    if style == 0:
        half = rng.choice(palette, size=(half_h, n))
    elif style == 1:
        weights = rng.dirichlet(np.ones(len(palette)))+.04
        half = rng.choice(palette, size=(half_h, n), p=weights/weights.sum())
    elif style == 2:
        side = int(rng.integers(2, 4))
        core = rng.choice(palette, size=((half_h+side-1)//side, (n+side-1)//side))
        half = np.repeat(np.repeat(core, side, 0), side, 1)[:half_h, :n].copy()
    else:
        motif = rng.choice(palette, size=(2, 3))
        rr, cc = np.indices((half_h, n))
        half = motif[rr % 2, cc % 3].copy()
    # Rare colors are witnessed too; symmetry is applied after anchoring them.
    locations = rng.choice(half.size, len(palette), replace=False)
    half.flat[locations] = palette
    return half[np.minimum(np.arange(n), n-1-np.arange(n))].astype(np.int16)


def _local_mask(rng, n, kind):
    extent = n//2
    length = int(rng.integers(1, extent+1))
    start = int(rng.integers(0, extent-length+1))
    if rng.integers(2):
        start = n-start-length
    across = int(rng.integers(1, n+1))
    offset = int(rng.integers(0, n-across+1))
    return ((start, start+length, offset, offset+across) if kind == 0
            else (offset, offset+across, start, start+length))


def _coupled_masks(rng, n, target_kind):
    # The symmetric band erases both local mates. The source rectangle hides
    # some diagonal mates on just one half, leaving the composed route intact.
    depth = int(rng.integers(1, n//2+1))
    span = 2*depth + n % 2
    start = (n-span)//2
    width = int(rng.integers(1, n))
    offset = int(rng.integers(0, n-width+1))
    source_depth = int(rng.integers(1, depth+1))
    source_width = int(rng.integers(1, width+1))
    source_offset = int(rng.integers(offset, offset+width-source_width+1))
    target = (start, start+span, offset, offset+width)
    source = (source_offset, source_offset+source_width, start, start+source_depth)
    if target_kind == 1:
        target = target[2], target[3], target[0], target[1]
        source = source[2], source[3], source[0], source[1]
    return target, source


def _scene(rng, n, counts, colors, style, witness_kind=None):
    bg, tag, mask, palette = colors
    witness = witness_kind is not None
    shape, positions = _positions(rng, n, sum(counts), witness)
    if witness:
        kinds = [witness_kind, witness_kind, 1-witness_kind, 1-witness_kind, witness_kind]
    else:
        kinds = list(map(int, rng.permutation([0]*counts[0]+[1]*counts[1])))
    cards = [dict(top=r, left=c, kind=k) for (r, c), k in zip(positions, kinds)]
    queues = _queues(cards)
    patches, masks = [None]*len(cards), [None]*len(cards)
    shared = _pattern(rng, n, palette, style) if witness else None
    stripe_width = int(rng.integers(1, min(2, n-1)+1))
    stripe_start = int(rng.integers(0, n-stripe_width+1))
    for rank, (h, s) in enumerate(zip(*queues)):
        base = shared.copy() if witness else _pattern(rng, n, palette, style)
        if witness:
            rr = np.minimum(np.arange(n), n-1-np.arange(n))
            for j in range(stripe_width):
                base[:, stripe_start+j] = np.asarray(palette)[(rr+rank+j) % len(palette)]
        patches[h], patches[s] = base, base.T.copy()
        if witness:
            target, source = (h, s) if witness_kind == 0 else (s, h)
            extent = int(rng.integers(1, n//2+1))
            if witness_kind == 0:
                masks[target] = (0, n, stripe_start, stripe_start+stripe_width)
                masks[source] = (stripe_start, stripe_start+stripe_width, 0, extent)
            else:
                masks[target] = (stripe_start, stripe_start+stripe_width, 0, n)
                masks[source] = (0, extent, stripe_start, stripe_start+stripe_width)
        else:
            mode = 2 if rank == 0 else int(rng.integers(4))
            if mode in (2, 3):
                kind = int(rng.integers(2))
                target, source = (h, s) if kind == 0 else (s, h)
                masks[target], masks[source] = _coupled_masks(rng, n, kind)
            else:
                masks[h] = _local_mask(rng, n, 0)
                if mode == 1:
                    masks[s] = _local_mask(rng, n, 1)
    for i, o in enumerate(cards):
        if patches[i] is None:
            base = _pattern(rng, n, palette, style)
            patches[i] = base if o['kind'] == 0 else base.T.copy()
            if rng.random() < .8:
                masks[i] = _local_mask(rng, n, o['kind'])
    full = np.full(shape, bg, dtype=np.int16)
    for o, p in zip(cards, patches):
        r, c = o['top'], o['left']
        full[r:r+3, c:c+3] = _badge(o['kind'], tag, bg)
        full[r:r+n, c+4:c+4+n] = p
    inp = full.copy()
    for o, box in zip(cards, masks):
        if box is not None:
            r, b, c, d = box
            inp[o['top']+r:o['top']+b, o['left']+4+c:o['left']+4+d] = mask
    return dict(input=inp, output=full)


def _make_pair(rng, n, counts, colors, style, seen, witness_kind=None):
    for _ in range(64):
        pair = _scene(rng, n, counts, colors, style, witness_kind)
        if _key(pair['input']) in seen:
            continue
        try:
            a = _analyze(pair['input'])
        except ValueError:
            continue
        if not np.array_equal(_render(a, a['restored']), pair['output']):
            continue
        if witness_kind is not None:
            e = _evidence(a)
            if not all(e[k] for k in ('local', 'horizontal', 'greedy', 'nearest', 'raw',
                                     'twins', 'ambiguous_twins', 'ties', 'composed')):
                continue
        return pair
    raise RuntimeError("bounded construction could not produce a valid distinct scene")


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    profile = int(rng.integers(2))  # 0: more ranks; 1: larger local coordinate chart
    vary_colors = bool(rng.random() < .8)
    shared_colors, style = _colors(rng), int(rng.integers(4))
    first_kind = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(num_train+num_test):
        train = index < num_train
        witness_kind = (first_kind if index == 0 else 1-first_kind) if index < 2 else None
        if index < 2:
            counts = (3, 2) if witness_kind == 0 else (2, 3)
        elif train:
            choices = [(h, s) for h in range(1, 5) for s in range(1, 5)
                       if profile == 1 or min(h, s) <= 2]
            counts = choices[int(rng.integers(len(choices)))]
        elif profile == 0:
            counts = None
        else:
            counts = [(2, 2), (2, 3), (3, 2), (3, 3)][int(rng.integers(4))]
        n = int(rng.integers(4, 7)) if train or profile == 0 else int(rng.integers(7, 9))
        if counts is None:
            choices = [(h, s) for h in range(3, 6) for s in range(3, 6)
                       if min(h, s) <= 4 and h+s <= _capacity(n)]
            counts = choices[int(rng.integers(len(choices)))]
        colors = _colors(rng) if vary_colors else shared_colors
        pair_style = int(rng.integers(4)) if rng.random() < .55 else style
        pair = _make_pair(rng, n, counts, colors, pair_style, seen, witness_kind)
        seen.add(_key(pair['input']))
        puzzle['train' if train else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Authored input-only invariants, exact labels and demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = _render(a, a['restored'])
            assert np.array_equal(expected, out), "wrong restoration"
            assert out.shape == inp.shape
            m, n = a['mask_color'], a['size']
            assert not np.any(out == m)
            assert np.array_equal(inp != out, inp == m), "changed a visible cell"
            for o, p in zip(a['cards'], a['restored']):
                assert _rectangle(o['patch'], m) is not False
                mate = p[::-1] if o['kind'] == 0 else p[:, ::-1]
                assert np.array_equal(p, mate), "class-dependent mirror conflict"
            for h, s in _pairing(a):
                assert np.array_equal(a['restored'][h], a['restored'][s].T)
            ids = _ids(a, _pairing(a))
            patches = np.stack([o['patch'] for o in a['cards']])
            visible_count = np.bincount(ids[patches != m].ravel(), minlength=int(ids.max())+1)
            assert np.all(visible_count[np.unique(ids)] > 0), "fully erased orbit"
            records[split].append((a, _evidence(a)))
    training = records['train']
    for concept in ('local', 'horizontal', 'greedy', 'nearest', 'reverse', 'raw', 'twins',
                    'ambiguous_twins', 'composed'):
        assert any(e[concept] for _, e in training), f"missing {concept} demonstration"
    differences = [len(a['queues'][0])-len(a['queues'][1]) for a, _ in training]
    assert min(differences) < 0 < max(differences), "both unmatched classes must be taught"
    if any(e['ties'] for _, e in records['test']):
        assert any(e['ties'] for _, e in training), "undemonstrated tie convention"


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded tests of 42 seeds, meaningful variation and negative controls."""
    signatures, shapes, normalized, sizes, roles, counts, layouts, mask_shapes = (
        set() for _ in range(8))
    profile_counts = [0, 0]
    pairs, composed, twin_pairs, tie_pairs = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        train_a = [_analyze(p['input']) for p in puzzle['train']]
        test_a = [_analyze(p['input']) for p in puzzle['test']]
        count_shift = min(min(map(len, a['queues'])) for a in test_a) > max(
            min(map(len, a['queues'])) for a in train_a)
        scale_shift = min(a['size'] for a in test_a) > max(a['size'] for a in train_a)
        assert count_shift or scale_shift, "missing structural test shift"
        profile_counts[int(scale_shift)] += 1
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                a, e = _analyze(pair['input']), _evidence(_analyze(pair['input']))
                pairs += 1
                composed += e['composed']
                twin_pairs += e['twins']
                tie_pairs += e['ties']
                shapes.add(a['grid'].shape)
                sizes.add(a['size'])
                roles.add((a['background'], a['badge_color'], a['mask_color']))
                counts.add(tuple(map(len, a['queues'])))
                layouts.add((a['size'], tuple((o['top'], o['left'], o['kind']) for o in a['cards'])))
                palette = {}
                normalized.add((a['grid'].shape, tuple(palette.setdefault(int(v), len(palette))
                                                      for v in a['grid'].ravel())))
                for o in a['cards']:
                    box = _rectangle(o['patch'], a['mask_color'])
                    if box is not None:
                        r, b, c, d = box
                        mask_shapes.add((b-r, d-c))
    assert len(signatures) == 42
    assert sizes == set(range(4, 9))
    assert len(shapes) >= 25 and len(normalized) >= 180 and len(layouts) >= 180
    assert len(counts) >= 15 and len(mask_shapes) >= 25
    assert all({r[i] for r in roles} == set(range(10)) for i in range(3))
    assert min(profile_counts) >= 10 and composed >= 126 and twin_pairs >= 84 and tie_pairs >= 84
    # Every legal explicit combination, including the two-demonstration minimum.
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(17, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # A color bijection preserves the semantic roles and input-only prediction.
    perm = np.random.default_rng(81723).permutation(10)
    recolored = {s: [{k: perm[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(recolored)
    # Orbit destruction is rejected even when the mask remains rectangular.
    a = _analyze(p['train'][0]['input'])
    destroyed = a['grid'].copy()
    h, s = _pairing(a)[0]
    n = a['size']
    for i in (h, s):
        o = a['cards'][i]
        destroyed[o['top']:o['top']+n, o['left']+4:o['left']+4+n] = a['mask_color']
    try:
        solve(destroyed)
    except ValueError:
        pass
    else:
        raise AssertionError("fully erased cross-card orbit accepted")
    # Solver support is semantic, not limited to the generator's sampled scale.
    rng = np.random.default_rng(9012)
    larger = _make_pair(rng, 10, (1, 1), _colors(rng), 0, set())
    assert np.array_equal(solve(larger['input']), larger['output'])
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), normalized_structures=len(normalized),
                layouts=len(layouts), sizes=sorted(sizes), count_patterns=len(counts),
                mask_shapes=len(mask_shapes), count_shift_puzzles=profile_counts[0],
                scale_shift_puzzles=profile_counts[1], composed_pairs=composed,
                twin_pairs=twin_pairs, tied_pairs=tie_pairs, legal_counts=legal_counts,
                independent_semantic_verification=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
