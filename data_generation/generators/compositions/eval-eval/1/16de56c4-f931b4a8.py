"""Woven enlargement: component-local sparse thread recipes over replicated maps.

No file I/O, parent imports, persistent state, or seed-dependent solving rules.
"""

import hashlib
import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= d <= 30 for d in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _components(source):
    """4-connected components; diagonal contact does not join two regions."""
    seen = np.zeros(source.shape, dtype=bool)
    result = []
    for r in range(source.shape[0]):
        for c in range(source.shape[1]):
            if seen[r, c]:
                continue
            ink = int(source[r, c])
            seen[r, c] = True
            todo, points = [(r, c)], []
            while todo:
                y, x = todo.pop()
                points.append((y, x))
                for a, b in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= a < source.shape[0] and 0 <= b < source.shape[1]
                            and not seen[a, b] and source[a, b] == ink):
                        seen[a, b] = True
                        todo.append((a, b))
            result.append((ink, points))
    return result


def _thread(line, bg):
    pos = np.flatnonzero(line != bg)
    if not len(pos):
        return None
    colors, counts = np.unique(line[pos], return_counts=True)
    repeat = colors[counts > 1]
    if len(repeat) != 1 or int(counts[counts > 1][0]) != 2:
        raise ValueError("thread must have exactly one equal-colored pair")
    ink = int(repeat[0])
    a, b = map(int, np.flatnonzero(line == ink))
    extras = [int(x) for x in pos if x not in (a, b)]
    terminals = [x for x in extras if (x-a) % (b-a) == 0]
    if len(terminals) > 1 or any(a < t < b for t in terminals):
        raise ValueError("ambiguous terminal")
    t = terminals[0] if terminals else None
    return dict(a=a, b=b, t=t, ink=ink,
                paint=ink if t is None else int(line[t]),
                extras=tuple((x, int(line[x])) for x in extras if x != t))


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    # A full-width/height uniform non-background fence determines presentation.
    candidates = []
    for transpose in (False, True):
        q = g.T if transpose else g
        for r in range(2, q.shape[0]-1):
            if int(q[r, 0]) != bg and np.all(q[r] == q[r, 0]):
                candidates.append((transpose, r))
    if len(candidates) != 1:
        raise ValueError("need one unambiguous separating fence")
    transposed, sep = candidates[0]
    q = g.T if transposed else g
    fence = int(q[sep, 0])
    m = sep-1
    # The source has no background holes; the right-hand padding does.
    occupied = np.flatnonzero(q[1, 1:] != bg)
    if not len(occupied):
        raise ValueError("empty carrier")
    n = int(occupied[-1])+1
    source = q[1:sep, 1:n+1].copy()
    if (np.any(source == bg) or np.any(source == fence)
            or not np.all(q[1:sep, n+1:] == bg)):
        raise ValueError("carrier must be a solid rectangle of material cells")
    if set(np.unique(q[0])) - {bg, fence} or set(np.unique(q[:sep, 0])) - {bg, fence}:
        raise ValueError("invalid scale rulers")
    xt = np.flatnonzero(q[0] == fence)
    yt = np.flatnonzero(q[:sep, 0] == fence)
    if len(xt) != 2 or len(yt) != 2 or xt[0] != 1 or yt[0] != 1:
        raise ValueError("each ruler needs two ticks starting at position one")
    sx, sy = int(xt[1]-xt[0]), int(yt[1]-yt[0])
    if sx >= n or sy >= m or max(m*sy, n*sx) > 30:
        raise ValueError("invalid enlargement")
    recipes, previous = {}, None
    for row in q[sep+1:]:
        label = int(row[0])
        if label not in set(map(int, np.unique(source))) or row[1] != fence:
            raise ValueError("invalid material key or recipe delimiter")
        if label != previous:
            if label in recipes:
                raise ValueError("a material's recipe rows must be consecutive")
            recipes[label] = []
        if np.any(row[2:] == fence):
            raise ValueError("fence cannot be thread ink")
        recipes[label].append(_thread(row[2:], bg))
        previous = label
    if set(recipes) != set(map(int, np.unique(source))):
        raise ValueError("every material needs exactly one recipe")
    if any(not any(s is not None for s in recipe) for recipe in recipes.values()):
        raise ValueError("a recipe needs an active thread")
    return dict(source=source, sy=sy, sx=sx, recipes=recipes, bg=bg,
                fence=fence, transposed=transposed, length=q.shape[1]-2)


def _context(d):
    source, sy, sx = d['source'], d['sy'], d['sx']
    base = np.repeat(np.repeat(source, sy, axis=0), sx, axis=1)
    parts = []
    for ink, points in _components(source):
        rr, cc = [], []
        for r, c in points:
            for y in range(sy):
                for x in range(sx):
                    rr.append(r*sy+y)
                    cc.append(c*sx+x)
        rr, cc = np.array(rr), np.array(cc)
        parts.append((ink, rr, cc, int(rr.min()), int(cc.min())))
    return dict(d, base=base, parts=parts)


def _render(ctx, variant=None):
    """Vectorized placement, with explicit alternative-rule probes for teaching."""
    if variant == 'pre_scale':
        small = _render(_context(dict(ctx, sy=1, sx=1)))
        return np.repeat(np.repeat(small, ctx['sy'], axis=0), ctx['sx'], axis=1)
    if variant == 'tile_carrier':
        h, w = ctx['base'].shape
        src = ctx['source']
        tiled = src[np.arange(h)[:, None] % src.shape[0],
                    np.arange(w)[None, :] % src.shape[1]]
        return _render(_context(dict(ctx, source=tiled, sy=1, sx=1)))
    out = ctx['base'].copy()
    if variant == 'ignore_threads':
        return out
    for material, rr, cc, top, left in ctx['parts']:
        if variant == 'global_origin':
            top = left = 0
        elif variant == 'color_origin':
            top = min(p[3] for p in ctx['parts'] if p[0] == material)
            left = min(p[4] for p in ctx['parts'] if p[0] == material)
        v, u = rr-top, cc-left
        recipe = ctx['recipes'][material]
        if variant == 'ignore_keys':
            recipe = next(iter(ctx['recipes'].values()))
        row_index = v % len(recipe)
        if variant == 'scaled_rows':
            row_index = (v // ctx['sy']) % len(recipe)
        if variant == 'row_phase':
            row_index = (v+1) % len(recipe)
        if variant == 'opaque_blanks':
            out[rr, cc] = ctx['bg']
        for j, s in enumerate(recipe):
            if s is None:
                continue
            a, b, t = s['a'], s['b'], s['t']
            paint = s['paint']
            if variant == 'no_terminal':
                t, paint = None, s['ink']
            if variant == 'no_recolor':
                paint = s['ink']
            if variant == 'off_phase_terminal':
                extras = list(s['extras'])
                if t is not None:
                    extras.append((t, s['paint']))
                outside = [(x, c) for x, c in extras if x < a or x > b]
                if outside:
                    t, paint = min(outside, key=lambda z: (min(abs(z[0]-a), abs(z[0]-b)), z[0]))
            phase = u % (b-a) == 0 if variant == 'zero_phase' else (u-a) % (b-a) == 0
            if variant == 'solid_intervals':
                phase = np.ones(u.shape, dtype=bool)
            if variant == 'raw_marks':
                # Only the sparse visible seeds, instead of a completed cadence.
                active = (u == a) | (u == b)
                if t is not None:
                    active |= u == t
            else:
                active = phase
                if t is not None and variant != 'unbounded_terminal':
                    active &= (u >= min(a, t)) & (u <= max(b, t))
            active &= row_index == j
            out[rr[active], cc[active]] = paint
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    answer = _render(_context(d))
    return answer.T.copy() if d['transposed'] else answer


_REQUIRED = frozenset(('raw_marks', 'no_terminal', 'no_recolor',
                       'unbounded_terminal', 'solid_intervals', 'zero_phase',
                       'off_phase_terminal', 'global_origin', 'color_origin',
                       'scaled_rows', 'pre_scale', 'tile_carrier', 'ignore_keys',
                       'opaque_blanks', 'ignore_threads', 'row_phase'))


def _evidence(d, output):
    ctx = _context(d)
    canonical = output.T if d['transposed'] else output
    return {name for name in _REQUIRED
            if not np.array_equal(_render(ctx, name), canonical)}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _carrier(rng, m, n, palette):
    k = int(rng.integers(2, len(palette)+1))
    inks = rng.choice(palette, k, replace=False)
    r, c = np.indices((m, n))
    style = int(rng.integers(6))
    if style == 0:  # Unequal rectangular partitions.
        rows = sorted(rng.choice(np.arange(1, m), int(rng.integers(1, m)), replace=False))
        cols = sorted(rng.choice(np.arange(1, n), int(rng.integers(1, n)), replace=False))
        colors = rng.integers(k, size=(len(rows)+1, len(cols)+1))
        labels = colors[np.searchsorted(rows, r, side='right'), np.searchsorted(cols, c, side='right')]
    elif style == 1:  # Islands, holes, and off-center rectangles.
        labels = np.zeros((m, n), dtype=int)
        for _ in range(int(rng.integers(2, 7))):
            y0, y1 = sorted(rng.choice(m+1, 2, replace=False))
            x0, x1 = sorted(rng.choice(n+1, 2, replace=False))
            labels[y0:y1, x0:x1] = int(rng.integers(k))
    elif style == 2:  # Broad diagonal or chevron bands.
        labels = ((r + int(_pick(rng, (-1, 1)))*c) // int(rng.integers(1, 4))) % k
    elif style == 3:  # Voronoi-like irregular material regions.
        centers = rng.choice(m*n, int(rng.integers(k, min(m*n, 2*k)+1)), replace=False)
        distance = abs(r[..., None]-centers//n) + abs(c[..., None]-centers % n)
        labels = rng.integers(k, size=len(centers))[np.argmin(distance, axis=2)]
    elif style == 4:  # Unequal stripes, sometimes interrupted.
        line = rng.integers(k, size=n)
        labels = np.tile(line, (m, 1))
        if rng.random() < .7:
            y = int(rng.integers(m))
            labels[y] = rng.integers(k, size=n)
    else:
        labels = rng.choice(k, size=(m, n), p=rng.dirichlet(np.full(k, 1.4)))
    if len(np.unique(labels)) == 1:
        labels[int(rng.integers(m)), int(rng.integers(n))] = (int(labels[0, 0])+1) % k
    if rng.random() < .5:
        labels = labels[::-1]
    if rng.random() < .5:
        labels = labels[:, ::-1]
    return inks[labels].astype(np.int8)


def _sample_thread(rng, length, inks, bg, blank=False):
    row = np.full(length, bg, dtype=np.int8)
    if blank:
        return row, None
    p = int(rng.integers(1, min(6, (length-1)//2)+1))
    # All placements leave room for at least three in-bounds phase positions.
    a = int(rng.integers(length-2*p))
    if rng.random() < .35:
        a += p
    b = a+p
    ink = int(_pick(rng, inks))
    row[a] = row[b] = ink
    available = [int(c) for c in inks if c != ink]
    rng.shuffle(available)
    options = [x for x in range(length) if (x-a) % p == 0 and (x < a or x > b)]
    if options and rng.random() < .58:
        t = int(_pick(rng, options))
        row[t] = available.pop()
    off = [x for x in range(length) if (x-a) % p != 0]
    if off and available and rng.random() < .45:
        row[int(_pick(rng, off))] = available.pop()
    return row, _thread(row, bg)


def _literal_output(source, sy, sx, recipes):
    """Constructor label path: coarse-region cells expanded one at a time.

    The solver instead parses the actual bank and uses array masks. This is
    consistency evidence by the same author, not an independent solver.
    """
    answer = np.repeat(np.repeat(source, sy, axis=0), sx, axis=1)
    for key, points in _components(source):
        top, left = min(r for r, _ in points), min(c for _, c in points)
        rows = recipes[key]
        for r, c in points:
            for y in range(sy):
                v = (r-top)*sy+y
                s = rows[v % len(rows)]
                if s is None:
                    continue
                for x in range(sx):
                    u = (c-left)*sx+x
                    if (u-s['a']) % (s['b']-s['a']):
                        continue
                    if s['t'] is not None and not min(s['a'], s['t']) <= u <= max(s['b'], s['t']):
                        continue
                    answer[r*sy+y, c*sx+x] = s['paint']
    return answer


def _sample_pair(rng, params, test=False):
    bg, fence, materials, inks = params
    if test:
        scales = (4, int(rng.integers(1, 5)))
        sy, sx = scales if rng.random() < .5 else scales[::-1]
    else:
        sy, sx = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    m = int(rng.integers(max(3, sy+1), 8))
    n = int(rng.integers(max(3, sx+1), 8))
    source = _carrier(rng, m, n, materials)
    length = int(rng.integers(max(8, n), 20))
    keys = list(map(int, np.unique(source)))
    rng.shuffle(keys)
    bank, recipes = [], {}
    for key in keys:
        k = int(rng.integers(1, 4))
        rows, recipe = [], []
        blanks = rng.random(k) < .18
        if np.all(blanks):
            blanks[int(rng.integers(k))] = False
        for j in range(k):
            line, spec = _sample_thread(rng, length, inks, bg, blank=bool(blanks[j]))
            rows.append(np.r_[key, fence, line])
            recipe.append(spec)
        bank.extend(rows)
        recipes[key] = recipe
    g = np.full((m+2+len(bank), length+2), bg, dtype=np.int8)
    g[1:m+1, 1:n+1] = source
    g[0, [1, 1+sx]] = fence
    g[[1, 1+sy], 0] = fence
    g[m+1] = fence
    g[m+2:] = bank
    out = _literal_output(source, sy, sx, recipes)
    transposed = bool(rng.integers(2))
    if transposed:
        g, out = g.T.copy(), out.T.copy()
    return {'input': g, 'output': out}


def _counts(a, b):
    for value, low, high in ((a, 2, 10), (b, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("illegal example count")
    if a <= b:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)]
    bg, fence = colors[:2]
    materials = colors[2:2+int(rng.integers(2, 5))]
    # Ink may equal a material key, including the material being painted.
    inks = colors[2:]
    params = bg, fence, materials, inks
    candidates, covers, decoded = [], [], []
    chosen = None
    # A finite pool selects demonstrations by observable consequences, without
    # fixing any pair's picture, orientation, period, or terminal schedule.
    for _ in range(128):
        pair = _sample_pair(rng, params)
        d = _decode(pair['input'])
        cover = _evidence(d, pair['output'])
        for j in rng.permutation(len(candidates)):
            j = int(j)
            other = decoded[j]
            if (_REQUIRED <= (cover | covers[j])
                    and d['transposed'] != other['transposed']
                    and (d['sy'], d['sx']) != (other['sy'], other['sx'])
                    and _key(pair['input']) != _key(candidates[j]['input'])):
                chosen = [candidates[j], pair]
                break
        if chosen is not None:
            break
        candidates.append(pair)
        covers.append(cover)
        decoded.append(d)
    if chosen is None:
        raise RuntimeError("bounded teaching pool did not distinguish the stated alternatives")
    puzzle = {'train': chosen, 'test': []}
    seen = {_key(p['input']) for p in chosen}
    for split, count in (('train', num_train), ('test', num_test)):
        while len(puzzle[split]) < count:
            for _ in range(64):
                pair = _sample_pair(rng, params, test=split == 'test')
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, conventions, evidence = set(), set(), set()
    train_axes, train_scales = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _decode(inp)
            conventions.add((d['bg'], d['fence']))
            canonical = out.T if d['transposed'] else out
            assert canonical.shape == (d['source'].shape[0]*d['sy'], d['source'].shape[1]*d['sx'])
            assert np.array_equal(solve(inp), out), 'incorrect local texture placement'
            assert np.array_equal(_literal_output(d['source'], d['sy'], d['sx'], d['recipes']), canonical)
            # Every cell with a transparent local stencil entry retains its
            # original material; no thread can leak to another component.
            ctx = _context(d)
            for key, rr, cc, top, left in ctx['parts']:
                recipe = d['recipes'][key]
                for r, c in zip(rr, cc):
                    s = recipe[(int(r)-top) % len(recipe)]
                    u = int(c)-left
                    active = (s is not None and (u-s['a']) % (s['b']-s['a']) == 0
                              and (s['t'] is None or min(s['a'], s['t']) <= u <= max(s['b'], s['t'])))
                    expected = s['paint'] if active else key
                    assert int(canonical[r, c]) == expected
            if split == 'train':
                evidence |= _evidence(d, out)
                train_axes.add(d['transposed'])
                train_scales.add((d['sy'], d['sx']))
                assert max(d['sy'], d['sx']) <= 3
            else:
                assert max(d['sy'], d['sx']) == 4, 'test must transfer to a new replication factor'
    assert len(conventions) == 1, 'palette conventions changed within a puzzle'
    assert train_axes == {False, True} and len(train_scales) >= 2
    assert _REQUIRED <= evidence, 'missing teaching evidence: ' + str(_REQUIRED-evidence)


def _normalized(g):
    roles, seq = {}, []
    for c in g.flat:
        c = int(c)
        seq.append(roles.setdefault(c, len(roles)))
    return g.shape, hashlib.sha256(bytes(seq)).hexdigest()


def self_test():
    """Stream 42 puzzles; retain only bounded measurement sets, never grids."""
    shapes, structures, sources, outputs, palettes, scales = set(), set(), set(), set(), set(), set()
    periods, stops, directions, heights, counts, modes = set(), set(), set(), set(), set(), set()
    reset_cases = blank_rows = off_phase = black_ink = 0
    digests = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                g, out = pair['input'], pair['output']
                digest.update(str(g.shape).encode())
                digest.update(g.tobytes())
                digest.update(out.tobytes())
                d = _decode(g)
                shapes.add(g.shape)
                outputs.add(out.shape)
                structures.add(_normalized(out))
                sources.add(_normalized(d['source']))
                palettes.add((d['bg'], d['fence']))
                scales.add((d['sy'], d['sx']))
                components = _components(d['source'])
                counts.add(len(components))
                reset_cases += len(components) > len(d['recipes'])
                black_ink += d['bg'] != 0 and (np.any(g == 0) or np.any(out == 0))
                for recipe in d['recipes'].values():
                    heights.add(len(recipe))
                    for s in recipe:
                        blank_rows += s is None
                        if s is None:
                            continue
                        periods.add(s['b']-s['a'])
                        modes.add(s['t'] is None)
                        off_phase += bool(s['extras'])
                        if s['t'] is not None:
                            stops.add(min(abs(s['t']-s['a']), abs(s['t']-s['b'])) // (s['b']-s['a']))
                            directions.add(-1 if s['t'] < s['a'] else 1)
        digests.add(digest.digest())
    assert len(digests) == 42
    assert len(shapes) >= 40 and len(outputs) >= 30
    assert len(structures) >= 150 and len(sources) >= 120
    assert len(palettes) >= 25 and len(scales) >= 12
    assert {1, 2, 3, 4, 5, 6} <= periods
    assert directions == {-1, 1} and max(stops) >= 6
    assert heights == {1, 2, 3} and modes == {False, True}
    assert max(counts) >= 12 and min(counts) <= 3
    assert reset_cases > 60 and blank_rows > 15 and off_phase > 60 and black_ink > 20
    # Hand-calculated anchor: the same material in separated source cells
    # restarts its phase, and a terminal recolors only the local bounded span.
    source = np.array([[1, 1, 2, 1, 1], [1, 1, 2, 1, 1]], dtype=np.int8)
    s = dict(a=1, b=3, t=5, ink=6, paint=7, extras=((0, 8),))
    t = dict(a=0, b=1, t=None, ink=4, paint=4, extras=())
    d = dict(source=source, sy=2, sx=3, recipes={1: [s, None], 2: [t]}, bg=0)
    expected = np.array([[1,7,1,7,1,7,4,4,4,1,7,1,7,1,7],
                         [1,1,1,1,1,1,4,4,4,1,1,1,1,1,1],
                         [1,7,1,7,1,7,4,4,4,1,7,1,7,1,7],
                         [1,1,1,1,1,1,4,4,4,1,1,1,1,1,1]], dtype=np.int8)
    assert np.array_equal(_render(_context(d)), expected)
    # Exercise compatible members outside the production sampler: four-row
    # recipes, periods seven/eight, scale five, a blank first recipe row, and
    # the maximum 30-by-30 output. The input parser sees no construction state.
    source = np.ones((6, 6), dtype=np.int8)
    source[:, 3:] = 2
    broad = np.zeros((16, 26), dtype=np.int8)
    broad[0, [1, 6]] = 9
    broad[[1, 6], 0] = 9
    broad[1:7, 1:7] = source
    broad[7] = 9
    for group, key in enumerate((1, 2)):
        bank = broad[8+group*4:12+group*4]
        bank[:, 0], bank[:, 1] = key, 9
        bank[1, [3, 10]], bank[1, 17] = 3, 4
        bank[2, [4, 12]] = 5
        bank[3, [9, 12]], bank[3, 3] = 6, 7
    bd = _decode(broad)
    expected = _literal_output(source, 5, 5, bd['recipes'])
    assert expected.shape == (30, 30)
    assert np.array_equal(solve(broad), expected)
    assert np.array_equal(solve(broad.T), expected.T)
    p = generate_puzzle(12345, 2, 1)
    q = generate_puzzle(12345, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for split in p for a, b in zip(p[split], q[split]) for k in a)
    # Demonstrate input-only equivariance for all ten palette roles and both
    # visible orientations, without regenerating any labels from a seed.
    pair = p['train'][0]
    perm = np.array([4, 0, 9, 5, 1, 8, 3, 6, 2, 7])
    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    assert np.array_equal(solve(pair['input'].T), pair['output'].T)
    damaged = {split: [dict(input=x['input'].copy(), output=x['output'].copy()) for x in p[split]] for split in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(7000+a*10+b, a, b))
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,False), (2.5,1), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, input_shapes=len(shapes), output_shapes=len(outputs),
                output_structures=len(structures), carrier_structures=len(sources),
                palette_conventions=len(palettes), scale_pairs=len(scales),
                periods=sorted(periods), recipe_heights=sorted(heights),
                component_count_range=[min(counts), max(counts)],
                local_reset_pairs=reset_cases, blank_recipe_rows=blank_rows,
                off_phase_threads=off_phase, black_payload_pairs=int(black_ink))
