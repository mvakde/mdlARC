"""Jointly recover two damaged views of a motif, then stamp it into itself.

Only standard-library modules and NumPy are used. Generation writes no files.
"""
from hashlib import sha256
import numpy as np


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _orbits(h, w):
    return [tuple(sorted({(r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c)}))
            for r in range((h+1)//2) for c in range((w+1)//2)]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, part = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    part.append(q)
        result.append(part)
    return result


def _tight_connected(mask):
    return (mask[0].any() and mask[-1].any() and mask[:, 0].any()
            and mask[:, -1].any() and len(_components(mask)) == 1)


def _motif(rng, h, w, force_loose=False):
    """Mix connected tight silhouettes with free masks inside damaged cards."""
    orbits = _orbits(h, w)
    four = [o for o in orbits if len(o) == 4]
    if force_loose or rng.random() < 0.55:
        # Erased corners visibly delimit a card even when its ink does not.
        # Disconnected masks and empty outer bands remain legitimate payload.
        corners = four[0]
        mask = np.zeros((h, w), dtype=bool)
        density = float(rng.uniform(0.35, 0.85))
        band_options = ([0] if (h-2)*w >= 8 else []) + ([1] if h*(w-2) >= 8 else [])
        band_axis = _pick(rng, band_options) if force_loose else None
        allowed = [o for o in orbits if o != corners and
                   (band_axis is None or all(p[band_axis] not in (0, (h, w)[band_axis]-1) for p in o))]
        for orbit in allowed:
            if rng.random() < density:
                for p in orbit:
                    mask[p] = True
        for p in _pick(rng, [o for o in allowed if len(o) == 4]):
            mask[p] = True
        for index in rng.permutation(len(orbits)):
            if mask.sum() >= 8:
                break
            orbit = orbits[int(index)]
            if orbit in allowed:
                for p in orbit:
                    mask[p] = True
        return mask
    mask = np.ones((h, w), dtype=bool)
    candidates = []
    for orbit in four:
        trial = mask.copy()
        for p in orbit:
            trial[p] = False
        if trial.sum() >= 8 and _tight_connected(trial):
            candidates.append(orbit)
    # Dimensions 3..5, excluding 3x3, always admit corner-orbit removal.
    first_blank = _pick(rng, candidates)
    for p in first_blank:
        mask[p] = False
    target = max(8, int(round(h*w*float(rng.uniform(0.35, 0.92)))))
    for index in rng.permutation(len(orbits)):
        if int(mask.sum()) <= target:
            break
        orbit = orbits[int(index)]
        trial = mask.copy()
        for p in orbit:
            trial[p] = False
        if (trial.sum() >= 8 and _tight_connected(trial)
                and any(all(trial[p] for p in o) for o in four)):
            mask = trial
    return mask


def _raw_damage(rng, h, w):
    style = int(rng.integers(4))
    if style == 0:
        return rng.random((h, w)) < float(rng.uniform(0.04, 0.35))
    mask = np.zeros((h, w), dtype=bool)
    if style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        rh, cw = int(rng.integers(1, h+1)), int(rng.integers(1, w+1))
        mask[r:min(h, r+rh), c:min(w, c+cw)] = True
    elif style == 2:
        if rng.integers(2):
            mask[int(rng.integers(h)), :] = True
        else:
            mask[:, int(rng.integers(w))] = True
    else:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(2, h+w+1))):
            mask[r, c] = True
            dr, dc = _pick(rng, ((-1, 0), (1, 0), (0, -1), (0, 1)))
            r, c = min(h-1, max(0, r+dr)), min(w-1, max(0, c+dc))
    return mask


def _damage(rng, motif, anchor, force_fallback=False):
    """Each view loses a whole different orbit; their joint evidence suffices."""
    h, w = motif.shape
    orbits = _orbits(h, w)
    occupied = [o for o in orbits if len(o) == 4 and motif[o[0]]]
    vacant = [o for o in orbits if len(o) == 4 and not motif[o[0]]]
    x = _pick(rng, occupied)
    y = _pick(rng, vacant) if _tight_connected(motif) else orbits[0]
    donor = anchor if anchor in x else _pick(rng, x)
    zeros_known = list(y)
    rng.shuffle(zeros_known)
    zeros_known = zeros_known[:int(rng.integers(1, 4))]
    if not _tight_connected(motif):
        zeros_known = zeros_known[:1]
    ink_known = [tuple(p) for p in np.argwhere(motif) if tuple(p) not in x]
    rng.shuffle(ink_known)
    ink_known = ink_known[:3]
    map_ink_known = _pick(rng, [tuple(p) for p in np.argwhere(motif)
                              if tuple(p) not in x and tuple(p) != anchor])
    for attempt in (range(64, 65) if force_fallback else range(65)):
        if attempt < 64:
            a, b = _raw_damage(rng, h, w), _raw_damage(rng, h, w)
        else:
            # Finite connected-support fallback: source is a nearly full card.
            a, b = np.ones((h, w), dtype=bool), np.ones((h, w), dtype=bool)
            zeros_known = [zeros_known[0]]
        for p in x:
            a[p], b[p] = True, p != donor
        for p in y:
            a[p], b[p] = p not in zeros_known, True
        for p in ink_known:
            a[p] = False
        b[anchor] = False
        b[map_ink_known] = False
        for orbit in orbits:
            if all(a[p] and b[p] for p in orbit):
                p = _pick(rng, orbit)
                if attempt < 64 and rng.integers(2):
                    a[p] = False
                else:
                    b[p] = False
        map_support = motif | b
        if (_tight_connected(motif | a) and map_support[0].any()
                and map_support[-1].any() and map_support[:, 0].any()
                and map_support[:, -1].any()):
            return a, b
    raise AssertionError('unreachable damage fallback')


def _place(rng, motif, damage, anchor, colors):
    h, w = motif.shape
    ah, aw, ph, pw = h*h, w*w, 2*h-1, 2*w-1
    tr, tc = anchor[0]*h, anchor[1]*w
    rr, cc = np.meshgrid(np.arange(-ph-1, ah+2), np.arange(-pw-1, aw+2), indexing='ij')
    lowr, lowc = np.minimum(0, rr), np.minimum(0, cc)
    highr, highc = np.maximum(ah, rr+ph), np.maximum(aw, cc+pw)
    valid = ((rr+ph < tr) | (tr+h < rr) | (cc+pw < tc) | (tc+w < cc))
    valid &= (highr-lowr <= 30) & (highc-lowc <= 30)
    positions = np.argwhere(valid)
    if rng.random() < 0.4:
        areas = ((highr-lowr)*(highc-lowc))[valid]
        positions = positions[areas <= np.quantile(areas, 0.25)]
    i, j = _pick(rng, positions)
    pr, pc = int(rr[i, j]), int(cc[i, j])
    base_h, base_w = int(highr[i, j]-lowr[i, j]), int(highc[i, j]-lowc[i, j])
    pad = int(rng.integers(5))
    gh = base_h if pad == 0 else 30 if pad == 1 else int(rng.integers(base_h, 31))
    gw = base_w if pad == 0 else 30 if pad == 1 else int(rng.integers(base_w, 31))
    top = int(rng.integers(gh-base_h+1))-int(lowr[i, j])
    left = int(rng.integers(gw-base_w+1))-int(lowc[i, j])
    bg, ink, marker, erase = colors
    a, b = damage
    inp = np.full((gh, gw), bg, dtype=np.int8)
    card = np.where(motif, ink, bg).astype(np.int8)
    card[a] = erase
    inp[top+tr:top+tr+h, left+tc:left+tc+w] = card
    diagram = np.where(motif, marker, bg).astype(np.int8)
    diagram[b] = erase
    diagram[anchor] = ink
    inp[top+pr:top+pr+ph:2, left+pc:left+pc+pw:2] = diagram
    # Generator construction uses a Kronecker product; solve paints copies.
    out = np.full_like(inp, bg)
    stamp = np.full((ah, aw), bg, dtype=np.int8)
    stamp[np.kron(motif, motif)] = ink
    stamp[tr:tr+h, tc:tc+w][motif] = marker
    out[top:top+ah, left:left+aw] = stamp
    return {'input': inp, 'output': out}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 integer ARC grid with colors 0..9')
    return a


def _decode(input_grid):
    """Identify all roles and the two views entirely from the input pixels."""
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('expected four distinct roles and a dominant background')
    bg = int(values[np.argmax(counts)])
    parts = _components(inp != bg)
    cards = [part for part in parts if len(part) > 1]
    if len(cards) != 1:
        raise ValueError('expected one connected dense card and isolated map cells')
    coords = np.asarray(cards[0])
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    h, w = br-tr+1, bc-tc+1
    if not (3 <= h <= 5 and 3 <= w <= 5 and (h, w) != (3, 3)):
        raise ValueError('unsupported motif dimensions')
    card = inp[tr:br+1, tc:bc+1]
    inside_colors = [int(c) for c in np.unique(card) if c != bg]
    outside = inp.copy()
    outside[tr:br+1, tc:bc+1] = bg
    ink_candidates = [c for c in inside_colors if np.count_nonzero(outside == c) == 1]
    if len(inside_colors) != 2 or len(ink_candidates) != 1:
        raise ValueError('card/anchor color correspondence is ambiguous')
    ink = ink_candidates[0]
    erase = next(c for c in inside_colors if c != ink)
    markers = [int(c) for c in values if c not in (bg, ink, erase)]
    marker = markers[0]
    if np.count_nonzero(outside == erase) < 3 or np.count_nonzero(card == ink) < 3:
        raise ValueError('insufficient visible roles')
    points = np.argwhere(outside != bg)
    pr, pc = map(int, points.min(axis=0))
    er, ec = map(int, points.max(axis=0))
    if (er-pr, ec-pc) != (2*h-2, 2*w-2):
        raise ValueError('map and card must have the same logical dimensions')
    if not (er+1 < tr or br+1 < pr or ec+1 < tc or bc+1 < pc):
        raise ValueError('card and map need a separating background band')
    if any((r-pr) % 2 or (c-pc) % 2 for r, c in points):
        raise ValueError('map cells must lie on a pitch-two lattice')
    diagram = inp[pr:er+1:2, pc:ec+1:2]
    if np.count_nonzero(diagram == ink) != 1:
        raise ValueError('expected one exceptional anchor')
    anchor = tuple(map(int, np.argwhere(diagram == ink)[0]))
    source = np.full((h, w), -1, dtype=np.int8)
    source[card == bg], source[card == ink] = 0, 1
    coarse = np.full((h, w), -1, dtype=np.int8)
    coarse[diagram == bg] = 0
    coarse[(diagram == marker) | (diagram == ink)] = 1
    expected = np.full_like(inp, bg)
    expected[tr:br+1, tc:bc+1] = card
    expected[pr:er+1:2, pc:ec+1:2] = diagram
    if not np.array_equal(expected, inp):
        raise ValueError('unrelated foreground or malformed lattice')
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    if not (0 <= top <= inp.shape[0]-h*h and 0 <= left <= inp.shape[1]-w*w):
        raise ValueError('anchor-relative expansion would leave the canvas')
    return {'roles': (bg, ink, marker, erase), 'source': source, 'coarse': coarse,
            'anchor': anchor, 'template_top': tr, 'template_left': tc,
            'map_top': pr, 'map_left': pc, 'top': top, 'left': left}


def _recover(meta, reflect=True):
    a, b = meta['source'], meta['coarse']
    h, w = a.shape
    orbits = _orbits(h, w) if reflect else [((r, c),) for r in range(h) for c in range(w)]
    result = np.zeros((h, w), dtype=bool)
    for orbit in orbits:
        witnesses = {int(view[p]) for view in (a, b) for p in orbit if view[p] >= 0}
        if len(witnesses) != 1:
            if not reflect and not witnesses:
                continue  # Explicit ablation: unknown coincident cells become blank.
            raise ValueError('missing or contradictory joint reflection witnesses')
        for p in orbit:
            result[p] = next(iter(witnesses)) == 1
    return result


def _paint(shape, meta, motif):
    bg, ink, marker, _ = meta['roles']
    h, w = motif.shape
    tr, tc = meta['template_top'], meta['template_left']
    ar, ac = meta['anchor']
    out = np.full(shape, bg, dtype=np.int8)
    for r, c in np.argwhere(motif):
        rr, cc = tr+(int(r)-ar)*h, tc+(int(c)-ac)*w
        out[rr:rr+h, cc:cc+w][motif] = marker if (r, c) == (ar, ac) else ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No examples, seed, output label, cached motif, or generator state needed."""
    inp = _grid(input_grid)
    meta = _decode(inp)
    motif = _recover(meta)
    if not motif[meta['anchor']]:
        raise ValueError('invalid recovered stencil')
    return _paint(inp.shape, meta, motif)


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name+' is out of range')
    return int(value)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    shift_axis = int(rng.integers(2))
    fixed_palette = bool(rng.random() < 0.2)
    base_colors = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_dims = previous_anchor = first_colors = None
    box_demonstrated = False
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            options = [(h, w) for h in range(3, 6) for w in range(3, 6)
                       if (h, w) != (3, 3)
                       and ((h, w)[shift_axis] <= 4 if split == 'train' else (h, w)[shift_axis] == 5)
                       and (split != 'train' or index != 0 or h != w)
                       and (split != 'train' or index != 1 or
                            (h != previous_dims[0] and w != previous_dims[1]))]
            if split == 'train' and index == 0:
                options = [(h, w) for h, w in options if min(h, w) >= 4 or
                           any(a != h and b != w and min(a, b) >= 4
                               and (a, b)[shift_axis] <= 4 for a in range(3, 6) for b in range(3, 6))]
            if split == 'train' and index == 1 and not box_demonstrated:
                options = [d for d in options if min(d) >= 4]
            dims = _pick(rng, options)
            for _ in range(32):
                box_demo = split == 'train' and index < 2 and not box_demonstrated and min(dims) >= 4
                motif = _motif(rng, *dims, force_loose=box_demo)
                anchors = [tuple(map(int, p)) for p in np.argwhere(motif)
                           if (split != 'train' or index != 0 or
                               tuple(p) != ((dims[0]-1)//2, (dims[1]-1)//2))
                           and (split != 'train' or index != 1 or
                                (p[0] != previous_anchor[0] and p[1] != previous_anchor[1]))]
                if not anchors:
                    continue
                anchor = _pick(rng, anchors)
                colors = base_colors if fixed_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                if split == 'train' and index == 1 and not fixed_palette and colors == first_colors:
                    colors = colors[1:]+colors[:1]
                pair = _place(rng, motif, _damage(rng, motif, anchor), anchor, colors)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-pair construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and box_demo:
                box_demonstrated = True
            if split == 'train' and index == 0:
                previous_dims, previous_anchor, first_colors = dims, anchor, colors
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(inp, out):
    meta = _decode(inp)
    motif = _recover(meta)
    h, w = motif.shape
    a, b = meta['source'], meta['coarse']
    orbits = _orbits(h, w)
    assert motif.sum() >= 8
    assert motif[meta['anchor']]
    assert all(motif[p] == motif[o[0]] for o in orbits for p in o)
    assert any(all(a[p] == -1 for p in o) for o in orbits), 'source independently repairable'
    assert any(all(b[p] == -1 for p in o) for o in orbits), 'map independently repairable'
    sole_occupied = [o for o in orbits if len(o) == 4 and motif[o[0]] and
                     len({p for p in o if a[p] >= 0 or b[p] >= 0}) == 1]
    assert sole_occupied, 'both folds and joint witnesses must matter'
    common_unknown = (a == -1) & (b == -1)
    assert np.any(common_unknown & motif) and np.any(common_unknown & ~motif)
    assert np.any((a == -1) & ~motif) and np.any((b == -1) & motif)
    assert np.array_equal(_paint(inp.shape, meta, motif), out)
    bg, ink, marker, erase = meta['roles']
    n = int(motif.sum())
    assert np.count_nonzero(out != bg) == n*n
    assert np.count_nonzero(out == marker) == n
    assert np.count_nonzero(out == ink) == n*(n-1)
    assert not np.any(out == erase)
    tr, tc = meta['template_top'], meta['template_left']
    assert np.array_equal(out[tr:tr+h, tc:tc+w] == marker, motif)
    without_a = _paint(inp.shape, meta, _recover(meta, reflect=False))
    assert not np.array_equal(without_a, out), 'reflection ablation did not change answer'
    without_b = np.full_like(inp, bg)
    without_b[tr:tr+h, tc:tc+w][motif] = marker
    assert not np.array_equal(without_b, out), 'substitution ablation did not change answer'
    meta['motif'] = motif
    meta['population'] = n
    meta['common_unknown'] = int(common_unknown.sum())
    return meta


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, groups = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            groups[split].append(_analyze(inp, out))
    train, test = groups['train'], groups['test']
    assert len({m['anchor'][0] for m in train}) >= 2
    assert len({m['anchor'][1] for m in train}) >= 2
    assert len({m['motif'].shape[0] for m in train}) >= 2
    assert len({m['motif'].shape[1] for m in train}) >= 2
    assert any(m['motif'].shape[0] != m['motif'].shape[1] for m in train)
    assert any(not (m['motif'][0].any() and m['motif'][-1].any()
                   and m['motif'][:, 0].any() and m['motif'][:, -1].any()) for m in train)
    assert any(max(m['motif'].shape[axis] for m in train) <
               min(m['motif'].shape[axis] for m in test) for axis in (0, 1))


def _signature(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed study; no files or unbounded retained payloads."""
    signatures, shapes, motifs, anchors, palettes = set(), set(), set(), set(), set()
    damage_shapes, populations, roles, overlaps = set(), set(), [set() for _ in range(4)], 0
    loose = disconnected = 0
    axes = set()
    fallback_rng = np.random.default_rng(991)
    for dims in ((3, 4), (4, 4), (5, 5)):
        mask = _motif(fallback_rng, *dims)
        anchor = tuple(map(int, np.argwhere(mask)[0]))
        pair = _place(fallback_rng, mask, _damage(fallback_rng, mask, anchor, force_fallback=True),
                      anchor, (0, 1, 2, 3))
        _analyze(pair['input'], pair['output'])
    # Semantic members beyond the sampler's population/shape preferences.
    # These use four ink pixels, including a disconnected corner-only stencil.
    for kind in (0, 1):
        mask = np.zeros((3, 4) if kind == 0 else (4, 4), dtype=bool)
        a, b = np.ones_like(mask), np.zeros_like(mask)
        if kind == 0:
            mask[1, :] = True
            a[1, :], a[0, 0] = False, False
            b[0, :], b[-1, :] = True, True
            b[0, 1] = False
            anchor = (1, 1)
        else:
            mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
            a[0, 0] = a[0, -1] = a[-1, 0] = a[1, 1] = False
            b[1:3, 1:3] = True
            anchor = (3, 3)
        pair = _place(fallback_rng, mask, (a, b), anchor, (0, 1, 2, 3))
        assert np.array_equal(solve(pair['input']), pair['output'])
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_shapes = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, before), 'solver mutated its input'
                m = _analyze(inp, out)
                mask = m['motif']
                shapes.add(inp.shape)
                motifs.add(_key(mask))
                anchors.add((mask.shape, m['anchor']))
                palettes.add(m['roles'])
                populations.add(m['population'])
                loose += not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())
                disconnected += len(_components(mask)) > 1
                for values, color in zip(roles, m['roles']):
                    values.add(color)
                damage_shapes.add((mask.shape, (m['source'] == -1).tobytes(),
                                   (m['coarse'] == -1).tobytes()))
                if split == 'train':
                    train_shapes.append(mask.shape)
                else:
                    axes.update(axis for axis in (0, 1) if max(s[axis] for s in train_shapes) < mask.shape[axis])
                pr, pc = m['map_top'], m['map_left']
                diagram_output = out[pr:pr+2*mask.shape[0]-1:2, pc:pc+2*mask.shape[1]-1:2]
                overlaps += bool(np.any(diagram_output != m['roles'][0]))
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(motifs) >= 20
    assert len(anchors) >= 45 and len(palettes) >= 80
    assert len(damage_shapes) >= 160 and len(populations) >= 6
    assert all(values == set(range(10)) for values in roles)
    assert axes == {0, 1} and overlaps >= 10
    assert loose >= 42 and disconnected >= 3
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9000+nt*10+ns, nt, ns)
            assert (len(p['train']), len(p['test'])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A single bad answer and contradictory visible evidence must be rejected.
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    m = _decode(p['test'][0]['input'])
    orbit = next(o for o in _orbits(*m['source'].shape) if len(o) == 4)
    m['source'][orbit[0]], m['coarse'][orbit[-1]] = 0, 1
    try:
        _recover(m)
    except ValueError:
        pass
    else:
        raise AssertionError('contradictory joint witnesses accepted')
