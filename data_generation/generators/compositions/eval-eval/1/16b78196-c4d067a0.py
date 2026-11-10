"""Blueprint-registered jigsaw mosaics; all registration is recovered from pixels.

Only full-width, column-convex fragments are used. This is a deliberate finite
exact-cover family, not a solver for arbitrary polyomino tilings.
"""
from collections import Counter
from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 186827052, 221710613, 511079860, 567059491, 232613056,
              346842966, 66812342, 864642572, 665920339, 12744777, 747619920,
              96136211, 388722452, 537000276, 830363190, 570984134, 644167041,
              5139890, 480765376, 514985946, 537018651, 384216415, 88451832,
              1815119, 410893041, 736432305, 216955826, 154178229, 953033169,
              29457298, 918996068, 928632354, 9667854, 33506486, 537904001,
              272181965, 838543717, 490463252, 295212338, 266989077, 120707833)


def _counts(nt, ns):
    for name, n, low, high in (("num_train", nt, 2, 10), ("num_test", ns, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not low <= int(n) <= high:
            raise ValueError(name + " out of range")
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ns)


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, bg):
    unseen = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        color = int(a[start])
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen and a[q] == color:
                    unseen.remove(q)
                    stack.append(q)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _dilate(mask):
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    h, w = mask.shape
    for r in range(3):
        for c in range(3):
            result |= padded[r:r + h, c:c + w]
    return result


def _canonical(a, axis, flip):
    a = a if axis == 0 else a.T
    return a[::-1] if flip else a


def _restore(a, axis, flip):
    a = a[::-1] if flip else a
    return (a if axis == 0 else a.T).copy()


def _parse(inp):
    inp = _grid(inp)
    histogram = np.bincount(inp.ravel(), minlength=10)
    bg = int(histogram.argmax())
    assert histogram[bg] > np.partition(histogram, -2)[-2], "background tie"
    components = _components(inp, bg)
    bands = [(i, axis) for i, (cells, _) in enumerate(components)
             for axis in (0, 1)
             if len(np.unique(cells[:, 1 - axis])) == inp.shape[1 - axis]]
    assert len(bands) == 1, "one spanning band required"
    band_index, axis = bands[0]
    band = components[band_index][1]
    assert sum(c == band for _, c in components) == 1
    a = inp if axis == 0 else inp.T
    band_mask = a == band
    top = np.array([np.flatnonzero(band_mask[:, c])[0] for c in range(a.shape[1])])
    bottom = np.array([np.flatnonzero(band_mask[:, c])[-1] + 1 for c in range(a.shape[1])])
    assert (np.ptp(top) == 0) != (np.ptp(bottom) == 0), "one flat, one jagged edge"
    flip = bool(np.ptp(top))
    a = _canonical(inp, axis, flip).copy()
    mask = a == band
    top = np.array([np.flatnonzero(mask[:, c])[0] for c in range(a.shape[1])])
    bottom = np.array([np.flatnonzero(mask[:, c])[-1] + 1 for c in range(a.shape[1])])
    t = int(top[0])
    assert np.all(top == t)
    for c in range(a.shape[1]):
        assert np.all(mask[t:bottom[c], c])
    comps = _components(a, bg)
    points = np.array([cells[0] for cells, color in comps if len(cells) == 1])
    assert len(points) >= 4, "miniature needs isolated swatches"
    mr, mc = map(int, points.min(axis=0))
    er, ec = map(int, points.max(axis=0) + 1)
    assert np.all((points - (mr, mc)) % 2 == 0)
    plan = a[mr:er:2, mc:ec:2].copy()
    assert min(plan.shape) >= 2
    assert all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1))
    mini_mask = np.zeros_like(a, dtype=bool)
    mini_mask[mr:er, mc:ec] = True
    slots = np.zeros((er - mr, ec - mc), dtype=bool)
    slots[::2, ::2] = True
    assert np.all(a[mr:er, mc:ec][~slots] == bg)
    assert np.count_nonzero(plan != bg) == len(points)
    pieces = [(cells, color) for cells, color in comps if color != band and len(cells) > 1]
    assert pieces
    widths = {int(np.ptp(cells[:, 1]) + 1) for cells, _ in pieces}
    assert len(widths) == 1
    side = widths.pop()
    assert 3 <= side <= 6
    y = t + 2                 # The family has a visible two-cell solid spine.
    assert int(bottom.max()) == y + side
    profile = bottom - y
    sockets = {}
    c = 0
    while c < a.shape[1]:
        if profile[c] == side:
            c += 1
            continue
        end = c
        while end < a.shape[1] and profile[end] != side:
            end += 1
        assert end - c == side and np.all((profile[c:end] >= 1) & (profile[c:end] < side))
        assert np.ptp(profile[c:end]) > 0
        sockets[c] = tuple(map(int, profile[c:end]))
        c = end
    assert len(sockets) >= 2
    types, sources = Counter(), []
    forbidden = _dilate(mask | mini_mask)
    occupied = np.zeros_like(mask)
    for cells, color in pieces:
        assert not np.any(forbidden[cells[:, 0], cells[:, 1]])
        assert not np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]])
        occupied[cells[:, 0], cells[:, 1]] = True
        local = cells - cells.min(axis=0)
        lo, hi = [], []
        for x in range(side):
            rows = np.sort(local[local[:, 1] == x, 0])
            assert len(rows) and np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
            lo.append(int(rows[0]))
            hi.append(int(rows[-1]) + 1)
        assert max(hi) <= side
        signature = (color, tuple(lo), tuple(hi))
        types[signature] += 1
        sources.append((cells, signature))
    base = a.copy()
    base[occupied] = bg
    return dict(a=a, bg=bg, band=band, axis=axis, flip=flip, plan=plan,
                mini_mask=mini_mask, mini=(mr, mc, er, ec), side=side, y=y,
                sockets=sockets, types=types, sources=sources, cleared=base, mask=mask)


def _can_pack(demands, inventory, side):
    """Global exact cover over multiplicities of colored masks, not identities."""
    types = tuple(sorted(inventory))
    stock = tuple(inventory[k] for k in types)
    by_color = {}
    for i, (color, lo, hi) in enumerate(types):
        by_color.setdefault(color, []).append(i)
    cache = {}

    def chains(color, initial):
        key = color, initial
        if key in cache:
            return cache[key]
        found = set()

        def visit(current, use):
            if all(v == side for v in current):
                found.add(tuple(use))
                return
            offset = min(current)
            normalized = tuple(v - offset for v in current)
            for i in by_color.get(color, ()):
                if use[i] == stock[i] or types[i][1] != normalized:
                    continue
                nxt = tuple(offset + v for v in types[i][2])
                if max(nxt) > side:
                    continue
                use[i] += 1
                visit(nxt, use)
                use[i] -= 1
        visit(initial, [0] * len(types))
        cache[key] = tuple(sorted(found))
        return cache[key]

    options = [chains(color, initial) for color, initial in demands]
    if any(not o for o in options):
        return False
    failed = set()

    def cover(todo, remaining):
        if not todo:
            return not any(remaining)
        state = todo, remaining
        if state in failed:
            return False
        candidates = []
        for j in todo:
            feasible = [u for u in options[j] if all(x <= n for x, n in zip(u, remaining))]
            if not feasible:
                failed.add(state)
                return False
            candidates.append((len(feasible), j, feasible))
        _, j, feasible = min(candidates, key=lambda v: (v[0], v[1]))
        rest = tuple(k for k in todo if k != j)
        for usage in feasible:
            if cover(rest, tuple(n - x for n, x in zip(remaining, usage))):
                return True
        failed.add(state)
        return False
    return cover(tuple(range(len(demands))), stock)


def _layouts(info, geometry=True):
    """Enumerate registration first, then enforce colored-mask exact cover.

    geometry=False is an explicit ablation: it knows every color's total area,
    square scale, band and miniature, but forgets the fragment contours.
    """
    plan, side, y = info['plan'], info['side'], info['y']
    h, w = plan.shape
    H, W = info['a'].shape
    area = Counter()
    for cells, sig in info['sources']:
        area[sig[0]] += len(cells)
    answers = {}
    for row in range(h):
        filled = [c for c in range(w) if plan[row, c] != info['bg']]
        if len(filled) < 2:
            continue
        for gutter in range(2, 5):
            pitch = side + gutter
            br = y - row * pitch
            if br < 0 or br + (h - 1) * pitch + side > H:
                continue
            possible_x = sorted({x - filled[0] * pitch for x in info['sockets']})
            for bc in possible_x:
                if bc < 0 or bc + (w - 1) * pitch + side > W:
                    continue
                if any(bc + c * pitch not in info['sockets'] for c in filled):
                    continue
                er, ec = br + (h - 1) * pitch + side, bc + (w - 1) * pitch + side
                if np.any(info['mini_mask'][br:er, bc:ec]):
                    continue
                demands, wanted = [], Counter()
                sites, valid = [], True
                for (r, c), color in np.ndenumerate(plan):
                    if color == info['bg']:
                        continue
                    x, z = bc + c * pitch, br + r * pitch
                    structural = info['mask'][z:z + side, x:x + side]
                    if r == row:
                        initial = info['sockets'][x]
                    else:
                        if np.any(structural):
                            valid = False
                            break
                        initial = (0,) * side
                    color = int(color)
                    wanted[color] += side * side - sum(initial)
                    demands.append((color, initial))
                    sites.append((r, c, z, x, color, initial))
                if not valid or wanted != area:
                    continue
                if geometry and not _can_pack(demands, info['types'], side):
                    continue
                answer = info['cleared'].copy()
                for r, c, z, x, color, initial in sites:
                    region = answer[z:z + side, x:x + side]
                    assert np.all((region == info['bg']) | (region == info['band']))
                    region[~info['mask'][z:z + side, x:x + side]] = color
                key = answer.tobytes()
                answers.setdefault(key, (answer, dict(row=row, pitch=pitch, base=(br, bc), sites=sites)))
    return list(answers.values())


def _recover(inp):
    info = _parse(inp)
    layouts = _layouts(info)
    assert len(layouts) == 1, "require one distinct colored mosaic, not one piece permutation"
    canonical, layout = layouts[0]
    info.update(layout)
    return _restore(canonical, info['axis'], info['flip']), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone. Never reads files, seeds or stored examples."""
    return _recover(input_grid)[0]


def _cells(lo, hi):
    return np.array([(r, c) for c in range(len(lo)) for r in range(int(lo[c]), int(hi[c]))], dtype=int)


def _scatter(rng, base, mini_mask, pieces, bg):
    H, W = base.shape
    for _ in range(24):
        a = base.copy()
        forbidden = _dilate((base != bg) | mini_mask)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0]) + rng.random() * 5)
        for i in order:
            target, color = pieces[i]
            local = target - target.min(axis=0)
            ph, pw = local.max(axis=0) + 1
            valid = np.ones((H - ph + 1, W - pw + 1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            tr, tc = target.min(axis=0)
            if tr < valid.shape[0] and tc < valid.shape[1]:
                valid[tr, tc] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            origin = choices[int(rng.integers(len(choices)))]
            loc = local + origin
            a[loc[:, 0], loc[:, 1]] = color
            placed = np.zeros_like(forbidden)
            placed[loc[:, 0], loc[:, 1]] = True
            forbidden |= _dilate(placed)
        else:
            return a
    raise ValueError("scatter clearance")


def _make_pair(rng, axis, flip, test=False, demo=0):
    # Geometry is selected from feasible combinations, not assigned test offsets.
    options = []
    for h in range(2, 5):
        if demo or test:
            if h < 3:
                continue
        for side in range(3, 7):
            if demo == 1 and side != 3:
                continue
            if demo == 2 and side not in (4, 5):
                continue
            for w in range(2, min(4, side) + 1):
                for gutter in range(2, 5):
                    pitch = side + gutter
                    union_h, union_w = h * pitch + side, (w - 1) * pitch + side
                    for separation in (0, 1):
                        need_h = union_h + (2 * h + 1 if separation == 0 else 0)
                        need_w = union_w + (2 * w + 1 if separation == 1 else 0)
                        if max(need_h, need_w) <= 30:
                            options.append((h, w, side, pitch, need_h, need_w, separation))
    h, w, side, pitch, need_h, need_w, separation = options[int(rng.integers(len(options)))]
    H = int(rng.integers(max(18, need_h), 31))
    W = int(rng.integers(max(18, need_w), 31))
    colors = [int(x) for x in rng.permutation(10)]
    bg, band, palette = colors[0], colors[1], colors[2:]
    root = int(rng.integers(1, h - 1)) if test else int(rng.integers(h))
    alt = root + (1 if root == 0 else -1 if root == h - 1 else int(rng.choice([-1, 1])))
    sequence = np.array(rng.choice(palette[:min(w, 4)], size=w), dtype=np.int8)
    if len(set(map(int, sequence))) < 2:
        sequence[-1] = palette[1] if sequence[0] != palette[1] else palette[0]
    shuffled = sequence.copy()
    for _ in range(20):
        rng.shuffle(shuffled)
        if not np.array_equal(shuffled, sequence):
            break
    else:
        shuffled = np.roll(sequence, 1)
    plan = rng.choice(palette[:int(rng.integers(3, 7))], size=(h, w)).astype(np.int8)
    blanks = rng.random((h, w)) < (.30 if test or demo == 2 else .15)
    plan[blanks] = bg
    plan[root], plan[alt] = sequence, shuffled
    for r in (0, h - 1):
        if np.all(plan[r] == bg):
            plan[r, int(rng.integers(w))] = palette[2]
    # The second demonstration actually teaches an empty site. Its third row
    # also distinguishes literal transfer from color-order/reflection guesses.
    if demo == 2:
        other = next(r for r in range(h) if r not in (root, alt))
        plan[other, 0], plan[other, -1] = bg, palette[2]
    box_w = (w - 1) * pitch + side
    union_h = h * pitch + side
    mini_w, mini_h = 2 * w - 1, 2 * h - 1
    if separation == 0:
        margin = int(rng.integers(H - need_h + 1))
        if rng.integers(2):
            mr, union_top = margin, margin + mini_h + 2
        else:
            union_top, mr = margin, margin + union_h + 2
        bc = int(rng.integers(W - box_w + 1))
        mc = int(rng.integers(W - mini_w + 1))
    else:
        union_top = int(rng.integers(H - need_h + 1))
        margin = int(rng.integers(W - need_w + 1))
        if rng.integers(2):
            mc, bc = margin, margin + mini_w + 2
        else:
            bc, mc = margin, margin + box_w + 2
        y = union_top + max(root, alt) * pitch
        mini_rows = [r for r in range(H - mini_h + 1)
                     if r + mini_h < y - 2 or r > y + side]
        if not mini_rows:
            raise ValueError("miniature clearance")
        mr = mini_rows[int(rng.integers(len(mini_rows)))]
    y = union_top + max(root, alt) * pitch
    br = y - root * pitch
    base = np.full((H, W), bg, dtype=np.int8)
    base[y - 2:y + side, :] = band
    pattern_pool = list(combinations(range(side), side // 2))
    selected = rng.choice(len(pattern_pool), w, replace=False)
    profiles = []
    for c, k in enumerate(selected):
        p = np.ones(side, dtype=int)
        p[list(pattern_pool[int(k)])] = 2
        profiles.append(p)
        x = bc + c * pitch
        for dx in range(side):
            base[y + p[dx]:y + side, x + dx] = bg
    # Unused sockets are structural context. They are never automatic commands.
    occupied_x = set(x for c in range(w) for x in range(bc + c * pitch - 1, bc + c * pitch + side + 1))
    for _ in range(int(rng.integers(3))):
        choices = [x for x in range(W - side + 1)
                   if not set(range(x, x + side)) & occupied_x]
        if not choices:
            break
        x = choices[int(rng.integers(len(choices)))]
        p = np.ones(side, dtype=int)
        p[list(pattern_pool[int(rng.integers(len(pattern_pool)))])] = 2
        for dx in range(side):
            base[y + p[dx]:y + side, x + dx] = bg
        occupied_x.update(range(x - 1, x + side + 1))
    mini_mask = np.zeros_like(base, dtype=bool)
    mini_mask[mr:mr + mini_h, mc:mc + mini_w] = True
    base[mr:mr + mini_h:2, mc:mc + mini_w:2] = plan
    output = base.copy()
    pieces = []
    for (r, c), color in np.ndenumerate(plan):
        if color == bg:
            continue
        p = profiles[c] if r == root else np.zeros(side, dtype=int)
        cuts = [p]
        if r == root:
            if side >= 5 and rng.random() < .45:
                cuts.append(rng.integers(3, side, size=side))
        elif side == 6 and rng.random() < .25:
            cuts += [rng.integers(1, 3, size=side), rng.integers(3, 5, size=side)]
        elif rng.random() < .75:
            cuts.append(rng.integers(1, side, size=side))
        cuts.append(np.full(side, side))
        origin = np.array([br + r * pitch, bc + c * pitch])
        for low, high in zip(cuts, cuts[1:]):
            cells = _cells(low, high) + origin
            local_mask = np.zeros((side, side), dtype=np.int8)
            local = cells - origin
            local_mask[local[:, 0], local[:, 1]] = 1
            assert len(_components(local_mask, 0)) == 1
            pieces.append((cells, int(color)))
            output[cells[:, 0], cells[:, 1]] = color
    inp = _scatter(rng, base, mini_mask, pieces, bg)
    if rng.random() < .5:
        inp, output = inp[:, ::-1], output[:, ::-1]
    pair = {"input": _restore(inp, axis, flip), "output": _restore(output, axis, flip)}
    expected, info = _recover(pair['input'])
    if not np.array_equal(expected, pair['output']):
        raise ValueError("construction/search disagreement")
    if len(_layouts(info, geometry=False)) < 2:
        raise ValueError("shape-free registration must have competing answers")
    return pair


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, metas = set(), {"train": [], "test": []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "incorrect mosaic or changed unaffected cell"
            assert np.array_equal(inp == info['band'], out == info['band'])
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10)), "color mass changed"
            canonical = _canonical(out, info['axis'], info['flip'])
            assert np.array_equal(canonical[info['mini_mask']], info['a'][info['mini_mask']])
            for r, c, z, x, color, initial in info['sites']:
                region = canonical[z:z + info['side'], x:x + info['side']]
                structural = info['mask'][z:z + info['side'], x:x + info['side']]
                assert np.all(region[~structural] == color)
                assert np.all(region[structural] == info['band'])
            assert len(_layouts(info, geometry=False)) >= 2, "area shortcut not distinguished"
            metas[split].append(info)
    assert {m['axis'] for m in metas['test']} <= {m['axis'] for m in metas['train']}
    assert len({m['side'] for m in metas['train']}) >= 2
    assert any(np.any(m['plan'] == m['bg']) for m in metas['train'])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))        # horizontal, vertical, or demonstrated mixed
    axes = [mode] if mode < 2 else [0, 1]
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            axis = train_axes[i] if split == 'train' else int(rng.choice(axes))
            flip = bool(i % 2) if split == 'train' and i < 2 else bool(rng.integers(2))
            demo = i + 1 if split == 'train' and i < 2 else 0
            for _ in range(100):
                try:
                    pair = _make_pair(rng, axis, flip, test=split == 'test', demo=demo)
                except (ValueError, AssertionError):
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded host-seed audit, including counterfactuals and negative checks."""
    features, normalized, signatures, miniature_sides = [], set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train'] + puzzle['test']))
        for pair in puzzle['train'] + puzzle['test']:
            expected, m = _recover(pair['input'])
            assert np.array_equal(expected, pair['output'])
            assert len(_layouts(m, geometry=False)) >= 2
            features.append((pair['input'].shape, m['axis'], m['flip'], m['plan'].shape,
                             m['side'], m['pitch'], m['row'], len(m['sources']),
                             int(np.count_nonzero(m['plan'] == m['bg'])), len(m['sockets'])))
            br, bc = m['base']
            er = br + (m['plan'].shape[0] - 1) * m['pitch'] + m['side']
            ec = bc + (m['plan'].shape[1] - 1) * m['pitch'] + m['side']
            mr, mc, me, ne = m['mini']
            miniature_sides.update(name for name, present in
                                   (('above', me <= br), ('below', mr >= er),
                                    ('left', ne <= bc), ('right', mc >= ec)) if present)
            palette, norm = {}, []
            for v in pair['input'].flat:
                norm.append(palette.setdefault(int(v), len(palette)))
            normalized.add((pair['input'].shape, tuple(norm)))
    assert len(signatures) == 42 and len(normalized) >= 200
    for index, minimum in ((0, 35), (1, 2), (2, 2), (3, 8), (4, 4), (5, 6),
                           (6, 3), (7, 8), (8, 3), (9, 3)):
        assert len({f[index] for f in features}) >= minimum, (index, minimum)
    assert miniature_sides == {'above', 'below', 'left', 'right'}
    assert any(f[3][1] == 4 for f in features)
    p = generate_puzzle(0, 2, 1)
    replay = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], replay[s])
               for k in ('input', 'output'))
    # B ablation: rearrange two differently colored, unanchored swatches. Area
    # and the root row stay fixed. All unanchored sites are full squares, so
    # their colored fragment chains can travel to the changed locations.
    witnessed = False
    for seed in HOST_SEEDS[:8]:
        for pair in generate_puzzle(seed)['train']:
            _, m = _recover(pair['input'])
            row = m['row']
            points = [(r, c) for r in range(m['plan'].shape[0]) if r != row
                      for c in range(m['plan'].shape[1]) if m['plan'][r, c] != m['bg']]
            for first, second in combinations(points, 2):
                if m['plan'][first] == m['plan'][second]:
                    continue
                changed = m['a'].copy()
                mr, mc, _, _ = m['mini']
                a = (mr + 2 * first[0], mc + 2 * first[1])
                b = (mr + 2 * second[0], mc + 2 * second[1])
                changed[a], changed[b] = changed[b], changed[a]
                changed = _restore(changed, m['axis'], m['flip'])
                try:
                    altered = solve(changed)
                except AssertionError:
                    continue
                assert not np.array_equal(altered, pair['output'])
                witnessed = True
                break
            if witnessed:
                break
        if witnessed:
            break
    assert witnessed, "blueprint must independently change a valid answer"
    # Duplicate the geometrically recovered color sequence on another full
    # row with the same color totals. Two origins now fit; no tie-break is legal.
    _, m = _recover(p['train'][0]['input'])
    root_row = m['plan'][m['row']]
    alternatives = [r for r in range(m['plan'].shape[0]) if r != m['row']
                    and Counter(map(int, m['plan'][r])) == Counter(map(int, root_row))]
    assert alternatives
    changed = m['a'].copy()
    mr, mc, _, _ = m['mini']
    changed[mr + 2 * alternatives[0], mc:mc + 2 * len(root_row):2] = root_row
    changed = _restore(changed, m['axis'], m['flip'])
    assert len(_layouts(_parse(changed))) >= 2
    try:
        solve(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous miniature accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == '__main__':
    self_test()
