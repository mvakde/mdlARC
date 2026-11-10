"""Keyed carriers: rank matches cargo; complementary contours route carriers.

Only the standard library and NumPy are used. No parent code or data is loaded.
See task_spec.md for the complete, fixed family conventions.
"""

import copy
import json

import numpy as np


HOST_SEEDS = (0, 714885587, 196763261, 336152810, 725553339, 551726193,
              280002401, 117666152, 355827385, 76912287, 105490484,
              761502353, 991227103, 768592193, 52455614, 274978818,
              875872476, 514480394, 677992343, 620249404, 867226462,
              916021795, 302731918, 152750213, 122205636, 123697266,
              492484060, 178255735, 471735337, 381932671, 527585653,
              566987293, 915450184, 388245076, 537991692, 248971593,
              923717724, 866831504, 61931112, 32252868, 262890580,
              163508751)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    """Four-connected sets of cells, in deterministic reading order."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, cells = [first], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _holes(mask):
    """Background components enclosed by this mask (four-neighbor topology)."""
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if not np.any((cells[:, 0] == 0) | (cells[:, 0] == h-1)
                          | (cells[:, 1] == 0) | (cells[:, 1] == w-1))]


def _norm(cells):
    return cells - cells.min(axis=0)


def _same_shape(a, b):
    return np.array_equal(_norm(a), _norm(b))


def _parse(grid):
    """Input-only segmentation and exhaustive docking-site enumeration."""
    g = _grid(grid)
    bg = int(g[0, 0])
    assert all(int(g[r, c]) == bg for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))), "background corners"
    objects = []
    # Objects have background clearance even if their colors differ.
    for cells in _components(g != bg):
        colors = np.unique(g[cells[:, 0], cells[:, 1]])
        assert len(colors) == 1, "touching objects or multicolor component"
        origin = cells.min(axis=0)
        shape = tuple(cells.max(axis=0)-origin+1)
        local = cells-origin
        mask = np.zeros(shape, dtype=bool)
        mask[local[:, 0], local[:, 1]] = True
        objects.append(dict(cells=cells, color=int(colors[0]),
                            origin=origin, mask=mask, local=local))
    spanning = [(i, axis) for i, o in enumerate(objects) for axis in (0, 1)
                if len(np.unique(o['cells'][:, axis])) == g.shape[axis]]
    assert len(spanning) == 1, "one fixed spanning rail"
    rail_index, span_axis = spanning[0]
    rail = objects[rail_index]
    rail_mask = np.zeros_like(g, dtype=bool)
    rc = rail['cells']
    rail_mask[rc[:, 0], rc[:, 1]] = True
    # Every cross-section is an interval; their intersection has a >=2 core.
    canonical = rail_mask if span_axis == 1 else rail_mask.T
    ends = []
    for c in range(canonical.shape[1]):
        rows = np.flatnonzero(canonical[:, c])
        assert len(rows) and np.array_equal(rows, np.arange(rows[0], rows[-1]+1))
        ends.append((int(rows[0]), int(rows[-1])+1))
    assert min(b for _, b in ends)-max(a for a, _ in ends) >= 2
    carriers, cargo = [], []
    for i, o in enumerate(objects):
        if i == rail_index:
            continue
        assert o['color'] != rail['color'], "rail color is reserved"
        r, c = o['origin']
        h, w = o['mask'].shape
        neighborhood = g[max(0, r-1):min(g.shape[0], r+h+1),
                         max(0, c-1):min(g.shape[1], c+w+1)].copy()
        neighborhood[r-max(0, r-1):r-max(0, r-1)+h,
                     c-max(0, c-1):c-max(0, c-1)+w] = bg
        assert np.all(neighborhood == bg), "objects need a full background gap"
        # A cavity may not contain another object.
        assert np.all(g[r:r+h, c:c+w][~o['mask']] == bg)
        holes = _holes(o['mask'])
        assert len(holes) <= 1, "at most one enclosed cavity"
        if holes:
            o['hole'] = holes[0]
            filled = o['mask'].copy()
            filled[holes[0][:, 0], holes[0][:, 1]] = True
            missing = ~filled
            assert np.any(missing), "a carrier needs a nonflat outer key"
            # The missing outer cells form a profile on one rail-facing side.
            cm = filled if span_axis == 1 else filled.T
            profile_sides = []
            for side in (-1, 1):
                oriented = cm if side == 1 else cm[::-1]
                cuts = [int(np.flatnonzero(oriented[:, j])[0])
                        for j in range(oriented.shape[1])]
                expect = np.arange(oriented.shape[0])[:, None] >= np.array(cuts)
                if np.array_equal(expect, oriented) and min(cuts) == 0 < max(cuts):
                    profile_sides.append((side, tuple(cuts)))
            assert len(profile_sides) == 1, "one complementary key face"
            o['side'], o['profile'] = profile_sides[0]
            # Search every legal translation, not a generator-supplied site list.
            valid = np.ones((g.shape[0]-h+1, g.shape[1]-w+1), dtype=bool)
            for rr, cc in np.argwhere(filled):
                valid &= ~rail_mask[rr:rr+valid.shape[0], cc:cc+valid.shape[1]]
            for rr, cc in np.argwhere(missing):
                valid &= rail_mask[rr:rr+valid.shape[0], cc:cc+valid.shape[1]]
            sites = np.argwhere(valid)
            assert len(sites) == 1, "outer contour must determine one docking translation"
            o['target'] = sites[0]
            carriers.append(o)
        else:
            cargo.append(o)
    assert carriers and cargo, "both queues must be present"
    for queue in (carriers, cargo):
        queue.sort(key=lambda o: (int(o['origin'][0]), int(o['origin'][1])))
        tops = [int(o['origin'][0]) for o in queue]
        assert len(set(tops)) == len(tops), "same-class top-row ties are excluded"
    cavity = carriers[0]['hole']
    assert all(_same_shape(o['hole'], cavity) for o in carriers)
    assert all(_same_shape(o['local'], cavity) for o in cargo), "cargo must exactly fit every cavity"
    # Destination rectangles cannot intersect. This also prevents rail-key sites
    # from silently representing two uses of the same physical socket.
    used = np.zeros_like(g, dtype=bool)
    for o in carriers:
        r, c = o['target']
        h, w = o['mask'].shape
        assert not np.any(used[r:r+h, c:c+w]), "overlapping carrier rectangles"
        used[r:r+h, c:c+w] = True
    return dict(grid=g, background=bg, rail=rail, rail_mask=rail_mask,
                orientation='horizontal' if span_axis == 1 else 'vertical',
                carriers=carriers, cargo=cargo)


def _rank_assignment(a):
    return {i: i for i in range(min(len(a['carriers']), len(a['cargo'])))}


def _greedy_assignment(a):
    """Competing single-pass row packing from B's review, for evidence checks."""
    items = [(tuple(o['origin']), kind, i) for kind, queue in
             enumerate((a['carriers'], a['cargo'])) for i, o in enumerate(queue)]
    rows = [[None, None]]
    for _, kind, i in sorted(items):
        if rows[-1][kind] is not None:
            rows.append([None, None])
        rows[-1][kind] = i
    return {h: s for h, s in rows if h is not None and s is not None}


def _render(a, assignment=None):
    """Translate whole masks; only cargo that has a recipient is removed."""
    if assignment is None:
        assignment = _rank_assignment(a)
    out = a['grid'].copy()
    bg = a['background']
    for o in a['carriers'] + [a['cargo'][i] for i in assignment.values()]:
        cells = o['cells']
        out[cells[:, 0], cells[:, 1]] = bg
    for o in a['carriers']:
        cells = o['local']+o['target']
        assert np.all(out[cells[:, 0], cells[:, 1]] == bg), "destination occupied by waiting cargo"
        out[cells[:, 0], cells[:, 1]] = o['color']
        holes = o['hole']+o['target']
        assert np.all(out[holes[:, 0], holes[:, 1]] == bg), "waiting cargo occupies a cavity"
    for i, j in assignment.items():
        carrier, item = a['carriers'][i], a['cargo'][j]
        cells = carrier['hole']+carrier['target']
        assert np.all(out[cells[:, 0], cells[:, 1]] == bg)
        out[cells[:, 0], cells[:, 1]] = item['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, queue ranks, and docking translations from input only."""
    return _render(_parse(input_grid))


def _witnesses(a):
    out = _render(a)
    # An alternative can be physically invalid because waiting cargo obstructs
    # it. In that case it is already distinguished by the visible scene.
    def different(mapping):
        try:
            return not np.array_equal(out, _render(a, mapping))
        except AssertionError:
            return True
    m = min(len(a['carriers']), len(a['cargo']))
    horizontal = []
    for reverse in (False, True):
        indices = sorted(range(len(a['cargo'])),
                         key=lambda j: tuple(a['cargo'][j]['origin'][::-1]),
                         reverse=reverse)
        horizontal.append(different(dict(zip(range(m), indices[:m]))))
    target_order = sorted(range(len(a['carriers'])),
                          key=lambda i: tuple(a['carriers'][i]['target']))
    nearest_wrong = any(
        min(range(len(a['carriers'])), key=lambda j:
            int(np.abs(o['origin']-a['carriers'][j]['target']).sum())) != i
        for i, o in enumerate(a['carriers']))
    nearest_cargo_wrong = any(
        min(range(len(a['cargo'])), key=lambda j:
            int(np.abs(a['carriers'][i]['origin']-a['cargo'][j]['origin']).sum())) != i
        for i in range(m))
    return dict(greedy=different(_greedy_assignment(a)),
                horizontal_ascending=horizontal[0],
                horizontal_descending=horizontal[1],
                reverse_cargo=different({i: len(a['cargo'])-1-i for i in range(m)}),
                geometry_color_ambiguity=len({o['color'] for o in a['cargo'][:m]}) >= 2,
                color_matching_wrong=any(a['carriers'][i]['color'] != a['cargo'][i]['color']
                                         for i in range(m)),
                nearest_cargo_wrong=nearest_cargo_wrong,
                docking_permutation=target_order != list(range(len(target_order))),
                nearest_wrong=nearest_wrong)


def _cavity(rng):
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
    mask = np.ones((h, w), dtype=bool)
    # Corner erosion yields convex/stepped cavities, including rectangles.
    for r, c in rng.permutation([(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)])[:int(rng.integers(4))]:
        candidate = mask.copy()
        candidate[r, c] = False
        cells = np.argwhere(candidate)
        if (len(cells) >= 3 and len(_components(candidate)) == 1
                and tuple(cells.max(axis=0)+1) == (h, w)
                and np.all(cells.min(axis=0) == 0) and not _holes(candidate)):
            mask = candidate
    return mask


def _make_carrier(rng, cavity, side, depth):
    ch, cw = cavity.shape
    lm, rm = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    w = cw+lm+rm
    key = rng.integers(0, depth+1, size=w)
    key[int(rng.integers(w))] = 0
    key[(int(np.flatnonzero(key == 0)[0])+int(rng.integers(1, w))) % w] = depth
    collar, far = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    h = depth+collar+ch+far
    rows = np.arange(h)[:, None]
    outer = rows >= key if side == 1 else rows < h-key
    top = depth+collar if side == 1 else far
    hole = np.argwhere(cavity)+[top, lm]
    mask = outer.copy()
    mask[hole[:, 0], hole[:, 1]] = False
    assert len(_components(mask)) == 1 and len(_holes(mask)) == 1
    return dict(mask=mask, hole=hole, side=side, key=key)


def _scatter(rng, base, masks, kinds, bg):
    """Free rectangle placement; neither shelf sites nor output sites are used."""
    h, w = base.shape
    rail = base != bg
    padded = np.pad(rail, 1)
    forbidden = np.zeros_like(rail)
    for dr in range(3):
        for dc in range(3):
            forbidden |= padded[dr:dr+h, dc:dc+w]
    summed = np.pad(forbidden.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for _ in range(20):
        # Full rectangles are reserved, so input cargo never nests in a carrier.
        boxes = []
        origins = [None]*len(masks)
        tops = {0: set(), 1: set()}
        order = sorted(range(len(masks)), key=lambda i: -masks[i].size+rng.random())
        for i in order:
            mh, mw = masks[i].shape
            # Summed-area queries test every rectangle against the rail moat.
            legal = (summed[mh:, mw:]-summed[:-mh, mw:]
                     -summed[mh:, :-mw]+summed[:-mh, :-mw]) == 0
            rr, cc = np.indices(legal.shape)
            for r, c, bh, bw in boxes:
                legal &= ((rr+mh < r) | (rr > r+bh) | (cc+mw < c) | (cc > c+bw))
            for top in tops[kinds[i]]:
                if top < legal.shape[0]:
                    legal[top] = False
            # All four corners visibly encode background.
            legal[0, 0] = legal[0, -1] = legal[-1, 0] = legal[-1, -1] = False
            sites = np.argwhere(legal)
            if not len(sites):
                break
            r, c = sites[int(rng.integers(len(sites)))]
            origins[i] = np.array([int(r), int(c)])
            tops[kinds[i]].add(int(r))
            boxes.append((int(r), int(c), mh, mw))
        else:
            return origins
    raise ValueError("scatter clearance")


def _construct(rng, nh, ns, depth, orientation, teach=False, demo=None):
    bg, rail_color = map(int, rng.choice(10, 2, replace=False))
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, rail_color)])))
    cavity = _cavity(rng)
    sides = list(map(int, rng.choice([-1, 1], size=nh)))
    if teach or nh == 4:
        sides[0], sides[-1] = -1, 1
    specs = [_make_carrier(rng, cavity, side, depth) for side in sides]
    needed = max(sum(s['mask'].shape[1]+2 for s in specs if s['side'] == side)-2
                 for side in (-1, 1))
    if needed > 28:
        raise ValueError("too many carriers on one side")
    width = int(rng.integers(max(17, needed+2), 31))
    top_extent = max([s['mask'].shape[0] for s in specs if s['side'] == -1] or [1])
    bottom_extent = max([s['mask'].shape[0] for s in specs if s['side'] == 1] or [1])
    thickness = int(rng.integers(2, 5))
    min_height = max(22 if nh <= 2 else 25, top_extent+bottom_extent+thickness)
    if min_height > 30:
        raise ValueError("insufficient perpendicular space")
    height = int(rng.integers(min_height, 31))
    b = int(rng.integers(top_extent, height-bottom_extent-thickness+1))
    base = np.full((height, width), bg, dtype=np.int8)
    base[b:b+thickness] = rail_color
    occupied = {-1: set(), 1: set()}
    for s in specs:
        h, w = s['mask'].shape
        choices = [c for c in range(width-w+1)
                   if not set(range(c-1, c+w+1)) & occupied[s['side']]]
        if not choices:
            raise ValueError("attachment spacing")
        c = choices[int(rng.integers(len(choices)))]
        occupied[s['side']].update(range(c, c+w))
        r = b-h if s['side'] == -1 else b+thickness
        s['target'] = np.array([r, c])
        for j, k in enumerate(s['key']):
            if s['side'] == -1:
                base[b-int(k):b, c+j] = rail_color
            else:
                base[b+thickness:b+thickness+int(k), c+j] = rail_color
    # Unused irregular boundary detail is structural context, not a command.
    for side in (-1, 1):
        candidates = [c for c in range(width) if c not in occupied[side]]
        if candidates:
            for c in rng.choice(candidates, size=min(len(candidates), int(rng.integers(1, 5))), replace=False):
                if side == -1:
                    base[b-1, c] = rail_color
                else:
                    base[b+thickness, c] = rail_color
    masks = [s['mask'] for s in specs]+[cavity]*ns
    if orientation == 'vertical':
        base = base.T.copy()
        masks = [m.T.copy() for m in masks]
        for s in specs:
            s['target'] = s['target'][::-1].copy()
            s['hole'] = s['hole'][:, ::-1]
            s['hole'] = s['hole'][np.lexsort((s['hole'][:, 1], s['hole'][:, 0]))]
    origins = _scatter(rng, base, masks, [0]*nh+[1]*ns, bg)
    if teach:
        hc, sc = palette[:nh], palette[:ns]
        # This literal color occurs in both topological classes in one scene.
        if demo is not None and demo not in (bg, rail_color):
            for colors in (hc, sc):
                if demo in colors:
                    colors[0], colors[colors.index(demo)] = colors[colors.index(demo)], colors[0]
                else:
                    colors[0] = demo
    else:
        pool = palette[:int(rng.integers(2, 9))]
        hc = list(map(int, rng.choice(pool, nh)))
        sc = list(map(int, rng.choice(pool, ns)))
    colors = hc+sc
    inp = base.copy()
    for mask, origin, color in zip(masks, origins, colors):
        cells = np.argwhere(mask)+origin
        inp[cells[:, 0], cells[:, 1]] = color
    h_order = sorted(range(nh), key=lambda i: tuple(origins[i]))
    s_order = sorted(range(nh, nh+ns), key=lambda i: tuple(origins[i]))
    # Construction truth uses known target masks, not the parser/search.
    out = base.copy()
    for i, s in enumerate(specs):
        cells = np.argwhere(masks[i])+s['target']
        out[cells[:, 0], cells[:, 1]] = hc[i]
    for rank, i in enumerate(s_order):
        if rank < nh:
            s = specs[h_order[rank]]
            cells = s['hole']+s['target']
        else:
            cells = np.argwhere(masks[i])+origins[i]
        if np.any(out[cells[:, 0], cells[:, 1]] != bg):
            raise ValueError("waiting cargo overlaps a docking rectangle")
        out[cells[:, 0], cells[:, 1]] = colors[i]
    analysis = _parse(inp)
    assert np.array_equal(_render(analysis), out), "construction disagrees with visible inference"
    return dict(input=inp, output=out), analysis


def _checked_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("integer counts must be train=2..10 and test=1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _checked_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first_orientation = str(rng.choice(['horizontal', 'vertical']))
    first_tail = int(rng.integers(2))
    demo = int(rng.integers(10))
    puzzle, seen = dict(train=[], test=[]), set()
    for index in range(num_train+num_test):
        training = index < num_train
        teach = index < 2
        if teach:
            nh, ns = ((3, 2) if (index+first_tail) % 2 == 0 else (2, 3))
            depth = index+1
            orientation = first_orientation if index == 0 else (
                'vertical' if first_orientation == 'horizontal' else 'horizontal')
        else:
            nh = int(rng.integers(1, 5))
            ns = int(rng.integers(1, 6))
            depth = int(rng.integers(1, 3)) if training else 3
            orientation = str(rng.choice(['horizontal', 'vertical']))
        for attempt in range(240):
            try:
                pair, a = _construct(rng, nh, ns, depth, orientation, teach, demo)
                _validate_pair(pair)
                key = _key(pair['input'])
                if key in seen:
                    continue
                if teach:
                    witnesses = _witnesses(a)
                    if not (witnesses['greedy'] and witnesses['geometry_color_ambiguity']
                            and witnesses['docking_permutation'] and witnesses['nearest_wrong']
                            and witnesses['color_matching_wrong'] and witnesses['nearest_cargo_wrong']):
                        continue
                    # Three cargo pieces can zigzag; two cannot refute both
                    # horizontal directions in the same example.
                    if ns == 3 and not (witnesses['horizontal_ascending']
                                        and witnesses['horizontal_descending']):
                        continue
                break
            except (AssertionError, ValueError):
                continue
        else:
            raise RuntimeError(f"bounded construction exhausted at example {index}")
        seen.add(key)
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    a = _parse(inp)
    assert np.array_equal(_render(a), out), "wrong output"
    assert np.array_equal(inp == a['rail']['color'], out == a['rail']['color']), "rail changed"
    assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                          np.bincount(out.ravel(), minlength=10)), "color area was not conserved"
    occupied = a['rail_mask'].copy()
    for rank, o in enumerate(a['carriers']):
        cells = o['local']+o['target']
        assert not np.any(occupied[cells[:, 0], cells[:, 1]])
        occupied[cells[:, 0], cells[:, 1]] = True
        assert np.all(out[cells[:, 0], cells[:, 1]] == o['color'])
        assert np.any(o['target'] != o['origin']), "carrier did not move"
        holes = o['hole']+o['target']
        expected_color = a['cargo'][rank]['color'] if rank < len(a['cargo']) else a['background']
        assert np.all(out[holes[:, 0], holes[:, 1]] == expected_color)
        if rank < len(a['cargo']):
            item = a['cargo'][rank]
            assert _same_shape(holes, item['cells']), "cargo was changed or rotated"
            assert np.any(holes.min(axis=0) != item['origin']), "assigned cargo did not move"
        # The entire destination rectangle is exactly fixed key + carrier +
        # cargo, or the same exact cover with its one unmatched cavity empty.
        r, c = o['target']
        h, w = o['mask'].shape
        block = out[r:r+h, c:c+w]
        if rank < len(a['cargo']):
            assert not np.any(block == a['background'])
        else:
            assert np.count_nonzero(block == a['background']) == len(o['hole'])
    for o in a['cargo'][len(a['carriers']):]:
        cells = o['cells']
        assert np.all(out[cells[:, 0], cells[:, 1]] == o['color']), "surplus cargo moved"
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _checked_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _validate_pair(pair)
            key = _key(pair['input'])
            assert key not in seen, "duplicate input"
            seen.add(key)
            (training if split == 'train' else tests).append(a)
    imbalances = [len(a['carriers'])-len(a['cargo']) for a in training]
    assert min(imbalances) < 0 < max(imbalances), "demonstrate both unmatched tails"
    assert {a['orientation'] for a in training} == {'horizontal', 'vertical'}
    witnesses = [_witnesses(a) for a in training]
    for name in ('greedy', 'horizontal_ascending', 'horizontal_descending',
                 'reverse_cargo', 'geometry_color_ambiguity', 'docking_permutation',
                 'nearest_wrong', 'color_matching_wrong', 'nearest_cargo_wrong'):
        assert any(w[name] for w in witnesses), f"missing demonstration: {name}"
    assert any({o['color'] for o in a['carriers']} & {o['color'] for o in a['cargo']}
               for a in training), "show that color does not choose geometry class"
    assert any({o['side'] for o in a['carriers']} == {-1, 1} for a in training)
    # The intended shift is new key depth, not a secret split-dependent rule.
    assert max(max(o['profile']) for a in training for o in a['carriers']) <= 2
    assert all(max(max(o['profile']) for o in a['carriers']) == 3 for a in tests)


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded authored checks over the 42 supplied host seeds, plus mutations."""
    signatures, structures, shapes, cavity_shapes = set(), set(), set(), set()
    counts, sides, depths, palettes = set(), set(), set(), set()
    paired = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for pair in p['train']+p['test']:
            a = _validate_pair(pair)
            assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
            roles, normalized = {}, []
            for value in pair['input'].flat:
                normalized.append(roles.setdefault(int(value), len(roles)))
            structures.add((pair['input'].shape, tuple(normalized)))
            shapes.add(pair['input'].shape)
            cavity_shapes.add(tuple(map(tuple, _norm(a['carriers'][0]['hole']))))
            counts.add((len(a['carriers']), len(a['cargo'])))
            sides.update((a['orientation'], o['side']) for o in a['carriers'])
            depths.update(max(o['profile']) for o in a['carriers'])
            palettes.add((a['background'], a['rail']['color']))
            paired += 1
    assert len(signatures) == 42
    assert len(structures) >= 200 and len(shapes) >= 35
    assert len(cavity_shapes) >= 15 and len(counts) >= 15 and len(palettes) >= 30
    assert sides == {(axis, side) for axis in ('horizontal', 'vertical') for side in (-1, 1)}
    assert depths == {1, 2, 3}
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    # The host also exercises every legal explicit combination. Keep this
    # built-in check small enough to run whenever verification calls it.
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(123+train+test, train, test)
        assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    broken = copy.deepcopy(original)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity mislabel accepted")
    # A cargo-color swap conserves all areas when the two plugs have equal
    # geometry; the rank check must still reject it.
    broken = copy.deepcopy(original)
    a = _parse(broken['train'][0]['input'])
    out = broken['train'][0]['output']
    first, second = a['carriers'][:2]
    for carrier, item in ((first, a['cargo'][1]), (second, a['cargo'][0])):
        cells = carrier['hole']+carrier['target']
        out[cells[:, 0], cells[:, 1]] = item['color']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("rank-swapped cargo accepted")
    return dict(seeds=42, pairs=paired, structures=len(structures),
                input_shapes=len(shapes), cavity_shapes=len(cavity_shapes),
                count_combinations=len(counts), role_palettes=len(palettes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
