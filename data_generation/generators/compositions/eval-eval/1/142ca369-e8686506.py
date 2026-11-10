"""Inverse optical assembly: pack the mirrors so the light can escape."""
from __future__ import annotations

from collections import Counter
import numpy as np


class InvalidScene(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise InvalidScene(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, comp = [start], []
        while todo:
            r, c = todo.pop()
            comp.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(tuple(sorted(comp)))
    return result


def _norm(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _mirror(cells, color):
    cells = tuple(sorted(cells))
    rr, cc = zip(*cells)
    n = len(cells)
    _need(n % 2 == 1 and (len(set(rr)) == 1 or len(set(cc)) == 1)
          and max(max(rr)-min(rr), max(cc)-min(cc)) == n-1, "a mirror must be an odd straight bar")
    return {"cells": cells, "color": int(color),
            "mid": ((min(rr)+max(rr))//2, (min(cc)+max(cc))//2),
            "axis": 2 if n == 1 else (0 if len(set(rr)) == 1 else 1)}


def _source(cells, color):
    rr, cc = zip(*cells)
    _need(len(cells) == 3 and max(rr)-min(rr) == max(cc)-min(cc) == 1, "source must be an L triomino")
    missing = next(iter({(r, c) for r in range(min(rr), max(rr)+1)
                         for c in range(min(cc), max(cc)+1)} - set(cells)))
    elbow = (min(rr)+max(rr)-missing[0], min(cc)+max(cc)-missing[1])
    return {"cells": tuple(sorted(cells)), "color": int(color), "elbow": elbow,
            "direction": (elbow[0]-missing[0], elbow[1]-missing[1])}


def _source_cells(p, d):
    return (p, (p[0]-d[0], p[1]), (p[0], p[1]-d[1]))


def _bar(mid, axis, n):
    return tuple((mid[0]+(k if axis == 1 else 0), mid[1]+(k if axis == 0 else 0))
                 for k in range(-(n//2), n//2+1))


def _trace(shape, sources, mirrors):
    """Trace only optical objects; the board's neutral floor can be painted."""
    h, w = shape
    objects = [o for o in sources] + [m for m in mirrors]
    occupied = {p for o in objects for p in o['cells']}
    _need(sum(len(o['cells']) for o in objects) == len(occupied), "overlapping optics")
    painted, routes = {}, []
    for s in sources:
        d, color, anchor = s['direction'], s['color'], s['elbow']
        p = (anchor[0]+d[0], anchor[1]+d[1])
        visited, path, contacts = set(), [], []
        while 0 <= p[0] < h and 0 <= p[1] < w:
            _need((p, d) not in visited, "ray loop")
            visited.add((p, d))
            _need(p not in occupied, "off-midpoint collision")
            hits = []
            for i, m in enumerate(mirrors):
                for axis in ((0, 1) if m['axis'] == 2 else (m['axis'],)):
                    other = 1-axis
                    if p[other] == m['mid'][other] and m['mid'][axis]-p[axis] == d[axis]:
                        hits.append((i, axis))
            _need(len(hits) <= 1, "competing mirror faces")
            if hits:
                i, axis = hits[0]
                color = mirrors[i]['color']
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
                contacts.append((i, axis, p, color))
            _need(p not in painted or painted[p] == color, "incompatible beams")
            painted[p] = color
            path.append((p, color))
            p = p[0]+d[0], p[1]+d[1]
        routes.append({'path': path, 'contacts': contacts})
    return painted, routes


def _parse(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _need(counts[bg] > inp.size//2, "background must be a strict majority")
    area = int(np.count_nonzero(inp != bg))
    candidates = []
    for color in range(10):
        if color == bg or not counts[color]:
            continue
        rr, cc = np.where(inp == color)
        r, c, h, w = int(rr.min()), int(cc.min()), int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)
        if h < 5 or w < 5 or h*w != area:
            continue
        board = inp[r:r+h, c:c+w]
        if all(np.all(edge == color) for edge in (board[0], board[-1], board[:, 0], board[:, -1])):
            candidates.append((r, c, h, w, color))
    _need(len(candidates) == 1, "expected one solid floor frame with conserved area")
    r, c, h, w, floor = candidates[0]
    board = inp[r:r+h, c:c+w].copy()
    sources, fixed = [], []
    for color in range(10):
        if color in (bg, floor):
            continue
        for comp in _components(map(tuple, np.argwhere(board == color))):
            sig = _norm(comp)
            if len(comp) == 3 and max(p[0] for p in sig) == max(p[1] for p in sig) == 1:
                sources.append(_source(comp, color))
            else:
                fixed.append(_mirror(comp, color))
    _need(sources, "no fixed light source")
    outside = inp.copy()
    outside[r:r+h, c:c+w] = bg
    pieces = []
    for comp in _components(map(tuple, np.argwhere(outside != bg))):
        colors = {int(outside[p]) for p in comp}
        _need(len(colors) == 1, "loose pieces must be separated and monochrome")
        color = colors.pop()
        _need(color != floor, "floor color used outside frame")
        sig = _norm(comp)
        _mirror(sig, color)
        pieces.append((color, sig))
    holes = tuple(sorted(map(tuple, np.argwhere(board == bg))))
    _need(1 <= len(pieces) <= 8 and holes, "one to eight loose mirrors required")
    _need(sum(len(s) for color, s in pieces) == len(holes), "inventory area mismatch")
    return bg, floor, (r, c, h, w), board, sources, fixed, pieces, holes


def _assemble(inp, stop_on_ambiguity=True):
    """Search translation packings and test their optics jointly, input only."""
    bg, floor, bbox, board, sources, fixed, pieces, holes = _parse(inp)
    inventory = Counter(pieces)
    types = sorted(inventory)
    ids = {p: i for i, p in enumerate(holes)}
    by_cell = [[] for p in holes]
    for t, (color, shape) in enumerate(types):
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        for r in range(board.shape[0]-sh+1):
            for c in range(board.shape[1]-sw+1):
                cells = tuple((r+dr, c+dc) for dr, dc in shape)
                if all(p in ids for p in cells):
                    mask = sum(1 << ids[p] for p in cells)
                    move = (t, mask, _mirror(cells, color))
                    for p in cells:
                        by_cell[ids[p]].append(move)
    answers, assemblies = {}, set()
    leaves = nodes = 0
    counts = [inventory[t] for t in types]

    def search(remaining, selected):
        nonlocal leaves, nodes
        nodes += 1
        if not remaining:
            leaves += 1
            full = board.copy()
            for m in selected:
                for p in m['cells']:
                    full[p] = m['color']
            assemblies.add(_key(full))
            try:
                paint, routes = _trace(board.shape, sources, fixed+selected)
            except InvalidScene:
                return
            out = full.copy()
            for p, color in paint.items():
                out[p] = color
            answers.setdefault(_key(out), (out, full, fixed+selected[:], paint, routes))
            return
        options = None
        bits = remaining
        while bits:
            bit = bits & -bits
            choices = [m for m in by_cell[bit.bit_length()-1]
                       if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return
            if options is None or len(choices) < len(options):
                options = choices
                if len(options) == 1:
                    break
            bits ^= bit
        for t, mask, mirror in options:
            counts[t] -= 1
            search(remaining ^ mask, selected+[mirror])
            counts[t] += 1
            if stop_on_ambiguity and len(answers) >= 2:
                return

    search((1 << len(holes))-1, [])
    _need(len(answers) == 1, f"expected one optical answer, found {len(answers)}")
    out, full, mirrors, paint, routes = next(iter(answers.values()))
    return {'output': out, 'full': full, 'mirrors': mirrors, 'paint': paint, 'routes': routes,
            'bg': bg, 'floor': floor, 'bbox': bbox, 'board': board, 'sources': sources,
            'fixed': fixed, 'pieces': pieces, 'holes': holes, 'assemblies': len(assemblies),
            'packings': leaves, 'nodes': nodes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _assemble(_grid(input_grid))['output'].copy()


def _inside_optics(cells, shape):
    return all(1 <= r < shape[0]-1 and 1 <= c < shape[1]-1 for r, c in cells)


def _add_launch(rng, shape, sources, mirrors, palette):
    for _ in range(120):
        elbow = (int(rng.integers(2, shape[0]-2)), int(rng.integers(2, shape[1]-2)))
        direction = (int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))
        s = _source(_source_cells(elbow, direction), int(rng.choice(palette)))
        if not _inside_optics(s['cells'], shape):
            continue
        occupied = {p for o in sources+mirrors for p in o['cells']}
        if any(p in occupied or any(q in occupied for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))) for p in s['cells']):
            continue
        try:
            paint, routes = _trace(shape, sources+[s], mirrors)
        except InvalidScene:
            continue
        if len(routes[-1]['path']) >= 4:
            return sources+[s]
    return None


def _extend(rng, shape, sources, mirrors, cmap, active_source=0, force_axis=None,
            movable=True, join_mode='rail'):
    _, routes = _trace(shape, sources, mirrors)
    route = routes[active_source]
    source = sources[active_source]
    anchor, d = source['elbow'], source['direction']
    for i, axis, p, color in route['contacts']:
        anchor = p
        d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
    occupied = {p for o in sources+mirrors for p in o['cells']}
    choices = []
    for step in range(2, max(shape)):
        contact = anchor[0]+step*d[0], anchor[1]+step*d[1]
        if not _inside_optics([contact], shape):
            break
        for axis in ([force_axis] if force_axis is not None else [0, 1]):
            for n in (3, 5, 7):
                for passive in ((1, 3, 5) if movable else (0,)):
                    if n != passive:
                        for side in (-1, 1):
                            choices.append((contact, axis, n, passive, side))
    rng.shuffle(choices)
    for contact, axis, n, passive, side in choices[:600]:
        mid = (contact[0]+(d[0] if axis == 0 else 0), contact[1]+(d[1] if axis == 1 else 0))
        cells = _bar(mid, axis, n)
        additions = [_mirror(cells, cmap[(axis, n)])]
        if passive:
            if join_mode == 'rail':
                passive_axis = axis
                shift = side*(n+passive)//2
                pmid = (mid[0]+(shift if axis == 1 else 0),
                        mid[1]+(shift if axis == 0 else 0))
            else:
                # Turn a corner at one end of the active bar, away from the
                # approaching beam. The two supplied bars stay disjoint.
                passive_axis = 1-axis
                pmid = list(mid)
                pmid[1-axis] += side*(n//2)
                pmid[axis] += d[axis]*(1+passive//2)
                pmid = tuple(pmid)
            signature = (2 if passive == 1 else passive_axis, passive)
            additions.append(_mirror(_bar(pmid, passive_axis, passive), cmap[signature]))
        allcells = {p for m in additions for p in m['cells']}
        if not _inside_optics(allcells, shape):
            continue
        if any(p in occupied or any(q in occupied for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))) for p in allcells):
            continue
        try:
            _, after = _trace(shape, sources, mirrors+additions)
        except InvalidScene:
            continue
        expected = route['contacts']+[(len(mirrors), axis, contact, additions[0]['color'])]
        if after[active_source]['contacts'] == expected:
            return mirrors+additions, additions
    return None


def _scatter(rng, board, loose, bg):
    h, w = board.shape
    dimensions = [(ch, cw) for ch in range(h+3, 31) for cw in range(w+3, 31) if ch*cw > 2*h*w]
    if not dimensions:
        return None
    for _ in range(20):
        ch, cw = dimensions[int(rng.integers(len(dimensions)))]
        br, bc = int(rng.integers(ch-h+1)), int(rng.integers(cw-w+1))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[br:br+h, bc:bc+w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0,br-1):min(ch,br+h+1), max(0,bc-1):min(cw,bc+w+1)] = True
        items = sorted(loose, key=lambda m: (-len(m['cells']), float(rng.random())))
        for m in items:
            sig = _norm(m['cells'])
            sh, sw = max(r for r,c in sig)+1, max(c for r,c in sig)+1
            options = [(r,c) for r in range(ch-sh+1) for c in range(cw-sw+1)
                       if all(not blocked[r+dr,c+dc] for dr,dc in sig)]
            if not options:
                break
            r,c = options[int(rng.integers(len(options)))]
            for dr,dc in sig:
                p = r+dr,c+dc
                canvas[p] = m['color']
                for rr,cc in (p,(p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)):
                    if 0 <= rr < ch and 0 <= cc < cw:
                        blocked[rr,cc] = True
        else:
            return canvas
    return None


def _make_pair(rng, patches=2, source_count=1, first_axis=None, fixed_count=0,
               mixed=False, elbow_chance=.3):
    for attempt in range(160):
        shape = (int(rng.integers(12,22)), int(rng.integers(12,22)))
        colors = list(map(int, rng.permutation(10)))
        bg, floor = colors[:2]
        palette = colors[2:2+int(rng.integers(3,8))]
        sigs = [(2,1)] + [(axis,n) for axis in (0,1) for n in (3,5,7)]
        cmap = {sig: palette[i % len(palette)] for i,sig in enumerate(sigs)}
        sources = _add_launch(rng, shape, [], [], palette)
        if sources is None:
            continue
        mirrors, loose = [], []
        good = True
        for i in range(patches+fixed_count):
            axis = first_axis if i == 0 else (1-first_axis if mixed and i == 1 else None)
            join_mode = 'elbow' if 0 < i < patches and rng.random() < elbow_chance else 'rail'
            extended = _extend(rng, shape, sources, mirrors, cmap, force_axis=axis,
                               movable=i < patches, join_mode=join_mode)
            if extended is None:
                good = False
                break
            mirrors, new = extended
            if i < patches:
                loose.extend(new)
        if not good:
            continue
        for _ in range(source_count-1):
            sources = _add_launch(rng, shape, sources, mirrors, palette)
            if sources is None:
                good = False
                break
        if not good:
            continue
        full = np.full(shape, floor, dtype=np.int8)
        for o in sources+mirrors:
            for p in o['cells']:
                full[p] = o['color']
        board = full.copy()
        for m in loose:
            for p in m['cells']:
                board[p] = bg
        inp = _scatter(rng, board, loose, bg)
        if inp is None:
            continue
        paint, _ = _trace(shape, sources, mirrors)
        expected = full.copy()
        for p,color in paint.items():
            expected[p] = color
        try:
            info = _assemble(inp)
        except InvalidScene:
            continue
        features = _features(info)
        if (info['assemblies'] >= 2 and np.array_equal(info['output'], expected)
                and features['changed_contacts'] > 0
                and features['covered_patches'] == features['patches']
                and (not mixed or len(features['directions']) >= 2)):
            return {'input': inp, 'output': expected}, info
    raise RuntimeError(f'could not generate {patches} patches, {source_count} sources')


def _check_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'),
                                    (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One rule for all seeds; the seed selects only visible scene variation."""
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    # Two compact lessons remain present even at the minimum legal counts.
    # Their first repaired contacts establish the two reflection axes.
    first_axis = int(rng.integers(2))
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train' and index == 0:
                settings = dict(patches=1, source_count=1, first_axis=first_axis)
            elif split == 'train' and index == 1:
                settings = dict(patches=2, source_count=2, first_axis=1-first_axis,
                                mixed=True)
            elif split == 'train':
                settings = dict(patches=int(rng.integers(1, 4)),
                                source_count=int(rng.integers(1, 3)),
                                fixed_count=int(rng.random() < .35))
            else:
                settings = dict(patches=int(rng.choice([3, 3, 4])),
                                source_count=int(rng.integers(1, 3)),
                                fixed_count=int(rng.random() < .35), elbow_chance=.45)
            for _ in range(12):
                pair, _ = _make_pair(rng, **settings)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not produce a distinct input')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    holes = set(info['holes'])
    patches = _components(holes)
    fixed_n = len(info['fixed'])
    contacts = [hit for route in info['routes'] for hit in route['contacts']]
    moved = info['mirrors'][fixed_n:]
    contacted = {i for i, axis, p, color in contacts}
    covered_patches = sum(any(i+fixed_n in contacted and set(m['cells']) <= set(comp)
                              for i, m in enumerate(moved)) for comp in patches)
    sizes = tuple(sorted(len(s) for color, s in info['pieces']))
    return {'shape': info['board'].shape, 'background': info['bg'],
            'floor': info['floor'], 'pieces': len(info['pieces']), 'sizes': sizes,
            'patches': len(patches), 'bent_patches': sum(len({r for r,c in p}) > 1
                and len({c for r,c in p}) > 1 for p in patches),
            'sources': len(info['sources']), 'fixed': fixed_n,
            'axes': {axis for i,axis,p,color in contacts},
            'directions': {s['direction'] for s in info['sources']},
            'bounces': tuple(len(route['contacts']) for route in info['routes']),
            'covered_patches': covered_patches,
            'assemblies': info['assemblies'], 'packings': info['packings'],
            'nodes': info['nodes'], 'colors': len({color for color,s in info['pieces']}),
            'repeated_piece': len(set(info['pieces'])) < len(info['pieces']),
            'same_color_shapes': any(len({s for col,s in info['pieces'] if col == c}) > 1
                                    for c,s in info['pieces']),
            'changed_contacts': sum(color != (route['contacts'][j-1][3] if j else source['color'])
                for source,route in zip(info['sources'],info['routes'])
                for j, (i,axis,p,color) in enumerate(route['contacts'])),
            'unchanged_contacts': sum(color == (route['contacts'][j-1][3] if j else source['color'])
                for source,route in zip(info['sources'],info['routes'])
                for j, (i,axis,p,color) in enumerate(route['contacts'])),
            'repeated_axis': any(a[1] == b[1] for route in info['routes']
                                 for a,b in zip(route['contacts'],route['contacts'][1:]))}


def validate_generated_puzzle(puzzle) -> None:
    """Check conservation, joint uniqueness, ray legality and teaching evidence."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _assemble(inp)
            f = _features(info)
            _need(np.array_equal(out, info['output']), 'output differs from unique optical packing')
            _need(out.shape == info['board'].shape, 'wrong crop')
            _need(info['assemblies'] >= 2, 'geometry alone should leave colored alternatives')
            _need(f['covered_patches'] == f['patches'], 'a repaired patch has no optical role')
            _need(f['changed_contacts'] > 0, 'must show reflector-controlled recoloring')
            full, board = info['full'], info['board']
            _need(not np.any(full == info['bg']), 'unfilled gap')
            fixed_mask = board != info['bg']
            _need(np.array_equal(board[fixed_mask], full[fixed_mask]), 'assembly changed fixed cells')
            for color in range(10):
                supply = sum(len(shape) for c,shape in info['pieces'] if c == color)
                _need(np.count_nonzero(full[board == info['bg']] == color) == supply,
                      'inventory was copied, discarded or recolored')
            for obj in info['sources']+info['mirrors']:
                _need(all(out[p] == obj['color'] for p in obj['cells']), 'optic cell overwritten')
            _need(all(board[p] == info['floor'] for p in info['paint']), 'ray must occupy neutral floor only')
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), 'invented color')
            # Omitting B leaves the holes empty and cannot produce the assembly.
            _need(any(out[p] != info['bg'] for p in info['holes']), 'assembly contributes nothing')
            # Omitting A gives a filled, unlit board, which is always different.
            _need(not np.array_equal(out, full), 'optics contributes nothing')
            if split == 'train':
                training.append(f)
    _need(set.union(*(f['axes'] for f in training)) == {0, 1}, 'training misses a reflection axis')
    _need(len(set.union(*(f['directions'] for f in training))) >= 2,
          'training needs more than one launch direction')
    _need(any(max(f['bounces']) >= 2 for f in training), 'training must demonstrate chained repair')
    _need(len({f['pieces'] for f in training}) >= 2, 'training must vary inventory multiplicity')


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
        for x,y in zip(a[s], b[s]) for k in ('input','output')) for s in ('train','test'))


def self_test() -> dict:
    """Bounded, measured coverage of 42 seeds plus direct rule regressions."""
    puzzles, records, structures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _assemble(pair['input'])
                records.append((split, _features(info)))
                # Normalize colors without removing shape or placement.
                mapping = {}
                code = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair['input'].flat)
                structures.add((pair['input'].shape, code))
                _need(np.array_equal(solve(pair['input']), pair['output']), 'solver mismatch')
    _need(len(structures) >= 200, 'insufficient structural variation')
    fs = [f for split,f in records]
    _need(len({f['shape'] for f in fs}) >= 55, 'insufficient board dimensions')
    _need({f['background'] for f in fs} == set(range(10)), 'background roles do not vary')
    _need({f['floor'] for f in fs} == set(range(10)), 'floor roles do not vary')
    for split in ('train','test'):
        local = [f for s,f in records if s == split]
        _need(set.union(*(f['directions'] for f in local)) == {(-1,-1),(-1,1),(1,-1),(1,1)}, 'launch rotations missing')
        _need(set.union(*(set(f['sizes']) for f in local)) >= {1,3,5,7}, 'mirror lengths missing')
        for feature in ('bent_patches','repeated_piece','same_color_shapes','unchanged_contacts','repeated_axis','fixed'):
            _need(any(f[feature] for f in local), 'variation missing: '+feature)
        _need(any(f['sources'] == 2 for f in local), 'multiple rays missing')
    _need(_puzzles_equal(puzzles[0], generate_puzzle(0)), 'nondeterministic generation')
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            puzzle = generate_puzzle(1000+a*10+b, a, b)
            _need(len(puzzle['train']) == a and len(puzzle['test']) == b, 'ignored explicit counts')
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _regressions(puzzles[0])
    return {'seeds': 42, 'pairs': len(records), 'structures': len(structures),
            'board_shapes': len({f['shape'] for f in fs}),
            'max_search_nodes': max(f['nodes'] for f in fs),
            'bent_pocket_pairs': sum(bool(f['bent_patches']) for f in fs),
            'packing_range': [min(f['packings'] for f in fs), max(f['packings'] for f in fs)]}


def _regressions(puzzle):
    # The same eight-cell gap and inventory have opposite correct seams when
    # the launch moves one column. These answers are specified directly.
    for source_col, short_left in ((2, True), (3, False)):
        board = np.full((10,14), 8, dtype=np.int8)
        launcher = _source_cells((2,source_col), (1,1))
        for p in launcher:
            board[p] = 1
        board[6,4:12] = 0
        inp = np.zeros((18,20), dtype=np.int8)
        inp[2:12,3:17] = board
        inp[15,1:4] = 2
        inp[17,12:17] = 3
        expected = board.copy()
        if short_left:
            expected[6,4:7], expected[6,7:12] = 2, 3
        else:
            expected[6,4:9], expected[6,9:12] = 3, 2
        for step in (1,2):
            expected[2+step,source_col+step] = 1
        contact_col = source_col+3
        for step in range(6):
            expected[5-step,contact_col+step] = 2 if short_left else 3
        info = _assemble(inp)
        _need(info['assemblies'] == 2 and info['packings'] == 2,
              'joint fixture should have two geometry-only packings')
        _need(np.array_equal(solve(inp), expected), 'inverse packing seam regression')
        turned = inp.copy()
        turned[15,1:4] = 0
        turned[14:17,1] = 2
        try:
            solve(turned)
        except InvalidScene:
            pass
        else:
            raise AssertionError('solver rotated a loose bar to fit a hole')
    # Hand-enumerated geometry: blue L, red H3, green V3, two contacts.
    s = _source(((1,1),(0,1),(1,0)), 1)
    h = _mirror(((5,3),(5,4),(5,5)), 2)
    v = _mirror(((1,7),(2,7),(3,7)), 3)
    paint, routes = _trace((9,10), [s], [h,v])
    expected = {(2,2):1,(3,3):1,(4,4):2,(3,5):2,(2,6):3,
                (1,5):3,(0,4):3}
    _need(paint == expected, 'contact recoloring or reflection geometry is wrong')
    _need([hit[1] for hit in routes[0]['contacts']] == [0,1], 'wrong reflection components')
    point_paint, _ = _trace((9,10), [s], [_mirror(((5,4),),2),v])
    _need(point_paint == expected, 'point reflector should use the contacted midpoint face')
    # At an off-center incidence, the next diagonal step hits a mirror cell.
    bad_h = _mirror(((5,4),(5,5),(5,6)), 2)
    try:
        _trace((9,10), [s], [bad_h])
    except InvalidScene:
        pass
    else:
        raise AssertionError('off-midpoint impact accepted')
    sample = puzzle['test'][0]
    for k in (1,2,3):
        _need(np.array_equal(solve(np.rot90(sample['input'],k)), np.rot90(sample['output'],k)),
              'whole-scene rotation changes the rule')
    # Rigid bars must keep their supplied orientation. Input inventory damage
    # and output corruption must both fail the family validator.
    for mode in ('inventory','output'):
        broken = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in puzzle[s]]
                  for s in ('train','test')}
        p = broken['test'][0]
        if mode == 'output':
            info = _assemble(p['input'])
            position = next(iter(info['paint']))
            p['output'][position] = info['floor']
        else:
            bg, floor, bbox, *_ = _parse(p['input'])
            r,c,h,w = bbox
            outside = p['input'] != bg
            outside[r:r+h,c:c+w] = False
            p['input'][tuple(np.argwhere(outside)[0])] = bg
        try:
            validate_generated_puzzle(broken)
        except (InvalidScene, AssertionError):
            pass
        else:
            raise AssertionError('corruption accepted')


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
