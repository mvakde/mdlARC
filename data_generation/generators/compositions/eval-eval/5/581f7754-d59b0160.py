"""Docked collective records: projection defines the objects to be filtered.

Only standard-library modules and NumPy are used. No files are read or written.
The solver reconstructs the scene from its input; construction records are used
only by the sampler, which paints its own answers before calling the validator.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 516714177, 628407068, 492927580, 200545240, 496668881,
              652102133, 344398646, 680195502, 699188954, 939659877, 376909796,
              732152132, 838102440, 953592532, 984467513, 554212813, 838071518,
              288037120, 333405375, 59814753, 359039190, 344039904, 594816204,
              24437733, 559189213, 279475095, 118894333, 182967636, 153531467,
              765522353, 781227389, 68272213, 907306958, 166784571, 560054281,
              427742374, 205383431, 928562308, 282805324, 345673072, 41457578)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger cells')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _view(a, corner):
    if corner & 1:
        a = a[:, ::-1]
    if corner & 2:
        a = a[::-1, :]
    return a


def _selector(grid):
    """The six equal padding cells identify background, even in dense scenes."""
    _require(min(grid.shape) >= 6, 'scene too small for the selector')
    candidates = []
    for corner in range(4):
        v = _view(grid, corner)
        frame = int(v[3, 3])
        if not (np.all(v[3, :4] == frame) and np.all(v[:4, 3] == frame)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        backgrounds = colors[counts == 6]
        if len(backgrounds) != 1:
            continue
        bg = int(backgrounds[0])
        query = frozenset(int(c) for c in colors if c != bg)
        if (len(query) == 3 and frame not in query and frame != bg
                and np.count_nonzero(grid == frame) == 7):
            candidates.append((corner, bg, frame, query))
    _require(len(candidates) == 1, 'selector must be uniquely recognizable')
    corner, bg, frame, query = candidates[0]
    h, w = grid.shape
    fixed = {(h - 1 - r if corner & 2 else r,
              w - 1 - c if corner & 1 else c) for r in range(4) for c in range(4)}
    return bg, frame, query, fixed


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    bg, frame, query, selector = _selector(grid)
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != bg)} - selector
    parts = _components(occupied)
    anchors, carriers = {}, []
    for part in parts:
        p = next(iter(part))
        if len(part) == 1 and (p[0] in (0, h - 1) or p[1] in (0, w - 1)):
            anchors.setdefault(int(grid[p]), set()).add(p)
        else:
            carriers.append(part)
    _require(anchors and carriers, 'anchors and carriers required')
    _require(not (set(anchors) & (query | {bg, frame})), 'conflicting color roles')
    systems = {}
    for color, points in anchors.items():
        rows, cols = {r for r, _ in points}, {c for _, c in points}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, _ in points)
        _require(horizontal != vertical, 'ambiguous boundary guide')
        axis = 0 if horizontal else 1
        guide = next(iter(rows if horizontal else cols))
        _require(0 < guide < grid.shape[axis] - 1, 'corner anchor')
        systems[color] = {'axis': axis, 'guide': guide, 'anchors': points}
    movers, ownership = [], {}
    anchor_points = set().union(*anchors.values())
    for part in carriers:
        regs = [p for p in part if int(grid[p]) in systems]
        _require(len(regs) == 1, 'exactly one registration cell per carrier')
        reg = regs[0]
        marker = int(grid[reg])
        system = systems[marker]
        delta = [0, 0]
        delta[system['axis']] = system['guide'] - reg[system['axis']]
        target = {(r + delta[0], c + delta[1]): int(grid[r, c]) for r, c in part}
        _require(all(0 <= r < h and 0 <= c < w for r, c in target), 'clipping is outside this family')
        _require(not (set(target) & (selector | anchor_points)), 'docking into a fixed control')
        _require(not (set(target) & set(ownership)), 'overlapping projected carriers')
        i = len(movers)
        ownership.update({p: i for p in target})
        labels = frozenset(int(grid[p]) for p in part if int(grid[p]) in query)
        movers.append({'points': part, 'reg': reg, 'marker': marker,
                       'delta': tuple(delta), 'target': target, 'query_labels': labels})
    _require(all(any(m['marker'] == c for m in movers) for c in systems), 'unused anchor')
    # Group projected material, excluding anchors and selector. A label-free
    # carrier participates in connectivity just as a colored one does.
    assemblies = []
    for points in _components(ownership):
        members = frozenset(ownership[p] for p in points)
        labels = frozenset(ownership_color for p in points
                           if (ownership_color := movers[ownership[p]]['target'][p]) in query)
        assemblies.append({'points': points, 'members': members, 'labels': labels,
                           'erase': query <= labels})
    adjacency = {i: set() for i in range(len(movers))}
    for p, i in ownership.items():
        for dr, dc in _DIRS:
            j = ownership.get((p[0] + dr, p[1] + dc))
            if j is not None and i != j:
                adjacency[i].add(j)
    return {'grid': grid, 'background': bg, 'frame': frame, 'query': query,
            'selector': selector, 'systems': systems, 'anchors': anchor_points,
            'movers': movers, 'assemblies': assemblies, 'adjacency': adjacency}


def _render(info, *, mode='collective'):
    grid, bg = info['grid'], info['background']
    result = np.full(grid.shape, bg, dtype=grid.dtype)
    for p in info['selector'] | info['anchors']:
        result[p] = grid[p]
    if mode == 'individual':
        erased = {i for i, m in enumerate(info['movers']) if info['query'] <= m['query_labels']}
    elif mode == 'guide':
        erased = set()
        for color in info['systems']:
            members = {i for i, m in enumerate(info['movers']) if m['marker'] == color}
            labels = set().union(*(info['movers'][i]['query_labels'] for i in members))
            if info['query'] <= labels:
                erased.update(members)
    elif mode == 'same_marker':
        erased = set()
        for color in info['systems']:
            owners = {p: i for i, m in enumerate(info['movers']) if m['marker'] == color
                      for p in m['target']}
            for points in _components(owners):
                members = {owners[p] for p in points}
                labels = set().union(*(info['movers'][i]['query_labels'] for i in members))
                if info['query'] <= labels:
                    erased.update(members)
    elif mode == 'neighbors':
        erased = {i for i in range(len(info['movers']))
                  if info['query'] <= set().union(*(info['movers'][j]['query_labels']
                     for j in info['adjacency'][i] | {i}))}
    else:
        erased = set().union(*(a['members'] for a in info['assemblies'] if a['erase']))
    for i, m in enumerate(info['movers']):
        if i not in erased:
            for p, c in m['target'].items():
                result[p] = c
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer controls, project all carriers, then filter contact-defined records."""
    return _render(_decode(input_grid))


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _fragment_shapes(rng, n, width, base, guides, bridge=False):
    """Partition a connected silhouette into connected pieces with visible handles.

    Free partitions grow from separate guide-row seeds, so their boundaries are
    irregular and can branch. The special bridge witness has a path graph.
    """
    top = base - int(rng.integers(2))
    bottom = base + 3 + int(rng.integers(2))
    silhouette = {(r, c) for r in range(top, bottom) for c in range(width)
                  if base <= r <= base + 2 or rng.random() < .62}
    if bridge:
        if width < 2 * n:
            return None
        cuts = [0] + sorted(map(int, rng.choice(range(2, width - 1), n - 1, replace=False))) + [width]
        if any(b - a < 2 for a, b in zip(cuts, cuts[1:])):
            return None
        parts = [{p for p in silhouette if a <= p[1] < b} for a, b in zip(cuts, cuts[1:])]
    else:
        cols = sorted(map(int, rng.choice(width, n, replace=False)))
        seeds = [(int(rng.choice(guides)), c) for c in cols]
        parts = [{p} for p in seeds]
        remaining = silhouette - set(seeds)
        while remaining:
            frontiers = []
            for i, cells in enumerate(parts):
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS} & remaining)
                if frontier:
                    frontiers.append((i, frontier))
            if not frontiers:
                return None
            least = min(len(parts[i]) for i, _ in frontiers)
            choices = [(i, f) for i, f in frontiers if len(parts[i]) <= least + 1]
            i, frontier = choices[int(rng.integers(len(choices)))]
            p = frontier[int(rng.integers(len(frontier)))]
            parts[i].add(p)
            remaining.remove(p)
    if any(len(p) < 4 for p in parts):
        return None
    return parts


def _paint_group(rng, parts, guides, markers, filler, query, extras, kind, missing):
    registers, cells = [], []
    for index, part in enumerate(parts):
        options = sorted(p for p in part if p[0] in guides)
        if kind == 'bridge' and index in (0, len(parts) - 1):
            required_row = guides[0 if index == 0 else -1]
            options = [p for p in options if p[0] == required_row]
        if not options:
            return None
        reg = options[int(rng.integers(len(options)))]
        registers.append(reg)
        colored = {p: filler for p in part}
        colored[reg] = markers[guides.index(reg[0])]
        cells.append(colored)
    n = len(parts)
    assignments = [[] for _ in parts]
    q = list(map(int, rng.permutation(query)))
    if kind in ('exact', 'positive', 'bridge'):
        if n < 2:
            return None
        # No individual piece has all three query identities.
        assignments[0] = [q[0]]
        assignments[-1] = [q[1], q[2]]
        if kind in ('positive', 'bridge'):
            assignments[0].append(int(extras[0]))
            if len(cells[-1]) >= 5:
                assignments[-1].append(q[1])
        if kind == 'positive':
            for i in range(1, n - 1):
                assignments[i] = [int(rng.choice(q)), int(rng.choice(extras))]
    elif kind == 'empty':
        pass
    else:
        keep = [c for c in query if c != missing]
        # Even a negative can have four different label identities, exceeding
        # the exact positive's three. Some also repeat labels densely.
        marks = keep + [int(extras[0])]
        if kind == 'rich_negative':
            marks += [int(extras[-1])] + keep + [int(extras[0])]
        for j, color in enumerate(marks):
            assignments[j % n].append(int(color))
    for i, (colored, reg, marks) in enumerate(zip(cells, registers, assignments)):
        spots = sorted(set(colored) - {reg})
        if len(marks) > len(spots):
            # Preserve the required identities, trimming repetitions only.
            unique = list(dict.fromkeys(marks))
            if len(unique) > len(spots):
                return None
            marks = unique + marks[:max(0, len(spots) - len(unique))]
        order = list(map(int, rng.permutation(len(spots))))
        for j, color in enumerate(marks):
            colored[spots[order[j]]] = int(color)
        # Free repetition is independent of qualification. A bridge stays blank.
        if kind not in ('exact', 'empty', 'bridge') and marks:
            for j in order[len(marks):]:
                if rng.random() < .22:
                    colored[spots[j]] = int(rng.choice(marks))
    return cells


def _place_sources(rng, target_pieces, grid, fixed):
    """Bounded backtracking over vertical translations; never save search traces."""
    h, w = grid.shape
    order = sorted(range(len(target_pieces)), key=lambda i: -len(target_pieces[i]))
    candidates = {}
    for i, piece in enumerate(target_pieces):
        r0, r1 = min(r for r, _ in piece), max(r for r, _ in piece)
        offsets = list(map(int, rng.permutation(np.arange(-r0, h - r1))))
        # Zero shifts are valid, but most fragments should visibly move.
        if 0 in offsets and rng.random() < .85:
            offsets.remove(0)
        candidates[i] = offsets
    used = set(fixed)
    solution, nodes = {}, [0]

    def visit(k):
        if k == len(order):
            return True
        if nodes[0] >= 1800:
            return False
        i = order[k]
        for delta in candidates[i]:
            if nodes[0] >= 1800:
                return False
            nodes[0] += 1
            cells = {(r + delta, c) for r, c in target_pieces[i]}
            if cells & used or any((r + dr, c + dc) in used for r, c in cells for dr, dc in _DIRS):
                continue
            used.update(cells)
            solution[i] = delta
            if visit(k + 1):
                return True
            used.difference_update(cells)
            solution.pop(i)
        return False

    if not visit(0):
        return False
    for i, piece in enumerate(target_pieces):
        for (r, c), color in piece.items():
            grid[r + solution[i], c] = color
    return True


def _scene(rng, colors, package, test, orientation):
    bg, frame, filler = map(int, colors[:3])
    # One or two parallel systems. The solver also supports mixed axes.
    nsystems = 2 if package == 'bridge' else 1 if rng.random() < .38 else 2
    markers = list(map(int, colors[3:3 + nsystems]))
    labels = list(map(int, colors[5:]))
    query = sorted(labels[:3])
    extras = labels[3:]
    if package == 'identity':
        plans = [('exact', 2, None), ('negative', 2, query[0]), ('rich_negative', 2, query[1])]
    elif package == 'bridge':
        plans = [('bridge', 3, None), ('rich_negative', 2, query[2]), ('empty', 1, None)]
    else:
        big = int(rng.integers(4, 6)) if test else int(rng.integers(2, 4))
        plans = [('positive' if rng.random() < .6 else 'rich_negative', big, int(rng.choice(query))),
                 ('positive', 2, None), ('rich_negative', int(rng.integers(1, 3)), int(rng.choice(query)))]
        if not test and rng.random() < .3:
            plans = plans[:2]
    rng.shuffle(plans)
    widths = []
    for kind, n, _ in plans:
        low = 2 * n if kind == 'bridge' else max(n + 1, (4 * n + 2) // 3)
        widths.append(low + int(rng.integers(0, 3)))
    gaps = list(map(int, rng.integers(1, 4, size=len(plans) - 1)))
    minimum_width = sum(widths) + sum(gaps) + 4
    if minimum_width > 30:
        return None
    w = int(rng.integers(max(18, minimum_width), 31))
    h = int(rng.integers(25 if test else 18, 31))
    base = int(rng.integers(6, h - 4))
    guides = [base] if nsystems == 1 else [base, base + 2]
    inp = np.full((h, w), bg, np.int8)
    out = inp.copy()
    inp[3, :4] = out[3, :4] = frame
    inp[:4, 3] = out[:4, 3] = frame
    for p, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        inp[int(p) // 3, int(p) % 3] = out[int(p) // 3, int(p) % 3] = int(color)
    fixed = {(r, c) for r in range(4) for c in range(4)}
    for guide, marker in zip(guides, markers):
        sides = [0, w - 1] if rng.random() < .35 else [int(rng.choice([0, w - 1]))]
        for c in sides:
            inp[guide, c] = out[guide, c] = marker
            fixed.add((guide, c))
    slack = w - 4 - sum(widths) - sum(gaps)
    x = 2 + int(rng.integers(slack + 1))
    pieces = []
    for j, ((kind, n, missing), width) in enumerate(zip(plans, widths)):
        parts = None
        for _ in range(24):
            parts = _fragment_shapes(rng, n, width, base, guides, kind == 'bridge')
            if parts is not None:
                break
        if parts is None:
            return None
        colored = _paint_group(rng, parts, guides, markers, filler, query, extras, kind, missing)
        if colored is None:
            return None
        for part in colored:
            p = {(r, c + x): color for (r, c), color in part.items()}
            pieces.append(p)
        # Construction answer uses planned groups, not the solver's decoder.
        actual_labels = {color for part in colored for color in part.values()}
        if not set(query) <= actual_labels:
            for part in pieces[-n:]:
                for p, color in part.items():
                    out[p] = color
        x += width + (gaps[j] if j < len(gaps) else 0)
    # A sampled system must actually have a registration in at least one piece.
    if any(not any(marker in p.values() for p in pieces) for marker in markers):
        return None
    if not _place_sources(rng, pieces, inp, fixed):
        return None
    # Rotations and reflections transform controls along with material.
    turns, reflect = orientation
    def transform(a):
        a = np.rot90(a, turns)
        return (a[:, ::-1] if reflect else a).copy()
    pair = {'input': transform(inp), 'output': transform(out)}
    try:
        info = _decode(pair['input'])
        if len(info['assemblies']) != len(plans):
            return None
        if not np.array_equal(_render(info), pair['output']):
            return None
        if not any(m['delta'] != (0, 0) for a in info['assemblies'] if not a['erase']
                   for i in a['members'] for m in [info['movers'][i]]):
            return None
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .3)
    base_colors = rng.permutation(10)
    first_turn = int(rng.integers(4))
    # Both orientations are taught, including for the two-pair minimum.
    orientations = [(first_turn, bool(rng.integers(2))),
                    ((first_turn + 1) % 4, bool(rng.integers(2)))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for k in range(a + b):
        test = k >= a
        package = 'identity' if k == 0 else 'bridge' if k == 1 else 'free'
        colors = base_colors if stable else rng.permutation(10)
        orientation = orientations[k] if k < 2 else (int(rng.integers(4)), bool(rng.integers(2)))
        for _ in range(180):
            pair = _scene(rng, colors, package, test, orientation)
            if pair is not None and _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('bounded scene construction exhausted')
        seen.add(_key(pair['input']))
        puzzle['test' if test else 'train'].append(pair)
    order = rng.permutation(a)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _assembly_labels(info, assembly):
    structural = set(info['systems']) | {info['background'], info['frame']}
    counts = Counter(int(info['grid'][p]) for i in assembly['members'] for p in info['movers'][i]['points'])
    # Sampling diagnostics only: the solver never needs to identify filler.
    # The dominant ordinary material is recovered across the whole scene.
    all_counts = Counter(int(info['grid'][p]) for m in info['movers'] for p in m['points']
                         if int(info['grid'][p]) not in structural | info['query'])
    filler = max(all_counts, key=lambda c: (all_counts[c], -c))
    return {c for c in counts if c not in structural | {filler}}, counts, filler


def _bridge_witness(info):
    for assembly in info['assemblies']:
        if not assembly['erase']:
            continue
        members = set(assembly['members'])
        for i in members:
            if info['movers'][i]['query_labels']:
                continue
            rest = members - {i}
            groups = []
            while rest:
                start = min(rest)
                rest.remove(start)
                group, stack = {start}, [start]
                while stack:
                    j = stack.pop()
                    for neighbor in info['adjacency'][j] & rest:
                        rest.remove(neighbor)
                        group.add(neighbor)
                        stack.append(neighbor)
                groups.append(group)
            if len(groups) > 1 and all(not info['query'] <= set().union(
                    *(info['movers'][j]['query_labels'] for j in group)) for group in groups):
                return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(_render(info), out), 'wrong assembly filtering or projection')
            _require(not np.array_equal(inp, out), 'identity scene')
            _require(any(a['erase'] for a in info['assemblies']) and
                     any(not a['erase'] for a in info['assemblies']), 'need erased and surviving assemblies')
            # Every sampled deletion is collective, never a single-piece match.
            _require(all(m['query_labels'] != info['query'] for m in info['movers']), 'individual match shortcut')
            _require(not np.array_equal(_render(info, mode='individual'), out), 'B ablation gives same answer')
            _require(any(m['delta'] != (0, 0) for a in info['assemblies'] if not a['erase']
                         for i in a['members'] for m in [info['movers'][i]]), 'no visible A witness')
            # Since input carriers are separated and incomplete, filtering before
            # projection would leave the original scene entirely unchanged.
            infos[split].append(info)
    train = infos['train']
    _require(any(_bridge_witness(i) for i in train), 'missing transitive blank bridge demonstration')
    _require({s['axis'] for i in train for s in i['systems'].values()} == {0, 1}, 'both axes must be taught')
    _require(any(not np.array_equal(_render(i, mode='guide'), _render(i)) for i in train),
             'missing same-guide disconnected counterexample')
    for shortcut in ('same_marker', 'neighbors'):
        _require(any(not np.array_equal(_render(i, mode=shortcut), _render(i)) for i in train),
                 'missing counterexample to ' + shortcut)
    # Each query identity has a near miss within at least one training scene.
    for omitted in range(3):
        _require(any(any(a['labels'] == i['query'] - {sorted(i['query'])[omitted]}
                             for a in i['assemblies']) for i in train), 'missing query-identity near miss')
    _require(any(len(a['members']) >= 2 and not a['erase'] for i in train for a in i['assemblies']),
             'no surviving multi-piece alignment witness')
    # The exact positive has three marks; a negative has just as many distinct
    # ordinary colors and more query-colored cells. Count thresholds cannot
    # replace identity containment. This check uses only returned cells.
    def ordinary_colors(i, a):
        return {int(i['grid'][p]) for j in a['members'] for p in i['movers'][j]['points']} - set(i['systems'])
    _require(any(any(p['erase'] and len(ordinary_colors(i, p)) == 4 and
                         any(not n['erase'] and len(ordinary_colors(i, n)) >= 4
                             and sum(int(i['grid'][x]) in i['query'] for j in n['members']
                                     for x in i['movers'][j]['points']) > 3
                             for n in i['assemblies']) for p in i['assemblies']) for i in train),
             'missing color-count and repeated-query counterexamples')
    _require(all(max(len(a['members']) for a in i['assemblies']) >= 4 for i in infos['test']),
             'test must extend assembly length')


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.int8).tobytes())
    return h.hexdigest()


def _extension_case():
    """Hand-specified mixed-axis, three-guide and diagonal-contact regression."""
    inp = np.zeros((16, 18), np.int8)
    inp[3, :4] = 8
    inp[:4, 3] = 8
    inp[0, 0], inp[1, 1], inp[2, 2] = 1, 2, 3
    inp[8, 0], inp[0, 11], inp[0, 13] = 4, 5, 6
    out = inp.copy()
    # Their targets touch across columns 8/9 and collectively contain 1,2,3.
    inp[5:7, 7:9] = [[1, 7], [4, 7]]
    inp[7:9, 12:15] = [[2, 3, 7], [7, 7, 5]]
    # Its target touches the erased record diagonally, never along an edge.
    inp[9:11, 15:17] = [[1, 7], [7, 6]]
    out[9:11, 12:14] = [[1, 7], [7, 6]]
    return {'input': inp, 'output': out}


def self_test():
    """42 supplied seeds, semantic mutations and metamorphisms; memory bounded.

    The small returned summary is useful to a caller, but this function saves no
    payloads, logs, traces, checkpoints, or temporary files.
    """
    shapes, layouts, signatures, palettes, fragment_shapes = set(), set(), set(), set(), set()
    sizes, systems, degrees, guide_failures = set(), set(), set(), 0
    selector_changes = anchor_changes = 0
    positives = negatives = bridges = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if seed in HOST_SEEDS[:3]:
            _require(_fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle), 'nondeterminism')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(info['grid'].shape)
                systems.add(len(info['systems']))
                palettes.add((info['background'], info['frame'], tuple(sorted(info['systems'])), tuple(sorted(info['query']))))
                layouts.add((info['grid'].shape, tuple(sorted(tuple(sorted(m['points'])) for m in info['movers']))))
                bridges += _bridge_witness(info)
                guide_failures += not np.array_equal(_render(info, mode='guide'), pair['output'])
                for a in info['assemblies']:
                    sizes.add(len(a['members']))
                    positives += a['erase']
                    negatives += not a['erase']
                for i, m in enumerate(info['movers']):
                    r0, c0 = min(r for r, _ in m['points']), min(c for _, c in m['points'])
                    fragment_shapes.add(tuple(sorted((r - r0, c - c0) for r, c in m['points'])))
                    degrees.add(len(info['adjacency'][i]))
        # Whole-scene recoloring and D4 transformations preserve the rule.
        pair = puzzle['train'][0]
        rng = np.random.default_rng(seed + 91)
        palette = rng.permutation(10)
        turns = seed % 4
        changed = palette[np.rot90(pair['input'], turns)[:, ::-1]]
        expected = palette[np.rot90(pair['output'], turns)[:, ::-1]]
        _require(np.array_equal(solve(changed), expected), 'geometric/color covariance')
        # Moving selector query cells changes no record membership.
        original = pair['input']
        corner = next(c for c in range(4) if int(_view(original, c)[3, 3]) == _decode(original)['frame']
                      and np.all(_view(original, c)[3, :4] == _decode(original)['frame'])
                      and np.all(_view(original, c)[:4, 3] == _decode(original)['frame']))
        changed, expected = original.copy(), pair['output'].copy()
        _view(changed, corner)[:3, :3] = np.roll(_view(changed, corner)[:3, :3].copy(), 1, axis=0)
        _view(expected, corner)[:3, :3] = _view(changed, corner)[:3, :3]
        _require(np.array_equal(solve(changed), expected), 'selector positions wrongly matter')
        # Change only the selector to the labels of a former three-label near
        # miss. Candidate coordinates, guides and all their colors stay fixed.
        found = False
        for pair in puzzle['train']:
            info = _decode(pair['input'])
            for assembly in info['assemblies']:
                labels, _, _ = _assembly_labels(info, assembly)
                if not assembly['erase'] and len(labels) == 3 and len(labels & info['query']) == 2:
                    changed = pair['input'].copy()
                    positions = sorted(p for p in info['selector'] if int(changed[p]) in info['query'])
                    for p, c in zip(positions, sorted(labels)):
                        changed[p] = c
                    new_info = _decode(changed)
                    old_decisions = {frozenset(a['points']): a['erase'] for a in info['assemblies']}
                    new_decisions = {frozenset(a['points']): a['erase'] for a in new_info['assemblies']}
                    _require(old_decisions != new_decisions, 'selector does not affect membership')
                    _require(next(a for a in new_info['assemblies'] if a['points'] == assembly['points'])['erase'],
                             'new selector failed to erase former near miss')
                    selector_changes += 1
                    found = True
                    break
            if found:
                break
        _require(found, 'no selector counterfactual')
        # Move one boundary guide, keeping all material and query cells fixed.
        # In a two-system scene this can separate the former cooperative match.
        found = False
        for pair in puzzle['train']:
            info = _decode(pair['input'])
            if not _bridge_witness(info):
                continue
            for color, system in info['systems'].items():
                for shift in (-5, 5, -6, 6):
                    changed = pair['input'].copy()
                    moved = {(r + shift, c) if system['axis'] == 0 else (r, c + shift)
                             for r, c in system['anchors']}
                    if any(not (0 <= r < changed.shape[0] and 0 <= c < changed.shape[1])
                           or int(changed[r, c]) != info['background'] for r, c in moved):
                        continue
                    for p in system['anchors']:
                        changed[p] = info['background']
                    for p in moved:
                        changed[p] = color
                    try:
                        new_info = _decode(changed)
                    except AssertionError:
                        continue
                    old_erased = set().union(*(a['members'] for a in info['assemblies'] if a['erase']))
                    new_erased = set().union(*(a['members'] for a in new_info['assemblies'] if a['erase']))
                    if old_erased != new_erased:
                        anchor_changes += 1
                        found = True
                        break
                if found:
                    break
            if found:
                break
    _require(len(signatures) == 42 and len(layouts) == 210, 'repeated puzzle or layout')
    _require(len(shapes) >= 45 and len(fragment_shapes) >= 150 and len(palettes) >= 60, 'insufficient structural variation')
    _require({1, 2} <= systems and {1, 2, 3, 4, 5} <= sizes, 'assembly/system count coverage')
    _require(3 in degrees and bridges >= 42 and guide_failures >= 42, 'contact graph/shortcut coverage')
    _require(selector_changes == 42 and anchor_changes >= 20, 'control counterfactual coverage')
    extension = _extension_case()
    _require(np.array_equal(solve(extension['input']), extension['output']),
             'mixed-axis or diagonal-contact regression')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(812, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Output mutations must be rejected by the cell-derived validator.
    puzzle = generate_puzzle(42, 2, 1)
    for mode in ('control', 'survivor', 'deleted'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        pair = bad['test'][0]
        info = _decode(pair['input'])
        if mode == 'control':
            p = min(info['anchors'])
            pair['output'][p] = info['background']
        else:
            a = next(a for a in info['assemblies'] if a['erase'] == (mode == 'deleted'))
            p = min(a['points'])
            pair['output'][p] = info['background'] if mode == 'survivor' else next(iter(info['query']))
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'distinct_layouts': len(layouts), 'canvas_shapes': len(shapes),
            'binary_fragment_shapes': len(fragment_shapes), 'role_assignments': len(palettes),
            'assembly_sizes': sorted(sizes), 'system_counts': sorted(systems),
            'contact_degrees': sorted(degrees), 'blank_bridge_scenes': bridges,
            'selector_counterfactuals': selector_changes, 'anchor_counterfactuals': anchor_changes,
            'mixed_axis_diagonal_case': True,
            'guide_pooling_failures': guide_failures, 'erased_records': positives,
            'surviving_records': negatives}
