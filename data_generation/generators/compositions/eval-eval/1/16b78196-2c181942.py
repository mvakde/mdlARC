"""Inward port docking constrained by complementary cavity completion."""
from __future__ import annotations

import numpy as np

NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        components.append(np.array(sorted(cells), dtype=int))
    return components


def _turn(cells, turns):
    result = cells.copy()
    for _ in range(turns % 4):
        result = np.column_stack((-result[:, 1], result[:, 0]))
    return result - result.min(axis=0)


def _parse(value):
    a = _grid(value)
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("background must be a strict majority")
    components = _components(a, bg)
    frames = []
    for i, cells in enumerate(components):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        r, c = map(int, lo)
        h, w = map(int, hi - lo + 1)
        if min(h, w) < 6:
            continue
        block = a[r:r + h, c:c + w]
        edge = np.zeros(block.shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        corners = [int(block[x, y]) for x, y in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))]
        if np.all(block[edge] != bg) and len(set(corners)) == 1 and np.any(block[1:-1, 1:-1] == bg):
            frames.append((i, r, c, h, w, corners[0]))
    if len(frames) != 1:
        raise ValueError("expected one enclosing mold")
    fi, r, c, h, w, wall = frames[0]
    block = a[r:r+h, c:c+w]
    gates = []
    side_cells = ([(0, x) for x in range(1, w-1)],
                  [(x, w-1) for x in range(1, h-1)],
                  [(h-1, x) for x in range(1, w-1)],
                  [(x, 0) for x in range(1, h-1)])
    inward = ((1, 0), (0, -1), (-1, 0), (0, 1))
    outer_label_cells = set()
    for side, line in enumerate(side_cells):
        k = 0
        while k < len(line):
            color = int(block[line[k]])
            if color == wall:
                k += 1
                continue
            end = k + 1
            while end < len(line) and int(block[line[end]]) == color:
                end += 1
            if end-k != 2:
                raise ValueError("gates must be two-cell bars")
            if (k == 0 or end == len(line) or int(block[line[k-1]]) != wall
                    or int(block[line[end]]) != wall):
                raise ValueError("gates require a wall cell on both ends")
            port = np.array(line[k:end], dtype=int) + (r, c)
            target = port + inward[side]
            if np.any(a[target[:, 0], target[:, 1]] != bg):
                raise ValueError("gate mouths must be empty")
            outer_label_cells.update(map(tuple, port))
            gates.append(dict(side=side, color=color, port=port, target=target))
            k = end
    if not 2 <= len(gates) <= 6:
        raise ValueError("expected 2..6 gates")
    frame_cells = components[fi]
    if any(int(a[tuple(p)]) != wall and tuple(p) not in outer_label_cells for p in frame_cells):
        raise ValueError("non-wall mold cells must belong to perimeter gates")
    cavity = set(map(tuple, np.argwhere(block == bg) + (r, c)))
    if not cavity:
        raise ValueError("empty cavity")
    labels = {g['color'] for g in gates}
    pieces = []
    occupied = set(map(tuple, frame_cells))
    for i, cells in enumerate(components):
        if i == fi:
            continue
        if any(r <= x < r+h and c <= y < c+w for x, y in cells):
            raise ValueError("loose fragments must be outside the mold")
        if any((x+dr, y+dc) in occupied for x, y in cells
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            raise ValueError("loose fragments require eight-neighbor clearance")
        occupied.update(map(tuple, cells))
        colors, counts = np.unique(a[cells[:, 0], cells[:, 1]], return_counts=True)
        if len(colors) != 2 or np.count_nonzero(counts == 2) != 1 or max(counts) < 3:
            raise ValueError("a fragment needs one two-cell tag and a larger monochrome body")
        tag_color = int(colors[np.flatnonzero(counts == 2)[0]])
        body_color = int(colors[np.argmax(counts)])
        if body_color in labels or body_color in (bg, wall) or tag_color in (bg, wall):
            raise ValueError("tag, body, background and wall roles must be separate")
        tag = a[cells[:, 0], cells[:, 1]] == tag_color
        tc = cells[tag]
        if np.abs(tc[0] - tc[1]).sum() != 1:
            raise ValueError("tag must be adjacent")
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        sides = []
        for side, axis, v in ((0, 0, lo[0]), (1, 1, hi[1]), (2, 0, hi[0]), (3, 1, lo[1])):
            extreme = cells[:, axis] == v
            if np.array_equal(extreme, tag):
                sides.append(side)
        if len(sides) != 1:
            raise ValueError("tag must be the entire unique extreme edge")
        pieces.append(dict(cells=cells, local=cells-lo, tag=tag, tag_color=tag_color,
                           body_color=body_color, source_side=sides[0],
                           colors=a[cells[:, 0], cells[:, 1]].copy()))
    if not pieces:
        raise ValueError("no loose fragments")
    base = np.full_like(a, bg)
    base[frame_cells[:, 0], frame_cells[:, 1]] = a[frame_cells[:, 0], frame_cells[:, 1]]
    cavity_bits = sum(1 << (int(x)*a.shape[1]+int(y)) for x, y in cavity)
    return dict(grid=a, background=bg, wall=wall, frame_cells=frame_cells,
                bbox=(r, c, h, w), gates=gates, pieces=pieces, labels=labels,
                cavity=cavity, cavity_bits=cavity_bits, base=base)


def _candidates(scene, use_labels=True):
    by_gate = [[] for _ in scene['gates']]
    for i, p in enumerate(scene['pieces']):
        for j, g in enumerate(scene['gates']):
            if use_labels and p['tag_color'] != g['color']:
                continue
            turns = (p['source_side'] - g['side']) % 4
            rotated = _turn(p['local'], turns)
            coords = rotated + g['target'].min(axis=0) - rotated[p['tag']].min(axis=0)
            if set(map(tuple, coords[p['tag']])) != set(map(tuple, g['target'])):
                continue
            if not set(map(tuple, coords)) <= scene['cavity']:
                continue
            bits = sum(1 << (int(x)*scene['grid'].shape[1]+int(y)) for x, y in coords)
            by_gate[j].append(dict(piece=i, gate=j, turns=turns, coords=coords, bits=bits))
    return by_gate


def _search(scene, use_labels=True, enforce_packing=True, limit=2):
    by_gate = _candidates(scene, use_labels)
    matched = {i for i, p in enumerate(scene['pieces']) if p['tag_color'] in scene['labels']}
    if use_labels and len(matched) != len(scene['gates']):
        return []
    if use_labels and enforce_packing and sum(len(scene['pieces'][i]['cells']) for i in matched) != len(scene['cavity']):
        return []
    solutions = {}
    full_gates = (1 << len(by_gate)) - 1

    def visit(used_gates, used_pieces, occupied, chosen):
        if len(solutions) >= limit:
            return
        if used_gates == full_gates:
            if enforce_packing and occupied != scene['cavity_bits']:
                return
            out = scene['base'].copy()
            for candidate in sorted(chosen, key=lambda x: x['gate']):
                coords = candidate['coords']
                out[coords[:, 0], coords[:, 1]] = scene['pieces'][candidate['piece']]['colors']
            solutions.setdefault(out.tobytes(), (out, chosen))
            return
        available = []
        for j, options in enumerate(by_gate):
            if used_gates & (1 << j):
                continue
            valid = [v for v in options if not used_pieces & (1 << v['piece'])
                     and (not enforce_packing or not occupied & v['bits'])]
            available.append((j, valid))
        j, options = min(available, key=lambda x: len(x[1]))
        for candidate in options:
            visit(used_gates | (1 << j), used_pieces | (1 << candidate['piece']),
                  occupied | candidate['bits'], chosen + [candidate])

    visit(0, 0, 0, [])
    return list(solutions.values())


def _topology(scene):
    """Gate ordering and color equality, independent of literal color values."""
    roles = {}
    return tuple((g['side'], roles.setdefault(g['color'], len(roles)))
                 for g in scene['gates'])


def _crossed_or_repeated_wall(scene):
    gates = scene['gates']
    return (len({g['side'] for g in gates}) < len(gates)
            or any(g['color'] == q['color'] and g['side'] % 2 != q['side'] % 2
                   for i, g in enumerate(gates) for q in gates[:i]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _parse(input_grid)
    solutions = _search(scene)
    if len(solutions) != 1:
        raise ValueError("mold must have exactly one completed colored assembly")
    return solutions[0][0].copy()


def _dilate(mask):
    result = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ra, rb = max(0, -dr), min(h, h-dr)
            ca, cb = max(0, -dc), min(w, w-dc)
            result[ra+dr:rb+dr, ca+dc:cb+dc] |= mask[ra:rb, ca:cb]
    return result


def _gate_geometry(side, start, h, w):
    if side == 0:
        return [(0, start), (0, start+1)], [(1, start), (1, start+1)], [(2, start), (2, start+1)]
    if side == 1:
        return [(start, w-1), (start+1, w-1)], [(start, w-2), (start+1, w-2)], [(start, w-3), (start+1, w-3)]
    if side == 2:
        return [(h-1, start), (h-1, start+1)], [(h-2, start), (h-2, start+1)], [(h-3, start), (h-3, start+1)]
    return [(start, 0), (start+1, 0)], [(start, 1), (start+1, 1)], [(start, 2), (start+1, 2)]


def _partition(rng, central, seeds):
    assigned = {}
    bodies = []
    for i, points in enumerate(seeds):
        if not set(points) <= central or set(points) & set(assigned):
            raise ValueError("colliding gate seeds")
        bodies.append(set(points))
        for p in points:
            assigned[p] = i
    unused = central - set(assigned)
    while unused:
        progress = False
        for i in rng.permutation(len(bodies)):
            options = sorted({(r+dr, c+dc) for r, c in bodies[i] for dr, dc in NEIGHBORS} & unused)
            if options:
                p = options[int(rng.integers(len(options)))]
                bodies[i].add(p)
                unused.remove(p)
                progress = True
        if not progress:
            raise ValueError("unreachable cavity cells")
    if min(map(len, bodies)) < 3:
        raise ValueError("body too small to distinguish its tag")
    return bodies


def _scatter(rng, block, targets, bg, deltas):
    objects = []
    for (cells, values), delta in zip(targets, deltas):
        local = _turn(cells, -int(delta))
        objects.append((local, values.copy()))
    bh, bw = block.shape
    initial = (int(rng.integers(max(18, bh+7), 29)), int(rng.integers(max(18, bw+7), 29)))
    for attempt in range(24):
        shape = tuple(min(30, x + (attempt // 6)*2) for x in initial)
        a = np.full(shape, bg, dtype=np.int8)
        # Try both arbitrary interior positions and boundary-biased positions
        # from the first attempt, rather than reaching interior positions only
        # after a failed corner scatter.
        position_mode = int(rng.integers(3))
        if position_mode == 0:
            r = int(rng.choice((0, shape[0]-bh)))
            c = int(rng.choice((0, shape[1]-bw)))
        else:
            r = int(rng.integers(shape[0]-bh+1))
            c = int(rng.integers(shape[1]-bw+1))
            if position_mode == 1:
                if rng.integers(2):
                    r = int(rng.choice((0, shape[0]-bh)))
                else:
                    c = int(rng.choice((0, shape[1]-bw)))
        a[r:r+bh, c:c+bw] = block
        occupied = np.zeros(shape, dtype=bool)
        occupied[r:r+bh, c:c+bw] = True  # No source fragment may start inside the cavity.
        blocked = _dilate(occupied)
        order = list(rng.permutation(len(objects)))
        order.sort(key=lambda i: -np.prod(objects[i][0].max(axis=0)+1))
        good = True
        for i in order:
            local, values = objects[i]
            ph, pw = local.max(axis=0) + 1
            if ph > shape[0] or pw > shape[1]:
                good = False
                break
            valid = np.ones((shape[0]-ph+1, shape[1]-pw+1), dtype=bool)
            for rr, cc in local:
                valid &= ~blocked[rr:rr+valid.shape[0], cc:cc+valid.shape[1]]
            positions = np.argwhere(valid)
            if not len(positions):
                good = False
                break
            coords = local + positions[int(rng.integers(len(positions)))]
            a[coords[:, 0], coords[:, 1]] = values
            occupied[coords[:, 0], coords[:, 1]] = True
            blocked = _dilate(occupied)
        if good:
            return a, (r, c)
    raise ValueError("scatter space exhausted")


def _make_pair(rng, config, palette):
    hard = config['hard']
    n = config['count']
    if hard:
        h, w = map(int, rng.choice((8, 10, 12), 2))
        # Opposite centered mouths give the repeated-key fragments two legal
        # individual destinations. The other fragments break the global swap.
        specs = [(0, w//2-1, 0), (2, w//2-1, 0)]
        anchors = [3 if config.get('left', True) else 1] if n == 3 else [1, 3]
        specs.extend((s, h//2-1, i+1) for i, s in enumerate(anchors))
        label_count = n-1
    else:
        h, w = map(int, rng.integers(9 if n <= 4 else 11, 15, size=2))
        label_count = int(rng.integers(1 if n <= 3 else 2, min(3, n-1)+1))
        choices = [(side, start) for side in range(4)
                   for start in range(2, (w if side % 2 == 0 else h)-3)]
        rng.shuffle(choices)
        specs = []
        for side, start in choices:
            if all(side != s or abs(start-x) >= 3 for s, x, _ in specs):
                specs.append((side, start, len(specs) % label_count))
            if len(specs) == n:
                break
        if len(specs) < n:
            raise ValueError("gate spacing")
        keys = list(range(label_count)) + list(map(int, rng.integers(label_count, size=n-label_count)))
        rng.shuffle(keys)
        specs = [(side, start, key) for (side, start, _), key in zip(specs, keys)]
    bg, wall = palette[:2]
    labels = palette[2:2+label_count]
    body_pool = palette[2+label_count:-1]
    absent = palette[-1]
    body_colors = [body_pool[i % len(body_pool)] for i in rng.permutation(n)]
    if hard and body_colors[0] == body_colors[1]:
        body_colors[1] = body_pool[(body_pool.index(body_colors[0])+1) % len(body_pool)]
    block = np.full((h, w), wall, dtype=np.int8)
    central = {(r, c) for r in range(2, h-2) for c in range(2, w-2)}
    # Connected inward wall projections are paired under 180 degrees in hard
    # witnesses, retaining local opposite-port feasibility.
    if config.get('pegs', False):
        depth = int(rng.integers(1, 3))
        removed = {(2+k, 2) for k in range(depth)}
        if hard:
            removed |= {(h-1-r, w-1-c) for r, c in removed}
        elif rng.random() < .5:
            removed |= {(h-3, w-3-k) for k in range(depth)}
        central -= removed
    for p in central:
        block[p] = bg
    necks, seeds = [], []
    for side, start, key in specs:
        port, neck, body_seed = _gate_geometry(side, start, h, w)
        for p in port:
            block[p] = labels[key]
        for p in neck:
            block[p] = bg
        necks.append(neck)
        seeds.append(body_seed)
    bodies = _partition(rng, central, seeds)
    targets, masks = [], []
    output_block = block.copy()
    for i, body in enumerate(bodies):
        cells = np.array(sorted(body | set(necks[i])), dtype=int)
        values = np.array([labels[specs[i][2]] if tuple(p) in set(necks[i]) else body_colors[i] for p in cells], dtype=np.int8)
        mask = np.zeros((h, w), dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        masks.append(mask)
        output_block[cells[:, 0], cells[:, 1]] = values
    if np.any(output_block == bg):
        raise ValueError("construction left a cavity gap")
    mirror = bool(rng.integers(2))
    turns = config.get('scene_turns', int(rng.integers(4)))
    if mirror:
        block, output_block = block[:, ::-1], output_block[:, ::-1]
        masks = [m[:, ::-1] for m in masks]
    block, output_block = np.rot90(block, turns), np.rot90(output_block, turns)
    masks = [np.rot90(m, turns) for m in masks]
    for mask in masks:
        cells = np.argwhere(mask)
        targets.append((cells, output_block[cells[:, 0], cells[:, 1]]))
    deltas = list(config.get('deltas', rng.integers(4, size=n)))
    for _ in range(config['distractors']):
        i = int(rng.integers(n))
        cells, values = targets[i]
        extra = values.copy()
        extra[extra == labels[specs[i][2]]] = absent
        # Change the payload too, so ignoring the tag cannot be rescued by
        # erasing/recoloring tags after an otherwise identical geometric fit.
        new_body = int(rng.choice([x for x in body_pool if x != body_colors[i]]))
        extra[values == body_colors[i]] = new_body
        targets.append((cells.copy(), extra))
        deltas.append(int(rng.integers(4)))
    inp, (r, c) = _scatter(rng, block, targets, bg, deltas)
    expected = np.full_like(inp, bg)
    expected[r:r+block.shape[0], c:c+block.shape[1]] = output_block
    scene = _parse(inp)
    solutions = _search(scene)
    if len(solutions) != 1 or not np.array_equal(solutions[0][0], expected):
        raise ValueError("ambiguous or inconsistent assembly")
    if hard and len(_search(scene, enforce_packing=False)) < 2:
        raise ValueError("missing packing witness")
    return dict(input=inp, output=expected)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.integers(2))
    scene_turns = int(rng.integers(4))
    mode = int(rng.integers(2))
    configs = [dict(hard=True, count=3, distractors=1, left=True, pegs=False,
                    scene_turns=scene_turns, deltas=[0, 1, 2]),
               dict(hard=True, count=4, distractors=int(rng.integers(1, 3)), pegs=True,
                    scene_turns=scene_turns, deltas=[3, 0, 1, 2])]
    configs += [dict(hard=False, count=int(rng.integers(2, 5 if mode == 0 else 6)),
                     distractors=int(rng.integers(3)), pegs=bool(rng.integers(2)))
                for _ in range(num_train-2)]
    rng.shuffle(configs)
    configs += [dict(hard=False, count=int(rng.integers(5, 7) if mode == 0 else rng.integers(3, 6)),
                     distractors=int(rng.integers(1, 3)), pegs=bool(rng.integers(2)))
                for _ in range(num_test)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    train_topologies = set()
    for i, config in enumerate(configs):
        palette = shared if stable else tuple(map(int, rng.permutation(10)))
        errors = {}
        for _ in range(160):
            try:
                pair = _make_pair(rng, config, palette)
                key = _key(pair['input'])
                if key in seen:
                    continue
                scene = _parse(pair['input'])
                if i >= num_train and mode == 1:
                    if _topology(scene) in train_topologies or not _crossed_or_repeated_wall(scene):
                        continue
                break
            except ValueError as error:
                message = str(error)
                errors[message] = errors.get(message, 0)+1
        else:
            raise RuntimeError(f"could not construct scene: {config}, {errors}")
        seen.add(key)
        if i < num_train:
            train_topologies.add(_topology(scene))
        puzzle['train' if i < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
        raise ValueError("pair must contain exactly input/output")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    if inp.shape != out.shape or np.array_equal(inp, out):
        raise ValueError("expected a changed grid on the same canvas")
    scene = _parse(inp)
    solutions = _search(scene)
    if len(solutions) != 1 or not np.array_equal(solutions[0][0], out):
        raise ValueError("answer is not the unique complete assembly")
    _, chosen = solutions[0]
    bg = scene['background']
    r, c, h, w = scene['bbox']
    outside = np.ones(inp.shape, dtype=bool)
    outside[r:r+h, c:c+w] = False
    if np.any(out[outside] != bg) or np.any(out[r:r+h, c:c+w] == bg):
        raise ValueError("answer must be one solid mold rectangle")
    fixed = scene['frame_cells']
    if not np.array_equal(inp[fixed[:, 0], fixed[:, 1]], out[fixed[:, 0], fixed[:, 1]]):
        raise ValueError("mold changed")
    matched = {i for i, p in enumerate(scene['pieces']) if p['tag_color'] in scene['labels']}
    if {v['piece'] for v in chosen} != matched or len({v['gate'] for v in chosen}) != len(scene['gates']):
        raise ValueError("fragments and gates must correspond one to one")
    histogram = np.bincount(inp.ravel(), minlength=10)
    erased_cells = 0
    for i, p in enumerate(scene['pieces']):
        if i not in matched:
            histogram -= np.bincount(p['colors'], minlength=10)
            erased_cells += len(p['cells'])
    histogram[bg] += erased_cells
    if not np.array_equal(histogram, np.bincount(out.ravel(), minlength=10)):
        raise ValueError("selected colors or pixel counts were not preserved")
    occupied = set()
    for v in chosen:
        p = scene['pieces'][v['piece']]
        rotated = _turn(p['local'], v['turns'])
        if not np.array_equal(rotated, v['coords']-v['coords'].min(axis=0)):
            raise ValueError("fragment is not a rigid quarter-turn translation")
        target = set(map(tuple, v['coords']))
        if target & occupied or not np.array_equal(out[v['coords'][:, 0], v['coords'][:, 1]], p['colors']):
            raise ValueError("fragment cells overlap or are recolored")
        occupied |= target
    if occupied != scene['cavity']:
        raise ValueError("cavity not covered exactly")
    scene['chosen'] = chosen
    return scene


def validate_generated_puzzle(puzzle):
    """Check returned arrays, input-derived uniqueness and teaching witnesses."""
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must contain exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("illegal example counts")
    analyses = {'train': [], 'test': []}
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            scene = _analyze_pair(pair)
            key = _key(scene['grid'])
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            analyses[split].append(scene)
    train = analyses['train']
    if {g['side'] for s in train for g in s['gates']} != set(range(4)):
        raise ValueError("training must show all gate directions")
    if {v['turns'] for s in train for v in s['chosen']} != set(range(4)):
        raise ValueError("training must show all four turn amounts")
    if not any(len(_search(s, enforce_packing=False)) > 1 for s in train):
        raise ValueError("training lacks a witness requiring complementary packing")
    if not any(len(_search(s, use_labels=False)) > 1 for s in train):
        raise ValueError("training lacks a color-filtering witness")
    topologies = {_topology(s) for s in train}
    max_count = max(len(s['gates']) for s in train)
    for s in analyses['test']:
        if len(s['gates']) > max_count:
            continue
        if _topology(s) in topologies or not _crossed_or_repeated_wall(s):
            raise ValueError("test must increase gate count or recombine gate topology")


def _shape_signature(cells):
    return min((tuple(map(int, _turn(cells, t).max(axis=0)+1)),
                tuple(sorted(map(tuple, _turn(cells, t))))) for t in range(4))


def _equal(left, right):
    return all(len(left[s]) == len(right[s]) and
               all(all(np.array_equal(p[k], q[k]) for k in ('input', 'output'))
                   for p, q in zip(left[s], right[s])) for s in ('train', 'test'))


def _symmetric_fixture(same_body=False):
    """Two opposite ports: different payloads are ambiguous; equal ones are not."""
    rng = np.random.default_rng(123)
    block = np.full((8, 8), 6, dtype=np.int8)
    block[2:6, 2:6] = 0
    block[0, 3:5] = block[7, 3:5] = 2
    block[1, 3:5] = block[6, 3:5] = 0
    targets = []
    for i, rows in enumerate((range(2, 4), range(4, 6))):
        body = {(r, c) for r in rows for c in range(2, 6)}
        neck = {(1 if i == 0 else 6, 3), (1 if i == 0 else 6, 4)}
        cells = np.array(sorted(body | neck))
        values = np.array([2 if tuple(p) in neck else (3 if same_body or i == 0 else 4) for p in cells], dtype=np.int8)
        targets.append((cells, values))
    return _scatter(rng, block, targets, 0, [0, 1])[0]


def self_test():
    """Bounded 42-seed variation, semantic ablations and mutation regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, mold_shapes, silhouettes, counts, backgrounds = set(), set(), set(), set(), set()
    topologies, role_assignments, turns, body_sizes = set(), set(), set(), set()
    gate_histogram = {split: {} for split in ('train', 'test')}
    ablation_witnesses = {'packing': 0, 'color': 0}
    count_shift = topology_shift = boundary_molds = off_center_gates = projection_molds = 0
    same_wall = crossed_keys = internal_holes = non_convex = 0
    puzzle_signatures = set()
    for puzzle in puzzles:
        puzzle_signatures.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                s = _analyze_pair(pair)
                local.append(s)
                shapes.add(pair['input'].shape)
                r, c, h, w = s['bbox']
                mold_shapes.add((h, w))
                n = len(s['gates'])
                counts.add(n)
                gate_histogram[split][n] = gate_histogram[split].get(n, 0)+1
                backgrounds.add(s['background'])
                topologies.add(_topology(s))
                role_assignments.add((s['background'], s['wall'], tuple(sorted(s['labels']))))
                turns.update(v['turns'] for v in s['chosen'])
                boundary_molds += r == 0 or c == 0 or r+h == pair['input'].shape[0] or c+w == pair['input'].shape[1]
                same_wall += len({g['side'] for g in s['gates']}) < n
                crossed_keys += any(g['color'] == q['color'] and g['side'] % 2 != q['side'] % 2
                                    for i, g in enumerate(s['gates']) for q in s['gates'][:i])
                off_center_gates += any((g['target'][:, 1].mean() != c+(w-1)/2 if g['side'] % 2 == 0
                                        else g['target'][:, 0].mean() != r+(h-1)/2) for g in s['gates'])
                core = pair['input'][r+2:r+h-2, c+2:c+w-2]
                projection_molds += np.any(core == s['wall'])
                for p in s['pieces']:
                    silhouettes.add(_shape_signature(p['local']))
                    body_sizes.add(int(np.count_nonzero(~p['tag'])))
                    # Measure actual row/column nonconvexity, rather than hashes.
                    coords = p['local']
                    non_convex += any(len(v := np.sort(coords[coords[:, axis] == x, 1-axis])) > 1
                                      and np.any(np.diff(v) > 1)
                                      for axis in (0, 1) for x in np.unique(coords[:, axis]))
                    mask = np.zeros(tuple(coords.max(axis=0)+3), dtype=bool)
                    mask[coords[:, 0]+1, coords[:, 1]+1] = True
                    empty = set(map(tuple, np.argwhere(~mask)))
                    stack = [(0, 0)]
                    empty.remove((0, 0))
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in NEIGHBORS:
                            point = (x+dx, y+dy)
                            if point in empty:
                                empty.remove(point)
                                stack.append(point)
                    internal_holes += bool(empty)
                if split == 'train':
                    ablation_witnesses['packing'] += len(_search(s, enforce_packing=False)) > 1
                    ablation_witnesses['color'] += len(_search(s, use_labels=False)) > 1
                else:
                    train_max = max(len(_parse(p['input'])['gates']) for p in puzzle['train'])
                    count_shift += n > train_max
                    topology_shift += n <= train_max
        assert len({_shape_signature(p['local']) for s in local for p in s['pieces']}) >= 5
    assert len(puzzle_signatures) == 42
    assert counts == set(range(2, 7)) and turns == set(range(4))
    assert backgrounds == set(range(10))
    assert len(shapes) >= 40 and len(mold_shapes) >= 15
    assert len(silhouettes) >= 150 and len(body_sizes) >= 20
    assert len(topologies) >= 30 and len(role_assignments) >= 40
    assert ablation_witnesses['packing'] >= 84 and ablation_witnesses['color'] >= 84
    assert count_shift >= 8 and topology_shift >= 8
    assert same_wall >= 20 and crossed_keys >= 20 and non_convex >= 100
    assert 20 <= boundary_molds < 190
    assert off_center_gates >= 50 and projection_molds >= 42
    # Reject genuinely different colored completions; collapse equivalent
    # identity permutations when the complete colored answer is identical.
    ambiguous = _symmetric_fixture()
    assert len(_search(_parse(ambiguous))) == 2
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colored outputs accepted")
    assert len(_search(_parse(_symmetric_fixture(True)))) == 1
    solve(_symmetric_fixture(True))
    # Geometric and palette conventions remain entirely visible in the input.
    p = puzzles[0]['test'][0]
    for reflection in (False, True):
        for t in range(4):
            a, b = p['input'], p['output']
            if reflection:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(np.rot90(a, t)), np.rot90(b, t))
    palette = np.random.default_rng(889).permutation(10)
    assert np.array_equal(solve(palette[p['input']]), palette[p['output']])
    first = generate_puzzle(991, 6, 3)
    assert _equal(first, generate_puzzle(991, 6, 3))
    assert not _equal(first, generate_puzzle(992, 6, 3))
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(2000+a, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Moved shape corruption and a changed fixed wall must both be caught.
    for location in ('wall', 'body'):
        damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()}
                       for pair in first[s]] for s in ('train', 'test')}
        pair = damaged['test'][0]
        scene = _parse(pair['input'])
        candidate = _search(scene)[0][1][0]
        fragment = scene['pieces'][candidate['piece']]
        cell = (tuple(scene['frame_cells'][0]) if location == 'wall'
                else tuple(candidate['coords'][np.flatnonzero(~fragment['tag'])[0]]))
        pair['output'][cell] = scene['background']
        try:
            validate_generated_puzzle(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("damaged answer accepted")
    report = dict(seeds=42, pairs=210, canvas_shapes=len(shapes), mold_shapes=len(mold_shapes),
                  rotation_normalized_silhouettes=len(silhouettes), gate_counts=gate_histogram,
                  gate_topologies=len(topologies), role_assignments=len(role_assignments),
                  body_sizes=sorted(body_sizes), ablation_witnesses=ablation_witnesses,
                  count_shift_tests=count_shift, topology_shift_tests=topology_shift,
                  same_wall_scenes=same_wall, crossed_key_scenes=crossed_keys,
                  non_convex_fragments=non_convex, internally_holed_fragments=internal_holes,
                  boundary_molds=boundary_molds, off_center_gate_scenes=off_center_gates,
                  wall_projection_scenes=int(projection_molds),
                  semantic_verification='generator-owned only')
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
