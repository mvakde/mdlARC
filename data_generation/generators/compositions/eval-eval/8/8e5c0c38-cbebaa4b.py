"""Symmetry chooses an anchored docking assembly and erases unsupported payload.

Every loose wing has two pins. All wings dock directly to the beaded root,
whose long axis is the reflection axis. Exact pin cover gives candidate rigid
translations; maximum same-color mirror support chooses between them. Unlike
local denoising followed by assembly, a wing's support comes from another wing
whose position is still unknown. Only the root contains color 9.
"""
from __future__ import annotations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g.astype(np.int8, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    component.add(q)
        result.append(component)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in N4} - set(cells)


def _buffer(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _reflect(q, orientation, axis2):
    r, c = q
    return (r, axis2 - c) if orientation == 0 else (axis2 - r, c)


def _parse(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size / 2 or bg == 9:
        raise ValueError("the background must be a majority and cannot be the bead")
    cc = _components(map(tuple, np.argwhere(grid != bg).tolist()))
    if len(cc) < 5:
        raise ValueError("expected a root and at least four wings")
    palettes = [{int(grid[q]) for q in part} for part in cc]
    common = set.intersection(*palettes)
    if len(common) != 1:
        raise ValueError("no unique common pin color")
    marker = common.pop()
    roots = [i for i, colors in enumerate(palettes) if 9 in colors]
    if len(roots) != 1 or marker == 9:
        raise ValueError("exactly one root must carry the bead")
    root_index = roots[0]
    root_cells = cc[root_index]
    root_pins = {q for q in root_cells if int(grid[q]) == marker}
    root_body = root_cells - root_pins
    root_colors = palettes[root_index] - {marker, 9}
    if len(root_colors) != 1:
        raise ValueError("root must have one material color and a bead")
    root_color = next(iter(root_colors))
    if len(_components(root_body)) != 1:
        raise ValueError("root body must be connected")
    t, l, b, r = _box(root_body)
    if b - t == r - l:
        raise ValueError("the root must have a unique long direction")
    orientation = 0 if b - t > r - l else 1
    axis2 = l + r if orientation == 0 else t + b
    if {_reflect(q, orientation, axis2) for q in root_body} != root_body:
        raise ValueError("the root body must be symmetric about its long axis")
    if any(grid[_reflect(q, orientation, axis2)] != grid[q] for q in root_body):
        raise ValueError("root bead and material must also reflect by color")
    wings = []
    for i, cells in enumerate(cc):
        if i == root_index:
            continue
        colors = palettes[i] - {marker}
        if len(colors) != 1 or colors & {root_color, 9}:
            raise ValueError("each wing must have one non-root payload color")
        color = next(iter(colors))
        body = {q for q in cells if int(grid[q]) == color}
        pins = cells - body
        if len(pins) != 2 or len(_components(body)) != 1:
            raise ValueError("a connected wing must carry exactly two pins")
        wings.append(dict(body=body, pins=pins, color=color))
    if len(root_pins) != 2 * len(wings):
        raise ValueError("each root pin must be paired with one wing pin")
    if len(wings) not in (4, 6, 8):
        raise ValueError("expected four, six, or eight wings")
    return dict(bg=bg, marker=marker, root_body=root_body, root_pins=root_pins,
                root_color=root_color, orientation=orientation, axis2=axis2,
                wings=wings)


def _docking_search(grid):
    """Exhaust all collision-free root dockings, then compare mirror support.

    Pin pairs are not identified by connectivity, hue, area or a slot index.
    Enumerating every exact translated subset also considers accidental keys
    made from pins belonging to two different intended sockets.
    """
    info = _parse(grid)
    root_body, root_pins = info['root_body'], info['root_pins']
    h, w = grid.shape
    options = []
    for wing in info['wings']:
        choices = []
        a = min(wing['pins'])
        for q in sorted(root_pins):
            d = (q[0] - a[0], q[1] - a[1])
            pins = _shift(wing['pins'], d)
            if not pins <= root_pins:
                continue
            body = _shift(wing['body'], d)
            if body & (root_body | root_pins):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in body | pins):
                continue
            choices.append((d, body, pins))
        if not choices:
            raise ValueError("a wing has no legal docking translation")
        options.append(choices)
    n = len(options)
    placed = {}
    count, nodes, best_score, runner_up = 0, 0, -1, -1
    winners = []

    def visit(used_pins, occupied):
        nonlocal count, nodes, best_score, runner_up, winners
        nodes += 1
        if nodes > 100000:
            raise ValueError("docking search exceeded its complete-search budget")
        if len(placed) == n:
            if used_pins != root_pins:
                return
            count += 1
            by_color = {}
            for i, (_, body, _) in placed.items():
                by_color.setdefault(info['wings'][i]['color'], set()).update(body)
            keep = {color: {q for q in cells
                            if _reflect(q, info['orientation'], info['axis2']) in cells}
                    for color, cells in by_color.items()}
            score = sum(map(len, keep.values()))
            if score > best_score:
                runner_up = best_score
                best_score, winners = score, []
            elif score < best_score:
                runner_up = max(runner_up, score)
            if score == best_score:
                winners.append((dict(placed), keep))
            return
        selected, available = None, None
        for i, choices in enumerate(options):
            if i in placed:
                continue
            legal = [item for item in choices
                     if not (item[2] & used_pins or item[1] & occupied)]
            if not legal:
                return
            if available is None or len(legal) < len(available):
                selected, available = i, legal
        for item in available:
            placed[selected] = item
            visit(used_pins | item[2], occupied | item[1])
            del placed[selected]

    visit(set(), set())
    if not winners or len(winners) != 1:
        raise ValueError("no unique maximum-support docking arrangement")
    positions, keep = winners[0]
    out = np.full(grid.shape, info['bg'], dtype=np.int8)
    for q in root_body:
        out[q] = grid[q]
    for q in root_pins:
        out[q] = info['marker']
    for color, cells in keep.items():
        for q in cells:
            out[q] = color
    info.update(positions=positions, keep=keep, candidate_count=count,
                score=best_score, runner_up=runner_up, search_nodes=nodes,
                discarded=sum(len(x['body']) for x in info['wings']) - best_score)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input only: no demonstrations, seed, cache, or generation state."""
    return _docking_search(_grid(input_grid))[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _core(rng, height, width):
    # The inward-facing column keeps every sparse pin attached to the payload.
    inward = {(r, width - 1) for r in range(height)}
    kind = int(rng.integers(5))
    if kind == 0:
        return {(r, c) for r in range(height) for c in range(width)}
    if kind == 1:
        return {(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)}
    if kind == 2:
        rows = {0, height - 1, int(rng.integers(height))}
        return inward | {(r, c) for r in rows for c in range(width)}
    if kind == 3:
        # Varying row lengths create steps, fingers and concave outlines.
        return {(r, c) for r in range(height)
                for c in range(int(rng.integers(width)), width)}
    body = set(inward)
    target = int(rng.integers(height, height * width + 1))
    universe = {(r, c) for r in range(height) for c in range(width)}
    while len(body) < target:
        body.add(_pick(rng, sorted(_neighbors(body) & universe)))
    return body


def _scatter(rng, shape, root_map, wing_maps, bg):
    h, w = shape
    # Scatter order is randomized, then large pieces are packed first.
    for _ in range(32):
        forbidden = _buffer(root_map)
        placed = []
        order = list(range(len(wing_maps)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(wing_maps[i]))
        for i in order:
            mapping = wing_maps[i]
            t, l, b, r = _box(mapping)
            success = False
            for _ in range(320):
                d = (int(rng.integers(h - (b - t))) - t,
                     int(rng.integers(w - (r - l))) - l)
                moved = _shift(mapping, d)
                if d == (0, 0) or moved & forbidden:
                    continue
                placed.append({(q[0] + d[0], q[1] + d[1]): color
                               for q, color in mapping.items()})
                forbidden |= _buffer(moved)
                success = True
                break
            if not success:
                break
        if len(placed) == len(wing_maps):
            inp = np.full(shape, bg, dtype=np.int8)
            for mapping in [root_map, *placed]:
                for q, color in mapping.items():
                    inp[q] = color
            return inp
    return None


def _construct(rng, pairs, roles, same_color, noise_high):
    bg, marker, root_color = roles
    palette = [c for c in range(9) if c not in roles]
    base = int(rng.integers(1, 3))  # odd/even axis parity, independently sampled
    # One side has distinct geometric keys; the opposite side repeats a key.
    gaps = list(map(int, rng.permutation(np.arange(1, pairs + 2))))
    repeated_gap, distinct_gaps = gaps[0], gaps[1:]
    heights = [int(rng.integers(max(2, g + 1, repeated_gap + 1), 8))
               for g in distinct_gaps]
    separators = [int(rng.integers(1, 3)) for _ in range(pairs - 1)]
    # At four pairs the available canvas requires thinner necks.
    if sum(heights) + sum(separators) > 28:
        separators = [1] * (pairs - 1)
    if sum(heights) + sum(separators) > 28:
        return None
    top_pad, bottom_pad = int(rng.integers(2)), int(rng.integers(2))
    root_h = top_pad + sum(heights) + sum(separators) + bottom_pad
    if root_h > 30:
        return None
    root_body = {(r, c) for r in range(root_h) for c in range(base)}
    root_pins, wings, true_bodies = set(), [], []
    material = _pick(rng, palette)
    cursor = top_pad
    shapes = set()
    for index, (height, gap) in enumerate(zip(heights, distinct_gaps)):
        extra = int(rng.integers(3))
        root_body |= {(r, c) for r in range(cursor, cursor + height)
                      for c in range(-extra, base + extra)}
        if extra == 2 and rng.random() < .4:
            # Paired cavities preserve both the center spine and outer walls.
            root_body -= {(r, c) for r in range(cursor + 1, cursor + height - 1)
                          for c in (-1, base)}
        width = int(rng.integers(1, 6))
        local = _core(rng, height, width)
        # Same-color wings must still be distinguishable by their geometry.
        signature = tuple(sorted(local))
        if signature in shapes:
            return None
        shapes.add(signature)
        color = material if same_color else _pick(rng, palette)
        left_core = {(cursor + r, -extra - 1 - width + c) for r, c in local}
        right_core = {_reflect(q, 0, base - 1) for q in left_core}
        start_l = int(rng.integers(height - gap))
        start_r = int(rng.integers(height - repeated_gap))
        lp = {(cursor + start_l, -extra - 1),
              (cursor + start_l + gap, -extra - 1)}
        rp = {(cursor + start_r, base + extra),
              (cursor + start_r + repeated_gap, base + extra)}
        root_pins |= lp | rp
        for body, pins in ((left_core, lp), (right_core, rp)):
            wings.append(dict(body=set(body), pins=pins, color=color))
            true_bodies.append(set(body))
        if index < pairs - 1:
            cursor += height + separators[index]
    # A transverse bead sits in a neck when available, visibly marking the root.
    bead_row = top_pad + heights[0]
    if bead_row >= root_h:
        bead_row = root_h // 2
    bead = {(bead_row, c) for c in range(base)}
    root_map = {q: (9 if q in bead else root_color) for q in root_body}
    root_map.update({q: marker for q in root_pins})
    occupied = root_body | root_pins | set().union(*true_bodies)
    for i, wing in enumerate(wings):
        # Additions may fill holes or form attached outer protrusions. Partners
        # are forbidden even when other wings use the same payload color.
        amount = int(rng.integers(noise_high + 1))
        if i == 0:
            amount = max(1, amount)
        for _ in range(amount):
            choices = []
            for q in sorted(_neighbors(wing['body']) - occupied):
                reflected = _reflect(q, 0, base - 1)
                if any(reflected in z['body'] and z['color'] == wing['color'] for z in wings):
                    continue
                # Keep vertical extents inside their own layer. Geometry can
                # vary freely within the layer, including its enclosed holes.
                t, _, b, _ = _box(true_bodies[i])
                if t <= q[0] <= b:
                    choices.append(q)
            if not choices:
                break
            q = _pick(rng, choices)
            wing['body'].add(q)
            occupied.add(q)
    t, l, b, r = _box(occupied)
    extent_h, extent_w = b - t + 1, r - l + 1
    h = min(30, max(18, extent_h) + int(rng.integers(0, 7)))
    w = min(30, max(22, extent_w) + int(rng.integers(0, 7)))
    offset = (int(rng.integers(h - extent_h + 1)) - t,
              int(rng.integers(w - extent_w + 1)) - l)
    root_map = {(q[0] + offset[0], q[1] + offset[1]): c for q, c in root_map.items()}
    wing_maps = []
    for wing in wings:
        mapping = {q: wing['color'] for q in wing['body']}
        mapping.update({q: marker for q in wing['pins']})
        wing_maps.append({(q[0] + offset[0], q[1] + offset[1]): c for q, c in mapping.items()})
    inp = _scatter(rng, (h, w), root_map, wing_maps, bg)
    if inp is None:
        return None
    # The label is drawn from complete mirrored cores, not from solve().
    out = np.full((h, w), bg, dtype=np.int8)
    for q, color in root_map.items():
        out[q] = color
    for wing, core in zip(wings, true_bodies):
        for q in _shift(core, offset):
            out[q] = wing['color']
    # Whole-scene transformations preserve all per-piece translations.
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    try:
        predicted, info = _docking_search(inp)
    except ValueError:
        return None
    if (not np.array_equal(predicted, out) or info['candidate_count'] < 2
            or info['discarded'] < 1 or any(d == (0, 0) for d, _, _ in info['positions'].values())):
        return None
    if not _asymmetric_retained_wings(info):
        return None
    return dict(input=inp, output=out)


def _checked_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _checked_count(num_train, 'num_train', 2, 10)
    b = _checked_count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(9, 3, replace=False)))
    shift = _pick(rng, ['count', 'noise'])
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', a), ('test', b)):
        train_rate = None
        if split == 'test' and shift == 'noise':
            training_records = [measure_pair(p) for p in puzzle['train']]
            train_rate = max(r['discarded'] / r['wings'] for r in training_records)
        for index in range(count):
            if split == 'train':
                pairs = 2 if index == 0 else 3 if index == 1 else int(rng.integers(2, 4))
                noise_high = 2
                same_color = index == 1 or rng.random() < .4
            else:
                pairs = 4 if shift == 'count' else int(rng.integers(2, 4))
                noise_high = 2 if shift == 'count' else 4
                same_color = rng.random() < .7
            for _ in range(240):
                pair = _construct(rng, pairs, roles, same_color, noise_high)
                if pair is None or _key(pair['input']) in seen:
                    continue
                if split == 'test' and shift == 'noise':
                    _, info = _docking_search(pair['input'])
                    if info['discarded'] / len(info['wings']) <= train_rate:
                        continue
                break
            else:
                raise RuntimeError("bounded construction failed to find an informative unique puzzle")
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _asymmetric_retained_wings(info):
    result = 0
    for i, (_, body, _) in info['positions'].items():
        kept = body & info['keep'][info['wings'][i]['color']]
        if not kept:
            continue
        t, l, b, r = _box(kept)
        local_axis2 = l + r if info['orientation'] == 0 else t + b
        if {_reflect(q, info['orientation'], local_axis2) for q in kept} != kept:
            result += 1
    return result


def measure_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    prediction, info = _docking_search(inp)
    assert np.array_equal(prediction, out), "output differs from maximum supported assembly"
    root = info['root_body'] | info['root_pins']
    assert all(inp[q] == out[q] for q in root), "root moved or changed"
    assert np.count_nonzero(inp == 9) == np.count_nonzero(out == 9)
    assert np.count_nonzero(inp == info['marker']) == 2 * np.count_nonzero(out == info['marker'])
    occupied = set(root)
    supported = set().union(*info['keep'].values())
    for i, (d, body, pins) in info['positions'].items():
        assert d != (0, 0), "a loose wing was already stationary"
        assert not body & occupied and pins <= info['root_pins']
        occupied |= body
        color = info['wings'][i]['color']
        retained = body & info['keep'][color]
        assert retained, "a whole wing disappeared"
        assert len(_components(retained)) == 1
        for q in retained:
            assert out[q] == color
        for q in body - supported:
            assert out[q] == info['bg']
    for color, cells in info['keep'].items():
        assert {_reflect(q, info['orientation'], info['axis2']) for q in cells} == cells
        assert np.count_nonzero(out == color) == len(cells)
    # Omitting movement and only reflecting the scattered input cannot work:
    # at least one retained payload cell must be drawn on an empty input cell.
    assert any(inp[q] == info['bg'] for q in supported)
    body_shapes, boxes, pin_spacings = [], [], []
    for wing in info['wings']:
        t, l, b, r = _box(wing['body'])
        boxes.append((b - t + 1, r - l + 1))
        body_shapes.append(tuple(sorted((x - t, y - l) for x, y in wing['body'])))
        p, q = sorted(wing['pins'])
        pin_spacings.append(abs(p[0] - q[0]) + abs(p[1] - q[1]))
    return dict(shape=inp.shape, wings=len(info['wings']), orientation=info['orientation'],
                parity=info['axis2'] % 2, roles=(info['bg'], info['marker'], info['root_color']),
                candidates=info['candidate_count'], score_gap=info['score'] - info['runner_up'],
                discarded=info['discarded'], boxes=boxes, body_shapes=body_shapes,
                pin_spacings=pin_spacings, colors=len(info['keep']),
                asymmetric_retained=_asymmetric_retained_wings(info),
                root_box=tuple(x for x in _box(info['root_body'])), search_nodes=info['search_nodes'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, records = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            record = measure_pair(pair)
            assert 4 <= record['wings'] <= 8 and record['wings'] % 2 == 0
            assert record['candidates'] >= 2 and record['score_gap'] > 0
            assert record['discarded'] > 0
            assert record['asymmetric_retained'] > 0
            # All keys have equal area; geometric spacing must actually vary.
            assert len(set(record['pin_spacings'])) >= 3
            records[split].append(record)
    roles = {r['roles'] for split in records for r in records[split]}
    assert len(roles) == 1
    assert records['train'][0]['wings'] == 4 and records['train'][1]['wings'] == 6
    assert records['train'][1]['colors'] == 1, "training must disambiguate shape with repeated colors"
    # The selected shift is visible in every query, without a hidden mode bit.
    train_rate = max(r['discarded'] / r['wings'] for r in records['train'])
    assert all(r['wings'] > max(x['wings'] for x in records['train'])
               or r['discarded'] / r['wings'] > train_rate for r in records['test'])
    return None


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def self_test():
    """Bounded checks on 42 seeds, structural diversity, and semantic mutations."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                records.append(measure_pair(pair))
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
        if seed < 2:
            assert _same(puzzle, generate_puzzle(seed))
    assert len(fingerprints) == 42
    assert {r['orientation'] for r in records} == {0, 1}
    assert {r['parity'] for r in records} == {0, 1}
    assert {r['wings'] for r in records} == {4, 6, 8}
    assert len({r['shape'] for r in records}) >= 35
    assert len({r['roles'] for r in records}) >= 30
    assert len({s for r in records for s in r['body_shapes']}) >= 150
    assert len({s for r in records for s in r['boxes']}) >= 20
    assert max(r['candidates'] for r in records) >= 24
    assert max(r['discarded'] for r in records) >= 12
    assert min(r['score_gap'] for r in records) > 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + a * 10 + b, a, b))
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None), (None, 1)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    out = bad['test'][0]['output']
    _, info = _docking_search(bad['test'][0]['input'])
    q = min(set().union(*info['keep'].values()))
    out[q] = info['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted deletion of a supported pixel")
    # A damaged input key must not acquire an invented missing pin.
    broken = generate_puzzle(0, 2, 1)['test'][0]['input'].copy()
    parsed = _parse(broken)
    broken[min(parsed['wings'][0]['pins'])] = parsed['bg']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("solver accepted a missing input pin")
    # Explicit tie: interchangeable identical right wings give two equally
    # supported rigid assignments. No input-order preference may select one.
    ambiguous = np.zeros((24, 24), dtype=np.int8)
    ambiguous[8:19, 10] = 1
    ambiguous[13, 10] = 9
    for q in [(8, 9), (10, 9), (14, 9), (17, 9),
              (8, 11), (9, 11), (14, 11), (15, 11)]:
        ambiguous[q] = 5
    for top, left, right_facing, gap in [(0, 0, True, 2), (20, 0, True, 3),
                                         (0, 20, False, 1), (20, 20, False, 1)]:
        body_left = left if right_facing else left + 1
        ambiguous[top:top + 4, body_left:body_left + 2] = 2
        pin_col = left + 2 if right_facing else left
        ambiguous[top, pin_col] = ambiguous[top + gap, pin_col] = 5
    try:
        solve(ambiguous)
    except ValueError as error:
        assert 'maximum-support' in str(error)
    else:
        raise AssertionError("solver accepted tied interchangeable wings")
    # Visible roles, not literal material/pin hues, govern prediction.
    pair = generate_puzzle(2, 2, 1)['test'][0]
    remap = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0, 9], dtype=np.int8)
    assert np.array_equal(solve(remap[pair['input']]), remap[pair['output']])
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(fingerprints),
                canvas_shapes=len({r['shape'] for r in records}),
                wing_shapes=len({s for r in records for s in r['body_shapes']}),
                max_docking_candidates=max(r['candidates'] for r in records),
                max_discarded=max(r['discarded'] for r in records))
