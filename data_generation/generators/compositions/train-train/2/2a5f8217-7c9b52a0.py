"""Transparent wall layers create cavities classified by an exact-shape legend.

Only standard Python and NumPy are used. Ground truth is drawn from sampled
room masks; solve reconstructs rooms from the visible input's boundary topology.
"""
from collections import deque
import json

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer arrays, size 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    """Four-connected foreground components, in row-major discovery order."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        todo, cells = [(sr, sc)], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in _D4:
                rr, cc = r + dr, c + dc
                if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(tuple(sorted(cells)))
    return result


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _box(cells):
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _transform(cells, turn=0, reflect=False):
    cells = [(r, -c if reflect else c) for r, c in cells]
    for _ in range(turn):
        cells = [(c, -r) for r, c in cells]
    return _normalize(cells)


def _shell(cells):
    s = set(cells)
    return {(r + dr, c + dc) for r, c in s
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - s


def _cavities(walls):
    """Enclosed blank components and the boundary-reachable exterior mask."""
    h, w = walls.shape
    exterior = np.zeros((h, w), dtype=bool)
    todo = deque()
    for r, c in ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
                 + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)]):
        if not walls[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            todo.append((r, c))
    while todo:
        r, c = todo.popleft()
        for dr, dc in _D4:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not walls[rr, cc]
                    and not exterior[rr, cc]):
                exterior[rr, cc] = True
                todo.append((rr, cc))
    return _components(~walls & ~exterior), exterior


def _extract(input_grid):
    """Parse rectangles and monochrome keys without consulting any examples."""
    a = _grid(input_grid)
    outer = int(a[0, 0])
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == outer):
        raise ValueError("the outer border must be uniform")
    panels, positions, keys, key_records = [], [], {}, []
    for cells in _components(a != outer):
        rs, cs = zip(*cells)
        top, left = min(rs), min(cs)
        crop = a[top:max(rs) + 1, left:max(cs) + 1]
        colors = set(int(a[r, c]) for r, c in cells)
        if len(colors) == 2 and len(cells) == crop.size:
            panels.append(crop.copy())
            positions.append((top, left))
        elif len(colors) == 1:
            color = next(iter(colors))
            shape = _normalize(cells)
            if shape in keys and keys[shape] != color:
                raise ValueError("conflicting colors for the same exact key mask")
            keys[shape] = color
            key_records.append((shape, color, top, left))
        else:
            raise ValueError("non-outer regions must be layers or monochrome keys")
    if len(panels) < 2 or not keys:
        raise ValueError("need at least two layers and one key")
    if len({p.shape for p in panels}) != 1:
        raise ValueError("layers must be congruent")
    corners = [int(p[r, c]) for p in panels
               for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    if len(set(corners)) != 1:
        raise ValueError("all layer corners must expose the common blank")
    blank = corners[0]
    palettes = [set(int(v) for v in np.unique(p)) for p in panels]
    if len({tuple(sorted(p)) for p in palettes}) != 1:
        raise ValueError("all layers must share blank and wall roles")
    wall = next(iter(palettes[0] - {blank}))
    if any(color in (outer, blank, wall) for color in keys.values()):
        raise ValueError("legend labels must differ from structural colors")
    walls = np.logical_or.reduce([p == wall for p in panels])
    holes, exterior = _cavities(walls)
    if not holes:
        raise ValueError("the mold must enclose at least one cavity")
    for cells in holes:
        if _normalize(cells) not in keys:
            raise ValueError("every enclosed cavity needs an exact oriented key")
    return dict(outer=outer, blank=blank, wall=wall, panels=panels,
                positions=positions, keys=keys, key_records=key_records,
                walls=walls, holes=holes, exterior=exterior)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Union wall layers; color only enclosed components by exact mask lookup."""
    info = _extract(input_grid)
    result = np.full(info['walls'].shape, info['blank'], dtype=np.int8)
    result[info['walls']] = info['wall']
    for cells in info['holes']:
        color = info['keys'][_normalize(cells)]
        for r, c in cells:
            result[r, c] = color
    return result


def _random_shape(rng, large=False, tiny=False):
    if tiny:
        h, w = int(rng.integers(1, 3)), int(rng.integers(2, 4))
        target = int(rng.integers(2, min(5, h * w) + 1))
    elif large:
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
        target = int(rng.integers(8, min(13, h * w) + 1))
    else:
        if rng.random() < 0.20:
            n = int(rng.integers(1, 6))
            return _transform(tuple((0, c) for c in range(n)), int(rng.integers(4)))
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        target = int(rng.integers(3, min(7, h * w) + 1))
    s = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(s) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in s for dr, dc in _D4
                           if 0 <= r + dr < h and 0 <= c + dc < w} - s)
        s.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize(s)


def _lesson_shapes(rng):
    for _ in range(100):
        first = _random_shape(rng, tiny=True)
        relatives = sorted({_transform(first, t, f)
                            for t in range(4) for f in (False, True)} - {first})
        relatives = [s for s in relatives if _box(s) == _box(first)]
        if relatives:
            return [first, relatives[int(rng.integers(len(relatives)))]]
    # Two oriented Ls are a guaranteed equal-area/equal-box distinction.
    return [((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (1, 1))]


def _same_holes(walls, planned):
    holes, _ = _cavities(walls)
    return {frozenset(c) for c in holes} == {frozenset(c) for c in planned}


def _local_mold(rng, shapes, room_types, open_bay):
    """Place intended negative-space masks, shell them, then connect walls."""
    all_types = list(room_types) + ([0] if open_bay else [])
    min_h = max(_box(shapes[i])[0] + 2 for i in all_types)
    min_w = max(_box(shapes[i])[1] + 2 for i in all_types)
    for _ in range(100):
        # Four small chambers fit on rectangular or square canvases; mask-based
        # spacing permits common walls instead of enforcing a room-slot layout.
        lo = 8 if len(all_types) >= 4 else 6
        h = int(rng.integers(max(lo, min_h), 12))
        w = int(rng.integers(max(lo, min_w), 12))
        occupied, placed = set(), []
        for typ in all_types:
            sh = shapes[typ]
            sh_h, sh_w = _box(sh)
            candidates = [(r, c) for r in range(1, h - sh_h)
                          for c in range(1, w - sh_w)]
            rng.shuffle(candidates)
            chosen = None
            for top, left in candidates:
                s = {(r + top, c + left) for r, c in sh}
                if not (s & occupied or _shell(s) & occupied):
                    chosen = s
                    break
            if chosen is None:
                break
            occupied.update(chosen)
            placed.append(chosen)
        if len(placed) != len(all_types):
            continue
        planned = placed[:len(room_types)]
        walls = np.zeros((h, w), dtype=bool)
        for cells in placed:
            for r, c in _shell(cells):
                walls[r, c] = True
        if any(walls[r, c] for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
            continue
        gate = None
        if open_bay:
            candidates = sorted(_shell(placed[-1]))
            rng.shuffle(candidates)
            for r, c in candidates:
                trial = walls.copy()
                trial[r, c] = False
                if _same_holes(trial, planned):
                    walls, gate = trial, (r, c)
                    break
            if gate is None:
                continue
        elif not _same_holes(walls, planned):
            continue
        # Connect separate wall pieces or extend strokes. Keep a bridge only
        # when it changes no cavity; topology is the acceptance condition.
        for _ in range(int(rng.integers(1, 5))):
            wall_cells = np.argwhere(walls)
            a, b = wall_cells[rng.choice(len(wall_cells), 2, replace=False)]
            ar, ac = map(int, a)
            br, bc = map(int, b)
            if rng.random() < 0.5:
                path = [(r, ac) for r in range(min(ar, br), max(ar, br) + 1)]
                path += [(br, c) for c in range(min(ac, bc), max(ac, bc) + 1)]
            else:
                path = [(ar, c) for c in range(min(ac, bc), max(ac, bc) + 1)]
                path += [(r, bc) for r in range(min(ar, br), max(ar, br) + 1)]
            corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
            if any(p in occupied or p == gate or p in corners for p in path):
                continue
            trial = walls.copy()
            for r, c in path:
                trial[r, c] = True
            if _same_holes(trial, planned):
                walls = trial
        if _same_holes(walls, planned):
            return walls, planned
    return None


def _split_walls(rng, walls, count):
    coords = np.argwhere(walls)
    for _ in range(160):
        masks = [np.zeros(walls.shape, dtype=bool) for _ in range(count)]
        overlap_rate = float(rng.choice((0.0, 0.08, 0.20, 0.30)))
        for r, c in coords:
            owner = int(rng.integers(count))
            masks[owner][r, c] = True
            if rng.random() < overlap_rate:
                other = (owner + int(rng.integers(1, count))) % count
                masks[other][r, c] = True
        if (all(np.any(m) and not _cavities(m)[0] for m in masks)
                and len({m.tobytes() for m in masks}) == count):
            return masks
    return None


def _free(occupied, cells):
    h, w = occupied.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _pack(rng, panels, legend, outer):
    """Scatter rectangles and key masks; use shelf placement only as fallback."""
    ph, pw = panels[0].shape
    count = len(panels)
    for attempt in range(16):
        if attempt < 10:
            H = int(rng.integers(max(ph + 2, 17), 31))
            W = int(rng.integers(max(pw + 2, 17), 31))
        else:
            H = W = 30
        canvas = np.full((H, W), outer, dtype=np.int8)
        occupied = np.zeros((H, W), dtype=bool)
        if attempt >= 12:
            # At most four 11x11 layers fit in a 2x2 block; a five-cell
            # strip on the right and bottom remains available to small keys.
            slots = [(1, 1), (1, pw + 2), (ph + 2, 1), (ph + 2, pw + 2)]
            rng.shuffle(slots)
            panel_positions = slots[:count]
        else:
            panel_positions = None
        failed = False
        for i, panel in enumerate(panels):
            if panel_positions is None:
                candidates = [(r, c) for r in range(1, H - ph)
                              for c in range(1, W - pw)]
                rng.shuffle(candidates)
            else:
                candidates = [panel_positions[i]]
            pos = None
            for top, left in candidates:
                cells = [(top + r, left + c) for r in range(ph) for c in range(pw)]
                if _free(occupied, cells):
                    pos = top, left
                    break
            if pos is None:
                failed = True
                break
            top, left = pos
            occupied[top:top + ph, left:left + pw] = True
            canvas[top:top + ph, left:left + pw] = panel
        if failed:
            continue
        order = list(legend)
        rng.shuffle(order)
        order.sort(key=lambda item: -len(item[0]))
        for shape, color in order:
            sh, sw = _box(shape)
            candidates = [(r, c) for r in range(1, H - sh)
                          for c in range(1, W - sw)]
            rng.shuffle(candidates)
            chosen = None
            for top, left in candidates:
                cells = [(top + r, left + c) for r, c in shape]
                if _free(occupied, cells):
                    chosen = cells
                    break
            if chosen is None:
                failed = True
                break
            for r, c in chosen:
                canvas[r, c] = color
                occupied[r, c] = True
        if not failed:
            return canvas
    return None


def _make_pair(rng, kind, layer_count, roles):
    outer, blank, wall = roles
    palette = [c for c in range(10) if c not in roles]
    for _ in range(100):
        if kind == 'orientation':
            shapes = _lesson_shapes(rng)
            room_types, open_bay = [0, 1], False
            labels = [int(c) for c in rng.choice(palette, 2, replace=False)]
        elif kind == 'enclosure':
            shapes = []
            while len(shapes) < 2:
                s = _random_shape(rng, tiny=True)
                if s not in shapes:
                    shapes.append(s)
            room_types, open_bay = [0, 0, 1], True
            labels = [int(rng.choice(palette))] * 2
        else:
            large = kind == 'large'
            type_count = int(rng.integers(1, 3 if large else 4))
            shapes = []
            while len(shapes) < type_count:
                s = _random_shape(rng, large=large, tiny=kind == 'many')
                if s not in shapes:
                    shapes.append(s)
            room_types = list(range(type_count))
            target = 4 if kind == 'many' else int(rng.integers(type_count, 4))
            while len(room_types) < target:
                room_types.append(int(rng.integers(type_count)))
            open_bay = kind != 'many' and not large and rng.random() < 0.30
            labels = [int(c) for c in rng.choice(palette, type_count, replace=False)]
            if type_count >= 2 and rng.random() < 0.35:
                labels[-1] = labels[0]
        local = _local_mold(rng, shapes, room_types, open_bay)
        if local is None:
            continue
        walls, holes = local
        masks = _split_walls(rng, walls, layer_count)
        if masks is None:
            continue
        panels = [np.where(m, wall, blank).astype(np.int8) for m in masks]
        legend = list(zip(shapes, labels))
        # A spare exemplar breaks "use every legend object exactly once".
        if kind == 'enclosure' or rng.random() < 0.35:
            for _ in range(50):
                spare = _random_shape(rng, tiny=True)
                if spare not in shapes:
                    legend.append((spare, int(rng.choice(palette))))
                    break
        if rng.random() < 0.20:
            legend.append(legend[int(rng.integers(len(shapes)))])
        inp = _pack(rng, panels, legend, outer)
        if inp is None:
            continue
        # Construct answers directly from intended cavity masks and labels.
        out = np.full(walls.shape, blank, dtype=np.int8)
        out[walls] = wall
        for typ, cells in zip(room_types, holes):
            for r, c in cells:
                out[r, c] = labels[typ]
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded sampling exhausted while constructing a mold")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('layers', 'large', 'many')))
    shared_roles = bool(rng.integers(2))
    fixed_roles = tuple(int(c) for c in rng.permutation(10)[:3])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                kind = 'orientation' if i == 0 else 'enclosure' if i == 1 else 'normal'
                layers = 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4 if shift == 'layers' else 5))
            else:
                kind = 'normal' if shift == 'layers' else shift
                layers = 4 if shift == 'layers' else int(rng.integers(2, 5))
            for _ in range(12):
                roles = fixed_roles if shared_roles else tuple(int(c) for c in rng.permutation(10)[:3])
                pair = _make_pair(rng, kind, layers, roles)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not generate distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    x = _extract(inp)
    active = [_normalize(c) for c in x['holes']]
    x['active'] = active
    x['orientation_lesson'] = any(
        s != t and len(s) == len(t) and _box(s) == _box(t)
        and x['keys'][s] != x['keys'][t]
        and t in {_transform(s, k, f) for k in range(4) for f in (False, True)}
        for s in active for t in active)
    # Recover an explicit one-gap bay without access to sampled room positions.
    x['open_bay'] = False
    h, w = x['walls'].shape
    for s in x['keys']:
        sh, sw = _box(s)
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                cells = {(r + top, c + left) for r, c in s}
                if all(x['exterior'][r, c] for r, c in cells):
                    missing = sum(not x['walls'][r, c] for r, c in _shell(cells))
                    if missing == 1:
                        x['open_bay'] = True
                        break
            if x['open_bay']:
                break
        if x['open_bay']:
            break
    x['overlap'] = bool(np.any(np.sum([p == x['wall'] for p in x['panels']], axis=0) > 1))
    return x


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen, 'duplicate input'
            seen.add(sig)
            x = _features(inp) if split == 'train' and len(train_info) < 2 else _extract(inp)
            assert 2 <= len(x['panels']) <= 4
            assert all(not _cavities(p == x['wall'])[0] for p in x['panels']), 'a layer already encloses a cavity'
            assert np.array_equal(solve(inp), out), 'wrong cavity coloring'
            assert out.shape == x['walls'].shape
            assert np.array_equal(out == x['wall'], x['walls']), 'walls must be preserved'
            assert np.array_equal(out == x['blank'], x['exterior']), 'exactly exterior blank must remain'
            assert all(all(out[r, c] == x['keys'][_normalize(cells)] for r, c in cells)
                       for cells in x['holes'])
            # Without B, filling each sheet separately colors no cells.
            raw_union = np.where(x['walls'], x['wall'], x['blank'])
            assert not np.array_equal(raw_union, out)
            if split == 'train':
                train_info.append(x)
    first, second = train_info[:2]
    assert first['orientation_lesson'], 'need a full-mask orientation distinction'
    assert second['open_bay'], 'need a boundary-reachable one-gap bay lesson'
    masks = [_normalize(c) for c in second['holes']]
    assert len(set(masks)) < len(masks), 'need repeated use of a key'
    assert len(set(second['keys']) - set(masks)) > 0, 'need an unused legend key'
    assert len({len(x['panels']) for x in train_info}) >= 2
    # Without A, one universal cavity color fails on the first demonstration.
    assert len({first['keys'][_normalize(c)] for c in first['holes']}) >= 2


def _canonical(grid):
    a = np.asarray(grid)
    colors = {}
    return a.shape, tuple(colors.setdefault(int(v), len(colors)) for v in a.flat)


def self_test():
    """Bounded deterministic coverage, counterfactual, and metamorphic checks."""
    shapes, dimensions, local_shapes, structures, layouts, roles = (set() for _ in range(6))
    layers, rooms = set(), set()
    stats = dict(overlap=0, shared_wall=0, open_bay=0, duplicate_key=0,
                 repeated_room=0, reused_label=0, thin=0, large=0)
    signature_set = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        serial = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()}
                      for pair in p[s]] for s in ('train', 'test')}
        signature_set.add(json.dumps(serial, separators=(',', ':')))
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                x = _features(a)
                active = x['active']
                shapes.update(active)
                dimensions.add(a.shape)
                local_shapes.add(b.shape)
                structures.add(_canonical(b))
                layouts.add(tuple(x['positions']))
                roles.add((x['outer'], x['blank'], x['wall']))
                layers.add(len(x['panels']))
                rooms.add(len(active))
                stats['overlap'] += x['overlap']
                stats['open_bay'] += x['open_bay']
                stats['duplicate_key'] += len(x['key_records']) > len(x['keys'])
                stats['repeated_room'] += len(active) > len(set(active))
                stats['reused_label'] += len({x['keys'][s] for s in set(active)}) < len(set(active))
                stats['thin'] += any(min(_box(s)) == 1 for s in active)
                stats['large'] += any(len(s) >= 8 for s in active)
                stats['shared_wall'] += len(_components(x['walls'])) < len(active)
        # A global bijective palette substitution preserves the rule.
        a, b = p['test'][0]['input'], p['test'][0]['output']
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        assert np.array_equal(solve(permutation[a]), permutation[b])
        # Physical sheet order is irrelevant; registration uses crop coordinates.
        x = _extract(a)
        swapped = a.copy()
        h, w = b.shape
        for pos, panel in zip(x['positions'], x['panels'][1:] + x['panels'][:1]):
            r, c = pos
            swapped[r:r + h, c:c + w] = panel
        assert np.array_equal(solve(swapped), b)
        # Change only the two active label colors in the first demonstration.
        # Geometrically identical walls must now assign the swapped colors.
        first = p['train'][0]
        z = _extract(first['input'])
        c1, c2 = sorted({z['keys'][_normalize(c)] for c in z['holes']})
        exchange = np.arange(10)
        exchange[c1], exchange[c2] = c2, c1
        relabeled = exchange[first['input']]
        recolored = solve(relabeled)
        assert np.array_equal(recolored, exchange[first['output']])
        assert not np.array_equal(recolored, first['output'])
    assert len(signature_set) == 42
    assert len(shapes) >= 60 and len(dimensions) >= 40
    assert len(local_shapes) >= 20 and len(structures) >= 180
    assert len(layouts) >= 150 and len(roles) >= 80
    assert layers == {2, 3, 4} and rooms.issuperset({1, 2, 3, 4})
    assert all(n > 0 for n in stats.values())
    # Determinism, including counts; schema/type validation for all legal counts
    # is also exercised by verify.py.
    p, q = generate_puzzle(101, 2, 1), generate_puzzle(101, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in ('input', 'output'))
    generate_puzzle(102, 10, 4)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Four-connectivity: diagonal blank contact cannot leak through walls.
    ring = np.zeros((5, 5), dtype=bool)
    ring[1:4, 1:4] = True
    ring[2, 2] = False
    ring[1, 1] = False
    assert _cavities(ring)[0] == [((2, 2),)]
    ring[1, 2] = False
    assert not _cavities(ring)[0]
    # An explicit overlapping-sheet case distinguishes OR from XOR: removing
    # the shared top-middle wall opens the singleton cavity.
    p1 = np.array([[0, 0, 0, 0, 0], [0, 8, 8, 8, 0], [0, 8, 0, 0, 0],
                   [0, 8, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    p2 = np.array([[0, 0, 0, 0, 0], [0, 0, 8, 0, 0], [0, 0, 0, 8, 0],
                   [0, 8, 8, 8, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    inp = np.full((7, 17), 9, dtype=np.int8)
    inp[1:6, 1:6], inp[1:6, 7:12], inp[3, 14] = p1, p2, 3
    expected = np.where((p1 == 8) | (p2 == 8), 8, 0)
    expected[2, 2] = 3
    assert np.array_equal(solve(inp), expected)
    assert not _cavities((p1 == 8) ^ (p2 == 8))[0]
    # A cavity can itself have a hole: its central wall island is preserved,
    # and the key is an eight-cell ring rather than a filled bounding box.
    p1, p2 = np.zeros((7, 7), dtype=np.int8), np.zeros((7, 7), dtype=np.int8)
    p1[1, 1:6], p1[1:6, 1], p1[3, 3] = 8, 8, 8
    p2[5, 1:6], p2[1:6, 5] = 8, 8
    inp = np.full((9, 23), 9, dtype=np.int8)
    inp[1:8, 1:8], inp[1:8, 9:16] = p1, p2
    for r in range(3):
        for c in range(3):
            if (r, c) != (1, 1):
                inp[3 + r, 18 + c] = 3
    expected = np.where((p1 == 8) | (p2 == 8), 8, 0)
    expected[2:5, 2:5] = 3
    expected[3, 3] = 8
    assert np.array_equal(solve(inp), expected)
    return dict(seeds=42, pairs=210, exact_masks=len(shapes), input_dimensions=len(dimensions),
                output_dimensions=len(local_shapes), output_structures=len(structures),
                layouts=len(layouts), role_triples=len(roles), layers=sorted(layers),
                rooms=sorted(rooms), **stats)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
