"""Construct ARC grids whose answer is the number of whole foreground figures."""

import numpy as np

_SOURCE_INPUTS = (
    "0000000000/5550000000/5522222200/0020000200/4424333233/4024333233/4424333233/0022222233/0000333333/0000333333",
    "0000000000/0000000000/0660000000/0660444400/0000400400/0000400400/0000444400/0444000000/0444000000/0444000000",
    "00000000/04440000/00000800/03330800/00000800/00077777/00000800/00000800",
    "0000000000/0000000000/0066660000/0060060000/0060767700/0066760700/0000700700/0000777700/0000000000/0000000000",
    "7000000000/7000000000/7000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000",
    "000000000000/000000000000/040000030000/044000330000/004403300000/000443000000/000344044400/003304040400/003000044400/000044400000/000040400000/000044400000",
)
_SOURCE_KEYS = {(len(rows := s.split('/')), len(rows[0]), s) for s in _SOURCE_INPUTS}


def _key(grid):
    return grid.shape, grid.tobytes()


def _is_source(grid):
    return (grid.shape[0], grid.shape[1], '/'.join(''.join(map(str, row)) for row in grid)) in _SOURCE_KEYS


def _counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError('Illegal pair count')
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError('Training count must exceed test count')
    return choices[int(rng.integers(len(choices)))]


def _shape(rng, kind=None):
    if kind is None:
        kind = ('solid', 'outline', 'bar', 'diag', 'step')[int(rng.integers(5))]
    if kind == 'solid':
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        points = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 'outline':
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 8))
        points = {(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 'bar':
        length, thick = int(rng.integers(3, 9)), int(rng.integers(1, 3))
        h, w = (thick, length) if rng.integers(2) else (length, thick)
        points = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 'diag':
        n = int(rng.integers(3, 8))
        h = w = n
        rising = bool(rng.integers(2))
        points = {(i, n - 1 - i if rising else i) for i in range(n)}
    else:
        n = int(rng.integers(3, 6))
        h, w = n, n + 1
        rising = bool(rng.integers(2))
        points = {(i, n - 1 - i + j if rising else i + j)
                  for i in range(n) for j in (0, 1)}
    return {'kind': kind, 'h': h, 'w': w, 'local': points}


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = remaining.pop()
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in remaining:
                        remaining.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _components4(points):
    remaining = set(points)
    result = []
    while remaining:
        start = remaining.pop()
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _foreground_components(grid, bg):
    return len(_components(set(zip(*np.where(grid != bg)))))


def _line_signature(group):
    if len(group) < 2:
        return None
    for axis, values in enumerate(({y for y, x in group}, {x for y, x in group},
                                   {y - x for y, x in group}, {y + x for y, x in group})):
        if len(values) == 1:
            return axis, next(iter(values))
    return None


def _repairable(a, b, grid, bg, color):
    """Recognize a solid, outline, or stepped figure interrupted by another color."""
    together = a | b
    y0, y1 = min(y for y, x in together), max(y for y, x in together)
    x0, x1 = min(x for y, x in together), max(x for y, x in together)
    h, w = y1 - y0 + 1, x1 - x0 + 1
    if min(len(a), len(b)) >= 3 and h >= 3 and w >= 3:
        if np.all(grid[y0:y1 + 1, x0:x1 + 1] != bg):
            return True
        perimeter = ([(y0, x) for x in range(x0, x1 + 1)] +
                     [(y1, x) for x in range(x0, x1 + 1)] +
                     [(y, x0) for y in range(y0, y1 + 1)] +
                     [(y, x1) for y in range(y0, y1 + 1)])
        if all(grid[y, x] != bg for y, x in perimeter):
            return True

    def step_direction(group):
        ys = sorted({y for y, x in group})
        if len(ys) < 2 or ys != list(range(ys[0], ys[-1] + 1)):
            return 0
        rows = [[x for y, x in group if y == yy] for yy in ys]
        if any(max(row) - min(row) > 1 for row in rows):
            return 0
        shift = sum(rows[0]) / len(rows[0]) - sum(rows[-1]) / len(rows[-1])
        return int(np.sign(shift))

    direction = step_direction(a)
    if not direction or direction != step_direction(b):
        return False
    upper, lower = sorted((a, b), key=lambda group: min(y for y, x in group))
    bottom = max(y for y, x in upper)
    top = min(y for y, x in lower)
    if not 1 <= top - bottom <= 2:
        return False
    return any(grid[y, x] not in (bg, color) and
               any(max(abs(y - ay), abs(x - ax)) == 1 for ay, ax in upper) and
               any(max(abs(y - by), abs(x - bx)) == 1 for by, bx in lower)
               for y in range(bottom, top + 1)
               for x in range(max(0, x0 - 1), min(grid.shape[1], x1 + 2)))


def _visible_figure_count(grid, bg):
    """Count same-color figures, bridging only short colored gaps in straight lines."""
    total = 0
    for color in np.unique(grid):
        if color == bg:
            continue
        groups = []
        for group in _components(set(zip(*np.where(grid == color)))):
            four = _components4(group)
            if len(four) == 2 and min(map(len, four)) >= 4:
                groups.extend(four)
            else:
                groups.append(group)
        parent = list(range(len(groups)))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sig = _line_signature(groups[i])
                line_bridge = False
                if sig is not None and sig == _line_signature(groups[j]):
                    axis = sig[0]
                    coordinate = (lambda p: p[1]) if axis in (0, 2, 3) else (lambda p: p[0])
                    left, right = sorted((groups[i], groups[j]), key=lambda g: min(map(coordinate, g)))
                    a = max(left, key=coordinate)
                    b = min(right, key=coordinate)
                    gap = coordinate(b) - coordinate(a) - 1
                    if 1 <= gap <= 2:
                        dy, dx = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
                        line_bridge = all(0 <= a[0] + k * dy < grid.shape[0] and
                                          0 <= a[1] + k * dx < grid.shape[1] and
                                          grid[a[0] + k * dy, a[1] + k * dx] not in (bg, color)
                                          for k in range(1, gap + 1))
                if line_bridge or _repairable(groups[i], groups[j], grid, bg, color):
                    parent[root(j)] = root(i)
        total += len({root(i) for i in range(len(groups))})
    return total


def _near(a, b, radius=1):
    return any(abs(y - v) <= radius and abs(x - u) <= radius
               for y, x in a for v, u in b)


def _draw(objects, h, w, bg):
    grid = np.full((h, w), bg, dtype=np.int8)
    for obj in objects:
        for y, x in obj['points']:
            grid[y, x] = obj['color']
    return grid


def _scene_ok(objects, grid, bg):
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    if frequencies[bg] <= max(frequencies[c] for c in range(10) if c != bg):
        return False
    for i, obj in enumerate(objects):
        hidden = set().union(*(other['points'] for other in objects[i + 1:])) if i + 1 < len(objects) else set()
        visible = obj['points'] - hidden
        if len(visible) < max(2, int(0.68 * len(obj['points']))):
            return False
        fragments = _components(visible)
        if len(fragments) > 2:
            return False
        if len(fragments) == 2 and obj['kind'] not in ('bar', 'diag'):
            if not _repairable(fragments[0], fragments[1], grid, bg, obj['color']):
                return False
    return _visible_figure_count(grid, bg) == len(objects)


def _make_scene(rng, count, mode, bg, palette, force_edge=False, first_relation=None):
    if count == 1:
        h, w = int(rng.integers(6, 15)), int(rng.integers(6, 15))
    elif mode in ('separate', 'same_touch', 'twin'):
        low = 11 if count == 4 else 9
        high = 19 if count == 4 else 15
        h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
    elif mode == 'crowded':
        if rng.random() < 0.12:
            h, w = int(rng.integers(8, 12)), int(rng.integers(8, 12))
        else:
            h, w = int(rng.integers(10, 18)), int(rng.integers(10, 18))
    else:
        h, w = int(rng.integers(9, 18)), int(rng.integers(9, 18))
    if count > 1 and rng.random() < 0.15:
        if rng.integers(2):
            h = min(22, h + int(rng.integers(2, 6)))
        else:
            w = min(22, w + int(rng.integers(2, 6)))
    objects = []
    for index in range(count):
        kind = None
        if first_relation == 'outline_outline' and index < 2:
            kind = 'outline'
        elif mode == 'same_touch' and index < 2:
            kind = ('outline', 'solid')[index]
        elif first_relation == 'line_outline' and index < 2:
            kind = ('outline', 'bar')[index]
        elif first_relation == 'line_line' and index < 2:
            kind = ('bar', 'diag')[index]
        elif mode == 'crowded' and index == 2:
            kind = 'bar'
        elif mode == 'crowded' and index == 3 and first_relation == 'outline_outline':
            kind = 'diag'
        shape = _shape(rng, kind)
        if mode == 'twin' and index == 1:
            first = objects[0]
            shape = {k: first[k] for k in ('kind', 'h', 'w', 'local')}
        if shape['h'] > h or shape['w'] > w:
            return None
        if index == 0:
            relation = 'first'
        elif mode in ('separate', 'twin'):
            relation = 'separate'
        elif mode == 'same_touch':
            relation = 'corner'
        elif mode in ('overlap', 'crowded') and index == 1:
            relation = 'overlap'
        elif mode == 'crowded':
            relation = 'overlap' if first_relation == 'outline_outline' else 'touch'
        else:
            relation = 'touch' if index == 1 or rng.integers(2) else 'separate'
        for attempt in range(260):
            if index and relation == 'corner':
                anchor = objects[-1]
                ay = min(y for y, x in anchor['points'])
                ax = min(x for y, x in anchor['points'])
                y = ay + anchor['h'] if rng.integers(2) else ay - shape['h']
                x = ax + anchor['w'] if rng.integers(2) else ax - shape['w']
                if not (0 <= y <= h - shape['h'] and 0 <= x <= w - shape['w']):
                    continue
            elif index and relation != 'separate':
                anchor = objects[-1]
                ay = min(y for y, x in anchor['points'])
                ax = min(x for y, x in anchor['points'])
                y = int(np.clip(ay + rng.integers(-shape['h'] + 1, anchor['h']), 0, h - shape['h']))
                x = int(np.clip(ax + rng.integers(-shape['w'] + 1, anchor['w']), 0, w - shape['w']))
            else:
                y = int(rng.integers(h - shape['h'] + 1))
                x = int(rng.integers(w - shape['w'] + 1))
                if force_edge and index == 0:
                    edge = int(rng.integers(4))
                    if edge == 0:
                        y = 0
                    elif edge == 1:
                        y = h - shape['h']
                    elif edge == 2:
                        x = 0
                    else:
                        x = w - shape['w']
                    if rng.integers(4) == 0:
                        if edge < 2:
                            x = 0 if rng.integers(2) else w - shape['w']
                        else:
                            y = 0 if rng.integers(2) else h - shape['h']
            points = {(y + r, x + c) for r, c in shape['local']}
            if index:
                hits = [len(points & obj['points']) for obj in objects]
                if relation == 'separate':
                    if any(_near(points, obj['points']) for obj in objects):
                        continue
                elif relation == 'overlap':
                    if not 1 <= hits[-1] <= 3:
                        continue
                elif relation == 'corner':
                    if hits[-1] or not _near(points, objects[-1]['points']):
                        continue
                elif hits[-1] or not _near(points, objects[-1]['points']):
                    continue
                if any(hit > 3 for hit in hits):
                    continue
            forbidden = {obj['color'] for obj in objects if _near(points, obj['points'], 2)}
            if relation == 'corner':
                forbidden.discard(objects[-1]['color'])
            if mode == 'twin' and index == 1:
                forbidden.add(objects[0]['color'])
            available = [c for c in palette if c not in forbidden]
            if not available:
                continue
            reused = [c for c in available if any(obj['color'] == c for obj in objects)]
            color = objects[-1]['color'] if relation == 'corner' else (int(reused[int(rng.integers(len(reused)))]) if reused and rng.random() < 0.65 else int(available[int(rng.integers(len(available)))]))
            obj = dict(shape, points=points, color=color)
            trial = objects + [obj]
            grid = _draw(trial, h, w, bg)
            if _scene_ok(trial, grid, bg):
                objects.append(obj)
                break
        else:
            return None
    grid = _draw(objects, h, w, bg)
    cc = _foreground_components(grid, bg)
    if mode == 'crowded' and cc != 1:
        return None
    if mode in ('separate', 'twin') and cc != count:
        return None
    if mode == 'overlap' and cc >= count:
        return None
    if mode == 'same_touch' and cc != 1:
        return None
    return grid


def _pair(rng, count, mode, bg, palette, force_edge=False, first_relation=None):
    for _ in range(200):
        grid = _make_scene(rng, count, mode, bg, palette, force_edge, first_relation)
        if grid is not None:
            return {'input': grid, 'output': np.full((1, count), bg, dtype=np.int8)}
    raise RuntimeError('Could not place clear figures')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    base_bg = int(rng.integers(10))
    varying_bg = bool(rng.integers(3) == 0)
    specs = [(4, 'crowded', 'outline_outline'), (3, 'separate', None)]
    if train_count >= 3:
        specs.append((2, 'same_touch', None))
    for _ in range(train_count - len(specs)):
        n = int(rng.integers(1, 5))
        mode = 'separate' if n == 1 else ('separate', 'overlap', 'mixed', 'crowded', 'twin' if n == 2 else 'separate')[int(rng.integers(5))]
        if n == 2 and mode == 'crowded':
            mode = 'overlap'
        specs.append((n, mode, None))
    rng.shuffle(specs)
    puzzle = {'train': [], 'test': []}
    seen, train_bgs = set(), []
    for split, split_specs in (('train', specs), ('test', [None] * test_count)):
        for index, spec in enumerate(split_specs):
            if spec is None:
                n = int(rng.integers(1, 5))
                mode = 'separate' if n == 1 else ('overlap', 'mixed', 'crowded', 'separate', 'same_touch' if train_count >= 3 and n == 2 else 'twin', 'twin' if n == 2 else 'overlap')[int(rng.integers(6))]
                if n == 2 and mode == 'crowded':
                    mode = 'overlap'
                spec = (n, mode, None)
            n, mode, relation = spec
            for _ in range(30):
                bg = int(rng.integers(10)) if varying_bg else base_bg
                if varying_bg and split == 'train' and index == 1 and bg == train_bgs[0]:
                    bg = (bg + 1) % 10
                palette = [c for c in range(10) if c != bg]
                rng.shuffle(palette)
                palette = palette[:int(rng.integers(3, 7))]
                if mode == 'separate' and n == 3 and split == 'train':
                    palette = palette[:2]
                pair = _pair(rng, n, mode, bg, palette, force_edge=(n == 1 or rng.random() < 0.32), first_relation=relation)
                k = _key(pair['input'])
                if k not in seen and not _is_source(pair['input']):
                    seen.add(k)
                    puzzle[split].append(pair)
                    if split == 'train':
                        train_bgs.append(bg)
                    break
            else:
                raise RuntimeError('Could not make a unique pair')
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, train_counts = set(), set()
    crowded_train = repeated_train = same_touch_train = same_touch_test = False
    for split, pairs in (('train', train), ('test', test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in 'iu'
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            k = _key(inp)
            assert k not in seen and not _is_source(inp)
            seen.add(k)
            frequencies = np.bincount(inp.ravel().astype(int), minlength=10)
            bg = int(np.argmax(frequencies))
            assert frequencies[bg] > max(frequencies[c] for c in range(10) if c != bg)
            count = _visible_figure_count(inp, bg)
            assert 1 <= count <= 4 and out.shape == (1, count)
            assert np.all(out == bg) and not np.array_equal(inp, out)
            same_touch = (count == 2 and len(np.unique(inp)) == 2 and
                          _foreground_components(inp, bg) == 1 and
                          len(_components4(set(zip(*np.where(inp != bg))))) == 2)
            if split == 'train':
                train_counts.add(count)
                crowded_train |= count == 4 and _foreground_components(inp, bg) == 1
                repeated_train |= count > len(np.unique(inp)) - 1
                same_touch_train |= same_touch
            else:
                same_touch_test |= same_touch
    assert {3, 4} <= train_counts and crowded_train and repeated_train
    assert len(train) < 3 or same_touch_train
    assert not same_touch_test or same_touch_train


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
                   for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, inputs, counts, train_lengths, test_lengths = set(), set(), set(), set(), set()
    crowded = edges = corners = small = repeats = dense = 0
    same_touch = separate_two = contact_two = compact_four = 0
    different_color_twins = 0
    background_variations = unseen_test_colors = 0
    shapes_by_count = {n: set() for n in range(1, 5)}
    full_rectangles, outlines, line_lengths, thick_bars = set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_lengths.add(len(puzzle['train']))
        test_lengths.add(len(puzzle['test']))
        backgrounds, train_colors, test_colors = set(), set(), set()
        train_ids = {id(pair) for pair in puzzle['train']}
        for pair in puzzle['train'] + puzzle['test']:
            inp = pair['input']
            bg = int(np.argmax(np.bincount(inp.ravel().astype(int), minlength=10)))
            n = _visible_figure_count(inp, bg)
            assert pair['output'].shape == (1, n)
            backgrounds.add(bg)
            foreground_colors = {int(c) for c in np.unique(inp) if c != bg}
            if id(pair) in train_ids:
                train_colors.update(foreground_colors)
            else:
                test_colors.update(foreground_colors)
            shapes.add(inp.shape)
            shapes_by_count[n].add(inp.shape)
            inputs.add(_key(inp))
            counts.add(n)
            cc = _foreground_components(inp, bg)
            crowded += n > cc
            edges += any(np.any(edge != bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
            corners += any(inp[y, x] != bg for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
            small += inp.size <= 120
            compact_four += n == 4 and inp.size <= 100
            dense += np.mean(inp != bg) >= 0.3
            repeats += n > len(np.unique(inp)) - 1
            same_touch += n == 2 and len(foreground_colors) == 1 and cc == 1
            separate_two += n == 2 and cc == 2
            contact_two += n == 2 and cc == 1
            if n == 2 and len(foreground_colors) == 2:
                normalized = []
                for color in foreground_colors:
                    groups = _components(set(zip(*np.where(inp == color))))
                    if len(groups) != 1:
                        break
                    cells = groups[0]
                    y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
                    normalized.append({(y - y0, x - x0) for y, x in cells})
                if len(normalized) == 2 and normalized[0] == normalized[1]:
                    different_color_twins += 1
            for color in foreground_colors:
                for cells in _components(set(zip(*np.where(inp == color)))):
                    y0, y1 = min(y for y, x in cells), max(y for y, x in cells)
                    x0, x1 = min(x for y, x in cells), max(x for y, x in cells)
                    h, w = y1 - y0 + 1, x1 - x0 + 1
                    local = {(y - y0, x - x0) for y, x in cells}
                    if h >= 2 and w >= 2 and len(cells) == h * w:
                        full_rectangles.add((h, w))
                        if min(h, w) == 2 and max(h, w) >= 4:
                            thick_bars.add((h, w))
                    if h >= 3 and w >= 3 and local == {
                        (y, x) for y in range(h) for x in range(w)
                        if y in (0, h - 1) or x in (0, w - 1)
                    }:
                        outlines.add((h, w))
                    if len(cells) >= 3 and _line_signature(cells) is not None:
                        line_lengths.add(len(cells))
        background_variations += len(backgrounds) > 1
        unseen_test_colors += bool(test_colors - train_colors)
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert counts == {1, 2, 3, 4} and len(shapes) >= 35 and len(inputs) >= 150
    assert crowded >= 42 and edges >= 50 and corners >= 10 and small >= 15
    assert repeats >= 42 and dense >= 5 and compact_four >= 1
    assert same_touch >= 30 and separate_two >= 10 and contact_two >= 40
    assert different_color_twins >= 5
    assert background_variations >= 5 and unseen_test_colors >= 10
    assert all(len(shapes_by_count[n]) >= 20 for n in range(1, 5))
    assert len(full_rectangles) >= 12 and len(outlines) >= 12
    assert len({(h, w) for h, w in outlines if h != w}) >= 8
    assert {3, 4, 5, 6, 7, 8} <= line_lengths and len(thick_bars) >= 6
    assert _same_puzzle(generate_puzzle(19), generate_puzzle(19))
    assert _same_puzzle(generate_puzzle(19, 6, 3), generate_puzzle(19, 6, 3))
    assert not _same_puzzle(generate_puzzle(19), generate_puzzle(20))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(81, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle['train']) == a
        assert b is None or len(puzzle['test']) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(81, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts were accepted')
