"""Cast miniature cavities into cargo and slide the silhouettes to their banks.

Only standard-library modules and NumPy are used. No generation metadata is
consulted by solve. Diagnostics stay in memory; this module writes no files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,719402063,136514310,943341631,661883982,915384658,836410803,396635261,836325869,968703629,260685198,275636157,830611089,878473380,347831887,23606201,856013738,258427935,796812902,187123050,340931646,372147139,776932707,419056588,225492614,886123080,714654252,768197093,173868341,160635233,6508798,206287081,737955537,345599735,833427796,681346852,314117663,629025601,161046564,228061583,26905733,410422131)


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _orientations(shape, reflection=True):
    result = set()
    for flip in range(2 if reflection else 1):
        cells = [(r, -c if flip else c) for r, c in shape]
        for _ in range(4):
            result.add(_norm(cells))
            cells = [(c, -r) for r, c in cells]
    return tuple(sorted(result))


def _key(shape):
    return _orientations(shape)[0]


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _signature(grid):
    return bytes(grid.shape)+grid.astype(np.uint8).tobytes()


def _decode(input_grid):
    """Recover the tray, scale, assignments and source locations from pixels."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg]*2 > grid.size, "input background must be a strict majority"
    candidates = []
    for cells in _components(grid != bg):
        r0, r1, c0, c1 = _bbox(cells)
        patch = grid[r0:r1, c0:c1]
        for axis in (1, 0):
            q = patch if axis == 1 else patch.T
            if min(q.shape) < 5:
                continue
            wall, low, high = int(q[0, 0]), int(q[1, 0]), int(q[1, -1])
            if len({bg, wall, low, high}) != 4:
                continue
            if not (np.all(q[0] == wall) and np.all(q[-1] == wall)
                    and np.all(q[1:-1, 0] == low) and np.all(q[1:-1, -1] == high)):
                continue
            if not np.all((q[1:-1, 1:-1] == wall) | (q[1:-1, 1:-1] == bg)):
                continue
            holes = _components(q == bg)
            if len(holes) >= 2:
                candidates.append((axis, (r0, r1, c0, c1), q.copy(), wall, low, high, holes))
    assert len(candidates) == 1, "one tray with two differently colored banks is required"
    axis, bounds, board, wall, low, high, holes = candidates[0]
    r0, r1, c0, c1 = bounds
    covered = grid == bg
    covered[r0:r1, c0:c1] = True
    cards = []
    for frame in (low, high):
        outside_frame = grid == frame
        outside_frame[r0:r1, c0:c1] = False
        for cells in _components(outside_frame):
            a, b, c, d = _bbox(cells)
            patch = grid[a:b, c:d]
            assert min(patch.shape) >= 4, "incomplete cargo card"
            assert np.all(patch[0] == frame) and np.all(patch[-1] == frame)
            assert np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
            inside = patch[1:-1, 1:-1]
            colors = set(map(int, np.unique(inside))) - {bg}
            assert len(colors) == 1, "one monochrome silhouette per card"
            color = next(iter(colors))
            assert color not in (wall, low, high)
            assert not np.any(covered[a:b, c:d] & (patch != bg)), "overlapping cards"
            covered[a:b, c:d] = True
            cards.append(dict(mask=inside == color, color=color,
                              d=-1 if frame == low else 1))
    assert np.all(covered) and len(cards) == len(holes), "unaccounted input objects"
    hole_area = sum(map(len, holes))
    cargo_area = sum(int(c['mask'].sum()) for c in cards)
    assert cargo_area % hole_area == 0
    scale = math.isqrt(cargo_area // hole_area)
    assert scale >= 2 and scale*scale*hole_area == cargo_area, "inconsistent scale"
    assert max(board.shape)*scale <= 30
    groups = {}
    for card in cards:
        mask = card['mask']
        h, w = mask.shape
        assert h % scale == w % scale == 0
        blocks = mask.reshape(h//scale, scale, w//scale, scale)
        small = blocks.all(axis=(1, 3))
        assert np.array_equal(mask, np.repeat(np.repeat(small, scale, 0), scale, 1))
        cells = tuple(map(tuple, np.argwhere(small)))
        assert cells and _size(_norm(cells)) == small.shape, "card must crop its silhouette tightly"
        assert len(_components(small)) == 1
        card['shape'] = _norm(cells)
        groups.setdefault(_key(cells), []).append(card)
    hole_groups = Counter(_key(cells) for cells in holes)
    assert set(groups) == set(hole_groups)
    assert all(len(v) == hole_groups[k] for k, v in groups.items())
    assert all(len({(p['color'], p['d']) for p in v}) == 1 for v in groups.values()), \
        "congruent cards must have identical color and direction"
    objects, reflection, reorientation = [], False, False
    for cells in holes:
        a, b, c, d = _bbox(cells)
        shape = _norm(cells)
        group = groups[_key(shape)]
        card = group[0]
        objects.append(dict(r=a, c=c, shape=shape, color=card['color'], d=card['d']))
        if len(group) == 1:
            reflection |= shape not in _orientations(card['shape'], False)
            reorientation |= shape != card['shape']
    # Source cavities are separated rectangles whenever their row spans overlap.
    # This makes destination-side order visible without any arbitrary tie break.
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            ah, aw = _size(a['shape'])
            bh, bw = _size(b['shape'])
            if max(a['r'], b['r']) < min(a['r']+ah, b['r']+bh):
                assert a['c']+aw < b['c'] or b['c']+bw < a['c']
    return dict(bg=bg, wall=wall, low=low, high=high, axis=axis, board=board,
                scale=scale, objects=objects, cards=cards, bounds=bounds,
                reflection=reflection, reorientation=reorientation)


def _cells(obj):
    return {(obj['r']+r, obj['c']+c) for r, c in obj['shape']}


def _settle(objects, width, method='stable'):
    """Stable row-profile constraints; alternate methods are diagnostics only."""
    result = [dict(o) for o in objects]
    for direction in (-1, 1):
        indices = [i for i, o in enumerate(objects) if o['d'] == direction]
        indices.sort(key=lambda i: (objects[i]['c'] if direction == -1
                                   else -objects[i]['c']-_size(objects[i]['shape'])[1],
                                   objects[i]['r']))
        settled = []
        for i in indices:
            o = result[i]
            h, w = _size(o['shape'])
            if method == 'edge_first':
                occupied = set().union(*(_cells(p) for p in settled))
                choices = range(1, width-w)
                if direction == 1:
                    choices = reversed(choices)
                for col in choices:
                    o['c'] = col
                    if not (_cells(o) & occupied):
                        break
                else:
                    raise ValueError('no space')
            else:
                limits = [1 if direction == -1 else width-w-1]
                rows = {r: [c for rr, c in o['shape'] if rr == r] for r, c in o['shape']}
                for p in settled:
                    ph, pw = _size(p['shape'])
                    if method == 'boxes':
                        if max(o['r'], p['r']) < min(o['r']+h, p['r']+ph):
                            limits.append(p['c']+pw if direction == -1 else p['c']-w)
                        continue
                    for pr, pc in p['shape']:
                        local_row = p['r']+pr-o['r']
                        if local_row in rows:
                            limits.append(p['c']+pc-min(rows[local_row])+1 if direction == -1
                                          else p['c']+pc-max(rows[local_row])-1)
                o['c'] = max(limits) if direction == -1 else min(limits)
            if not 1 <= o['c'] <= width-w-1:
                raise ValueError('bank capacity exceeded')
            settled.append(o)
    occupied = set()
    for o in result:
        cells = _cells(o)
        if cells & occupied:
            raise ValueError('opposite groups collide after settling')
        occupied.update(cells)
    return result


def _walk(objects, width):
    """Check the recurrence by sliding from source positions one unit at a time."""
    result = [dict(o) for o in objects]
    for direction in (-1, 1):
        group = [o for o in result if o['d'] == direction]
        group.sort(key=lambda o: o['c'] if direction == -1 else -o['c']-_size(o['shape'])[1])
        occupied = set()
        for o in group:
            for _ in range(width):
                candidate = dict(o, c=o['c']+direction)
                cs = _cells(candidate)
                if any(c < 1 or c >= width-1 for r, c in cs) or cs & occupied:
                    break
                o['c'] += direction
            else:
                raise AssertionError('movement did not terminate')
            occupied.update(_cells(o))
    return result


def _render(info, objects):
    board = np.full(info['board'].shape, info['bg'], dtype=np.int8)
    board[0] = board[-1] = info['wall']
    board[1:-1, 0], board[1:-1, -1] = info['low'], info['high']
    occupied = set()
    for obj in objects:
        cells = _cells(obj)
        assert not cells & occupied
        for r, c in cells:
            assert 0 < r < board.shape[0]-1 and 0 < c < board.shape[1]-1
            board[r, c] = obj['color']
        occupied.update(cells)
    out = np.repeat(np.repeat(board, info['scale'], 0), info['scale'], 1)
    return out if info['axis'] == 1 else out.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _render(info, _settle(info['objects'], info['board'].shape[1]))


def _grow(rng, area, span):
    cells = {(0, 0)}
    while len(cells) < area:
        options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _N4} - cells)
        options = [p for p in options if max(_size(_norm(cells | {p}))) <= span]
        if not options:
            break
        cells.add(options[int(rng.integers(len(options)))])
    return _norm(cells)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _place(rng, shapes, h, w):
    placed = []
    for shape in sorted(shapes, key=lambda s: -len(s)):
        sh, sw = _size(shape)
        positions = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
        for j in rng.permutation(len(positions)):
            r, c = positions[int(j)]
            if any(not (r+sh < p['r'] or p['r']+_size(p['shape'])[0] < r
                        or c+sw < p['c'] or p['c']+_size(p['shape'])[1] < c) for p in placed):
                continue
            placed.append(dict(r=r, c=c, shape=shape))
            break
        else:
            return None
    return placed


def _scene(rng, scale, purpose):
    limit = 30 // scale
    if purpose == 'chain':
        heights = [int(rng.integers(3, 5)) for _ in range(3)] if limit >= 12 else [3, 3, 3]
        a, b, c = heights
        h, w = int(rng.integers(sum(heights), limit+1)), int(rng.integers(10, limit+1))
        top = int(rng.integers(1, h-sum(heights)+2))
        gaps = rng.multinomial(w-10, [.25]*4)
        col0, col1 = 1+int(gaps[0]), 4+int(gaps[0]+gaps[1])
        col2 = 7+int(sum(gaps[:3]))
        shapes = [tuple((r, 0) for r in range(a))+((a-1, 1),),
                  tuple((r, 1) for r in range(b))+((int(rng.integers(1, b-1)), 0),),
                  tuple((r, q) for r in range(c-1) for q in (0, 1))+((c-1, 0),)]
        objects = [dict(r=top+c+b-2, c=col0, shape=_norm(shapes[0]), d=-1),
                   dict(r=top+c-1, c=col1, shape=_norm(shapes[1]), d=-1),
                   dict(r=top, c=col2, shape=_norm(shapes[2]), d=-1)]
        if rng.random() < .5:
            for o in objects:
                sh, sw = _size(o['shape'])
                o['r'] = h-o['r']-sh
                o['shape'] = _norm((sh-1-r, c) for r, c in o['shape'])
        if rng.random() < .5:
            for o in objects:
                sh, sw = _size(o['shape'])
                o['c'] = w-o['c']-sw
                o['shape'] = _norm((r, sw-1-c) for r, c in o['shape'])
                o['d'] = 1
        return (h, w), objects
    for _ in range(100):
        if purpose == 'match':
            area = int(rng.integers(4, 7))
            first = _grow(rng, area, 3)
            second = _grow(rng, area, 3)
            if len(_orientations(first)) != 8 or _key(first) == _key(second):
                continue
            shapes = [_pick(rng, _orientations(first)), _pick(rng, _orientations(second))]
        else:
            count = int(rng.integers(2, 7 if scale == 2 else 5))
            shapes = [_pick(rng, _orientations(_grow(rng, int(rng.integers(1, 9)),
                                                         4 if scale == 2 else 3))) for _ in range(count)]
        h, w = int(rng.integers(6, limit+1)), int(rng.integers(7, limit+1))
        objects = _place(rng, shapes, h, w)
        if objects is None:
            continue
        if purpose == 'match':
            a, b = objects
            if max(a['r'], b['r']) >= min(a['r']+_size(a['shape'])[0], b['r']+_size(b['shape'])[0]):
                continue
            # Put each class on the opposite source side. This teaches that
            # opposing groups pass through one another while sorting.
            a['d'], b['d'] = (-1, 1) if a['c'] > b['c'] else (1, -1)
        classes = {}
        for i, o in enumerate(objects):
            classes.setdefault(_key(o['shape']), o['d'] if purpose == 'match'
                               else int(rng.choice([-1, 1])))
            o['d'] = classes[_key(o['shape'])]
        try:
            settled = _settle(objects, w)
        except ValueError:
            continue
        if purpose == 'match' and not all(a['c'] != b['c'] for a, b in zip(objects, settled)):
            continue
        if any(a['c'] != b['c'] for a, b in zip(objects, settled)):
            return (h, w), objects
    return None


def _free_positions(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    sums = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:]-sums[:-h, w:]-sums[h:, :-w]+sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, patches, bg):
    order = [0]+sorted(range(1, len(patches)), key=lambda i: -patches[i].size)
    for attempt in range(30):
        h, w = ((int(rng.integers(21, 31)), int(rng.integers(21, 31)))
                if attempt < 16 else (30, 30))
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        for i in order:
            ph, pw = patches[i].shape
            positions = _free_positions(blocked, ph, pw)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+ph, c:c+pw] = patches[i]
            blocked[max(0, r-1):r+ph+1, max(0, c-1):c+pw+1] = True
        else:
            if np.count_nonzero(canvas == bg)*2 <= canvas.size:
                continue
            if rng.random() < .75:
                a, b, c, d = _bbox(list(zip(*np.nonzero(canvas != bg))))
                crop = canvas[max(0, a-int(rng.integers(3))):min(h, b+int(rng.integers(3))),
                              max(0, c-int(rng.integers(3))):min(w, d+int(rng.integers(3)))]
                if np.count_nonzero(crop == bg)*2 > crop.size:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, axis, scale, purpose, excluded):
    for _ in range(160):
        sampled = _scene(rng, scale, purpose)
        if sampled is None:
            continue
        shape, objects = sampled
        if excluded is not None and not ({_key(o['shape']) for o in objects} - excluded):
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, wall, low, high = palette[:4]
        colors = palette[4:4+int(rng.integers(1, 7))]
        assignments = {}
        for o in objects:
            assignments.setdefault(_key(o['shape']), (_pick(rng, colors), o['d']))
            o['color'], o['d'] = assignments[_key(o['shape'])]
            if purpose == 'match':
                o['color'] = colors[0]
        board = np.full(shape, wall, dtype=np.int8)
        board[1:-1, 0], board[1:-1, -1] = low, high
        for o in objects:
            for r, c in _cells(o):
                board[r, c] = bg
        patches = [board if axis == 1 else board.T.copy()]
        reflected = False
        for o in objects:
            choices = _orientations(o['shape'])
            if purpose == 'match' and not reflected and len(choices) == 8:
                choices = tuple(s for s in choices if o['shape'] not in _orientations(s, False))
                reflected = True
            loose = _pick(rng, choices)
            lh, lw = _size(loose)
            mask = np.zeros((lh, lw), dtype=bool)
            for r, c in loose:
                mask[r, c] = True
            mask = np.repeat(np.repeat(mask, scale, 0), scale, 1)
            patch = np.full((mask.shape[0]+2, mask.shape[1]+2),
                            low if o['d'] == -1 else high, dtype=np.int8)
            patch[1:-1, 1:-1] = np.where(mask, o['color'], bg)
            # Card orientation is free in canvas coordinates, independent of tray axis.
            patches.append(patch)
        canvas = _pack(rng, patches, bg)
        if canvas is None:
            continue
        info = dict(bg=bg, wall=wall, low=low, high=high, scale=scale,
                    axis=axis, board=board, objects=objects)
        output = _render(info, _settle(objects, shape[1]))
        decoded = _decode(canvas)
        if purpose == 'match' and not decoded['reflection']:
            continue
        return {'input': canvas, 'output': output}
    raise RuntimeError('bounded scene construction exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    axis, schedule = int(rng.integers(2)), int(rng.integers(3))
    train_scales = ([2]*num_train if schedule != 1 else
                    [3, 2]+[int(rng.choice([2, 3])) for _ in range(num_train-2)])
    test_scales = [3 if schedule == 0 else 2 if schedule == 2 else int(rng.choice([2, 3]))
                   for _ in range(num_test)]
    puzzle, seen, training_classes = {'train': [], 'test': []}, set(), set()
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, scale in enumerate(scales):
            purpose = 'match' if split == 'train' and i == 0 else 'chain' if split == 'train' and i == 1 else 'free'
            for _ in range(12):
                pair = _make_pair(rng, axis, scale, purpose, training_classes if split == 'test' else None)
                sig = _signature(pair['input'])
                if sig not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
            seen.add(sig)
            puzzle[split].append(pair)
            if split == 'train':
                training_classes.update(_key(o['shape']) for o in _decode(pair['input'])['objects'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _differs(info, method=None, reverse=False):
    objects = [dict(o, d=-o['d'] if reverse else o['d']) for o in info['objects']]
    correct = _render(info, _settle(info['objects'], info['board'].shape[1]))
    try:
        alternative = _render(info, _settle(objects, info['board'].shape[1], method or 'stable'))
    except (ValueError, AssertionError):
        return True
    return not np.array_equal(correct, alternative)


def measure_pair(pair):
    info = _decode(pair['input'])
    objects = info['objects']
    settled = _settle(objects, info['board'].shape[1])
    return dict(axis=info['axis'], scale=info['scale'], board=list(info['board'].shape),
                input_shape=list(np.asarray(pair['input']).shape), count=len(objects),
                classes=sorted({_key(o['shape']) for o in objects}),
                areas=sorted(len(o['shape']) for o in objects),
                directions=sorted(Counter(o['d'] for o in objects).values()),
                reflection=info['reflection'], boxes_differ=_differs(info, 'boxes'),
                jumping_differs=_differs(info, 'edge_first'),
                direction_matters=_differs(info, reverse=True),
                crossings=sum(a['d'] == -1 and b['d'] == 1 and a['c'] > b['c']
                              and max(a['r'], b['r']) < min(a['r']+_size(a['shape'])[0], b['r']+_size(b['shape'])[0])
                              for a in objects for b in objects),
                stationary=sum(a['c'] == b['c'] for a, b in zip(objects, settled)),
                travel=[abs(a['c']-b['c']) for a, b in zip(objects, settled)],
                palette=[info[k] for k in ('bg', 'wall', 'low', 'high')],
                cavity_geometry=tuple((o['r'], o['c'], o['shape']) for o in objects))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, axes, records, train_classes = set(), set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            info = _decode(inp)
            axes.add(info['axis'])
            settled = _settle(info['objects'], info['board'].shape[1])
            walked = _walk(info['objects'], info['board'].shape[1])
            assert [o['c'] for o in walked] == [o['c'] for o in settled], 'physical sliding disagrees'
            assert np.array_equal(_render(info, walked), out), 'incorrect casting or movement'
            # Full rim is preserved, interior mold material is erased, cargo area conserved.
            logical = (out if info['axis'] == 1 else out.T)[::info['scale'], ::info['scale']]
            assert np.array_equal(logical[[0, -1]], info['board'][[0, -1]])
            assert np.array_equal(logical[:, [0, -1]], info['board'][:, [0, -1]])
            assert np.count_nonzero(logical[1:-1, 1:-1] == info['wall']) == 0
            for color in {o['color'] for o in info['objects']}:
                assert np.count_nonzero(logical == color) == sum(len(o['shape']) for o in info['objects'] if o['color'] == color)
            classes = {_key(o['shape']) for o in info['objects']}
            if split == 'train':
                train_classes.update(classes)
            else:
                assert classes-train_classes, 'test must introduce a new shape class'
            records[split].append(measure_pair(pair))
    assert len(axes) == 1
    training = records['train']
    assert any(r['reflection'] for r in training), 'reflection must be demonstrated'
    assert any(r['boxes_differ'] and r['jumping_differs'] for r in training), 'shape and stable-order witness required'
    assert any(r['direction_matters'] for r in training)
    # Both frames must carry the same payload color in the first demonstration,
    # and equal area cannot stand in for exact shape matching.
    first = _decode(puzzle['train'][0]['input'])['objects']
    assert len(first) == 2 and {o['d'] for o in first} == {-1, 1}
    assert len({o['color'] for o in first}) == len({len(o['shape']) for o in first}) == 1
    assert len({_key(o['shape']) for o in first}) == 2
    first_info = _decode(puzzle['train'][0]['input'])
    first_settled = _settle(first, first_info['board'].shape[1])
    assert all(a['c'] != b['c'] for a, b in zip(first, first_settled))
    assert _differs(first_info, reverse=True), 'swapping equal-area card assignments must change the answer'
    assert training[0]['crossings'] == 1, 'opposite-class passing must be demonstrated'
    assert not np.array_equal(_render(first_info, first), _render(first_info, first_settled)), 'omitting movement must change the answer'


def self_test():
    """A streamed 42-seed check; retains summaries/hashes, never a grid corpus."""
    digests, rows = set(), []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                digest.update(_signature(pair['input']))
                digest.update(_signature(pair['output']))
                rows.append(measure_pair(pair))
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
        digests.add(digest.hexdigest())
    assert len(digests) == 42
    assert {r['axis'] for r in rows} == {0, 1}
    assert {r['scale'] for r in rows} == {2, 3}
    assert len({tuple(r['input_shape']) for r in rows}) >= 35
    assert len({tuple(r['board']) for r in rows}) >= 25
    assert len({s for r in rows for s in r['classes']}) >= 35
    assert {2, 3, 4} <= {r['count'] for r in rows}
    assert any(r['stationary'] for r in rows)
    assert any(r['directions'] == [len(r['areas'])] for r in rows)
    assert any(len(r['directions']) == 2 for r in rows)
    assert sum(r['boxes_differ'] and r['jumping_differs'] for r in rows) >= 42
    assert len({r['cavity_geometry'] for r in rows}) >= 150
    # Hand-computed concave chain: silhouette contact, bounding rectangles and
    # first-empty-slot repacking make three different predictions.
    chain = [dict(r=5, c=1, d=-1, color=2, shape=((0,0),(1,0),(2,0),(2,1))),
             dict(r=3, c=4, d=-1, color=3, shape=((0,1),(1,0),(1,1),(2,1))),
             dict(r=1, c=7, d=-1, color=4, shape=((0,0),(0,1),(1,0),(1,1),(2,0)))]
    assert [o['c'] for o in _settle(chain, 10)] == [1, 1, 3]
    assert [o['c'] for o in _walk(chain, 10)] == [1, 1, 3]
    assert [o['c'] for o in _settle(chain, 10, 'boxes')] == [1, 3, 5]
    assert [o['c'] for o in _settle(chain, 10, 'edge_first')] == [1, 1, 1]
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(123456+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Visible banks allow arbitrary color renaming and a change of axis, without
    # teaching a new color dictionary or consulting another example.
    permutation = np.array([7,2,5,0,9,3,8,1,6,4], dtype=np.int8)
    for pair in p['train']:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        assert np.array_equal(solve(pair['input'].T), pair['output'].T)
    broken = {s: [{'input': pair['input'].copy(), 'output': pair['output'].copy()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
