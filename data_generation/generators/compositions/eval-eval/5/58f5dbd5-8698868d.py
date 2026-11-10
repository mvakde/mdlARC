"""Tally-locked punches: choose stencil pose by the topology of the final cut.

No parent generator is imported. Prediction is an input-only geometric decoder.
All sampling, rejection loops, packing attempts and diagnostics are bounded.
"""
from __future__ import annotations

import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Capacity witnesses used only if the finite random mask search exhausts.
# Each has a unique successful identity cut and satisfies all strong witnesses.
_FALLBACK = {
    4: ('1010/0000/0000/0000', '0111/1000/0110/1000'),
    5: ('11010/00000/00010/10001/00000', '01100/10000/00000/11010/10111'),
    6: ('000100/000000/110100/110001/001110/010000', '100010/000000/100001/000010/000001/011100'),
    7: ('0000000/0011000/1011000/1001110/0000000/0000000/0000000', '0110010/1000001/1100011/1011101/0010100/0010100/1000011'),
    8: ('00001100/10100111/01011000/10000001/01001001/01101000/01000010/10100001', '10110000/11001000/00010111/00010010/10100001/00000101/00000000/10100001'),
}


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _N4
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                q = y + dy, x + dx
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _count(mask, diagonal=False):
    return len(_components(mask, diagonal))


def _poses(mask):
    """Four rotations, followed by four rotations of a left/right reflection."""
    return [np.rot90(mask, j) for j in range(4)] + [
        np.rot90(np.fliplr(mask), j) for j in range(4)]


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _span(mask):
    return all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))


def _stamp(rng, k):
    mode = int(rng.integers(10))
    m = np.zeros((k, k), dtype=bool)
    if mode == 0:
        np.fill_diagonal(m, True)
        if rng.random() < .5:
            m |= np.fliplr(m)
    elif mode == 1:
        m[0] = m[-1] = True
    elif mode == 2:
        m[int(rng.integers(k))] = True
        m[:, int(rng.integers(k))] = True
    elif mode == 3:
        m[0] = True
        m[:, 0] = m[:, -1] = True
    elif mode == 4:
        m[0, 0] = m[-1, -1] = True
        for _ in range(int(rng.integers(1, k + 1))):
            m[int(rng.integers(k)), int(rng.integers(k))] = True
    else:
        m = rng.random((k, k)) < float(rng.uniform(.12, .48))
        if mode == 5:
            m |= np.fliplr(m)
        # Full span visibly registers even disconnected stencils.
        for axis, edge in ((0, 0), (0, -1), (1, 0), (1, -1)):
            j = int(rng.integers(k))
            if axis == 0:
                m[edge, j] = True
            else:
                m[j, edge] = True
    return _pick(rng, _poses(m)).copy()


def _holes(rng, k):
    mode = int(rng.integers(4))
    if mode < 2:
        return rng.random((k, k)) < float(rng.uniform(.08, .30))
    m = np.zeros((k, k), dtype=bool)
    if mode == 2:
        for _ in range(int(rng.integers(1, 5))):
            y, x = int(rng.integers(k)), int(rng.integers(k))
            h = int(rng.integers(1, min(4, k - y) + 1))
            w = int(rng.integers(1, min(4, k - x) + 1))
            m[y:y+h, x:x+w] = True
    else:
        # Unequal, occasionally diagonal-touching connected patches.
        for _ in range(int(rng.integers(1, k + 1))):
            y, x = int(rng.integers(k)), int(rng.integers(k))
            m[y, x] = True
            for _ in range(int(rng.integers(4))):
                dy, dx = _pick(rng, _N4)
                y, x = min(k-1, max(0, y+dy)), min(k-1, max(0, x+dx))
                m[y, x] = True
    return m


def _pose_data(holes, stamp):
    poses = _poses(stamp)
    results = [holes | p for p in poses]
    counts = [_count(m) for m in results]
    groups = {}
    for j, (mask, count) in enumerate(zip(results, counts)):
        groups.setdefault(count, {})[mask.tobytes()] = j
    return poses, results, counts, groups


def _decoy_options(holes, stamp, pose, quota):
    """A wrong-color near-match that also meets the tally, with a different cut."""
    wanted = holes | _poses(stamp)[pose]
    options = []
    for i in range(stamp.size):
        other = stamp.copy()
        other.flat[i] ^= True
        if not _span(other) or other.all():
            continue
        result = holes | _poses(other)[pose]
        if _count(result) == quota and not np.array_equal(result, wanted):
            options.append(other)
    return options


def _sample_board(rng, k, pose_group=None, require_decoy=False):
    """Finite random search, then a checked witness of the same exact rule."""
    for _ in range(4096):
        holes, stamp = _holes(rng, k), _stamp(rng, k)
        if not holes.any() or not _span(stamp) or stamp.all():
            continue
        # The intact carrier frame and its interior are initially one object.
        if _count(np.pad(~holes, 1, constant_values=True)) != 1:
            continue
        poses, results, counts, groups = _pose_data(holes, stamp)
        candidates = []
        areas = [int(m.sum()) for m in results]
        for j, (punch, result, quota) in enumerate(zip(poses, results, counts)):
            if not 1 <= quota <= 7 or len(groups[quota]) != 1:
                continue
            if not ((holes & punch).any() and (holes & ~punch).any()
                    and (punch & ~holes).any()):
                continue
            if pose_group is not None:
                if j not in pose_group or len({m.tobytes() for m in results}) != 8:
                    continue
                # Observable witnesses against pixel counts, eight-neighbor
                # connectivity, fixed pose, and maximizing/minimizing overlap.
                if (quota < 2 or _count(result, True) == quota
                        or quota in (_count(holes), _count(stamp))
                        or not min(areas) < areas[j] < max(areas)):
                    continue
            candidates.append(j)
        if not candidates:
            continue
        j = _pick(rng, candidates)
        decoys = _decoy_options(holes, stamp, j, counts[j]) if require_decoy else []
        if require_decoy and not decoys:
            continue
        return {'holes': holes, 'stamp': stamp, 'result': results[j],
                'quota': counts[j], 'pose': j,
                'decoy': _pick(rng, decoys) if decoys else None}
    return _fallback_board(rng, k, pose_group, require_decoy)


def _fallback_board(rng, k, pose_group=None, require_decoy=False):
    holes, canonical = [np.array([[c == '1' for c in row] for row in text.split('/')])
                        for text in _FALLBACK[k]]
    j = 0 if pose_group is None else _pick(rng, list(pose_group))
    stamp = (np.rot90(canonical, -j) if j < 4
             else np.fliplr(np.rot90(canonical, -(j-4)))).copy()
    result = holes | canonical
    quota = _count(result)
    options = _decoy_options(holes, stamp, j, quota) if require_decoy else []
    assert not require_decoy or options
    return {'holes': holes, 'stamp': stamp, 'result': result, 'quota': quota,
            'pose': j, 'decoy': _pick(rng, options) if options else None}


def _marks(rng, k, quota):
    """Noncorner, pairwise four-isolated tally cells, distributed on any edges."""
    s = k + 2
    slots = ([(0, j) for j in range(1, s-1)]
             + [(s-1, j) for j in range(1, s-1)]
             + [(j, 0) for j in range(1, s-1)]
             + [(j, s-1) for j in range(1, s-1)])
    for _ in range(16):
        chosen = []
        for i in rng.permutation(len(slots)):
            y, x = slots[int(i)]
            if any(abs(y-a) + abs(x-b) == 1 for a, b in chosen):
                continue
            chosen.append((y, x))
            if len(chosen) == quota:
                return chosen
    # At k>=4, alternate noncorner cells on all four edges have capacity >=8.
    slots = [(y, x) for y, x in slots
             if ((x if y in (0, s-1) else y) % 2 == 1)]
    return [slots[int(i)] for i in rng.permutation(len(slots))[:quota]]


def _separate(first, second):
    y, x, h, w = first
    a, b, c, d = second
    return y+h < a or a+c < y or x+w < b or b+d < x


def _layout(rng, objects, rows):
    """Free placement with bounded attempts, then a guaranteed shuffled shelf."""
    if rng.random() < .75:
        order = sorted(range(len(objects)), key=lambda j: -objects[j].size)
        for _ in range(20):
            boxes = [None] * len(objects)
            placed = []
            for j in order:
                h, w = objects[j].shape
                free = [(y, x, h, w) for y in range(1, 30-h)
                        for x in range(1, 30-w)
                        if all(_separate((y, x, h, w), b) for b in placed)]
                if not free:
                    break
                boxes[j] = _pick(rng, free)
                placed.append(boxes[j])
            if all(b is not None for b in boxes):
                return boxes
    boxes = [None] * len(objects)
    y = 1
    for row_index in rng.permutation(len(rows)):
        row = list(rows[int(row_index)])
        row = [row[int(i)] for i in rng.permutation(len(row))]
        height = max(objects[j].shape[0] for j in row)
        width = sum(objects[j].shape[1] for j in row) + len(row) - 1
        x = int(rng.integers(1, 30-width))
        for j in row:
            h, w = objects[j].shape
            dy = int(rng.integers(height-h+1))
            boxes[j] = y+dy, x, h, w
            x += w + 1
        y += height + 1
    assert y <= 30
    return boxes


def _make_pair(rng, sizes, witness_group=None, with_decoy=False):
    n = len(sizes)
    # One distractor per shelf row at most; its height never exceeds its plate.
    d = int(rng.integers(min(2, n) + 1))
    if with_decoy:
        d = max(1, d)
    palette = list(map(int, rng.permutation(10)))
    bg, marker = palette[:2]
    carriers, distractor_colors = palette[2:2+n], palette[2+n:2+n+d]
    boards = [_sample_board(rng, k, witness_group if i == 0 else None,
                            with_decoy and i == 0) for i, k in enumerate(sizes)]
    objects, rows = [], []
    for k, color, b in zip(sizes, carriers, boards):
        plate = np.full((k+2, k+2), color, dtype=np.int8)
        plate[1:-1, 1:-1][b['holes']] = bg
        for y, x in _marks(rng, k, b['quota']):
            plate[y, x] = marker
        stamp = np.where(b['stamp'], color, bg).astype(np.int8)
        rows.append([len(objects), len(objects)+1])
        objects.extend([plate, stamp])
    for i, color in enumerate(distractor_colors):
        if i == 0 and with_decoy:
            mask = boards[0]['decoy']
        else:
            mask = _stamp(rng, int(rng.integers(4, min(6, sizes[i])+1)))
            if mask.all():
                mask[1, 1] = False
        rows[i].append(len(objects))
        objects.append(np.where(mask, color, bg).astype(np.int8))
    boxes = _layout(rng, objects, rows)
    canvas = np.full((30, 30), bg, dtype=np.int8)
    result = np.full_like(canvas, bg)
    for obj, (y, x, h, w) in zip(objects, boxes):
        canvas[y:y+h, x:x+w] = obj
    for i, (color, board) in enumerate(zip(carriers, boards)):
        y, x, h, w = boxes[2*i]
        plate = np.full((h, w), color, dtype=np.int8)
        plate[1:-1, 1:-1][board['result']] = bg
        result[y:y+h, x:x+w] = plate
    plate_boxes = [boxes[2*i] for i in range(n)]
    y0, x0 = min(b[0] for b in plate_boxes)-1, min(b[1] for b in plate_boxes)-1
    y1 = max(b[0]+b[2] for b in plate_boxes)+1
    x1 = max(b[1]+b[3] for b in plate_boxes)+1
    output = result[y0:y1, x0:x1].copy()
    # Variable canvas margins, always preserving a blank perimeter.
    low_y, low_x = min(b[0] for b in boxes)-1, min(b[1] for b in boxes)-1
    high_y = max(b[0]+b[2] for b in boxes)+1
    high_x = max(b[1]+b[3] for b in boxes)+1
    low_y, low_x = int(rng.integers(low_y+1)), int(rng.integers(low_x+1))
    high_y, high_x = int(rng.integers(high_y, 31)), int(rng.integers(high_x, 31))
    return {'input': canvas[low_y:high_y, low_x:high_x].copy(), 'output': output}


def _box(cells):
    a = np.asarray(cells)
    y, x = a.min(axis=0)
    yy, xx = a.max(axis=0) + 1
    return int(y), int(x), int(yy-y), int(xx-x)


def _frame(k):
    m = np.ones((k, k), dtype=bool)
    m[1:-1, 1:-1] = False
    return m


def _parse_marker(grid, bg, marker):
    foreground = _components(grid != bg)
    boards, plate_area = [], np.zeros(grid.shape, dtype=bool)
    for cells in foreground:
        if not any(grid[y, x] == marker for y, x in cells):
            continue
        y, x, h, w = _box(cells)
        assert h == w and 6 <= h <= 10
        plate = grid[y:y+h, x:x+w]
        frame = _frame(h)
        assert np.all(plate[frame] != bg)
        colors = set(map(int, np.unique(plate))) - {bg, marker}
        assert len(colors) == 1
        color = colors.pop()
        assert all(plate[a, b] == color for a, b in ((0, 0), (0, h-1), (h-1, 0), (h-1, h-1)))
        dots = plate == marker
        assert not dots[1:-1, 1:-1].any()
        assert not (dots[:-1] & dots[1:]).any() and not (dots[:, :-1] & dots[:, 1:]).any()
        quota = int(dots.sum())
        assert 1 <= quota <= 7
        holes = plate[1:-1, 1:-1] == bg
        assert holes.any()
        assert _count(np.pad(~holes, 1, constant_values=True)) == 1
        plate_area[y:y+h, x:x+w] = True
        boards.append({'box': (y, x, h, w), 'color': color, 'quota': quota, 'holes': holes.copy()})
    assert 1 <= len(boards) <= 3
    assert len({b['color'] for b in boards}) == len(boards)
    assert not np.any((grid == marker) & ~plate_area)
    stamps, stamp_boxes = {}, []
    external = ~plate_area & (grid != bg)
    for color in map(int, np.unique(grid[external])):
        assert color != marker
        cells = np.argwhere(external & (grid == color))
        y, x, h, w = _box(cells)
        assert h == w and 4 <= h <= 8
        assert not plate_area[y:y+h, x:x+w].any()
        register = grid[y:y+h, x:x+w]
        assert np.all((register == bg) | (register == color))
        mask = register == color
        assert _span(mask) and not mask.all()
        stamps[color] = mask.copy()
        stamp_boxes.append((y, x, h, w))
    assert {b['color'] for b in boards} <= set(stamps)
    boxes = [b['box'] for b in boards] + stamp_boxes
    assert all(_separate(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    expected = np.full_like(grid, bg)
    for b in boards:
        y, x, h, w = b['box']
        stamp = stamps[b['color']]
        assert stamp.shape == b['holes'].shape
        poses, results, counts, groups = _pose_data(b['holes'], stamp)
        quota = b['quota']
        assert quota in groups and len(groups[quota]) == 1, 'tally must determine one resulting cut'
        j = next(iter(groups[quota].values()))
        punch, result = poses[j], results[j]
        assert (b['holes'] & punch).any() and (b['holes'] & ~punch).any() and (punch & ~b['holes']).any()
        plate = np.full((h, w), b['color'], dtype=grid.dtype)
        plate[1:-1, 1:-1][result] = bg
        expected[y:y+h, x:x+w] = plate
        b.update(stamp=stamp, results=results, result=result, poses=poses, counts=counts,
                 matching_poses={i for i, m in enumerate(results) if np.array_equal(m, result)},
                 quota8=_count(result, True), initial_count=_count(b['holes']),
                 stamp_count=_count(stamp), areas=[int(m.sum()) for m in results])
    y0 = min(b['box'][0] for b in boards)-1
    x0 = min(b['box'][1] for b in boards)-1
    y1 = max(b['box'][0]+b['box'][2] for b in boards)+1
    x1 = max(b['box'][1]+b['box'][3] for b in boards)+1
    assert min(y0, x0) >= 0 and y1 <= grid.shape[0] and x1 <= grid.shape[1]
    info = {'background': bg, 'marker': marker, 'boards': boards, 'stamps': stamps,
            'distractors': len(stamps)-len(boards), 'crop': (y0, x0, y1, x1)}
    return expected[y0:y1, x0:x1].copy(), info


def _decode(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= dim <= 30 for dim in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    interpretations = []
    for marker in map(int, np.unique(grid)):
        if marker == bg:
            continue
        try:
            interpretations.append(_parse_marker(grid, bg, marker))
        except (AssertionError, IndexError, ValueError):
            continue
    assert len(interpretations) == 1, 'expected one visible marker/plate interpretation'
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer carriers, color matches, tally and legal poses using input only."""
    return _decode(np.asarray(input_grid))[0]


def _color_witness(info):
    for b in info['boards']:
        for color, stamp in info['stamps'].items():
            if color == b['color'] or stamp.shape != b['holes'].shape:
                continue
            for punch in _poses(stamp):
                wrong = b['holes'] | punch
                if _count(wrong) == b['quota'] and not np.array_equal(wrong, b['result']):
                    return True
    return False


def _training_coverage(infos):
    boards = [b for info in infos for b in info['boards']]
    fixed = set(range(8))
    for b in boards:
        fixed &= b['matching_poses']
    assert not fixed, 'training permits a fixed pose without reading tallies'
    assert any(b['quota8'] != b['quota'] for b in boards), 'four/eight adjacency not taught'
    assert any(b['quota'] not in (b['initial_count'], b['stamp_count']) for b in boards)
    assert any(min(b['areas']) < int(b['result'].sum()) < max(b['areas']) for b in boards)
    assert any(_color_witness(info) for info in infos), 'color matching not taught'
    # Carving the complement and replacing rather than unioning are both wrong.
    assert any(not np.array_equal(b['holes'] | ~p, b['result'])
               for b in boards for p in b['poses'] if np.array_equal(b['holes'] | p, b['result']))


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for g in pair.values():
                a = np.asarray(g)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in a.shape)
                assert np.all((0 <= a) & (a <= 9))
            key = _key(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            expected, info = _decode(pair['input'])
            assert np.array_equal(expected, pair['output']), 'incorrect punch, cleanup or crop'
            if split == 'train':
                training.append(info)
    _training_coverage(training)


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(name + ' must be an integer in its legal range')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1 if num_train is not None else 5))
    if num_train is None:
        num_train = int(rng.integers(num_test+1, 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(rng, num_train, num_test)
    curriculum = _pick(rng, ('more_plates', 'larger_registers', 'mixed_sizes'))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            witness = split == 'train' and index < 2
            if split == 'test' and curriculum != 'larger_registers':
                n = 3
                sizes = list(map(int, rng.integers(4, 7, n)))
                if curriculum == 'mixed_sizes':
                    sizes[0], sizes[1] = 4, 6
            else:
                n = int(rng.integers(1, 3))
                upper = 9 if split == 'test' else 7
                sizes = list(map(int, rng.integers(4, upper, n)))
                if split == 'test':
                    sizes[0] = int(rng.integers(7, 9))
                if witness:
                    sizes[0] = int(rng.integers(5, 7))
                if curriculum == 'mixed_sizes' and split == 'train':
                    sizes = [sizes[0]] * n
            group = range(4, 8) if witness and index == 0 else range(4) if witness else None
            for _ in range(8):
                pair = _make_pair(rng, sizes, group, with_decoy=witness and index == 0)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, cells = _key(pair[role])
                h.update(bytes(shape)); h.update(cells)
    return h.hexdigest()


def self_test():
    """42 sequential seeds; retain metrics and digests, never a puzzle corpus."""
    diagonal = np.array([[True, False], [False, True]])
    assert _count(diagonal) == 2 and _count(diagonal, True) == 1
    # A connecting cut merges two existing cavities rather than adding counts.
    old = np.array([[True, False, True]])
    punch = np.array([[False, True, False]])
    assert _count(old) == 2 and _count(punch) == 1 and _count(old | punch) == 1
    metrics = {'puzzles': 42, 'pairs': 0, 'plates': 0}
    fingerprints, inputs, outputs, sizes, quotas, backgrounds = set(), set(), set(), set(), set(), set()
    structures, cavity_shapes, populations, distractors, pose_sets = set(), set(), set(), set(), set()
    mixed_sizes = connected_stamps = disconnected_stamps = reflection_only = symmetric = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                metrics['pairs'] += 1
                expected, info = _decode(pair['input'])
                assert np.array_equal(expected, pair['output'])
                before = pair['input'].copy()
                assert np.array_equal(solve(before), expected) and np.array_equal(before, pair['input'])
                inputs.add(pair['input'].shape); outputs.add(expected.shape)
                backgrounds.add(info['background']); populations.add(len(info['boards']))
                distractors.add(info['distractors'])
                mixed_sizes += len({b['stamp'].shape for b in info['boards']}) > 1
                # Hash normalized structure; no retained reconstructible grid copies.
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                structures.add(hashlib.sha256(bytes(pair['input'].shape)+normalized).hexdigest())
                for b in info['boards']:
                    metrics['plates'] += 1
                    sizes.add(len(b['stamp'])); quotas.add(b['quota'])
                    pose_sets.add(tuple(sorted(b['matching_poses'])))
                    reflection_only += bool(b['matching_poses']) and not (b['matching_poses'] & set(range(4)))
                    conn = _count(b['stamp']) == 1
                    connected_stamps += conn; disconnected_stamps += not conn
                    symmetric += len({m.tobytes() for m in _poses(b['stamp'])}) < 8
                    canonical = min(m.tobytes() for m in _poses(b['result']))
                    cavity_shapes.add(hashlib.sha256(bytes(b['result'].shape)+canonical).hexdigest())
    assert len(fingerprints) == 42 and len(structures) >= 190
    assert len(inputs) >= 35 and len(outputs) >= 35
    assert sizes == set(range(4, 9)) and backgrounds == set(range(10))
    assert populations == {1, 2, 3} and {0, 1, 2} <= distractors and len(quotas) >= 5
    assert len(cavity_shapes) >= 150 and len(pose_sets) >= 8
    assert min(mixed_sizes, connected_stamps, disconnected_stamps, reflection_only, symmetric) >= 5
    # Exercise exceptional sampling directly, including all inverse D4 poses.
    for k in range(4, 9):
        for j in range(8):
            b = _fallback_board(np.random.default_rng(k), k, [j], True)
            poses, results, counts, groups = _pose_data(b['holes'], b['stamp'])
            assert len(groups[b['quota']]) == 1
            assert np.array_equal(results[j], b['result'])
            assert len({m.tobytes() for m in results}) == 8
            assert b['quota'] != _count(b['result'], True)
            assert b['quota'] not in (_count(b['holes']), _count(b['stamp']))
            areas = [int(m.sum()) for m in results]
            assert min(areas) < int(b['result'].sum()) < max(areas)
    p = generate_puzzle(123, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(123, 2, 1))
    # Relabel every visible role, preserving equality, then solve independently.
    permutation = np.array([7, 3, 8, 0, 2, 9, 1, 6, 4, 5])
    recolored = {s: [{k: permutation[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(recolored)
    # Same task under a global rotation: relative stamping convention is D4.
    for pair in p['train'] + p['test']:
        assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
        assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
    for a, b in ((2, 1), (10, 4), (None, 4), (2, None)):
        q = generate_puzzle(431, a, b)
        assert a is None or len(q['train']) == a
        assert b is None or len(q['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt crop accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    test_input = broken['test'][0]['input']
    _, info = _decode(test_input)
    y, x = np.argwhere(test_input == info['marker'])[0]
    board = next(b for b in info['boards'] if
                 b['box'][0] <= y < b['box'][0]+b['box'][2]
                 and b['box'][1] <= x < b['box'][1]+b['box'][3])
    test_input[y, x] = board['color']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed tally with unchanged answer accepted')
    metrics.update(input_shapes=len(inputs), output_shapes=len(outputs),
                   normalized_inputs=len(structures), cavity_patterns=len(cavity_shapes),
                   register_sizes=sorted(sizes), quotas=sorted(quotas),
                   mixed_size_pairs=mixed_sizes, connected_stamps=connected_stamps,
                   disconnected_stamps=disconnected_stamps, reflection_only=reflection_only,
                   symmetric_stamps=symmetric)
    return metrics
