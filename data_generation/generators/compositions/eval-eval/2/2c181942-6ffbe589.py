"""Rotating nursery: counted bud masks grow color-matched silhouettes.

Only NumPy and the standard library are used. The public solver reads the
scaffold, ports, counters and prototypes entirely from its input.
"""
from __future__ import annotations

import json
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1)
                   for c in (-1, 0, 1) if r or c)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer arrays of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def _components(occupied):
    remaining = set(map(tuple, np.argwhere(occupied).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _corners(n, arm):
    cells = set()
    for y in (0, n - 1):
        for x in (0, n - 1):
            dy, dx = (1 if y == 0 else -1), (1 if x == 0 else -1)
            cells.update((y + i * dy, x) for i in range(arm))
            cells.update((y, x + i * dx) for i in range(arm))
    return cells


def _rotate_cells(cells, n, turns):
    result = set(cells)
    for _ in range(turns % 4):
        result = {(c, n - 1 - r) for r, c in result}
    return result


def _port(n, side, offset):
    return _rotate_cells({(1, offset), (1, offset + 1)}, n, side)


def _port_side(cells, n):
    for side in range(4):
        normalized = _rotate_cells(cells, n, -side)
        if len(normalized) != 2:
            continue
        points = sorted(normalized)
        if (points[0][0] == points[1][0] == 1
                and points[1][1] == points[0][1] + 1
                and 2 <= points[0][1] <= n - 4):
            return side, points[0][1]
    raise ValueError("a bud must be a tangential two-cell bar on the inner rim")


def _attachments(mask):
    """Canonical attachment faces down; collapse identical placement choices."""
    candidates = {}
    for turn in range(4):
        m = np.rot90(mask, turn).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], edge):
            candidates[_key(m)] = m
    return list(candidates.values())


def _canonical(mask):
    candidates = _attachments(mask)
    if len(candidates) != 1 or len(_components(mask)) != 1:
        raise ValueError("prototype has missing or competing attachment geometry")
    return candidates[0]


def _stamp(mask, cells, side):
    """Put the neck ON the bud, with the body growing into the square."""
    neck = np.zeros_like(mask)
    w = mask.shape[1]
    neck[-1, w // 2 - 1:w // 2 + 1] = True
    turn = (2 - side) % 4
    oriented = np.rot90(mask, turn)
    attachment = np.argwhere(np.rot90(neck, turn))
    shift = np.min(list(cells), axis=0) - attachment.min(axis=0)
    assert set(map(tuple, (attachment + shift).tolist())) == set(cells)
    return set(map(tuple, (np.argwhere(oriented) + shift).tolist()))


def _parse(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if frequencies[bg] <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    frames = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(a == color)
        low, high = coords.min(axis=0), coords.max(axis=0)
        dims = high - low + 1
        if dims[0] != dims[1] or dims[0] < 8:
            continue
        local = set(map(tuple, (coords - low).tolist()))
        for arm in (2, 3):
            if local == _corners(int(dims[0]), arm):
                frames.append((int(low[0]), int(low[1]), int(dims[0]), color, arm))
    if len(frames) != 1:
        raise ValueError("one unique four-corner square scaffold is required")
    row, col, n, frame, arm = frames[0]
    square = a[row:row + n, col:col + n]
    occupied = (square != bg) & (square != frame)
    ports = {}
    for coords in _components(occupied):
        colors = set(map(int, square[coords[:, 0], coords[:, 1]]))
        if len(colors) != 1:
            raise ValueError("buds must be separated monochrome components")
        color = colors.pop()
        cells = set(map(tuple, coords.tolist()))
        side, offset = _port_side(cells, n)
        ports.setdefault(color, []).append((cells, side, offset))
    if not 1 <= len(ports) <= 4:
        raise ValueError("one to four bud colors are required")
    exterior = a != bg
    exterior[row:row + n, col:col + n] = False
    # Every exterior object is separated from the entire scaffold rectangle.
    if exterior[max(0, row - 1):row + n + 1,
                max(0, col - 1):col + n + 1].any():
        raise ValueError("exterior objects touch the scaffold")
    prototypes, counters, distractors = {}, {}, []
    orientations = {}
    for coords in _components(exterior):
        colors = set(map(int, a[coords[:, 0], coords[:, 1]]))
        if len(colors) != 1:
            raise ValueError("exterior objects must be separated and monochrome")
        color = colors.pop()
        low, high = coords.min(axis=0), coords.max(axis=0)
        dims = high - low + 1
        if min(dims) == 1:
            if len(coords) != max(dims) or len(coords) > 9 or color not in ports or color in counters:
                raise ValueError("invalid, repeated, or unmatched counter")
            counters[color] = len(coords)
            orientations[color] = 'single' if len(coords) == 1 else ('H' if dims[0] == 1 else 'V')
        else:
            mask = np.zeros(tuple(dims), dtype=bool)
            mask[coords[:, 0] - low[0], coords[:, 1] - low[1]] = True
            if color not in ports:
                distractors.append((color, mask))
                continue
            if color in prototypes:
                raise ValueError("only one prototype per bud color is permitted")
            prototypes[color] = _canonical(mask)
    if set(prototypes) != set(ports):
        raise ValueError("each bud color needs exactly one prototype")
    return dict(background=bg, row=row, col=col, size=n, frame=frame,
                arm=arm, ports=ports, prototypes=prototypes, counters=counters,
                orientations=orientations, distractors=distractors)


def _render(info, use_counts=True, grow=True, direction=1):
    n, bg = info['size'], info['background']
    output = np.full((n, n), bg, dtype=np.int8)
    for r, c in _corners(n, info['arm']):
        output[r, c] = info['frame']
    for color, ports in info['ports'].items():
        count = direction * info['counters'].get(color, 0) if use_counts else 0
        for cells, side, _ in ports:
            target = _rotate_cells(cells, n, count)
            cells = _stamp(info['prototypes'][color], target, (side + count) % 4) if grow else target
            for r, c in cells:
                if not (1 <= r < n - 1 and 1 <= c < n - 1) or output[r, c] != bg:
                    raise ValueError("grown copies clip or overlap")
                output[r, c] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count each color's bar, rotate its buds, and grow the matching prototype."""
    return _render(_parse(input_grid))


def _small_mask(handed=False):
    rows = [[1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] if handed else [[1, 1, 1, 1], [0, 1, 1, 0]]
    return np.array(rows, dtype=bool)


def _make_mask(rng, witness=False, handed=False):
    for _ in range(80):
        kind = int(rng.integers(4))
        width = int(rng.choice((2, 4, 6, 8), p=(.13, .43, .34, .10)))
        depth = int(rng.integers(2, 7))
        center = width // 2
        mask = np.zeros((depth, width), dtype=bool)
        mask[:, center - 1:center + 1] = True
        body = int(rng.integers(1, depth))
        if kind == 0:
            mask[:body] = True
        elif kind == 1:
            # Independent left/right shoulders, holes and notches.
            for r in range(body):
                lo = int(rng.integers(center))
                hi = int(rng.integers(center + 1, width + 1))
                mask[r, lo:hi] = True
            mask[0, :] = True
            for r, c in np.argwhere(mask[:body]):
                if c not in (center - 1, center) and rng.random() < .20:
                    mask[r, c] = False
        elif kind == 2 and width > 2:
            # Diagonal wings are a single eight-connected silhouette.
            wing = min(center - 1, depth - 1)
            mask[:] = False
            mask[wing:, center - 1:center + 1] = True
            for r in range(wing):
                reach = wing - r
                mask[r, center - 1 - reach] = True
                mask[r, center + reach] = True
                if rng.random() < .35:
                    mask[r, center - 1 - reach:center + reach + 1] = True
        else:
            mask[:body] = True
            if width > 2 and body > 1:
                mask[0, int(rng.choice((0, width - 1)))] = False
        rr, cc = np.nonzero(mask)
        mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        candidates = _attachments(mask)
        if len(candidates) != 1 or len(_components(mask)) != 1:
            continue
        canonical = candidates[0]
        if witness and (canonical.shape[1] < 4 or
                        sum(np.array_equal(canonical, np.rot90(canonical, k)) for k in range(4)) != 1):
            continue
        if handed and np.array_equal(canonical, np.fliplr(canonical)):
            continue
        return canonical
    return _small_mask(handed)


def _arrange(rng, n, masks, counts, multiplicities, witness):
    """Choose disjoint inward copies; reverse the counts to get visible buds."""
    all_slots = [(side, offset) for side in range(4) for offset in range(2, n - 3)]
    tasks = [(k, j) for k, amount in enumerate(multiplicities) for j in range(amount)]
    # Four required sides in witnesses, randomized by a common phase.
    required = {}
    motifs, anchors = {}, {}
    if not witness:
        for k, amount in enumerate(multiplicities):
            if amount in (2, 4) and rng.random() < .40:
                motifs[k] = 2 if amount == 2 else 1
    if witness:
        phase = int(rng.integers(4))
        for j, task in enumerate(tasks[:4]):
            required[task] = (phase + j) % 4
    rng.shuffle(tasks)
    used, source_halo, records = set(), set(), [[] for _ in masks]
    for k, j in tasks:
        options = list(rng.permutation(len(all_slots)))
        for index in options:
            side, offset = all_slots[int(index)]
            if (k, j) in required and side != required[k, j]:
                continue
            if k in anchors:
                base_side, base_offset = anchors[k]
                if side != (base_side + j * motifs[k]) % 4 or offset != base_offset:
                    continue
            target = _port(n, side, offset)
            copy = _stamp(masks[k], target, side)
            if (any(not (1 <= r < n - 1 and 1 <= c < n - 1) for r, c in copy)
                    or copy & used):
                continue
            source = _rotate_cells(target, n, -counts[k])
            if source & source_halo:
                continue
            # Witness count-two layer with two buds must actually move.
            if witness and j and counts[k] % 4 == 2:
                other = set().union(*(p[0] for p in records[k]))
                if _rotate_cells(source, n, 2) == other:
                    continue
            used.update(_halo(copy))
            source_halo.update(_halo(source))
            if k in motifs and k not in anchors:
                anchors[k] = ((side - j * motifs[k]) % 4, offset)
            source_side = (side - counts[k]) % 4
            records[k].append((source, source_side, offset))
            break
        else:
            return None
    return records


def _pack(rng, n, bg, frame, arm, colors, ports, masks, counts, distractors):
    objects = []
    for color, mask, count in zip(colors, masks, counts):
        objects.append((color, np.rot90(mask, int(rng.integers(4)))))
        if count:
            # The repeated two-cell witness bars use opposite orientations.
            orient = int(rng.integers(2))
            if len(colors) == 3 and counts == [1, 2, 2] and color in colors[1:]:
                orient = colors.index(color) % 2
            objects.append((color, np.ones((1, count) if orient == 0 else (count, 1), bool)))
    objects.extend(distractors)
    order = list(rng.permutation(len(objects)))
    order.sort(key=lambda i: -objects[i][1].size)
    for attempt in range(20):
        h = int(rng.integers(max(18, n + 5), 31)) if attempt < 12 else 30
        w = int(rng.integers(max(18, n + 5), 31)) if attempt < 12 else 30
        row, col = int(rng.integers(h - n + 1)), int(rng.integers(w - n + 1))
        a = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), bool)
        blocked[max(0, row - 1):min(h, row + n + 1),
                max(0, col - 1):min(w, col + n + 1)] = True
        for r, c in _corners(n, arm):
            a[row + r, col + c] = frame
        for color, layer in zip(colors, ports):
            for cells, _, _ in layer:
                for r, c in cells:
                    a[row + r, col + c] = color
        good = True
        for index in order:
            color, mask = objects[index]
            mh, mw = mask.shape
            local = np.argwhere(mask)
            yy, xx = np.indices((h - mh + 1, w - mw + 1))
            rr = yy.ravel()[:, None] + local[:, 0]
            cc = xx.ravel()[:, None] + local[:, 1]
            candidates = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
            if not len(candidates):
                good = False
                break
            chosen = int(rng.choice(candidates))
            a[rr[chosen], cc[chosen]] = color
            for r, c in _halo(zip(rr[chosen].tolist(), cc[chosen].tolist())):
                if 0 <= r < h and 0 <= c < w:
                    blocked[r, c] = True
        if good and np.count_nonzero(a == bg) > a.size // 2:
            return a
    return None


def _build(rng, index, split, previous_bg):
    witness = split == 'train' and index < 2
    if witness:
        counts = [1, 2, 2] if index == 0 else [3, 4, 0]
        multiplicities = [1, 2, 1] if index == 0 else [1, 1, 2]
        layer_count = 3
    else:
        layer_count = int(rng.integers(1, 5))
        cap = 4 if split == 'train' else 9
        counts = list(map(int, rng.integers(0, cap + 1, layer_count)))
        if split == 'test':
            counts[int(rng.integers(layer_count))] = int(rng.integers(5, 10))
        multiplicities = list(map(int, rng.integers(1, 5, layer_count)))
        # A finite square supports at most eight well-separated grown copies.
        while sum(multiplicities) > 8:
            k = int(rng.choice([j for j, amount in enumerate(multiplicities) if amount > 1]))
            multiplicities[k] -= 1
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    palette = list(map(int, rng.choice([c for c in range(10) if c != bg], layer_count + 1, replace=False)))
    frame, colors = palette[0], palette[1:]
    distractor_colors = [c for c in range(10) if c not in palette and c != bg]
    dcount = 1 if witness and index == 1 else int(rng.integers(3))
    distractors = [(int(rng.choice(distractor_colors)), np.rot90(_make_mask(rng), int(rng.integers(4))))
                   for _ in range(dcount)]
    # Bounded geometry retries retain counts, multiplicities and teaching roles.
    for attempt in range(160):
        n = int(rng.integers(12, 21)) if attempt < 80 else 20
        arm = int(rng.choice((2, 3)))
        masks = [_make_mask(rng, witness, handed=witness and index == 0 and k == 0)
                 for k in range(len(colors))]
        if attempt >= 120:
            masks = [_small_mask(handed=witness and index == 0 and k == 0)
                     for k in range(len(colors))]
        ports = _arrange(rng, n, masks, counts, multiplicities, witness)
        if ports is None:
            continue
        a = _pack(rng, n, bg, frame, arm, colors, ports, masks, counts, distractors)
        if a is None:
            continue
        info = dict(size=n, background=bg, frame=frame, arm=arm,
                    ports=dict(zip(colors, ports)), prototypes=dict(zip(colors, masks)),
                    counters={c: k for c, k in zip(colors, counts) if k})
        output = _render(info)
        # Acceptance re-parses actual pixels; it cannot rely on a chosen neck.
        parsed = _parse(a)
        if not np.array_equal(output, _render(parsed)):
            raise AssertionError("construction and input interpretation disagree")
        if witness and any(np.array_equal(_port_mask(parsed, c), np.rot90(_port_mask(parsed, c), 2))
                           for c in colors if parsed['counters'].get(c, 0) % 2):
            continue
        return dict(input=a, output=output)
    raise RuntimeError("bounded nursery construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for split, amount in (('train', num_train), ('test', num_test)):
        for index in range(amount):
            for _ in range(8):
                pair = _build(rng, index, split, previous_bg)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
            seen.add(key)
            previous_bg = int(np.argmax(np.bincount(pair['input'].ravel(), minlength=10)))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _port_mask(info, color):
    mask = np.zeros((info['size'], info['size']), bool)
    for cells, _, _ in info['ports'][color]:
        for r, c in cells:
            mask[r, c] = True
    return mask


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(a)
            assert b.shape == (info['size'], info['size'])
            assert np.array_equal(b, _render(info)), 'incorrect counted growth'
            assert len(info['distractors']) <= 2
            for color, mask in info['prototypes'].items():
                amount = len(info['ports'][color])
                assert 1 <= amount <= 4
                assert int(np.count_nonzero(b == color)) == int(mask.sum()) * amount, 'copy-area invariant'
            assert sum(map(len, info['ports'].values())) <= 8
            assert np.count_nonzero(b == info['frame']) == len(_corners(info['size'], info['arm']))
            infos[split].append(info)
    train = infos['train']
    demonstrated_counts = {a['counters'].get(c, 0) for a in train for c in a['ports']}
    assert set(range(5)) <= demonstrated_counts, 'show zero, one, two, three and a full cycle'
    assert any(any(len(p) > 1 for p in a['ports'].values()) for a in train), 'show replication'
    assert any(a['distractors'] for a in train), 'show filtering of dockable distractors'
    assert any(not np.array_equal(m, np.fliplr(m)) for a in train for m in a['prototypes'].values()), 'show handedness preservation'
    assert any({a['orientations'].get(c) for c in a['ports'] if a['counters'].get(c) == 2} >= {'H', 'V'}
               for a in train), 'equal counts in different orientations'
    sides = set()
    for a in train:
        for color, ports in a['ports'].items():
            k = a['counters'].get(color, 0)
            sides.update((side + k) % 4 for _, side, _ in ports)
    assert sides == set(range(4)), 'show all inward growth directions'
    # Compare each grown color in isolation as well as complete assemblies:
    # the ablation must change pixels, not merely fail an overlap assertion.
    assert any(not np.array_equal(_render(_one_layer(a, c)),
                                  _render(_one_layer(a, c), use_counts=False))
               for a in train for c in a['ports']), 'B ablation must change grown pixels'
    assert any(not np.array_equal(_render(a), _render(a, grow=False))
               for a in train), 'A ablation must change a training answer'
    assert any(_different_or_invalid(a, direction=-1) for a in train), 'show clockwise direction'
    assert all(max(a['counters'].values(), default=0) <= 4 for a in train)
    assert all(max(a['counters'].values(), default=0) >= 5 for a in infos['test'])


def _one_layer(info, color):
    result = dict(info)
    result['ports'] = {color: info['ports'][color]}
    return result


def _different_or_invalid(info, **kwargs):
    try:
        return not np.array_equal(_render(info), _render(info, **kwargs))
    except ValueError:
        return True


def self_test():
    """Bounded 42-seed structural, semantic, ablation and diversity checks."""
    fingerprints, backgrounds, sizes, shapes, positions = set(), set(), set(), set(), set()
    mask_signatures, multiplicities, lengths, layer_counts, symmetries = set(), set(), set(), set(), set()
    arms, orientations = set(), set()
    diagonal = cycle = repeated = reflected = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        for s in ('train', 'test'):
            for pair in p[s]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                backgrounds.add(info['background'])
                sizes.add(info['size'])
                shapes.add(tuple(pair['input'].shape))
                positions.add((info['row'], info['col']))
                layer_counts.add(len(info['ports']))
                arms.add(info['arm'])
                orientations.update(info['orientations'].values())
                lengths.update(info['counters'].values())
                raw = list(info['counters'].values())
                repeated += len(raw) > len(set(raw))
                for color, mask in info['prototypes'].items():
                    mask_signatures.add(_key(mask))
                    multiplicities.add(len(info['ports'][color]))
                    pm = _port_mask(info, color)
                    symmetries.add('quarter' if np.array_equal(pm, np.rot90(pm)) else
                                   'half' if np.array_equal(pm, np.rot90(pm, 2)) else 'asymmetric')
                    k = info['counters'].get(color, 0)
                    cycle += k in (4, 8)
                    reflected += not np.array_equal(mask, np.fliplr(mask))
                    # Four-connectivity versus the intended eight-connectivity.
                    cells = set(map(tuple, np.argwhere(mask).tolist()))
                    stack = [cells.pop()]
                    while stack:
                        r, c = stack.pop()
                        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if q in cells:
                                cells.remove(q)
                                stack.append(q)
                    diagonal += bool(cells)
    assert len(fingerprints) == 42
    assert backgrounds == set(range(10)) and layer_counts == {1, 2, 3, 4}
    assert len(sizes) >= 7 and len(shapes) >= 40 and len(positions) >= 25
    assert len(mask_signatures) >= 50 and multiplicities == {1, 2, 3, 4}
    assert lengths == set(range(1, 10)) and orientations == {'single', 'H', 'V'}
    assert arms == {2, 3} and diagonal >= 5 and reflected >= 20 and cycle >= 42 and repeated >= 42
    assert symmetries == {'asymmetric', 'half', 'quarter'}
    first, second = generate_puzzle(73, 2, 1), generate_puzzle(73, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(100 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), input_shapes=len(shapes),
                square_sizes=sorted(sizes), prototype_shapes=len(mask_signatures),
                port_multiplicities=sorted(multiplicities), bud_symmetries=sorted(symmetries),
                diagonal_prototypes=diagonal, asymmetric_bodies=reflected,
                full_cycle_layers=cycle, repeated_counter_pairs=repeated)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
