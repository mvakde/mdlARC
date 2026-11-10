"""Competing halos: local population elections on overlapping exterior contours.

Only NumPy and the standard library are used. The solver reads visible cells;
construction choices and random seeds are never consulted by it.
"""
from collections import deque
import json
import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
AUDIT_SEEDS = (0, 936980349, 670356092, 479946021, 53955095, 911277872,
               123813482, 854427784, 917743753, 937286467, 545830406,
               71061308, 847458945, 124332964, 846106541, 400159940,
               425602413, 478134433, 665690714, 129024734, 590958499,
               198243526, 694726289, 106228719, 125464871, 491806447,
               9821305, 933075043, 305884822, 178868126, 598424111,
               803689546, 839621499, 777317920, 999155100, 335622375,
               478596275, 822409910, 686050270, 411233590, 428472232,
               589008926)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=False)


def _dilate(mask):
    x = mask.copy()
    x[1:] |= mask[:-1]
    x[:-1] |= mask[1:]
    x[:, 1:] |= mask[:, :-1]
    x[:, :-1] |= mask[:, 1:]
    return x


def _interior(mask):
    p = np.pad(mask, 1)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _components(mask):
    remaining = mask.copy()
    result = []
    for row, col in zip(*np.nonzero(mask)):
        if not remaining[row, col]:
            continue
        found = np.zeros(mask.shape, dtype=bool)
        queue = deque([(int(row), int(col))])
        remaining[row, col] = False
        while queue:
            r, c = queue.popleft()
            found[r, c] = True
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and remaining[nr, nc]):
                    remaining[nr, nc] = False
                    queue.append((nr, nc))
        result.append(found)
    return result


def _hole_free(mask):
    # Padding joins all exterior background, including across canvas corners.
    return len(_components(~np.pad(mask, 1))) == 1


def _read_input(input_grid):
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(frequencies))
    if frequencies[background] * 2 <= grid.size:
        raise ValueError("background must occupy a strict majority")
    masks = _components(grid != background)
    if not 3 <= len(masks) <= 5:
        raise ValueError("expected three to five separated objects")
    objects = []
    for mask in masks:
        inside = _interior(mask)
        boundary_colors = np.unique(grid[mask & ~inside])
        colors = np.unique(grid[mask])
        if len(boundary_colors) != 1 or len(colors) != 2 or not _hole_free(mask):
            raise ValueError("each solid object needs a uniform shell and an inner accent")
        shell = int(boundary_colors[0])
        accent = int(next(c for c in colors if c != shell))
        core = mask & (grid == accent)
        score = int(core.sum())
        if np.any(core & ~inside) or score * 2 >= int(mask.sum()):
            raise ValueError("the core must be strictly internal and a local minority")
        objects.append({"mask": mask, "core": core, "score": score,
                        "area": int(mask.sum()), "shell": shell,
                        "accent": accent})
    if len({x['shell'] for x in objects}) != 1:
        raise ValueError("objects must share a shell color")
    if len({x['accent'] for x in objects}) != len(objects):
        raise ValueError("objects must have distinct core colors")
    rings = np.stack([_dilate(x['mask']) & (grid == background) for x in objects])
    return grid, background, objects, rings


def _election(background, objects, rings):
    scores = np.array([x['score'] for x in objects])[:, None, None]
    bids = rings * scores
    best = bids.max(axis=0)
    leaders = rings & (bids == best)
    unique = (leaders.sum(axis=0) == 1) & (best > 0)
    output = np.full(best.shape, background, dtype=np.int8)
    for i, obj in enumerate(objects):
        output[unique & leaders[i]] = obj['accent']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase objects; resolve each background contour cell by local core counts."""
    _, bg, objects, rings = _read_input(input_grid)
    return _election(bg, objects, rings)


def _crop(mask):
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()


def _shape(rng):
    """Four silhouette processes; orientation and core arrangement are separate."""
    for _ in range(100):
        mode = int(rng.integers(4))
        h, w = map(int, rng.integers(5, 9, size=2))
        if mode == 0:
            m = np.ones((h, w), dtype=bool)
            for r, c, dr, dc in ((0, 0, 1, 1), (0, w-1, 1, -1),
                                 (h-1, 0, -1, 1), (h-1, w-1, -1, -1)):
                cut = int(rng.integers(0, 3))
                for i in range(cut):
                    for j in range(cut-i):
                        m[r+dr*i, c+dc*j] = False
        elif mode == 1:
            radius = int(rng.integers(2, 5))
            y, x = np.indices((2*radius+1, 2*radius+1))
            m = abs(y-radius) + abs(x-radius) <= radius
        elif mode == 2:
            core = np.zeros((9, 9), dtype=bool)
            r = c = 4
            for _step in range(int(rng.integers(3, 10))):
                core[r, c] = True
                dr, dc = DIRECTIONS[int(rng.integers(4))]
                r = int(np.clip(r+dr, 2, 6))
                c = int(np.clip(c+dc, 2, 6))
            m = _dilate(_dilate(core))
        else:
            m = np.ones((h, w), dtype=bool)
            opening = int(rng.integers(1, 3))
            col = int(rng.integers(1, w-1))
            m[:opening, col:col+1] = False
        m = _crop(np.rot90(m, int(rng.integers(4))))
        if rng.integers(2):
            m = m[:, ::-1].copy()
        if (len(_components(m)) == 1 and _hole_free(m)
                and int(_interior(m).sum()) >= 4):
            return m
    raise RuntimeError("could not construct a silhouette")


def _placements(local, occupied, old_rings, require_triple):
    """All placements with no edge contact and at least one shared halo cell."""
    h, w = local.shape
    ny, nx = 31-h, 31-w
    invalid = np.zeros((ny, nx), dtype=bool)
    blocked = _dilate(occupied)
    for r, c in zip(*np.nonzero(local)):
        invalid |= blocked[r:r+ny, c:c+nx]
    # A padded local ring includes halo cells outside the local bounding box.
    padded = np.pad(local, 1)
    ring = _dilate(padded) & ~padded
    target = np.pad(old_rings >= (2 if require_triple else 1), 1)
    contact = np.zeros((ny, nx), dtype=bool)
    for r, c in zip(*np.nonzero(ring)):
        contact |= target[r:r+ny, c:c+nx]
    return np.argwhere(~invalid & contact)


def _cluster(rng, count, triple):
    for _attempt in range(160):
        masks = []
        occupied = np.zeros((30, 30), dtype=bool)
        old_rings = np.zeros((30, 30), dtype=np.int8)
        for i in range(count):
            local = _shape(rng)
            h, w = local.shape
            if i == 0:
                r, c = 15-h//2, 15-w//2
            else:
                positions = _placements(local, occupied, old_rings, triple and i == 2)
                if not len(positions):
                    break
                r, c = positions[int(rng.integers(len(positions)))]
            mask = np.zeros((30, 30), dtype=bool)
            mask[r:r+h, c:c+w] = local
            masks.append(mask)
            occupied |= mask
            old_rings += _dilate(mask) & ~mask
        if len(masks) != count:
            continue
        rows, cols = np.nonzero(occupied)
        # Empty canvas margins vary independently, including boundary clipping.
        top = max(0, int(rows.min()) - int(rng.integers(0, 5)))
        bottom = min(30, int(rows.max()) + 1 + int(rng.integers(0, 5)))
        left = max(0, int(cols.min()) - int(rng.integers(0, 5)))
        right = min(30, int(cols.max()) + 1 + int(rng.integers(0, 5)))
        cropped = [m[top:bottom, left:right].copy() for m in masks]
        union = np.logical_or.reduce(cropped)
        if union.sum() * 2 >= union.size:
            continue
        rings = np.stack([_dilate(m) & ~union for m in cropped])
        owners = rings.sum(axis=0)
        if triple and owners.max() < 3:
            continue
        if not triple and owners.max() != 2:
            continue
        if any(not np.any(rings[i] & (owners == 1)) for i in range(count)):
            continue
        return cropped, rings
    raise RuntimeError("could not arrange a connected cluster of objects")


def _contest_details(objects, rings):
    bids = rings * np.array([x['score'] for x in objects])[:, None, None]
    top = bids.max(axis=0)
    leaders = rings & (bids == top)
    contested = rings.sum(axis=0) >= 2
    strict = contested & (leaders.sum(axis=0) == 1)
    ties = contested & (leaders.sum(axis=0) >= 2)
    winners = {i for i in range(len(objects)) if np.any(strict & leaders[i])}
    smaller_wins = False
    larger_wins = False
    for i in range(len(objects)):
        for j in range(len(objects)):
            beats = strict & leaders[i] & rings[j]
            if i != j and np.any(beats):
                smaller_wins |= objects[i]['area'] < objects[j]['area']
                larger_wins |= objects[i]['area'] > objects[j]['area']
    return {'strict': strict, 'ties': ties, 'winners': winners,
            'smaller_wins': smaller_wins, 'larger_wins': larger_wins}


def _scores(rng, masks, rings, kind, limit):
    capacities = [min(int(_interior(m).sum()), (int(m.sum())-1)//2, limit)
                  for m in masks]
    for _ in range(250):
        if kind == 'strict_demo':
            high = int(rng.integers(4, min(capacities)+1))
            counts = [1, int(rng.integers(2, high)), high]
            rng.shuffle(counts)
        elif kind == 'tie_demo' or kind == 'tie_test':
            junction = np.argwhere(rings.sum(axis=0) >= 3)
            r, c = junction[int(rng.integers(len(junction)))]
            candidates = np.flatnonzero(rings[:, r, c])
            tied = rng.choice(candidates, 2, replace=False)
            high = int(rng.integers(2, min(capacities)+1))
            counts = [int(rng.integers(1, high)) for _ in masks]
            for i in tied:
                counts[int(i)] = high
        else:
            counts = [int(rng.integers(1, cap+1)) for cap in capacities]
        objects = [{'score': n, 'area': int(m.sum())} for n, m in zip(counts, masks)]
        details = _contest_details(objects, rings)
        if not np.any(details['strict']):
            continue
        if kind == 'strict_demo':
            if len(details['winners']) < 2 or not details['smaller_wins']:
                continue
        if kind in ('tie_demo', 'tie_test') and not np.any(details['ties']):
            continue
        if kind == 'strict_test' and np.any(details['ties']):
            continue
        if kind.endswith('test') and len(details['winners']) < 2:
            continue
        return counts
    raise ValueError("population profile incompatible with contact geometry")


def _core_sites(rng, inside, count):
    sites = [tuple(map(int, p)) for p in np.argwhere(inside)]
    mode = int(rng.integers(3))
    if mode == 0:
        return [sites[int(i)] for i in rng.choice(len(sites), count, replace=False)]
    selected = [sites[int(rng.integers(len(sites)))]]
    while len(selected) < count:
        remaining = [s for s in sites if s not in selected]
        distances = [min(abs(s[0]-p[0])+abs(s[1]-p[1]) for p in selected)
                     for s in remaining]
        target = min(distances) if mode == 1 else max(distances)
        eligible = [s for s, d in zip(remaining, distances) if d == target]
        selected.append(eligible[int(rng.integers(len(eligible)))])
    return selected


def _make_pair(rng, count, kind, limit, previous_palette=None):
    for _attempt in range(140):
        masks, rings = _cluster(rng, count, kind != 'strict_demo')
        try:
            counts = _scores(rng, masks, rings, kind, limit)
        except ValueError:
            continue
        for _ in range(40):
            palette = [int(x) for x in rng.choice(10, count+2, replace=False)]
            if previous_palette is None or (palette[0] != previous_palette[0]
                                             and palette[1] != previous_palette[1]):
                break
        bg, shell = palette[:2]
        grid = np.full(masks[0].shape, bg, dtype=np.int8)
        objects = []
        for i, (mask, score) in enumerate(zip(masks, counts)):
            grid[mask] = shell
            for r, c in _core_sites(rng, _interior(mask), score):
                grid[r, c] = palette[i+2]
            objects.append({'score': score, 'accent': palette[i+2]})
        return {'input': grid, 'output': _election(bg, objects, rings)}
    raise RuntimeError("could not construct an informative pair")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    limit = 8
    first_palette = None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                n = 3 if i < 2 else int(rng.integers(3, 5))
                kind = 'strict_demo' if i == 0 else ('tie_demo' if i == 1 else 'mixed')
            else:
                n = int(rng.integers(4, 6))
                kind = 'tie_test' if rng.integers(2) else 'strict_test'
            for _attempt in range(100):
                pair = _make_pair(rng, n, kind, limit,
                                  first_palette if split == 'train' and i == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and i == 0:
                _, bg, objects, _ = _read_input(pair['input'])
                first_palette = (bg, objects[0]['shell'])
        if split == 'train':
            limit = max(x['score'] for p in puzzle['train']
                        for x in _read_input(p['input'])[2])
    return puzzle


def _check_pair(pair):
    """Validate by counting adjacent component labels, separately from election code."""
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    grid, bg, objects, rings = _read_input(pair['input'])
    output = _grid(pair['output'])
    assert output.shape == grid.shape
    labels = np.full(grid.shape, -1, dtype=int)
    for i, obj in enumerate(objects):
        labels[obj['mask']] = i
    expected = np.full(grid.shape, bg, dtype=np.int8)
    graph = [set() for _ in objects]
    for r, c in zip(*np.nonzero(grid == bg)):
        claimants = set()
        for dr, dc in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                label = int(labels[nr, nc])
                if label >= 0:
                    claimants.add(label)
        for i in claimants:
            graph[i].update(claimants - {i})
        if claimants:
            maximum = max(objects[i]['score'] for i in claimants)
            winners = [i for i in claimants if objects[i]['score'] == maximum]
            if len(winners) == 1:
                expected[r, c] = objects[winners[0]]['accent']
    assert np.array_equal(output, expected), 'incorrect contour ownership'
    assert np.array_equal(solve(grid), expected)
    assert np.all(output[grid != bg] == bg), 'original objects must be erased'
    reached, pending = {0}, [0]
    for i in pending:
        for j in graph[i] - reached:
            reached.add(j)
            pending.append(j)
    assert len(reached) == len(objects), 'contact graph must be connected'
    owners = rings.sum(axis=0)
    assert all(np.any(rings[i] & (owners == 1)) for i in range(len(objects)))
    details = _contest_details(objects, rings)
    assert np.any(details['strict']), 'each example must expose population comparison'
    no_counts = [dict(obj, score=1) for obj in objects]
    assert not np.array_equal(_election(bg, no_counts, rings), expected)
    # A diagonal-only neighbor must remain blank; no eight-neighbor halo shortcut.
    union = grid != bg
    p = np.pad(union, 1)
    diagonal = p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    diagonal_only = diagonal & ~_dilate(union)
    assert np.any(diagonal_only) and np.all(output[diagonal_only] == bg)
    details.update({'objects': objects, 'rings': rings, 'background': bg,
                    'owners': owners, 'graph': graph})
    return details


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            key = _key(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info[split].append(_check_pair(pair))
    first, second = info['train'][:2]
    assert len(first['objects']) == len(second['objects']) == 3
    assert first['owners'].max() == 2 and not np.any(first['ties'])
    assert len(first['winners']) >= 2 and first['smaller_wins']
    assert second['owners'].max() >= 3 and np.any(second['ties'])
    assert first['background'] != second['background']
    assert first['objects'][0]['shell'] != second['objects'][0]['shell']
    trained_limit = max(x['score'] for d in info['train'] for x in d['objects'])
    for d in info['test']:
        assert 4 <= len(d['objects']) <= 5 and d['owners'].max() >= 3
        assert len(d['winners']) >= 2
        assert max(x['score'] for x in d['objects']) <= trained_limit


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded semantic, count, determinism and diversity checks over 42 seeds."""
    shapes, silhouettes, palettes, populations, counts = set(), set(), set(), set(), set()
    signatures = set()
    features = {'clipped': 0, 'interior': 0, 'disconnected_core': 0,
                'larger_wins': 0, 'tie_test': 0, 'strict_test': 0,
                'notched': 0, 'four_way': 0}
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sig = _signature(puzzle)
        assert sig not in signatures
        signatures.add(sig)
        if seed == 0:
            assert sig == _signature(generate_puzzle(seed))
        # Palette and orientation are semantic symmetries, including clipping.
        sample = puzzle['train'][0]
        turns = seed % 4
        assert np.array_equal(solve(np.rot90(sample['input'], turns)),
                              np.rot90(sample['output'], turns))
        permutation = np.array([7, 3, 9, 0, 8, 2, 5, 1, 6, 4], dtype=np.int8)
        assert np.array_equal(solve(permutation[sample['input']]),
                              permutation[sample['output']])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _check_pair(pair)
                grid = np.asarray(pair['input'])
                shapes.add(grid.shape)
                counts.add(len(d['objects']))
                palettes.add((d['background'], d['objects'][0]['shell'],
                              tuple(x['accent'] for x in d['objects'])))
                features['larger_wins'] += d['larger_wins']
                features['four_way'] += d['owners'].max() == 4
                if split == 'test':
                    features['tie_test' if np.any(d['ties']) else 'strict_test'] += 1
                for obj in d['objects']:
                    mask = obj['mask']
                    local = _crop(mask)
                    silhouettes.add((local.shape, np.packbits(local).tobytes()))
                    populations.add(obj['score'])
                    contact = any(np.any(edge) for edge in
                                  (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
                    features['clipped' if contact else 'interior'] += 1
                    features['disconnected_core'] += len(_components(obj['core'])) > 1
                    features['notched'] += any(
                        np.any(row) and np.ptp(np.flatnonzero(row))+1 != row.sum()
                        for row in local)
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(silhouettes) >= 60 and len(palettes) >= 80
    assert {3, 4, 5} <= counts and {1, 2, 3, 4} <= populations
    assert all(features[k] > 0 for k in features if k != 'four_way')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(314159, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted illegal counts {a}/{b}')
    # Explicit tie topology: two equal strongest competitors and a third weaker
    # one. The weaker third must not break the stalemate at the shared center.
    centers = ((4, 7), (7, 4), (7, 10))
    yy, xx = np.indices((15, 15))
    fixture = np.zeros((15, 15), dtype=np.int8)
    for i, ((r, c), n) in enumerate(zip(centers, (2, 2, 1))):
        mask = abs(yy-r)+abs(xx-c) <= 2
        fixture[mask] = 8
        for y, x in np.argwhere(_interior(mask))[:n]:
            fixture[y, x] = i+1
    assert solve(fixture)[7, 7] == 0
    assert solve(fixture)[0, 0] == 0
    # Change only populations of the lower two contestants in a strict demo.
    # Their shared border must change while geometry and literal roles stay fixed.
    first = generate_puzzle(0)['train'][0]
    changed = first['input'].copy()
    _, bg, objs, _ = _read_input(changed)
    low, middle, _ = sorted(objs, key=lambda obj: obj['score'])
    for obj, new_count in ((low, middle['score']), (middle, low['score'])):
        changed[obj['core']] = obj['shell']
        for r, c in np.argwhere(_interior(obj['mask']))[:new_count]:
            changed[r, c] = obj['accent']
    assert np.array_equal(changed != bg, first['input'] != bg)
    assert not np.array_equal(solve(changed), first['output'])
    return {'seeds': 42, 'puzzles': len(signatures), 'grid_shapes': len(shapes),
            'silhouettes': len(silhouettes), 'palettes': len(palettes),
            'core_counts': sorted(populations), 'object_counts': sorted(counts),
            'features': {k: int(v) for k, v in features.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
