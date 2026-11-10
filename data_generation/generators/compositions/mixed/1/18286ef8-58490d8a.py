"""Fill a keyed object tally, then relocate its entire addressed panel.

Only NumPy and the standard library are used. Generation labels are rendered
from planned counts; solve reconstructs counts and roles from the input alone.
No parent generator, disk payload, or puzzle-level hidden convention is used.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np


ADDRESSES = tuple((r, c) for r in range(3) for c in range(3))
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
CARDINALS = ((0, 1), (1, 0), (1, 2), (2, 1))
HOST_SEEDS = (
    0, 29054219, 337441158, 486192233, 524224256, 774166524,
    108874803, 560261721, 991629647, 561247494, 547528443,
    43978220, 379129644, 803995603, 74509426, 970037302,
    899338577, 739566402, 699071990, 972130510, 232158967,
    134826122, 834462130, 107361526, 52364092, 832079202,
    799295239, 18487146, 366527063, 53883983, 118939713,
    103898905, 750937035, 466354932, 903363251, 173490079,
    693137135, 197276970, 718383415, 716687193, 318286872,
    707676887,
)
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
SINGLE = ((0, 0),)
GLYPHS = (
    DIAMOND, SQUARE, SINGLE,
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= size <= 30 for size in grid.shape):
        raise ValueError("grids must be 2D with dimensions in 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or not np.all((0 <= grid) & (grid <= 9)):
        raise ValueError("grid colors must be integers in 0..9")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _components(mask, diagonal=True):
    """Monochrome component cells; deterministic iteration, including corners."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r)+abs(c) == 1)]
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                point = (r+dr, c+dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        groups.append(cells)
    return groups


def _bounds(address, height, width):
    r, c = address
    return (slice(r*(height+1), r*(height+1)+height),
            slice(c*(width+1), c*(width+1)+width))


def _blank(height, width, background, divider):
    result = np.full((3*height+2, 3*width+2), background, dtype=np.int8)
    result[[height, 2*height+1], :] = divider
    result[:, [width, 2*width+1]] = divider
    return result


def _inspect(input_grid):
    """Parse visible structure; no labels, RNG, or generation metadata."""
    grid = _grid(input_grid)
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    assert len(rows) == len(cols) == 2, "Exactly two full dividers per axis"
    height, width = rows[0], cols[0]
    assert 6 <= height <= 9 and 5 <= width <= 9
    assert grid.shape == (3*height+2, 3*width+2)
    assert rows == [height, 2*height+1] and cols == [width, 2*width+1]
    divider = int(grid[rows[0], 0])
    divmask = np.zeros(grid.shape, bool)
    divmask[rows, :] = True
    divmask[:, cols] = True
    assert np.all(grid[divmask] == divider)
    assert not np.any(grid[~divmask] == divider)
    frequencies = np.bincount(grid[~divmask], minlength=10)
    background = int(np.argmax(frequencies))
    assert frequencies[background] > np.count_nonzero(~divmask)*.55
    candidates = []
    for address in ADDRESSES:
        panel = grid[_bounds(address, height, width)]
        counts = np.bincount(panel.ravel(), minlength=10)
        field = int(np.argmax(counts))
        if field != background and counts[field] > panel.size/2:
            candidates.append((address, field, panel.copy()))
    assert len(candidates) == 1, "One filled tally panel"
    source, field, panel = candidates[0]
    # The upper-right local 3x3 register is part of the moving payload.
    register = panel[:3, width-3:]
    values, counts = np.unique(register, return_counts=True)
    assert sorted(map(int, counts)) == [1, 8]
    register_fill = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    destination = tuple(map(int, np.argwhere(register == marker)[0]))
    structural = (background, divider, field, register_fill, marker)
    assert len(set(structural)) == 5
    keymask = panel != field
    keymask[:3, width-3:] = False
    positions = [tuple(map(int, point)) for point in np.argwhere(keymask)]
    assert 1 <= len(positions) <= 3
    assert all(3 <= r <= height-2 and c == 1 for r, c in positions)
    assert all(b[0]-a[0] >= 2 for a, b in zip(positions, positions[1:]))
    keys = tuple(int(panel[p]) for p in positions)
    assert len(set(keys)) == len(keys) and not set(keys) & set(structural)
    assert np.count_nonzero(grid == marker) == 1
    assert np.count_nonzero(grid == register_fill) == 8
    outside = ~divmask
    outside[_bounds(source, height, width)] = False
    assert np.mean(grid[outside] == background) > .60
    assert not np.any(np.isin(grid[outside], (field, register_fill, marker)))
    colors = sorted(set(map(int, grid[outside]))-{background})
    objects, object_panels, owner = {}, {}, np.full(grid.shape, -1, int)
    serial = 0
    for color in colors:
        groups = _components((grid == color) & outside)
        objects[color] = tuple(_normalize(group) for group in groups)
        object_panels[color] = []
        for group in groups:
            addresses = {(r//(height+1), c//(width+1)) for r, c in group}
            assert len(addresses) == 1, "Objects cannot cross dividers"
            object_panels[color].append(next(iter(addresses)))
            for r, c in group:
                owner[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        near = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((near < 0) | (near == owner[r, c])), "Separate objects need a corner-inclusive halo"
    counts = {color: len(shapes) for color, shapes in objects.items()}
    assert set(keys) <= set(counts), "Every key has a positive object count"
    completed = panel.copy()
    for (r, c), color in zip(positions, keys):
        assert c+2*(counts[color]-1) <= width-2, "No tally clipping"
        for i in range(counts[color]):
            completed[r, c+2*i] = color
    return dict(grid=grid, height=height, width=width, divider_mask=divmask,
                background=background, divider=divider, field=field,
                register_fill=register_fill, marker=marker, source=source,
                destination=destination, panel=panel, completed=completed,
                positions=positions, keys=keys, counts=counts, objects=objects,
                object_panels=object_panels, outside=outside,
                distractors=tuple(color for color in colors if color not in keys))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count before clearing; move the completed rectangle without rotation."""
    meta = _inspect(input_grid)
    result = _blank(meta['height'], meta['width'], meta['background'], meta['divider'])
    result[_bounds(meta['destination'], meta['height'], meta['width'])] = meta['completed']
    return result


def _shape(rng):
    if rng.random() < .65:
        cells = GLYPHS[int(rng.integers(len(GLYPHS)))]
    elif rng.random() < .35:
        cells = tuple((0, c) for c in range(int(rng.integers(1, 6))))
    else:
        side = int(rng.integers(2, 5))
        target = int(rng.integers(2, min(10, side*side)+1))
        cells = {(side//2, side//2)}
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < side and 0 <= c+dc < side}-cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = tuple((r, -c) for r, c in cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    return _normalize(cells)


def _place(grid, blocked, color, shape, panels, height, width, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    options = []
    for address in panels:
        rs, cs = _bounds(address, height, width)
        # A second color-dominated compartment would obscure the payload cue.
        if np.count_nonzero(grid[rs, cs] == color)+len(shape) > height*width/2:
            continue
        for r in range(rs.start, rs.stop-sh+1):
            for c in range(cs.start, cs.stop-sw+1):
                if all(not blocked[r+dr, c+dc] for dr, dc in shape):
                    options.append((r, c))
    if not options:
        return False
    r0, c0 = options[int(rng.integers(len(options)))]
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _make_pair(rng, spec):
    height, width = spec['height'], spec['width']
    background, divider, field, register_fill, marker = spec['roles']
    keys, counts = spec['keys'], spec['counts']
    # Select from the finite legal separated row combinations, without retries.
    rowsets = [rows for rows in combinations(range(3, height-1), len(keys))
               if all(b-a >= 2 for a, b in zip(rows, rows[1:]))]
    positions = rowsets[int(rng.integers(len(rowsets)))]
    panel = np.full((height, width), field, np.int8)
    panel[:3, width-3:] = register_fill
    panel[spec['destination'][0], width-3+spec['destination'][1]] = marker
    for row, color in zip(positions, keys):
        panel[row, 1] = color
    completed = panel.copy()
    for row, color, count in zip(positions, keys, counts):
        completed[row, 1:2*count:2] = color
    panels = [address for address in ADDRESSES if address != spec['source']]
    # The first keyed category spans panels and includes the destination.
    first_panels = [spec['destination']] if spec['destination'] != spec['source'] else []
    remaining = [p for p in panels if p not in first_panels]
    rng.shuffle(remaining)
    first_panels.extend(remaining)
    objects = []
    for index, (color, count) in enumerate(zip(keys+spec['distractors'], counts+spec['dcounts'])):
        prototype = _shape(rng)
        for j in range(count):
            shape = _shape(rng) if rng.random() < .75 else prototype
            if spec['witness'] and index == 0 and j < 3:
                shape = (DIAMOND, SQUARE, SINGLE)[j]
            preferred = [first_panels[j % len(first_panels)]] if index == 0 else panels
            objects.append((color, shape, preferred, index == 0))
    # Reserve a strict background majority regardless of random shape draws.
    # Only shape size can change here; counts and mandatory witnesses cannot.
    area = sum(len(obj[1]) for obj in objects)
    budget = int(8*height*width*.35)
    for i in sorted(range(len(objects)), key=lambda i: -len(objects[i][1])):
        if area <= budget:
            break
        color, shape, allowed, first = objects[i]
        if spec['witness'] and first:
            continue
        objects[i] = color, SINGLE, allowed, first
        area -= len(shape)-1
    assert area <= budget
    # Place the distributed category first, then pack the other larger objects.
    objects.sort(key=lambda obj: (not obj[3], -len(obj[1])))
    for _ in range(16):
        grid = _blank(height, width, background, divider)
        grid[_bounds(spec['source'], height, width)] = panel
        blocked = grid == divider
        blocked[_bounds(spec['source'], height, width)] = True
        if not all(_place(grid, blocked, color, shape, allowed, height, width, rng)
                   for color, shape, allowed, _ in objects):
            continue
        output = _blank(height, width, background, divider)
        output[_bounds(spec['destination'], height, width)] = completed
        # Planned counts are never replaced with observed counts on packing failure.
        meta = _inspect(grid)
        assert all(meta['counts'][c] == n for c, n in zip(keys+spec['distractors'], counts+spec['dcounts']))
        assert np.array_equal(solve(grid), output)
        return {'input': grid, 'output': output}
    raise RuntimeError("Bounded object packing failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.permutation(10)[:5]))
    pool = [c for c in range(10) if c not in roles]
    rng.shuffle(pool)
    first_key, ignored_key = pool[:2]
    regime = int(rng.integers(3))  # address, count, category transfer
    heldout = []
    if regime == 0:
        # Keep enough corner/cardinal choices for the orientation witnesses.
        options = list(ADDRESSES)
        rng.shuffle(options)
        for address in options:
            if len(heldout) == num_test:
                break
            kept = [a for a in ADDRESSES if a not in heldout+[address]]
            if any(a in CORNERS for a in kept) and any(a in CARDINALS for a in kept):
                heldout.append(address)
    train_pool = [a for a in ADDRESSES if a not in heldout]
    corner = [a for a in train_pool if a in CORNERS]
    cardinal = [a for a in train_pool if a in CARDINALS]
    directions = [corner[int(rng.integers(len(corner)))], cardinal[int(rng.integers(len(cardinal)))]]
    directions += [train_pool[int(rng.integers(len(train_pool)))] for _ in range(num_train-2)]
    test_directions = heldout if heldout else [ADDRESSES[int(rng.integers(9))] for _ in range(num_test)]
    fixed_palette = rng.random() < .25
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, addresses in (('train', directions), ('test', test_directions)):
        for index, destination in enumerate(addresses):
            witness = split == 'train' and index == 0
            role_witness = split == 'train' and index == 1
            if witness or role_witness or fixed_palette:
                pair_roles, pair_pool = roles, pool.copy()
            else:
                palette = list(map(int, rng.permutation(10)))
                pair_roles, pair_pool = tuple(palette[:5]), palette[5:]
            if witness:
                key_count = 2
                pair_pool = [first_key]+[c for c in pair_pool if c not in (first_key, ignored_key)]
                keys, distractors = pair_pool[:2], [ignored_key]
            elif role_witness:
                key_count = 1
                keys, distractors = [ignored_key], [first_key]
            else:
                high = 2 if regime == 2 and split == 'train' else 3
                key_count = 3 if regime == 2 and split == 'test' else int(rng.integers(1, high+1))
                keys = pair_pool[:key_count]
                nd = int(rng.integers(0, min(2, len(pair_pool)-key_count)+1))
                distractors = pair_pool[key_count:key_count+nd]
            height = 9 if key_count == 3 else int(rng.integers(6 if key_count == 1 else 7, 10))
            width = int(rng.integers(7 if witness else 5, 10))
            if regime == 1 and split == 'test':
                width = 9
            cap = min((width-1)//2, 3 if regime == 1 and split == 'train' else 4)
            counts = [int(rng.integers(1, cap+1)) for _ in keys]
            if witness:
                counts[0] = 3
            if role_witness:
                counts[0] = 2
            if regime == 1 and split == 'test':
                counts[int(rng.integers(key_count))] = 4
            dcounts = [int(rng.integers(1, 5)) for _ in distractors]
            if role_witness:
                dcounts[0] = max(3, dcounts[0])
            # Some later examples self-address, but the two witnesses move.
            source_options = [a for a in ADDRESSES if a != destination]
            if not (witness or role_witness) and rng.random() < .12:
                source = destination
            else:
                source = source_options[int(rng.integers(len(source_options)))]
            spec = dict(height=height, width=width, roles=pair_roles, keys=keys,
                        counts=counts, distractors=distractors, dcounts=dcounts,
                        source=source, destination=destination, witness=witness)
            for _ in range(8):
                pair = _make_pair(rng, spec)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction failed")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(meta):
    keyed_shapes = [shape for color in meta['keys'] for shape in meta['objects'][color]]
    diagonal = False
    for shape in keyed_shapes:
        mask = np.zeros((1+max(r for r, c in shape), 1+max(c for r, c in shape)), bool)
        for point in shape:
            mask[point] = True
        diagonal |= len(_components(mask, False)) > 1
    mixed = any(len({len(shape) for shape in meta['objects'][color]}) > 1 for color in meta['keys'])
    witness = any({DIAMOND, SQUARE, SINGLE} <= set(meta['objects'][color]) for color in meta['keys'])
    return diagonal, mixed, witness


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, groups = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert _key(grid) not in seen
            seen.add(_key(grid))
            meta = _inspect(grid)
            assert output.shape == grid.shape
            assert np.array_equal(output, solve(grid))
            assert np.array_equal(grid[meta['divider_mask']], output[meta['divider_mask']])
            # The completed payload includes every register pixel and blank margin.
            assert np.array_equal(output[_bounds(meta['destination'], meta['height'], meta['width'])], meta['completed'])
            for address in ADDRESSES:
                if address != meta['destination']:
                    assert np.all(output[_bounds(address, meta['height'], meta['width'])] == meta['background'])
            assert np.array_equal(meta['panel'][:3, -3:], meta['completed'][:3, -3:])
            assert np.count_nonzero(output == meta['marker']) == 1
            for (row, col), color in zip(meta['positions'], meta['keys']):
                actual = list(map(int, np.flatnonzero(meta['completed'][row] == color)))
                assert actual == [col+2*i for i in range(meta['counts'][color])]
            assert not np.array_equal(grid, output)
            groups[split].append(meta)
    train, test = groups['train'], groups['test']
    assert any(all(_features(meta)) for meta in train)
    assert any(meta['destination'] in CORNERS and meta['source'] != meta['destination'] for meta in train)
    assert any(meta['destination'] in CARDINALS and meta['source'] != meta['destination'] for meta in train)
    assert any(set(a['keys']) & set(b['distractors']) for a in train for b in train)
    assert len({len(meta['keys']) for meta in train}) >= 2
    assert any(len(set(meta['object_panels'][c])) >= 3 for meta in train for c in meta['keys'])
    assert any(meta['destination'] in meta['object_panels'][c]
               for meta in train for c in meta['keys'])
    # Verify that there is a genuine transfer along at least one chosen axis.
    train_targets = {meta['destination'] for meta in train}
    train_max = max(meta['counts'][c] for meta in train for c in meta['keys'])
    train_keys = max(len(meta['keys']) for meta in train)
    assert all(meta['destination'] not in train_targets or
               max(meta['counts'][c] for c in meta['keys']) > train_max or
               len(meta['keys']) > train_keys for meta in test)


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 host seeds, semantic mutations and all count choices; no disk writes."""
    signatures, layouts, targets, sources, shapes, palettes = set(), set(), set(), set(), set(), set()
    key_counts, object_counts, self_address, shifts = set(), set(), False, set()
    role_colors = [set() for _ in range(5)]
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        assert signature not in signatures
        signatures.add(signature)
        assert signature == _digest(generate_puzzle(seed))
        train = [_inspect(pair['input']) for pair in puzzle['train']]
        known_targets = {meta['destination'] for meta in train}
        train_max = max(meta['counts'][c] for meta in train for c in meta['keys'])
        train_k = max(len(meta['keys']) for meta in train)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                meta = _inspect(before)
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), "solve mutated its input"
                layouts.add((meta['height'], meta['width']))
                targets.add(meta['destination'])
                sources.add(meta['source'])
                key_counts.add(len(meta['keys']))
                object_counts.update(meta['counts'][c] for c in meta['keys'])
                shapes.update(s for objects in meta['objects'].values() for s in objects)
                palette = tuple(meta[k] for k in ('background', 'divider', 'field', 'register_fill', 'marker'))
                palettes.add(palette)
                for colors, color in zip(role_colors, palette):
                    colors.add(color)
                self_address |= meta['source'] == meta['destination']
                if split == 'test':
                    if meta['destination'] not in known_targets:
                        shifts.add('address')
                    if max(meta['counts'][c] for c in meta['keys']) > train_max:
                        shifts.add('count')
                    if len(meta['keys']) > train_k:
                        shifts.add('categories')
                # Ablations: keeping completed tally at source or omitting counts
                # must fail on the required moving/extension demonstrations.
                if meta['source'] != meta['destination']:
                    stationary = _blank(meta['height'], meta['width'], meta['background'], meta['divider'])
                    stationary[_bounds(meta['source'], meta['height'], meta['width'])] = meta['completed']
                    assert not np.array_equal(stationary, pair['output'])
                if max(meta['counts'][c] for c in meta['keys']) > 1:
                    uncounted = _blank(meta['height'], meta['width'], meta['background'], meta['divider'])
                    uncounted[_bounds(meta['destination'], meta['height'], meta['width'])] = meta['panel']
                    assert not np.array_equal(uncounted, pair['output'])
        # Input mutations have predictable effects without using generation state.
        pair = next(p for p in puzzle['train'] if _features(_inspect(p['input']))[2])
        meta = _inspect(pair['input'])
        moved_source = pair['input'].copy()
        a = _bounds(meta['source'], meta['height'], meta['width'])
        other = next(address for address in ADDRESSES if address != meta['source'])
        b = _bounds(other, meta['height'], meta['width'])
        moved_source[a], moved_source[b] = moved_source[b].copy(), moved_source[a].copy()
        assert np.array_equal(solve(moved_source), pair['output'])
        color = next(c for c in meta['keys'] if {DIAMOND, SQUARE, SINGLE} <= set(meta['objects'][c]))
        removed = pair['input'].copy()
        group = _components((removed == color) & meta['outside'])[0]
        for point in group:
            removed[point] = meta['background']
        expected = pair['output'].copy()
        dest = _bounds(meta['destination'], meta['height'], meta['width'])
        row = meta['positions'][meta['keys'].index(color)][0]
        expected[dest[0].start+row,
                 dest[1].start+1+2*(meta['counts'][color]-1)] = meta['field']
        assert np.array_equal(solve(removed), expected)
        ignored = pair['input'].copy()
        ignored[(ignored == meta['distractors'][0]) & meta['outside']] = meta['background']
        assert np.array_equal(solve(ignored), pair['output'])
        redirected = pair['input'].copy()
        old_r, old_c = meta['destination']
        new_address = (old_r, (old_c+1) % 3)
        redirected[a[0].start+old_r, a[1].start+meta['width']-3+old_c] = meta['register_fill']
        redirected[a[0].start+new_address[0], a[1].start+meta['width']-3+new_address[1]] = meta['marker']
        payload = meta['completed'].copy()
        payload[old_r, meta['width']-3+old_c] = meta['register_fill']
        payload[new_address[0], meta['width']-3+new_address[1]] = meta['marker']
        expected = _blank(meta['height'], meta['width'], meta['background'], meta['divider'])
        expected[_bounds(new_address, meta['height'], meta['width'])] = payload
        assert np.array_equal(solve(redirected), expected)
        # Mutate one label; validator must reject it.
        wrong = {s: [{'input': p['input'], 'output': p['output'].copy()} for p in puzzle[s]]
                 for s in ('train', 'test')}
        wrong['test'][0]['output'][0, 0] = (int(wrong['test'][0]['output'][0, 0])+1) % 10
        try:
            validate_generated_puzzle(wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted label accepted")
    assert len(signatures) == 42 and len(layouts) >= 12 and len(palettes) >= 50
    assert targets == sources == set(ADDRESSES)
    assert key_counts == {1, 2, 3} and object_counts == {1, 2, 3, 4}
    assert all(colors == set(range(10)) for colors in role_colors)
    assert len(shapes) >= 40 and self_address
    assert shifts == {'address', 'count', 'categories'}
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(9000+10*a+b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'panel_sizes': len(layouts),
            'structural_palettes': len(palettes), 'object_shapes': len(shapes),
            'destinations': len(targets), 'sources': len(sources), 'shifts': sorted(shifts)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
