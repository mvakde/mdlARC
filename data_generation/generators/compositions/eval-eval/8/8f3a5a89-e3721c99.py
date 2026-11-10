"""Topological echo rooms: use a marked free-space room as a topology exemplar.

Only standard-library modules and NumPy are used. No files, examples, seeds, or
generation metadata are consulted by solve. Self-tests retain aggregates only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
AUDIT_SEEDS = (0, 473730296, 9373151, 221411141, 235913191, 912408453,
               549145367, 669771807, 719154612, 896891613, 91720916,
               289272064, 867428669, 579778804, 296480281, 166651968,
               486952478, 782198943, 208589598, 59677478, 426039229,
               142363763, 173597907, 377014554, 873001463, 601782875,
               508519807, 990394708, 527536269, 713387181, 526617195,
               491060678, 274739560, 787645509, 478444871, 262581739,
               612364168, 487952217, 185537621, 409395742, 134041897,
               750760588)


def _components(mask):
    """Four-connected components, returned as full-canvas Boolean masks."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        part = np.zeros_like(mask)
        stack = [(r, c)]
        seen[r, c] = part[r, c] = True
        while stack:
            x, y = stack.pop()
            for dr, dc in N4:
                a, b = x + dr, y + dc
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                    seen[a, b] = part[a, b] = True
                    stack.append((a, b))
        result.append(part)
    return result


def _spread(allowed, starts):
    """Flood with Boolean dilation; also works on the padded exterior."""
    reached = np.asarray(starts, dtype=bool) & allowed
    while True:
        expanded = reached.copy()
        expanded[1:] |= reached[:-1]
        expanded[:-1] |= reached[1:]
        expanded[:, 1:] |= reached[:, :-1]
        expanded[:, :-1] |= reached[:, 1:]
        expanded &= allowed
        if np.array_equal(expanded, reached):
            return reached
        reached = expanded


def _dilate(mask, radius=1):
    h, w = mask.shape
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask, dtype=bool)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            result |= padded[dr:dr + h, dc:dc + w]
    return result


def _erode(mask, radius=2):
    return ~_dilate(~np.pad(mask, radius), radius)[radius:-radius, radius:-radius]


def _topology(room):
    """Return bounded complement components, exterior, and external inner rim.

    In particular, an island is a hole of the *free-space object*. Its boundary
    is absent from the rim; the canvas edge is an exterior boundary.
    """
    padded = np.pad(room, 1)
    starts = np.zeros_like(padded)
    starts[0, 0] = True
    exterior = _spread(~padded, starts)
    holes = _components((~padded & ~exterior)[1:-1, 1:-1])
    rim = room & _dilate(exterior)[1:-1, 1:-1]
    return holes, exterior, rim


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return grid


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or sum(counts == 1) != 1 or sum(counts == counts.max()) != 1:
        raise ValueError('need three roles: dominant background, structure, singleton marker')
    bg = int(colors[np.argmax(counts)])
    ink = int(colors[counts == 1][0])
    wall = int(next(c for c in colors if c not in (bg, ink)))
    rooms = []
    target = None
    for mask in _components((grid == bg) | (grid == ink)):
        holes, exterior, rim = _topology(mask)
        info = dict(mask=mask, holes=holes, exterior=exterior, rim=rim,
                    count=len(holes), area=int(mask.sum()))
        if np.any(mask & (grid == ink)):
            target = len(rooms)
        rooms.append(info)
    if target is None:
        raise ValueError('marker must belong to a background compartment')
    return dict(grid=grid, background=bg, structure=wall, marker=ink,
                rooms=rooms, target=target, count=rooms[target]['count'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _analyze(input_grid)
    output = p['grid'].copy()
    for room in p['rooms']:
        if room['count'] == p['count']:
            output[room['rim']] = p['marker']
    return output


def _sites(core, number):
    """A conservative deterministic packing certificate for small islands."""
    available = core.copy()
    sites = []
    for r, c in np.argwhere(core):
        r, c = int(r), int(c)
        if available[r, c]:
            sites.append((r, c))
            available[max(0, r-1):r+2, max(0, c-1):c+2] = False
            if len(sites) == number:
                break
    return sites


def _roomy(mask):
    core = _erode(mask)
    return int(mask.sum()) >= 48 and len(_sites(core, 3)) == 3


def _cut(rng, room, jagged):
    """Try a seam through an existing room, rather than choosing a slot layout."""
    transposed = bool(rng.integers(2))
    work = room.T if transposed else room
    h, w = work.shape
    rr, cc = np.nonzero(work)
    left, right = int(cc.min()), int(cc.max())
    if right - left < 10:
        return None
    base = int(rng.integers(left + 5, right - 4))
    thickness = 1 + int(rng.random() < .18)
    profile = np.full(h, base, dtype=int)
    if jagged:
        cuts = sorted(set([0, h] + [int(x) for x in rng.integers(2, h-1, size=int(rng.integers(1, 5)))]))
        for start, stop in zip(cuts, cuts[1:]):
            profile[start:stop] = base + int(rng.integers(-2, 3))
    seam = np.zeros_like(work)
    for r, c in enumerate(profile):
        seam[r, c:c+thickness] = True
        if r:
            a, b = sorted((int(profile[r-1]), int(c)))
            seam[r, a:b+thickness] = True
    seam &= work
    if transposed:
        seam = seam.T
    children = _components(room & ~seam)
    if len(children) != 2 or not all(_roomy(c) for c in children):
        return None
    # No cut may accidentally leave an unconnected structural island.
    if any(_topology(c)[0] for c in children):
        return None
    return seam, children


def _partition(rng, h, w, wanted, rich):
    walls = np.zeros((h, w), dtype=bool)
    rooms = [~walls]
    for _ in range(100):
        if len(rooms) == wanted:
            return walls, rooms
        sizes = np.asarray([r.sum() for r in rooms], dtype=float)
        index = int(rng.choice(len(rooms), p=sizes / sizes.sum()))
        result = _cut(rng, rooms[index], bool(rng.random() < (.85 if rich else .4)))
        if result is not None:
            seam, children = result
            walls |= seam
            rooms[index:index+1] = children
    return None


def _intrude(rng, walls, rooms, attempts):
    """Grow wall-connected bays without making new rooms or new islands."""
    h, w = walls.shape
    for _ in range(attempts):
        index = int(rng.integers(len(rooms)))
        room = rooms[index]
        near = _dilate(walls) & room
        near[0] |= room[0]
        near[-1] |= room[-1]
        near[:, 0] |= room[:, 0]
        near[:, -1] |= room[:, -1]
        choices = np.argwhere(near)
        if not len(choices):
            continue
        start = tuple(int(x) for x in choices[int(rng.integers(len(choices)))])
        grown = np.zeros_like(walls)
        grown[start] = True
        size = int(rng.integers(2, 12))
        for _ in range(size - 1):
            frontier = np.zeros_like(walls)
            frontier[1:] |= grown[:-1]
            frontier[:-1] |= grown[1:]
            frontier[:, 1:] |= grown[:, :-1]
            frontier[:, :-1] |= grown[:, 1:]
            choices = np.argwhere(frontier & room & ~grown)
            if not len(choices):
                break
            grown[tuple(choices[int(rng.integers(len(choices)))])] = True
        trial = room & ~grown
        if len(_components(trial)) == 1 and _roomy(trial) and not _topology(trial)[0]:
            rooms[index] = trial
            walls |= grown
    return walls, rooms


def _islands(rng, room, sizes):
    """Grow solid, separate islands of specified areas, entirely inside a room."""
    core = _erode(room)
    if not sizes:
        return np.zeros_like(room)
    for _ in range(18):
        occupied = np.zeros_like(room)
        success = True
        # Larger islands go first; the resulting count does not depend on order.
        for size in sorted(sizes, reverse=True):
            allowed = core & ~_dilate(occupied)
            candidates = np.argwhere(allowed)
            if not len(candidates):
                success = False
                break
            island = np.zeros_like(room)
            island[tuple(candidates[int(rng.integers(len(candidates)))])] = True
            for _ in range(size - 1):
                fringe = np.zeros_like(room)
                fringe[1:] |= island[:-1]
                fringe[:-1] |= island[1:]
                fringe[:, 1:] |= island[:, :-1]
                fringe[:, :-1] |= island[:, 1:]
                choices = np.argwhere(fringe & allowed & ~island)
                if not len(choices):
                    success = False
                    break
                island[tuple(choices[int(rng.integers(len(choices)))])] = True
            if not success or _topology(island)[0]:
                success = False
                break
            occupied |= island
        if success and len(_components(room & ~occupied)) == 1:
            return occupied
    return None


def _fallback_partition(ordinal, wanted):
    """Finite roomy partition; distinct shapes for all fourteen possible pairs."""
    h, w = 27 + ordinal % 4, 27 + ordinal // 4
    walls = np.zeros((h, w), dtype=bool)
    walls[:, 8] = walls[:, 17] = True
    for i in range(wanted - 3):
        a, b = ((0, 8), (9, 17), (18, w))[i]
        walls[11 + i + ordinal % 3, a:b] = True
    # A visible boundary notch in every fallback room also guarantees genuine
    # diagonal-only rim cells, rather than relying on a rectangular fixture.
    for room in _components(~walls):
        rr, cc = np.nonzero(room)
        r, c = int(rr.min()), (int(cc.min()) + int(cc.max())) // 2
        walls[r:r+2, c] = True
    return walls, _components(~walls)


def _populate(rng, walls, rooms, genus, peers, witness, palette, marker_rim):
    # Choose peers with unequal areas. The marked sample alternates between a
    # smaller and a larger member, never defining a fixed position/size rule.
    areas = [int(r.sum()) for r in rooms]
    order = [int(x) for x in np.argsort(areas)]
    if max(areas) - min(areas) < 6:
        return None
    subsets = [list(group) for group in combinations(range(len(rooms)), peers)
               if max(areas[i] for i in group) - min(areas[i] for i in group) >= 6
               and (witness != 1 or order[-1] not in group)
               and (witness != 2 or order[0] not in group)]
    if not subsets:
        return None
    matched = subsets[int(rng.integers(len(subsets)))]
    matched.sort(key=lambda i: areas[i])
    target = matched[-1] if witness == 2 else matched[0]
    foil = next(i for i in rng.permutation(len(rooms)) if i not in matched)
    counts = [int(rng.integers(4)) for _ in rooms]
    # Keep the requested echo multiplicity exact in the sampling regime.
    for i in range(len(rooms)):
        if i not in matched and counts[i] == genus:
            counts[i] = (genus + 1) % 4
    for i in matched:
        counts[i] = genus
    counts[foil] = 2 if genus == 1 else 1
    sizes = [[int(rng.integers(1, 5)) for _ in range(k)] for k in counts]
    # Equal total solid area, unequal number of islands; a direct area-shortcut
    # counterexample in both indispensable teaching pairs.
    if genus == 1:
        sizes[target], sizes[foil] = [2], [1, 1]
        for i in matched:
            if i != target:
                sizes[i] = [int(rng.integers(4, 8))]
    elif genus == 2:
        sizes[target], sizes[foil] = [1, 1], [2]
        for i in matched:
            if i != target:
                sizes[i] = [2, int(rng.integers(2, 5))]
    elif genus == 3:
        sizes[target], sizes[foil] = [1, 2, 1], [4]
        for i in matched:
            if i != target:
                sizes[i] = [2, 2, 2]
    role = walls.copy()
    for room, wanted in zip(rooms, sizes):
        islands = _islands(rng, room, wanted)
        if islands is None:
            return None
        role |= islands
    bg, wall, ink = palette
    grid = np.full(role.shape, bg, dtype=np.int8)
    grid[role] = wall
    actual_target = rooms[target] & ~role
    _, _, rim = _topology(actual_target)
    choices = np.argwhere(actual_target & (rim if marker_rim else ~rim))
    if not len(choices):
        return None
    grid[tuple(choices[int(rng.integers(len(choices)))])] = ink
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == wall):
        return None
    # Write labels from the constructed region classes, then verify by parsing
    # the finished input. Neither representation is retained in returned pairs.
    output = grid.copy()
    for i in matched:
        _, _, boundary = _topology(rooms[i] & ~role)
        output[boundary] = ink
    assert np.array_equal(output, solve(grid))
    return dict(input=grid, output=output)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    changing_palette = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    shared_orientation = bool(rng.random() < .3)
    training_genera = [1, 2] + [int(rng.integers(4)) for _ in range(num_train - 2)]
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            ordinal = len(seen)
            witness = index + 1 if split == 'train' and index < 2 else 0
            genus = training_genera[index] if split == 'train' else int(rng.choice(training_genera))
            peers = 3 if split == 'test' else 2
            roles = tuple(int(x) for x in rng.choice(10, 3, replace=False)) if changing_palette else palette
            marker_rim = witness == 2 or (not witness and bool(rng.integers(2)))
            for attempt in range(28):
                wanted = int(rng.integers(4, 7)) if split == 'test' else int(rng.integers(3, 6))
                if attempt < 22:
                    h, w = int(rng.integers(19, 31)), int(rng.integers(19, 31))
                    if wanted >= 5:
                        h, w = max(h, 25), max(w, 25)
                    partition = _partition(rng, h, w, wanted, split == 'test' or witness == 2)
                    if partition is None:
                        continue
                    walls, rooms = _intrude(rng, *partition, attempts=int(rng.integers(1, 9)))
                else:
                    walls, rooms = _fallback_partition(ordinal, wanted)
                pair = _populate(rng, walls, rooms, genus, peers, witness, roles, marker_rim)
                if pair is None:
                    continue
                rotation = base_rotation if shared_orientation else int(rng.integers(4))
                reflection = bool(rng.integers(2))
                pair = {k: np.rot90(v, rotation).copy() for k, v in pair.items()}
                if reflection:
                    pair = {k: np.fliplr(v).copy() for k, v in pair.items()}
                if witness:
                    teaching = _features(pair['input'])
                    if not teaching['diagonal']:
                        continue
                    if witness == 1 and not teaching['largest_foil']:
                        continue
                    if witness == 2 and not teaching['smallest_foil']:
                        continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted; no partial puzzle returned')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _features(input_grid):
    p = _analyze(input_grid)
    target = p['rooms'][p['target']]
    matching = [r for r in p['rooms'] if r['count'] == p['count']]
    others = [r for r in p['rooms'] if r['count'] != p['count']]
    area_sum = lambda r: sum(int(h.sum()) for h in r['holes'])
    rim4 = np.zeros_like(p['grid'], dtype=bool)
    e = target['exterior']
    h, w = p['grid'].shape
    for dr, dc in N4:
        rim4 |= e[1+dr:1+dr+h, 1+dc:1+dc+w]
    marker_at = tuple(np.argwhere(p['grid'] == p['marker'])[0])
    return dict(shape=p['grid'].shape, palette=(p['background'], p['structure'], p['marker']),
                rooms=len(p['rooms']), genus=p['count'], matches=len(matching),
                genera=tuple(sorted(r['count'] for r in p['rooms'])),
                areas=tuple(sorted(r['area'] for r in p['rooms'])),
                island_areas=tuple(sorted(int(h.sum()) for r in p['rooms'] for h in r['holes'])),
                unequal_matches=len({r['area'] for r in matching}) > 1,
                area_foil=any(area_sum(r) == area_sum(target) for r in others),
                unequal_island_area=len({area_sum(r) for r in matching}) > 1,
                largest_foil=any(r['area'] > max(q['area'] for q in matching) for r in others),
                smallest_foil=any(r['area'] < min(q['area'] for q in matching) for r in others),
                diagonal=int(np.count_nonzero(target['rim'] & ~rim4 & (p['grid'] == p['background']))),
                marker_rim=bool(target['rim'][marker_at]),
                interior_rooms=sum(not (r['mask'][0].any() or r['mask'][-1].any()
                                         or r['mask'][:, 0].any() or r['mask'][:, -1].any())
                                   for r in p['rooms']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            p = _analyze(inp)
            assert len(p['rooms']) >= 3
            assert np.array_equal(out, solve(inp))
            # The only edits are background-to-marker cells; walls and all
            # islands remain intact. Foreign background rooms stay intact too.
            changed = inp != out
            assert changed.any() and np.all(inp[changed] == p['background'])
            assert np.all(out[changed] == p['marker'])
            assert np.array_equal(out[inp == p['structure']], inp[inp == p['structure']])
            for r in p['rooms']:
                assert r['mask'].any()
                for hole in r['holes']:
                    assert np.all(inp[hole] == p['structure']), 'islands must be solid structure'
                    assert len(_components(hole)) == 1 and not _topology(hole)[0]
                    # There is a visible band around the island left uncolored.
                    band = _dilate(hole) & r['mask'] & ~r['rim'] & (inp != p['marker'])
                    assert band.any() and np.all(out[band] == p['background'])
                if r['count'] != p['count']:
                    assert np.array_equal(inp[r['mask']], out[r['mask']])
            f = _features(inp)
            assert f['matches'] >= 2 and f['matches'] < f['rooms']
            assert f['unequal_matches']
            if split == 'train':
                training.append(f)
    # These are teaching requirements, separate from the semantic rule above.
    assert {1, 2}.issubset({f['genus'] for f in training})
    assert all(any(f['genus'] == k and f['area_foil'] and f['unequal_island_area']
                   for f in training) for k in (1, 2))
    assert any(f['marker_rim'] for f in training) and any(not f['marker_rim'] for f in training)
    assert any(f['diagonal'] for f in training)
    assert any(f['largest_foil'] for f in training)
    assert any(f['smallest_foil'] for f in training)
    assert all(_features(pair['input'])['genus'] in {f['genus'] for f in training}
               for pair in puzzle['test'])


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in a)


def _unit_checks():
    # Hand-specified full fixture: two different one-island rooms and a two-
    # island foil. Expected boundary locations are written without _topology.
    grid = np.zeros((11, 23), dtype=np.int8)
    grid[:, 7] = grid[:, 15] = 6
    grid[5, 3] = 6
    grid[4:6, 11:13] = 6
    grid[3, 19] = grid[7, 19] = 6
    grid[2, 2] = 9
    expected = grid.copy()
    for a, b in ((0, 7), (8, 15)):
        expected[0, a:b] = expected[-1, a:b] = 9
        expected[:, a] = expected[:, b-1] = 9
    assert np.array_equal(solve(grid), expected)
    # Move the exemplar inside its room; only the marker cell changes.
    moved = grid.copy()
    moved[2, 2], moved[8, 5] = 0, 9
    moved_out = expected.copy()
    moved_out[2, 2], moved_out[8, 5] = 0, 9
    assert np.array_equal(solve(moved), moved_out)
    # Move it across a wall: the two-island foil becomes the sole matching room.
    across = grid.copy()
    across[2, 2], across[2, 18] = 0, 9
    across_out = across.copy()
    across_out[0, 16:] = across_out[-1, 16:] = 9
    across_out[:, 16] = across_out[:, -1] = 9
    assert np.array_equal(solve(across), across_out)
    # Hand-specified concavity: (2,2) and (2,4) touch an outside intrusion
    # only diagonally. A separate internal singleton is deliberately ignored.
    room = np.ones((7, 7), dtype=bool)
    room[:2, 3] = False
    room[4, 3] = False
    rim = np.zeros_like(room)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    rim[1, 2] = rim[1, 4] = True
    rim[2, 2:5] = True
    rim &= room
    holes, _, actual = _topology(room)
    assert len(holes) == 1 and holes[0][4, 3]
    assert np.array_equal(actual, rim)
    # Joining two islands changes topology although their total area barely
    # changes; attaching an island to the room boundary makes it an intrusion.
    attached = grid.copy()
    attached[:5, 3] = 6
    assert _analyze(attached)['count'] == 0
    assert not np.array_equal(solve(attached), expected)
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert len(_topology(diagonal)[0]) == 1
    diagonal[0, 1] = False
    assert not _topology(diagonal)[0]
    for rotation in range(4):
        for reflected in (False, True):
            a, b = np.rot90(grid, rotation), np.rot90(expected, rotation)
            if reflected:
                a, b = np.fliplr(a), np.fliplr(b)
            before = a.copy()
            assert np.array_equal(solve(a), b) and np.array_equal(a, before)
    permutation = np.array([4, 0, 3, 7, 5, 2, 1, 8, 9, 6], dtype=np.int8)
    assert np.array_equal(solve(permutation[grid]), permutation[expected])


def self_test():
    """Bounded 42-seed audit; returns aggregates, writes no grids or logs."""
    _unit_checks()
    fallback = dict(train=[], test=[])
    fallback_rng = np.random.default_rng(6001)
    for ordinal, genus, peers, witness in ((0, 1, 2, 1), (1, 2, 2, 2), (2, 1, 3, 0)):
        for _ in range(20):
            pair = _populate(fallback_rng, *_fallback_partition(ordinal, 4 + (ordinal == 2)),
                             genus, peers, witness, (0, 6, 9), witness == 2)
            if pair is not None:
                f = _features(pair['input'])
                if witness == 1 and not f['largest_foil']:
                    continue
                if witness == 2 and not f['smallest_foil']:
                    continue
                break
        else:
            raise AssertionError('fallback population failed')
        fallback['train' if witness else 'test'].append(pair)
    validate_generated_puzzle(fallback)
    shapes, palettes, genera, room_counts, areas = set(), set(), set(), set(), set()
    signatures, normalized, island_sizes, interior = set(), set(), set(), 0
    marker_modes = Counter()
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        local_shapes, local_counts = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                f = _features(inp)
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                genera.add(f['genus'])
                room_counts.add(f['rooms'])
                areas.add(f['areas'])
                local_shapes.add(f['shape'])
                local_counts.add(f['genus'])
                island_sizes.update(f['island_areas'])
                marker_modes[f['marker_rim']] += 1
                interior += f['interior_rooms']
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                roles = np.full(inp.shape, 0, dtype=np.int8)
                roles[inp == f['palette'][1]] = 1
                roles[inp == f['palette'][2]] = 2
                normalized.add((inp.shape, hashlib.sha256(roles.tobytes()).hexdigest()))
        assert len(local_shapes) >= 2 and len(local_counts) >= 2
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 35 and len(palettes) >= 25
    assert genera == {0, 1, 2, 3} and {3, 4, 5, 6}.issubset(room_counts)
    assert len(areas) >= 150 and {1, 2, 3, 4, 5, 6, 7}.issubset(island_sizes)
    assert all(marker_modes.values())
    fixed = generate_puzzle(123, 2, 1)
    assert _same(fixed, generate_puzzle(123, 2, 1))
    # The host covers every legal count; keep this owned test to the extremes.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(100 + a*10 + b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal example counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in fixed[s]] for s in fixed}
    p = _analyze(bad['test'][0]['input'])
    cell = tuple(np.argwhere(bad['test'][0]['input'] == p['structure'])[0])
    bad['test'][0]['output'][cell] = p['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged structural cell accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                shapes=len(shapes), palettes=len(palettes),
                normalized_structures=len(normalized), genera=sorted(genera),
                room_counts=sorted(room_counts), island_areas=sorted(island_sizes),
                interior_rooms=interior)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
