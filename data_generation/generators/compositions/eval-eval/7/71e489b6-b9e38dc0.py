"""A spotlight exposes damaged rectangular screens; only reached holes glow.

Only standard Python and NumPy are used. No filesystem access, generation state,
training examples, or source fixtures are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


VERIFICATION_SEEDS = (
    0, 399845869, 178593014, 202971154, 973731604, 214575446, 76448580,
    820512755, 235247680, 447815438, 648434800, 447231631, 524933678,
    953291220, 772578342, 757057789, 587425342, 94901845, 186921095,
    567869123, 335636081, 522770830, 346682336, 428039895, 735421438,
    306690155, 518542842, 5561373, 970891130, 493435245, 351503901,
    572374950, 57833521, 232032053, 838195487, 825748692, 60032650,
    362803096, 60863133, 989396687, 409593371, 460924420,
)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return [(int(v[0]), int(v[-1])) for v in np.split(indices, cuts)]


def _groups(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, pending = [start], [start]
        while pending:
            y, x = pending.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
                    group.append(q)
        result.append(group)
    return result


def _halo(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= p[dy:dy+h, dx:dx+w]
    return result


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _fit_rail(values, start, sign):
    distance = np.arange(1, len(values)+1)
    fits = []
    for period in (1, 2):
        for phase in range(period):
            predicted = start + sign * ((distance + phase) // period)
            if (np.array_equal(values, predicted) and
                    abs(int(values[-1])-start) >= 2):
                fits.append((period, phase))
    if len(fits) != 1:
        raise ValueError("rail cadence is not uniquely readable")
    return fits[0]


def _inspect(input_grid):
    """Read the lamp, semantic palette, and screen rectangles from pixels."""
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 4 or counts.max()*2 <= inp.size:
        raise ValueError("four roles and a majority field are required")
    field = int(colors[np.argmax(counts)])
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        for wall in map(int, colors):
            if wall == field:
                continue
            pos = np.argwhere(a == wall)
            if len(pos) < 9:
                continue
            rear, mouth = int(pos[:, 1].min()), int(pos[:, 1].max())
            if mouth-rear < 3 or mouth >= a.shape[1]-3:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1):
                continue
            upper, lower = [], []
            for x in range(rear+1, mouth+1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[1]-rows[0] < 2:
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            if len(upper) != mouth-rear:
                continue
            try:
                up_fit = _fit_rail(upper, int(cap[0]), -1)
                low_fit = _fit_rail(lower, int(cap[-1]), 1)
            except ValueError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            yy = np.arange(a.shape[0])
            for x in range(rear+1, a.shape[1]):
                d = x-rear
                top = int(cap[0])-(d+up_fit[1])//up_fit[0]
                bottom = int(cap[-1])+(d+low_fit[1])//low_fit[0]
                footprint[:, x] = (yy > top) & (yy < bottom)
            chamber = footprint.copy()
            chamber[:, mouth+1:] = False
            patches = set(map(int, np.unique(a[chamber])))-{field, wall}
            if len(patches) != 1:
                continue
            payload = patches.pop()
            patch = a == payload
            if (np.any(patch & ~chamber) or
                    len(_groups(patch)) != 1 or not 1 <= int(patch.sum()) <= 5):
                continue
            material = (set(map(int, colors))-{field, wall, payload}).pop()
            observed = a == material
            # Whole-slab evidence identifies rectangles even when a through
            # slit disconnects their observed material into several pieces.
            slabs = _runs(observed.sum(axis=0) >= 4)
            if not 1 <= len(slabs) <= 3:
                continue
            base = np.zeros(a.shape, dtype=bool)
            boxes, readable = [], True
            for left, right in slabs:
                points = np.argwhere(observed[:, left:right+1])
                top, bottom = int(points[:, 0].min()), int(points[:, 0].max())
                if (left < mouth+3 or not 2 <= right-left+1 <= 5 or
                        bottom-top+1 < 9 or
                        not observed[top, left:right+1].all() or
                        not observed[bottom, left:right+1].all()):
                    readable = False
                    break
                base[top:bottom+1, left:right+1] = True
                boxes.append((top, bottom, left, right))
            if not readable:
                continue
            holes = base & ~observed
            extras = observed & ~base
            if (np.any(extras & chamber) or
                    np.any(extras.sum(axis=0) > 2) or
                    np.any(holes.sum(axis=0) > 4) or
                    any(np.any(observed[:, l:r+1].sum(axis=0) < 5)
                        for _, _, l, r in boxes)):
                continue
            matches.append(dict(a=a, rotation=rotation, field=field, wall=wall,
                                payload=payload, material=material, rear=rear,
                                mouth=mouth, periods=(up_fit[0], low_fit[0]),
                                phases=(up_fit[1], low_fit[1]),
                                footprint=footprint, chamber=chamber, patch=patch,
                                base=base, holes=holes, extras=extras, boxes=boxes))
    if len(matches) != 1:
        raise ValueError("input must identify one lamp and screen interpretation")
    return matches[0]


def _scan(info, *, erase_extras=True, halos=True, raw=False, occlusion=True):
    """Prediction: independent lane scans; no construction masks are passed."""
    a = info['a']
    field, payload = info['field'], info['payload']
    holes = np.zeros(a.shape, dtype=bool) if raw else info['holes']
    out = a.copy()
    if erase_extras and not raw:
        out[info['extras']] = field
    blockers = (a == info['material']) if raw or not erase_extras else info['base'] & ~holes
    active = np.zeros(a.shape, dtype=bool)
    shadow = np.zeros(a.shape, dtype=bool)
    for y in range(a.shape[0]):
        blocked = False
        for x in range(info['rear']+1, a.shape[1]):
            if not info['footprint'][y, x]:
                continue
            if blockers[y, x]:
                if occlusion:
                    blocked = True
            elif blocked:
                shadow[y, x] = True
            elif holes[y, x]:
                active[y, x] = True
            elif out[y, x] == field:
                out[y, x] = payload
    if halos:
        out[_halo(active) & ~holes] = payload
    return out, active, shadow


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from a valid family input."""
    info = _inspect(input_grid)
    out, _, _ = _scan(info)
    return np.rot90(out, info['rotation']).copy()


def _construct_output(inp, field, material, payload, footprint, base, holes, extras):
    """Construction renderer: downstream union, distinct from lane scanning."""
    shadow = np.zeros(inp.shape, dtype=bool)
    for y, x in np.argwhere(base & ~holes & footprint):
        shadow[y, x+1:] = True
    active = holes & footprint & ~shadow
    out = inp.copy()
    out[extras] = field
    out[footprint & ~base & ~shadow & (out == field)] = payload
    rim = np.zeros(inp.shape, dtype=bool)
    for y, x in np.argwhere(active):
        rim[max(0, y-1):y+2, max(0, x-1):x+2] = True
    out[rim & ~holes] = payload
    return out


def _grow_payload(rng, chamber):
    cells = np.argwhere(chamber)
    start = tuple(cells[int(rng.integers(len(cells)))])
    selected = {start}
    for _ in range(int(rng.integers(1, 6))-1):
        frontier = sorted({q for y, x in selected for q in
                           ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                           if 0 <= q[0] < chamber.shape[0] and
                           0 <= q[1] < chamber.shape[1] and
                           chamber[q] and q not in selected})
        if not frontier:
            break
        selected.add(frontier[int(rng.integers(len(frontier)))])
    patch = np.zeros(chamber.shape, dtype=bool)
    for q in selected:
        patch[q] = True
    return patch


def _build(rng, palette, rotation, screen_count, teaching):
    field, wall, material, payload = map(int, palette)
    periods = [int(rng.integers(1, 3)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    length = 2*max(periods)+1+int(rng.integers(2))
    rear = int(rng.integers(3))
    mouth = rear+length
    distances = [int(rng.integers(1, 4)) for _ in range(2)]
    extents = [d+(length+ph)//p for d, ph, p in zip(distances, phases, periods)]
    height = min(30, max(15, sum(extents)+1+int(rng.integers(1, 9))))
    center = int(rng.integers(extents[0], height-extents[1]))
    widths = [int(rng.integers(3, 6 if screen_count == 1 else 5))]
    widths += [int(rng.integers(2, 4 if screen_count == 3 else 5))
               for _ in range(screen_count-1)]
    if screen_count == 3:
        widths[0] = 3
    boxes, next_x = [], mouth+3
    for width in widths:
        rh = int(rng.integers(9, min(height, 22)+1))
        low = max(0, center-rh+4)
        high = min(center-3, height-rh)
        if low > high:
            raise ValueError("screen does not fit")
        top = int(rng.integers(low, high+1))
        boxes.append((top, top+rh-1, next_x, next_x+width-1))
        next_x += width+int(rng.integers(1, 3))
    end = boxes[-1][3]+1
    if end+3 > 30:
        raise ValueError("not enough downstream space")
    width = end+int(rng.integers(3, min(7, 30-end)+1))
    inp = np.full((height, width), field, dtype=np.int64)
    wall_mask = np.zeros(inp.shape, dtype=bool)
    footprint = np.zeros(inp.shape, dtype=bool)
    wall_mask[center-distances[0]:center+distances[1]+1, rear] = True
    for x in range(rear+1, width):
        delta = x-rear
        upper = center-distances[0]-(delta+phases[0])//periods[0]
        lower = center+distances[1]+(delta+phases[1])//periods[1]
        if x <= mouth:
            wall_mask[upper, x] = wall_mask[lower, x] = True
        footprint[max(0, upper+1):min(height, lower), x] = True
    inp[wall_mask] = wall
    chamber = footprint.copy()
    chamber[:, mouth+1:] = False
    patch = _grow_payload(rng, chamber)
    inp[patch] = payload
    base, holes = np.zeros(inp.shape, dtype=bool), np.zeros(inp.shape, dtype=bool)
    side = int(rng.choice((-1, 1)))
    lane_a, lane_b, branch = center-side, center+side, center+2*side
    for index, (top, bottom, left, right) in enumerate(boxes):
        base[top:bottom+1, left:right+1] = True
        # Anchors have an aligned transmission lane. Later-depth queries may
        # interrupt it, while retaining the same already demonstrated physics.
        if teaching or index == 0 or rng.random() < .65:
            holes[lane_a, left:right+1] = True
        elif rng.random() < .5:
            holes[lane_a, left:right] = True
        else:
            holes[lane_a, left+1:right+1] = True
        if index == 0:
            holes[lane_b, left:right] = True
            holes[branch, left+1] = True
        else:
            holes[lane_b, left:right+1] = True
        for _ in range(int(rng.integers(0, 5))):
            y = int(rng.integers(top+1, bottom))
            x = int(rng.integers(left, right+1))
            if y in (lane_a, lane_b, branch):
                continue
            if holes[:, x].sum() < min(4, bottom-top+1-5):
                holes[y, x] = True
    inp[base] = material
    inp[holes] = field
    extras = np.zeros(inp.shape, dtype=bool)
    slab_columns = base.any(axis=0)

    def place_extra(candidates):
        candidates = candidates & ~extras & ~wall_mask & ~patch & ~slab_columns[None, :]
        candidates &= (extras.sum(axis=0) < 2)[None, :]
        cells = np.argwhere(candidates)
        if not len(cells):
            raise ValueError("no readable speck site")
        y, x = cells[int(rng.integers(len(cells)))]
        extras[y, x] = True

    before = np.zeros(inp.shape, dtype=bool)
    before[:, mouth+1:mouth+3] = footprint[:, mouth+1:mouth+3]
    # Put the forced speck in the first of the two free columns, so there is
    # observable illumination beyond it before any real screen is reached.
    before[:, mouth+2:] = False
    place_extra(before)
    place_extra(~footprint & ~chamber)
    for _ in range(int(rng.integers(0, 5))):
        allowed = ~chamber
        place_extra(allowed)
    inp[extras] = material
    out = _construct_output(inp, field, material, payload, footprint, base, holes, extras)
    if np.count_nonzero(inp == field)*2 <= inp.size:
        raise ValueError("field must dominate")
    pair = {k: np.rot90(v, rotation).copy() for k, v in (('input', inp), ('output', out))}
    # Generation checks actual returned pixels, not just sampled intentions.
    decoded = _inspect(pair['input'])
    if not (np.array_equal(decoded['base'], base) and
            np.array_equal(decoded['holes'], holes) and
            np.array_equal(solve(pair['input']), pair['output'])):
        raise ValueError("construction and visible decoding disagree")
    return pair


def _features(pair):
    info = _inspect(pair['input'])
    expected, active, shadow = _scan(info)
    output = np.rot90(np.asarray(pair['output']), -info['rotation'])
    if not np.array_equal(expected, output):
        raise AssertionError("output disagrees with input-only prediction")
    holes, base = info['holes'], info['base']
    dark = holes & ~active
    mixed = False
    for group in _groups(holes):
        yy, xx = zip(*group)
        mixed |= bool(active[yy, xx].any() and dark[yy, xx].any())
    transmitted = hidden_slit = False
    for top, bottom, left, right in info['boxes']:
        for y in range(top+1, bottom):
            if holes[y, left:right+1].all():
                hidden_slit |= bool(dark[y, left:right+1].all())
                if active[y, left:right+1].all() and right+2 < output.shape[1]:
                    transmitted |= bool(output[y, right+2] == info['payload'] and
                                        not base[y, right+2] and
                                        not _halo(active)[y, right+2])
    no_recovery, _, _ = _scan(info, erase_extras=False)
    no_halo, _, _ = _scan(info, halos=False)
    no_a, _, _ = _scan(info, raw=True, halos=False)
    no_occlusion, _, _ = _scan(info, occlusion=False)
    unconditional_halo = no_halo.copy()
    unconditional_halo[_halo(holes) & ~holes] = info['payload']
    no_b = info['a'].copy()
    no_b[info['extras']] = info['field']
    no_b[_halo(holes) & ~holes] = info['payload']
    normalized = np.zeros(base.shape, dtype=np.uint8)
    normalized[base], normalized[holes] = 1, 2
    pos = np.argwhere(base)
    crop = normalized[pos[:, 0].min():pos[:, 0].max()+1,
                      pos[:, 1].min():pos[:, 1].max()+1]
    return dict(screens=len(info['boxes']), shape=tuple(info['a'].shape),
                rotation=info['rotation'], palette=tuple(info[k] for k in
                ('field', 'wall', 'material', 'payload')), periods=info['periods'],
                phases=info['phases'], holes=int(holes.sum()), active=int(active.sum()),
                dark=int(dark.sum()), extras=int(info['extras'].sum()),
                mixed_cavity=mixed, transmitted=transmitted, hidden_slit=hidden_slit,
                visible_shadow=bool((shadow & (output == info['field']) & ~holes & ~base).any()),
                halo_on_material=int((_halo(active) & base & ~holes).sum()),
                halo_outside_cone=int((_halo(active) & ~info['footprint'] & ~holes).sum()),
                omit_a_diff=int(np.count_nonzero(expected != no_a)),
                omit_recovery_diff=int(np.count_nonzero(expected != no_recovery)),
                omit_halo_diff=int(np.count_nonzero(expected != no_halo)),
                omit_b_diff=int(np.count_nonzero(expected != no_b)),
                omit_occlusion_diff=int(np.count_nonzero(expected != no_occlusion)),
                unconditional_halo_diff=int(np.count_nonzero(expected != unconditional_halo)),
                widths=tuple(r-l+1 for _, _, l, r in info['boxes']),
                heights=tuple(b-t+1 for t, b, _, _ in info['boxes']),
                patch_area=int(info['patch'].sum()),
                structure=hashlib.sha256(crop.tobytes()+bytes(crop.shape)).hexdigest())


def _teaches(features, depth):
    required = ('mixed_cavity', 'transmitted', 'visible_shadow', 'halo_on_material',
                'omit_a_diff', 'omit_recovery_diff', 'omit_halo_diff', 'omit_b_diff',
                'omit_occlusion_diff', 'unconditional_halo_diff')
    return (features['screens'] == depth and all(features[k] for k in required) and
            (depth == 1 or features['hidden_slit']))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.integers(2))
    palette = rng.choice(10, size=4, replace=False)
    held_direction = rng.random() < .3
    preferred = int(rng.integers(4))
    alternatives = [v for v in range(4) if v != preferred]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = split == 'train' and index < 2
            screens = (index+1 if teaching else
                       int(rng.integers(1, 3)) if split == 'train' else
                       2 if held_direction else 3)
            rotation = (preferred if split == 'train' else int(rng.choice(alternatives))) \
                if held_direction else int(rng.integers(4))
            colors = palette if shared_palette else rng.choice(10, size=4, replace=False)
            for _ in range(64):
                try:
                    pair = _build(rng, colors, rotation, screens, teaching)
                    features = _features(pair)
                except (ValueError, AssertionError):
                    continue
                if not (features['omit_occlusion_diff'] and features['unconditional_halo_diff']):
                    continue
                if teaching and not _teaches(features, screens):
                    continue
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded readable-screen construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError("puzzle requires exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise AssertionError("splits must be lists")
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
        seen, records = set(), []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                inp, out = _grid(pair['input']), _grid(pair['output'])
                assert inp.shape == out.shape and not np.array_equal(inp, out)
                key = _signature(inp)
                assert key not in seen, "duplicate input"
                seen.add(key)
                info = _inspect(inp)
                canonical = np.rot90(out, -info['rotation'])
                expected, active, _ = _scan(info)
                assert np.array_equal(canonical, expected)
                assert np.all(canonical[info['holes']] == info['field'])
                assert np.array_equal(canonical[info['a'] == info['wall']],
                                      info['a'][info['a'] == info['wall']])
                assert np.all(canonical[info['patch']] == info['payload'])
                assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
                # Except erased specks and the local rim, changes must be
                # unblocked field-to-payload illumination in the footprint.
                allowed = info['extras'] | _halo(active) | info['footprint']
                assert not np.any((info['a'] != canonical) & ~allowed)
                assert np.all(canonical[info['base'] & ~info['holes'] &
                                        ~_halo(active)] == info['material'])
                f = _features(pair)
                assert f['active'] and f['dark'] and f['extras']
                assert f['omit_a_diff'] and f['omit_b_diff']
                assert f['omit_occlusion_diff'] and f['unconditional_halo_diff']
                if split == 'train':
                    records.append(f)
        assert any(_teaches(f, 1) for f in records), "missing one-screen teaching witness"
        assert any(_teaches(f, 2) for f in records), "missing interacting-screen witness"
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise AssertionError(str(error)) from error


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def _hand_fixture():
    """Hand-authored thin cone and one branched aperture, without _build()."""
    a = np.zeros((17, 18), dtype=np.int64)
    a[6:11, 0] = 1
    for x, top in enumerate((6, 5, 5, 4, 4), 1):
        a[top, x] = a[16-top, x] = 1
    a[8, 2] = 3
    a[2:15, 8:12] = 2
    a[7, 8:12] = 0             # Full through slit.
    a[9, 8:11] = 0             # Front pocket blocked at column 11.
    a[10, 9] = 0               # Connected branch with a solid upstream cell.
    a[8, 6] = a[0, 2] = 2     # One influential and one exterior speck.
    return a


def self_test():
    """Bounded, sequential validation; no samples or logs are written."""
    fixture = _hand_fixture()
    result = solve(fixture)
    assert result[0, 2] == 0                     # Erased exterior noise.
    assert result[8, 7] == 3                     # Noise has no shadow.
    assert np.all(result[7, 8:12] == 0)          # Hole centers stay empty.
    assert result[7, 14] == 3                    # Through-slit transmission.
    assert result[9, 14] == 0                    # Front pocket does not transmit.
    assert result[10, 9] == 0                    # Dark branch stays empty.
    assert result[11, 9] == 2                    # No halo from the dark branch.
    assert result[10, 8] == 3                    # Halo from exposed pocket.
    assert result[3, 6] == 0 and result[4, 6] == 3  # Thin virtual stair phase.
    for k in range(4):
        assert np.array_equal(solve(np.rot90(fixture, k)), np.rot90(result, k))
    records, shapes, structures, palettes, directions = [], set(), set(), set(), set()
    puzzle_hashes, depth_shifts, orientation_shifts = set(), 0, 0
    for seed in VERIFICATION_SEEDS:
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        local = []
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _features(pair)
                records.append(f)
                local.append((split, f))
                shapes.add(f['shape'])
                structures.add(f['structure'])
                palettes.add(f['palette'])
                directions.add(f['rotation'])
                digest.update(pair['input'].astype(np.uint8).tobytes())
        puzzle_hashes.add(digest.hexdigest())
        tf, qf = [f for s, f in local if s == 'train'], [f for s, f in local if s == 'test']
        depth_shifts += int(min(f['screens'] for f in qf) > max(f['screens'] for f in tf))
        orientation_shifts += int(bool({f['rotation'] for f in qf}-{f['rotation'] for f in tf}))
        assert len({f['structure'] for _, f in local}) >= 3
        if seed in (0, 399845869):
            assert _equal(p, generate_puzzle(seed))
            pair = p['test'][0]
            before = pair['input'].copy()
            for k in range(4):
                assert np.array_equal(solve(np.rot90(before, k)), np.rot90(pair['output'], k))
            permutation = np.random.default_rng(seed+17).permutation(10)
            assert np.array_equal(solve(permutation[before]), permutation[pair['output']])
            assert np.array_equal(before, pair['input'])
    assert len(puzzle_hashes) == 42 and len(structures) >= 180
    assert len(shapes) >= 40 and len(palettes) >= 50 and directions == {0, 1, 2, 3}
    assert {f['screens'] for f in records} == {1, 2, 3}
    assert {p for f in records for p in f['periods']} == {1, 2}
    assert any(f['periods'][0] != f['periods'][1] for f in records)
    assert len({f['holes'] for f in records}) >= 15
    assert len({v for f in records for v in f['widths']}) == 4
    assert len({v for f in records for v in f['heights']}) >= 10
    assert {f['patch_area'] for f in records} == {1, 2, 3, 4, 5}
    assert any(f['halo_outside_cone'] for f in records)
    assert depth_shifts >= 10 and orientation_shifts >= 5
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(10000+10*tr+te, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(83, 2, 1)
    pair = damaged['test'][0]
    y, x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y, x] = pair['input'][y, x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged label accepted")
    return dict(seeds=42, default_pairs=len(records), structures=len(structures),
                shapes=len(shapes), palettes=len(palettes),
                depth_shift_puzzles=depth_shifts, new_direction_puzzles=orientation_shifts,
                hole_counts=sorted({f['holes'] for f in records}),
                active_counts=sorted({f['active'] for f in records}),
                dark_counts=sorted({f['dark'] for f in records}),
                minimum_omit_a_difference=min(f['omit_a_diff'] for f in records),
                minimum_omit_b_difference=min(f['omit_b_diff'] for f in records),
                minimum_omit_occlusion_difference=min(f['omit_occlusion_diff'] for f in records),
                minimum_unconditional_halo_difference=min(f['unconditional_halo_diff'] for f in records))
