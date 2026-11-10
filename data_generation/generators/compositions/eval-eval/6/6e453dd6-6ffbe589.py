"""Cavity light counters: first-hit rays control simultaneous color rotations.

No parent code is imported. Generation uses orbit territories; solve uses only
visible pixels and does not require that particular construction method.
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 train, 1..4 test, with train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _parse(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    bg = int(g[0, -1])
    candidates = []
    for col in range(1, w-1):
        rail = int(g[0, col])
        if rail == bg or not np.all(g[:, col] == rail):
            continue
        outside = np.argwhere((g == rail) & (np.indices(g.shape)[1] != col))
        if len(outside) != 4 or not np.all(outside[:, 1] > col):
            continue
        top, left = map(int, outside.min(axis=0))
        bottom, right = map(int, outside.max(axis=0))
        if bottom-top != right-left or bottom-top < 4:
            continue
        if set(map(tuple, outside.tolist())) != {(top, left), (top, right), (bottom, left), (bottom, right)}:
            continue
        candidates.append((col, rail, top+1, left+1, bottom-top-1))
    _need(len(candidates) == 1, "one full rail and four square corner bolts required")
    col, rail, row, left, n = candidates[0]
    source = g[row:row+n, left:left+n].copy()
    palette = tuple(int(c) for c in np.unique(source) if c != bg)
    _need(len(palette) >= 2 and rail not in palette, "receiver needs at least two content colors")
    exterior = g[:, col+1:].copy()
    exterior[row:row+n, left-col-1:left-col-1+n] = bg
    exterior[exterior == rail] = bg
    _need(np.all(exterior == bg), "unmarked cells outside receiver must be blank")
    work = g[:, :col]
    emitter_colors = set(map(int, np.unique(work))) - {bg}
    _need(len(emitter_colors) == 1, "one visible emitter color required")
    emitter = emitter_colors.pop()
    _need(emitter not in palette and emitter != rail, "device roles must be distinct")
    active, holes, naive, objects = set(), set(), set(), []
    for cells in _components(work == emitter):
        yy, xx = np.array(cells).T
        r0, c0 = int(yy.min()), int(xx.min())
        mask = np.zeros((int(yy.max())-r0+1, int(xx.max())-c0+1), dtype=bool)
        mask[yy-r0, xx-c0] = True
        local_holes, local_active, local_decoy = [], set(), set()
        for region in _components(~mask):
            if any(y in (0, mask.shape[0]-1) or x in (0, mask.shape[1]-1) for y, x in region):
                continue
            local_holes.append(region)
            rows = {y for y, x in region}
            contact = {y for y, x in region if x == mask.shape[1]-2 and mask[y, -1]}
            if contact:
                _need(contact == rows, "partial-contact cavities excluded")
                local_active |= rows
            else:
                local_decoy |= rows
        local_naive = {y for y in range(mask.shape[0]) if mask.shape[1] >= 3 and
                       not mask[y, -2] and mask[y, -1]}
        active |= {r0+y for y in local_active}
        holes |= {r0+y for region in local_holes for y, x in region}
        naive |= {r0+y for y in local_naive}
        objects.append(dict(row=r0, col=c0, mask=mask, active=local_active,
                            decoy=local_decoy, naive=local_naive, holes=local_holes))
    # Separated row bands prevent duplicate light rays from different emitters.
    objects.sort(key=lambda o: o['row'])
    _need(all(a['row']+a['mask'].shape[0] < b['row'] for a, b in zip(objects, objects[1:])),
          "emitter row bands must have a blank separating row")
    return dict(grid=g, background=bg, rail=rail, rail_col=col, row=row, left=left,
                size=n, source=source, palette=palette, emitter=emitter,
                active=active, holes=holes, naive=naive, objects=objects)


def _light_counts(info, rows=None, all_hits=False):
    counts = {c: 0 for c in info['palette']}
    for global_row in sorted(info['active'] if rows is None else rows):
        r = global_row - info['row']
        if not 0 <= r < info['size']:
            continue
        values = info['source'][r]
        hits = values[values != info['background']]
        if len(hits):
            if all_hits:
                for c in hits:
                    counts[int(c)] += 1
            else:
                counts[int(hits[0])] += 1
    return counts


def _rotate(source, bg, counts, coordinate=False):
    out = np.full_like(source, bg)
    n = len(source)
    for color in sorted(counts):
        if coordinate:
            for r, c in np.argwhere(source == color):
                y, x = int(r), int(c)
                for _ in range(counts[color] % 4):
                    y, x = x, n-1-y
                _need(out[y, x] == bg, "rotated layers overlap")
                out[y, x] = color
        else:
            mask = np.rot90(source == color, -counts[color] % 4)
            _need(np.all(out[mask] == bg), "rotated layers overlap")
            out[mask] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read cavity light, count first-hit colors, rotate, and crop the receiver."""
    info = _parse(input_grid)
    return _rotate(info['source'], info['background'], _light_counts(info))


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n-1-r
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _pattern(rng, n, bg, colors, row_colors, active_rows):
    """Orbit ownership prevents collisions; exact teaching rows expose blocking."""
    orbits = _orbits(n)
    index = {p: i for i, orbit in enumerate(orbits) for p in orbit}
    owners, forced = {}, {}
    # A shielded color receives no pulse, despite lying on an illuminated row.
    illuminated = [r for r in active_rows if row_colors[r] is not None]
    shield = None
    if illuminated:
        r = min(illuminated)
        shield = index[r, n-1]
        owners[shield] = colors[-1]
        forced[r, n-1] = colors[-1]
    # Match anchors AND asymmetric witnesses to distinct orbits, instead of
    # greedily consuming a later row's only useful territory. At n>=11 every
    # anchor offers n-3 cells, and >=2 unforced rows offer witness positions.
    # Each orbit contains <=4 cells. These bounds give Hall's condition even
    # after the single shield orbit is excluded (at most five witnesses).
    choices = {}
    for r, color in sorted(row_colors.items()):
        if color is not None:
            choices['row', r] = [(index[r, c], (r, c)) for c in range(n-3) if index[r, c] != shield]
    witness_choices = [(i, p) for i, orbit in enumerate(orbits) if i != shield and len(orbit) == 4
                       for p in orbit if p[0] not in row_colors]
    for color in colors:
        choices['witness', color] = witness_choices
    options = {}
    for task, positions in choices.items():
        ids = sorted({i for i, p in positions})
        options[task] = list(map(int, rng.permutation(ids)))
    occupant, assigned = {}, {}

    def match(task, visited):
        for i in options[task]:
            if i in visited:
                continue
            visited.add(i)
            if i not in occupant or match(occupant[i], visited):
                occupant[i], assigned[task] = task, i
                return True
        return False

    for task in sorted(choices, key=lambda t: len(options[t])):
        _need(match(task, set()), "teaching territory matching failed")
    witnesses = {}
    for task, i in assigned.items():
        positions = [p for j, p in choices[task] if j == i]
        p = positions[int(rng.integers(len(positions)))]
        color = row_colors[task[1]] if task[0] == 'row' else task[1]
        owners[i] = color
        if task[0] == 'row':
            forced[p] = color
        else:
            witnesses[i] = p
    density = {color: float(rng.uniform(.20, .85)) for color in colors}
    out = np.full((n, n), bg, dtype=np.int8)
    geometry = int(rng.integers(3))
    radial_order = list(rng.permutation(colors))
    for i, orbit in enumerate(orbits):
        if i not in owners:
            r0, c0 = orbit[0]
            if geometry == 0:
                depth = min(r0, c0, n-1-r0, n-1-c0)
                color = int(radial_order[min(len(colors)-1, depth*len(colors)//max(1, (n+1)//2))])
            elif geometry == 1:
                color = int(colors[(r0//2+c0//2) % len(colors)])
            else:
                color = int(rng.choice(colors))
            owners[i] = color
        color = owners[i]
        if i in witnesses:
            out[witnesses[i]] = color
            continue
        style = int(rng.integers(3))
        if style == 0:
            chosen = orbit if rng.random() < density[color] else []
        elif style == 1:
            phase = int(rng.integers(len(orbit)))
            chosen = orbit[phase::2] if rng.random() < density[color] else []
        else:
            chosen = [p for p in orbit if rng.random() < density[color]]
        for p in chosen:
            out[p] = color
    for r in row_colors:
        out[r] = bg
    for p, color in forced.items():
        out[p] = color
    return out


def _plan(rng, witness, test, palette_size):
    if witness == 0:
        n, counts = int(rng.integers(11, 20)), [1, 2]
    elif witness == 1:
        n, counts = int(rng.integers(15, 20)), [3, 4]
    else:
        n = int(rng.integers(15 if test else 11, 20))
        counts = [0]*palette_size
        marked = list(map(int, rng.permutation(palette_size)))[:int(rng.integers(1, palette_size+1))]
        budget = n-5
        for j, color in enumerate(marked):
            cap = min(7 if test and j == 0 else 3 if test else 4,
                      budget-(len(marked)-j-1))
            low = 5 if test and j == 0 else 1
            counts[color] = int(rng.integers(low, cap+1))
            budget -= counts[color]
        if not test and rng.random() < .12:
            counts = [0]*palette_size
    # Optional empty receiving row: an eligible ray can leave without a pulse.
    empty = int((test or witness < 0) and rng.random() < .5)
    length = sum(counts) + empty
    # No pulse is needed in occasional identity-crop demonstrations.
    if not length:
        empty = length = 1
    n = max(n, length+4)
    max_blocks = min(3, length, n-length-3)
    blocks = int(rng.integers(1, max_blocks+1))
    cuts = sorted(map(int, rng.choice(range(1, length), blocks-1, replace=False)))
    sizes = np.diff([0]+cuts+[length]).tolist()
    tokens = [('A', size) for size in sizes] + [('O', 1), ('D', 1)]
    rng.shuffle(tokens)
    minimum = length + 2 + len(tokens)-1
    slack = n-minimum
    extra = rng.multinomial(slack, np.full(len(tokens)+1, 1/(len(tokens)+1)))
    status = ['S'] * n
    row = int(extra[0])
    gaps = []
    for i, (kind, size) in enumerate(tokens):
        status[row:row+size] = [kind]*size
        row += size
        if i < len(tokens)-1:
            gap = 1+int(extra[i+1])
            gaps.append((row, gap))
            row += gap
    # A gap of three solid rows permits two walls and one blank separator.
    splits = [1+start+size//2 for start, size in gaps if size >= 3 and rng.random() < .55]
    active = [r for r, kind in enumerate(status) if kind == 'A']
    pulse_labels = [color for color, count in enumerate(counts) for _ in range(count)] + ([None] if empty else [])
    rng.shuffle(pulse_labels)
    labels = dict(zip(active, pulse_labels))
    labels.update({r: 0 if kind == 'O' else 1 for r, kind in enumerate(status) if kind in ('O', 'D')})
    return n, status, splits, labels


def _pair(rng, witness=-1, test=False):
    k = int(rng.integers(3, 6))
    n, status, splits, labels = _plan(rng, witness, test, k)
    roles = list(map(int, rng.choice(10, k+3, replace=False)))
    bg, rail, emitter, *colors = roles
    # The final palette color is unlit in the two deliberate teaching examples.
    row_colors = {r: None if label is None else colors[label] for r, label in labels.items()}
    active = {r for r, kind in enumerate(status) if kind == 'A'}
    source = _pattern(rng, n, bg, colors, row_colors, active)
    work = int(rng.integers(6, 26-n))
    gap = int(rng.integers(1, 27-work-n))
    # One blank outer column keeps the top-right background role observable.
    left = work+gap+2
    width = left+n+2
    height = int(rng.integers(n+4, 31))
    row = int(rng.integers(2, height-n-1))
    g = np.full((height, width), bg, dtype=np.int8)
    g[:, work] = rail
    for y in (row-1, row+n):
        for x in (left-1, left+n):
            g[y, x] = rail
    g[row:row+n, left:left+n] = source
    # Source components share no rows. They can have independently sampled
    # widths, gaps to the rail, stepped cavity backs, and irregular left edges.
    extended = ['S'] + status + ['S']
    boundaries = [-1] + sorted(splits) + [n+2]
    for lo, hi in zip(boundaries, boundaries[1:]):
        start, stop = lo+1, hi
        if start >= stop:
            continue
        segment = extended[start:stop]
        max_width = min(work, 8)
        w = int(rng.integers(4, max_width+1))
        mask = np.ones((len(segment), w), dtype=bool)
        for y, kind in enumerate(segment):
            if kind == 'A':
                back = int(rng.integers(1, w-1))
                mask[y, back:w-1] = False
            elif kind == 'O':
                mask[y, :w-1] = False
            elif kind == 'D':
                back = int(rng.integers(1, w-2))
                mask[y, back:w-2] = False
        # Recess a leading lobe by using a tail to establish the true extreme.
        # Only decoy-only components qualify; their holes remain inactive.
        if witness < 0 and 'D' in segment and 'A' not in segment and rng.random() < .5 and w < max_width:
            mask = np.pad(mask, ((0, 0), (0, 1)))
            mask[0, -1] = True
            w += 1
        # Shallow exterior notches do not alter any chamber or leading wall.
        for y in range(len(segment)):
            if (segment[y] == 'S' and (y == 0 or segment[y-1] == 'S') and
                    (y == len(segment)-1 or segment[y+1] == 'S') and rng.random() < .35):
                mask[y, 0] = False
        x = int(rng.integers(0, work-w+1))
        view = g[row-1+start:row-1+stop, x:x+w]
        view[mask] = emitter
    info = _parse(g)
    return {'input': g, 'output': _rotate(source, bg, _light_counts(info), coordinate=True)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            pair = _pair(rng, i if split == 'train' and i < 2 else -1, split == 'test')
            # Twenty distinct dihedral color permutations, against at most
            # thirteen earlier inputs. A reflection fixes at most two colors,
            # while every input uses at least six, so this fallback is total.
            for sign in (1, -1):
                for offset in range(10):
                    candidate = {name: (sign*value+offset) % 10 for name, value in pair.items()}
                    key = _key(candidate['input'])
                    if key not in seen:
                        break
                else:
                    continue
                break
            else:
                raise AssertionError("twenty color permutations cannot all be blocked")
            seen.add(key)
            puzzle[split].append(candidate)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, mode):
    if mode == 'all_holes':
        counts = _light_counts(info, info['holes'])
    elif mode == 'no_enclosure':
        counts = _light_counts(info, info['naive'])
    elif mode == 'all_hits':
        counts = _light_counts(info, all_hits=True)
    elif mode == 'no_rotation':
        return info['source'].copy()
    else:
        raise ValueError(mode)
    return _rotate(info['source'], info['background'], counts)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            counts = _light_counts(info)
            expected = _rotate(info['source'], info['background'], counts, coordinate=True)
            _need(np.array_equal(out, expected), "incorrect output")
            _need(np.array_equal(out, solve(inp.copy())), "solver disagreement")
            for color in info['palette']:
                _need(np.sum(info['source'] == color) == np.sum(out == color), "layer area changed")
            # Ownership is a generation invariant, stronger than no actual clash.
            for orbit in _orbits(info['size']):
                occupied = {int(info['source'][p]) for p in orbit} - {info['background']}
                _need(len(occupied) <= 1, "different layers share a rotation orbit")
            if split == 'train':
                teaching.append((info, out))
            else:
                _need(max(counts.values()) >= 5, "test must exercise longer pulse counting")
    observed = set()
    for info, out in teaching:
        for color, count in _light_counts(info).items():
            mask = info['source'] == color
            if all(not np.array_equal(mask, np.rot90(mask, turn)) for turn in (1, 2, 3)):
                observed.add(count)
    _need({0, 1, 2, 3, 4}.issubset(observed), "training needs observable counts zero through four")
    for mode in ('all_holes', 'no_enclosure', 'all_hits', 'no_rotation'):
        _need(any(not np.array_equal(_alternative(info, mode), out) for info, out in teaching),
              f"training does not distinguish {mode}")


def self_test():
    """42 seeds, streamed statistics, count boundaries, corruption, invariance.

    No files, trace payloads, retained puzzle corpus, or unbounded retry loops.
    """
    shapes, geometries, sizes, backgrounds, object_counts, pulse_counts = set(), set(), set(), set(), set(), set()
    signatures, masks, first_positions = set(), set(), set()
    total, multiple, empty_rays = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                total += 1
                shapes.add(pair['input'].shape)
                sizes.add(info['size'])
                backgrounds.add(info['background'])
                object_counts.add(len(info['objects']))
                pulse_counts.update(_light_counts(info).values())
                multiple += len(info['objects']) > 1
                geometries.add((info['size'], tuple((o['row'], o['col'], _key(o['mask'])) for o in info['objects'])))
                for o in info['objects']:
                    masks.add(_key(o['mask']))
                for y in info['active']:
                    r = y-info['row']
                    if 0 <= r < info['size']:
                        hits = np.flatnonzero(info['source'][r] != info['background'])
                        if len(hits):
                            first_positions.add(int(hits[0]))
                        else:
                            empty_rays += 1
    _need(len(signatures) == 42 and len(shapes) >= 20 and len(geometries) >= 100, "variation collapsed")
    _need(len(masks) >= 100 and len(sizes) >= 7, "geometry diversity collapsed")
    _need(backgrounds == set(range(10)) and len(object_counts) >= 2, "role or object count diversity collapsed")
    _need(set(range(8)).issubset(pulse_counts) and multiple >= 10 and empty_rays >= 10,
          "pulse, grouping, or empty-row diversity collapsed")
    _need(len(first_positions) >= 5, "first hits are pinned to a border")
    # A small hand-built member outside the sampler's size range. The central
    # singleton receives a pulse but is invariant; an asymmetric layer turns.
    fixture = np.zeros((9, 15), dtype=np.int8)
    fixture[:, 4] = 9
    for y in (1, 7):
        for x in (7, 13):
            fixture[y, x] = 9
    fixture[1:6, :3] = 8
    fixture[2:5, 1] = 0
    fixture[3, 8] = 1
    fixture[4, 10] = 2
    expected = np.zeros((5, 5), dtype=np.int8)
    expected[0, 3], expected[2, 2] = 1, 2
    before = fixture.copy()
    _need(np.array_equal(solve(fixture), expected) and np.array_equal(fixture, before),
          "small/symmetric member or input immutability failed")
    for variant in ('open', 'tail'):
        altered = fixture.copy()
        if variant == 'open':
            altered[3, 0] = 0
        else:
            altered[1, 3] = 8
        _need(np.array_equal(solve(altered), fixture[2:7, 8:13]),
              "hand-built open/recessed cavity activated")
    missed = fixture.copy()
    missed[missed == 8] = 0
    missed[:7, :3] = 8
    missed[1:6, 1] = 0
    _need(np.array_equal(solve(missed), expected), "ray missing receiver changed a count")
    a, b = generate_puzzle(981, 2, 1), generate_puzzle(981, 2, 1)
    _need(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    # Color equivariance: literal black can be any role, with no learned palette.
    lut = np.array([7, 3, 0, 8, 6, 1, 9, 2, 5, 4], dtype=np.int8)
    p = a['train'][0]
    _need(np.array_equal(solve(lut[p['input']]), lut[p['output']]), "palette dependence")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            generate_puzzle(710+train*5+test, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=total, shapes=len(shapes), geometries=len(geometries),
                emitter_masks=len(masks), sizes=sorted(sizes), backgrounds=sorted(backgrounds),
                object_counts=sorted(object_counts), pulse_counts=sorted(pulse_counts),
                multiple_emitter_pairs=multiple, empty_rays=empty_rays,
                first_hit_columns=sorted(first_positions), independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
