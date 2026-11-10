"""Registered tracing fragments: remove repeated ink, then classify topology.

The full line names the ink color. The multicolor panel glyph is a registration
mark; monochrome panel glyphs give local hole-count -> output-color mappings.
All decisions in solve are made from pixels, independently of construction.
"""
from collections import Counter
import json
import numpy as np


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points)+1, max(c for r, c in points)+1)


def _norm(points):
    r, c, _, _ = _bounds(points)
    return {(u-r, v-c) for u, v in points}


def _cavities(points):
    """Four-neighbor empty components, with an explicit exterior padding."""
    r, c, b, e = _bounds(points)
    empty = {(u, v) for u in range(r-1, b+1) for v in range(c-1, e+1)
             if (u, v) not in points}
    return [g for g in _groups(empty)
            if not any(u in (r-1, b) or v in (c-1, e) for u, v in g)]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=False)


def _parse_region(grid, bg, ink, panel, work):
    panel_groups = _groups(zip(*np.nonzero(panel & (grid != bg))))
    reference, mapping, order = None, {}, []
    for group in panel_groups:
        colors = {int(grid[p]) for p in group}
        if ink in colors:
            raise ValueError("ink is not a legend or reference color")
        if len(colors) == 2:
            if reference is not None:
                raise ValueError("multiple reference marks")
            r, c, b, e = _bounds(group)
            if len(group) != (b-r)*(e-c):
                raise ValueError("reference mark must be a filled rectangle")
            reference = {(u-r, v-c): int(grid[u, v]) for u, v in group}
        elif len(colors) == 1:
            color = next(iter(colors))
            k = len(_cavities(group))
            if k in mapping or color in mapping.values():
                raise ValueError("duplicate legend class or color")
            if any(grid[p] != bg for h in _cavities(group) for p in h):
                raise ValueError("legend cavities must contain background")
            mapping[k] = color
            order.append((_bounds(group)[:2], k))
        else:
            raise ValueError("unrecognized panel component")
    if reference is None or not 1 <= len(mapping) <= 5:
        raise ValueError("need a reference mark and a legend")
    marker_colors = set(reference.values())
    if marker_colors & set(mapping.values()):
        raise ValueError("legend and marker colors must be disjoint")
    if not set(map(int, np.unique(grid[work]))) <= {bg, ink} | marker_colors:
        raise ValueError("unexpected work color")
    parts = _groups(zip(*np.nonzero(work & (grid != bg))))
    if not 2 <= len(parts) <= 4:
        raise ValueError("need two to four separated tracing fragments")
    # One color belongs only to the copied reference; the other may extend
    # into guide strokes. Test both possible roles using visible mark equality.
    registrations = []
    for registration_color in sorted(marker_colors):
        ref_layer = {p for p, v in reference.items() if v == registration_color}
        if len(ref_layer) < 2:
            continue
        rr, rc, _, _ = _bounds(ref_layer)
        offsets, valid = [], True
        for part in parts:
            layer = {p for p in part if int(grid[p]) == registration_color}
            if not layer or _norm(layer) != _norm(ref_layer):
                valid = False
                break
            pr, pc, _, _ = _bounds(layer)
            origin = pr-rr, pc-rc
            if any((u+origin[0], v+origin[1]) not in part
                   or int(grid[u+origin[0], v+origin[1]]) != color
                   for (u, v), color in reference.items()):
                valid = False
                break
            if not any(int(grid[p]) == ink for p in part):
                valid = False
                break
            offsets.append(origin)
        if valid:
            registrations.append(offsets)
    if not registrations:
        raise ValueError("no exact reference registration")
    if any(x != registrations[0] for x in registrations[1:]):
        raise ValueError("ambiguous registration")
    offsets = registrations[0]
    layers = [{(r-dr, c-dc) for r, c in part if int(grid[r, c]) == ink}
              for part, (dr, dc) in zip(parts, offsets)]
    counts = Counter(p for layer in layers for p in layer)
    r, c, b, e = _bounds(counts)
    if b-r > 30 or e-c > 30:
        raise ValueError("registered ink crop exceeds ARC bounds")
    single = {p for p, n in counts.items() if n == 1}
    if not single:
        raise ValueError("all ink is repeated")
    objects = _groups(single)
    holes = [len(_cavities(obj)) for obj in objects]
    return dict(background=bg, ink=ink, marker_colors=marker_colors,
                reference=reference, mapping=mapping, parts=parts,
                offsets=offsets, layers=layers, counts=counts, single=single,
                objects=objects, holes=holes, order=order)


def _analyze_input(input_grid):
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    if frequencies.max()*2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    bg = int(colors[frequencies.argmax()])
    h, w = grid.shape
    candidates = []
    for axis, length in ((0, h), (1, w)):
        for index in range(1, length-1):
            line = grid[index, :] if axis == 0 else grid[:, index]
            ink = int(line[0])
            if ink == bg or not np.all(line == ink):
                continue
            before = np.zeros(grid.shape, dtype=bool)
            after = before.copy()
            if axis == 0:
                before[:index] = True
                after[index+1:] = True
            else:
                before[:, :index] = True
                after[:, index+1:] = True
            for panel, work, side in ((before, after, 'before'),
                                      (after, before, 'after')):
                try:
                    parsed = _parse_region(grid, bg, ink, panel, work)
                    parsed['separator'] = (axis, index, side)
                    candidates.append(parsed)
                except ValueError:
                    continue
    if len(candidates) != 1:
        raise ValueError("input needs one unambiguous panel/work interpretation")
    return candidates[0]


def _render(parsed, *, cancel=True, classify=True):
    r, c, b, e = _bounds(parsed['counts'])
    bg = parsed['background']
    output = np.full((b-r, e-c), bg, dtype=np.int8)
    material = parsed['single'] if cancel else set(parsed['counts'])
    for obj in _groups(material):
        k = len(_cavities(obj))
        color = parsed['mapping'].get(k, bg) if classify else parsed['ink']
        for u, v in obj:
            output[u-r, v-c] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register exact marks, remove ink seen more than once, decode topology."""
    return _render(_analyze_input(input_grid))


def _mask_points(mask):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}


def _object(k, rng, expanded=False, plain=False):
    """Construct final material S and optional repeatedly drawn material D.

    D closes a zero-hole notch, divides a chamber, or plugs a hole. Removing D
    thus changes topology in both directions, rather than only deleting noise.
    """
    if k == 0:
        h = 3 + int(expanded)
        w = int(rng.integers(3, 6)) if not plain else 3
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = mask[:, -1] = mask[-1, :] = True
        duplicate = {(0, c) for c in range(1, w-1)}
    else:
        widths = [1]*k
        if not plain:
            room = int(rng.integers(k))
            widths[room] += int(rng.integers(0, 3))
        h = 3 + int(expanded)
        w = 1 + sum(x+1 for x in widths)
        mask = np.ones((h, w), dtype=bool)
        col = 1
        for width in widths:
            mask[1:-1, col:col+width] = False
            col += width+1
        # A corner cut leaves a diagonal-only opening: still a four-neighbor
        # cavity. A small wall thickening makes unequal/nonrectangular holes.
        if not plain and rng.random() < .35:
            mask[int(rng.choice([0, h-1])), int(rng.choice([0, w-1]))] = False
        points = _mask_points(mask)
        cavities = _cavities(points)
        cavity = cavities[int(rng.integers(len(cavities)))]
        cr, cc, cb, ce = _bounds(cavity)
        if ce-cc >= 3 and rng.random() < .65:
            duplicate = {p for p in cavity if p[1] == (cc+ce-1)//2}
        else:
            duplicate = set(cavity)
        if (not plain and expanded and len(cavity) >= 4 and rng.random() < .5):
            # Add one boundary-adjacent wall cell without altering hole count.
            for p in sorted(cavity):
                candidate = points | {p}
                if len(_groups(candidate)) == 1 and len(_cavities(candidate)) == k:
                    mask[p] = True
                    duplicate.discard(p)
                    break
    points = _mask_points(mask)
    assert len(_groups(points)) == 1 and len(_cavities(points)) == k
    assert not (points & duplicate)
    # Turn some individual shapes in the blueprint; their fragments are then
    # cut in that orientation, never individually rotated during solving.
    if not plain and rng.random() < .22 and max(mask.shape) <= 4:
        points = {(c, mask.shape[0]-1-r) for r, c in points}
        duplicate = {(c, mask.shape[0]-1-r) for r, c in duplicate}
    return points, duplicate


def _scene(rng, classes, plain=False):
    objects = [_object(k, rng, expanded=(i == 0 and not plain), plain=plain)
               for i, k in enumerate(classes)]
    # Packing final objects has no bearing on fragment registration. Bounding
    # boxes have a blank moat so a deleted divider cannot join foreign objects.
    h, w = (8, 13) if not plain else (7, 13)
    for attempt in range(30):
        blocked, placed = set(), []
        order = sorted(range(len(objects)),
                       key=lambda i: -len(objects[i][0] | objects[i][1]))
        for i in order:
            s, d = objects[i]
            _, _, bh, bw = _bounds(s | d)
            candidates = [(r, c) for r in range(h-bh+1) for c in range(w-bw+1)
                          if all((u, v) not in blocked
                                 for u in range(r, r+bh) for v in range(c, c+bw))]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            placed.append((i, {(u+r+3, v+c) for u, v in s},
                           {(u+r+3, v+c) for u, v in d}))
            blocked.update((u, v) for u in range(r-1, r+bh+1)
                           for v in range(c-1, c+bw+1))
        if len(placed) == len(objects):
            return [(s, d) for i, s, d in sorted(placed)]
    return None


def _reference(rng, a, b):
    width = int(rng.integers(3, 5))
    mark = np.full((2, width), a, dtype=np.int8)
    available = list(range(1, mark.size-1))
    mark.flat[0] = b
    for p in rng.choice(available, size=int(rng.integers(1, len(available))), replace=False):
        mark.flat[int(p)] = b
    return mark


def _guide(part, start, end, color, rng):
    r, c = start
    u, v = end
    if rng.random() < .5:
        path = [(r, x) for x in range(min(c, v), max(c, v)+1)]
        path += [(x, v) for x in range(min(r, u), max(r, u)+1)]
    else:
        path = [(x, c) for x in range(min(r, u), max(r, u)+1)]
        path += [(u, x) for x in range(min(c, v), max(c, v)+1)]
    for p in path:
        part.setdefault(p, color)


def _fragments(scene, n, mark, ink, a, rng, triple):
    layers = [set() for _ in range(n)]
    for index, (single, duplicate) in enumerate(scene):
        axis = int(rng.integers(2))
        coords = sorted({p[axis] for p in single})
        cut = coords[len(coords)//2]
        owners = [index % n, (index+1) % n]
        if rng.random() < .5:
            owners.reverse()
        for p in single:
            layers[owners[int(p[axis] >= cut)]].add(p)
        if duplicate:
            support = 3 if triple and n >= 3 and index == 0 else 2
            chosen = rng.choice(n, support, replace=False)
            for i in chosen:
                layers[int(i)].update(duplicate)
    # Four fragments with three objects can leave one owner empty. Move a
    # proper arc to it, preserving every intended support multiplicity.
    for i, layer in enumerate(layers):
        if not layer:
            donor = max(range(n), key=lambda j: len(layers[j]))
            candidates = sorted(p for p in layers[donor]
                                if sum(p in x for x in layers) == 1)
            moved = set(candidates[:max(1, len(candidates)//3)])
            if not moved:
                raise ValueError("cannot give each fragment ink")
            layers[donor] -= moved
            layers[i] |= moved
    anchor_col = int(rng.integers(14-mark.shape[1]))
    start_candidates = [(r, c+anchor_col) for r in range(mark.shape[0])
                        for c in range(mark.shape[1]) if int(mark[r, c]) == a]
    parts = []
    for layer in layers:
        part = {p: ink for p in layer}
        part.update({(r, c+anchor_col): int(mark[r, c])
                     for r in range(mark.shape[0]) for c in range(mark.shape[1])})
        start = start_candidates[int(rng.integers(len(start_candidates)))]
        for group in _groups(layer):
            target = sorted(group)[int(rng.integers(len(group)))]
            _guide(part, start, target, a, rng)
        assert len(_groups(part)) == 1
        r, c, _, _ = _bounds(part)
        parts.append({(u-r, v-c): color for (u, v), color in part.items()})
    return parts


def _glyph(k, rng):
    h, w = 3, max(3, 2*k+1)
    glyph = np.ones((h, w), dtype=bool)
    glyph[1, 1:2*k:2] = False
    if k == 0 and rng.random() < .5:
        glyph = np.ones((2, int(rng.integers(2, 5))), dtype=bool)
    return glyph


def _pack(parts, mapping, mark, bg, ink, rng):
    entries = list(mapping)
    rng.shuffle(entries)
    items = [(_glyph(k, rng), mapping[k]) for k in entries]
    insert = int(rng.integers(len(items)+1))
    items.insert(insert, (mark, None))
    panel_width = sum(mask.shape[1] for mask, color in items) + len(items)+1
    shapes = [_bounds(p)[2:] for p in parts]
    n = len(parts)
    # Free rectangular placement; all fragments include their entire reference
    # mark. A guaranteed two-row packing is used only after bounded retries.
    for attempt in range(25):
        if attempt < 24:
            wh = int(rng.integers(max(h for h, w in shapes), 24))
            width = int(rng.integers(max(panel_width, max(w for h, w in shapes)), 31))
            positions = []
            blocked = np.zeros((wh, width), dtype=bool)
            order = sorted(range(n), key=lambda i: -shapes[i][0]*shapes[i][1])
            for i in order:
                h, w = shapes[i]
                options = [(r, c) for r in range(wh-h+1) for c in range(width-w+1)
                           if not blocked[r:r+h, c:c+w].any()]
                if not options:
                    break
                r, c = options[int(rng.integers(len(options)))]
                positions.append((i, r, c))
                blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
            if len(positions) != n:
                continue
        else:
            wh, width = 23, max(29, panel_width)
            if width > 30:
                raise ValueError("panel too wide")
            order = list(rng.permutation(n))
            slots = [(0, 0), (0, 15), (12, 0), (12, 15)]
            positions = [(int(i), r, c) for i, (r, c) in zip(order, slots)]
        grid = np.full((wh+7, width), bg, dtype=np.int8)
        grid[5, :] = ink
        col = int(rng.integers(width-panel_width+1))+1
        for mask, color in items:
            hh, ww = mask.shape
            row = int(rng.integers(1, 5-hh))
            if color is None:
                grid[row:row+hh, col:col+ww] = mask
            else:
                grid[row:row+hh, col:col+ww][mask] = color
            col += ww+1
        for i, r, c in positions:
            for (u, v), color in parts[i].items():
                grid[u+r+7, v+c] = color
        if np.count_nonzero(grid == bg)*2 > grid.size:
            return grid
    raise ValueError("packing did not preserve a dominant background")


def _int_count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _int_count(num_train, 'num_train', 2, 10)
    nq = _int_count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Only distribution preferences persist. Each pair visibly states every
    # color convention, reference mark, and topology mapping it requires.
    stable_roles = bool(rng.random() < .35)
    base_palette = [int(x) for x in rng.permutation(10)]
    shift = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(40):
                palette = base_palette if stable_roles else [int(x) for x in rng.permutation(10)]
                bg, ink, a, b = palette[:4]
                if split == 'train' and index < 2:
                    classes = [0, 1, 2]
                    mapping_keys = [0, 2] if index == 0 else [1, 2]
                    n = 2 if index == 0 else 3
                else:
                    population = int(rng.integers(3, 5))
                    classes = [int(x) for x in rng.integers(0, 3, population)]
                    if population == 3 and rng.random() < .45:
                        classes[int(rng.integers(population))] = 3
                    mapping_keys = [int(x) for x in rng.choice(4, int(rng.integers(2, 4)), replace=False)]
                    if not set(classes) & set(mapping_keys):
                        mapping_keys[0] = classes[int(rng.integers(len(classes)))]
                    n = int(rng.integers(3, 5) if split == 'test' else rng.integers(2, 4))
                    if split == 'test' and index == 0:
                        n = 4  # Every puzzle has an explicit multiplicity shift.
                scene = _scene(rng, classes, plain=attempt >= 25)
                if scene is None:
                    continue
                mapping = dict(zip(mapping_keys, palette[4:4+len(mapping_keys)]))
                mark = _reference(rng, a, b)
                try:
                    parts = _fragments(scene, n, mark, ink, a, rng,
                                       triple=(index == 1 or split == 'test'))
                    inp = _pack(parts, mapping, mark, bg, ink, rng)
                    # The first two examples already demonstrate whole-example
                    # orientation changes; test shift increases fragment count.
                    rotation = (index + shift) % 4
                    inp = np.rot90(inp, rotation).copy()
                    parsed = _analyze_input(inp)
                    # Compare the solver's recovered geometry against the
                    # independently constructed target survivor set.
                    intended = set().union(*(s for s, d in scene))
                    rotated = intended
                    for _ in range(rotation):
                        rotated = {(-c, r) for r, c in rotated}
                    if _norm(parsed['single']) != _norm(rotated):
                        raise ValueError("registered material differs from construction")
                    all_ink = set().union(*(s | d for s, d in scene))
                    r0, c0, bottom, right = _bounds(all_ink)
                    expected = np.full((bottom-r0, right-c0), bg, dtype=np.int8)
                    for (single, duplicate), k in zip(scene, classes):
                        for r, c in single:
                            expected[r-r0, c-c0] = mapping.get(k, bg)
                    out = np.rot90(expected, rotation).copy()
                    if not np.array_equal(_render(parsed), out):
                        raise ValueError("pixel solution disagrees with constructed label")
                    if split == 'train' and index == 0:
                        visible = _mask_points(out != bg)
                        vr, vc, vb, ve = _bounds(visible)
                        if (vr, vc, vb, ve) == (0, 0, *out.shape):
                            # Make the saved pre-erasure crop observable even
                            # in a puzzle with only two demonstrations.
                            continue
                    key = _signature(inp)
                    if key in seen:
                        continue
                    seen.add(key)
                    puzzle[split].append({'input': inp, 'output': out})
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded generation attempts exhausted")
    return puzzle


def _signature(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return tuple(grid.shape), grid.tobytes()


def _features(pair):
    p = _analyze_input(pair['input'])
    unregistered_counts = Counter((r+dr, c+dc)
                                  for layer, (dr, dc) in zip(p['layers'], p['offsets'])
                                  for r, c in layer)
    unregistered = dict(p, counts=unregistered_counts,
                        single={cell for cell, n in unregistered_counts.items() if n == 1})
    before_holes = sorted(len(_cavities(g)) for g in _groups(p['counts']))
    source_holes = tuple(len(_cavities(g)) for layer in p['layers'] for g in _groups(layer))
    out = np.asarray(pair['output'])
    visible = _mask_points(out != p['background'])
    crop_witness = not visible or _bounds(visible) != (0, 0, *out.shape)
    return dict(shape=list(np.asarray(pair['input']).shape),
                output_shape=list(np.asarray(pair['output']).shape),
                fragments=len(p['parts']), objects=len(p['objects']),
                classes=sorted(p['holes']), keys=sorted(p['mapping']),
                union_classes=before_holes, source_classes=list(source_holes),
                support=sorted(set(p['counts'].values())),
                repeated=sum(n > 1 for n in p['counts'].values()),
                areas=sorted(len(h) for obj in p['objects'] for h in _cavities(obj)),
                removed=sum(k not in p['mapping'] for k in p['holes']),
                background=p['background'], separator_axis=p['separator'][0],
                rank_witness=any(k != j for j, (_, k) in enumerate(sorted(p['order']))),
                registration_witness=not np.array_equal(_render(unregistered), pair['output']),
                crop_witness=crop_witness,
                cancellation_witness=not np.array_equal(_render(p, cancel=False), pair['output']),
                classification_witness=not np.array_equal(_render(p, classify=False), pair['output']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            p = _analyze_input(inp)
            assert np.array_equal(_render(p), out), "wrong registered topology output"
            assert p['single'] and any(n > 1 for n in p['counts'].values())
            # Guides/registration are excluded; neutral support-one cells alone
            # determine connected components. An object is recolored in full.
            r, c, b, e = _bounds(p['counts'])
            assert out.shape == (b-r, e-c), "wrong pre-cancellation crop"
            for obj, k in zip(p['objects'], p['holes']):
                color = p['mapping'].get(k, p['background'])
                assert all(int(out[u-r, v-c]) == color for u, v in obj)
            assert all(int(out[u-r, v-c]) == p['background']
                       for (u, v), n in p['counts'].items() if n > 1)
            if split == 'train':
                teaching.append(_features(pair))
    assert any(f['cancellation_witness'] for f in teaching), "cancellation not demonstrated"
    assert all(f['registration_witness'] for f in teaching), "registration not needed"
    assert all(f['classification_witness'] for f in teaching), "legend not needed"
    assert any(f['removed'] for f in teaching), "erasure not demonstrated"
    assert len(set(k for f in teaching for k in f['classes'] if k in f['keys'])) >= 2
    assert any(f['rank_witness'] for f in teaching), "legend rank shortcut unchallenged"
    assert any(f['crop_witness'] for f in teaching), "pre-erasure cropping not demonstrated"
    assert any(3 in f['support'] for f in teaching), "three-copy deletion not taught"


def self_test():
    """Bounded semantic regressions and measured diversity over 42 seeds."""
    diagonal = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert len(_cavities(diagonal)) == 1
    assert len(_cavities(diagonal - {(0, 1)})) == 0
    ring = {(r, c) for r in range(3) for c in range(5)
            if r in (0, 2) or c in (0, 4)}
    divider = {(1, 2)}
    assert len(_cavities(ring)) == 1
    assert len(_cavities(ring | divider)) == 2
    assert len(_cavities(ring | {(1, 1), (1, 2), (1, 3)})) == 0
    assert len(_cavities(ring - {(0, 2)})) == 0
    features, signatures = [], set()
    local_variation = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_signature(pair['input']) for s in p for pair in p[s]))
        local = [_features(pair) for s in p for pair in p[s]]
        features.extend(local)
        local_variation += int(len({tuple(f['classes']) for f in local}) > 1
                               and len({f['fragments'] for f in local}) > 1)
    assert len(signatures) == 42
    assert local_variation >= 35
    assert len({tuple(f['shape']) for f in features}) >= 20
    assert len({tuple(f['output_shape']) for f in features}) >= 12
    assert {f['fragments'] for f in features} == {2, 3, 4}
    assert len({tuple(f['classes']) for f in features}) >= 12
    assert {f['background'] for f in features} == set(range(10))
    assert {f['separator_axis'] for f in features} == {0, 1}
    assert any(3 in f['classes'] for f in features)
    assert any(f['objects'] >= 4 for f in features)
    assert any(max(f['areas'], default=0) >= 4 for f in features)
    assert sum(f['classes'] != f['union_classes'] for f in features) >= 100
    # The exact-once rule differs from parity when ink has three copies.
    example = generate_puzzle(0, 2, 1)
    p = _analyze_input(example['train'][1]['input'])
    assert any(n == 3 for n in p['counts'].values())
    parity = {cell for cell, n in p['counts'].items() if n % 2}
    assert parity != p['single']
    assert not np.array_equal(_render(dict(p, single=parity)), example['train'][1]['output'])
    # Palette permutation and transposition require no new solving convention.
    perm = np.array([7, 2, 9, 1, 0, 6, 3, 8, 5, 4], dtype=np.int8)
    for pair in example['train'] + example['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
        assert np.array_equal(solve(pair['input'].T), pair['output'].T)
        assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    again = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in example
               for x, y in zip(example[s], again[s]) for k in x)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(100+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{'input': x['input'].copy(), 'output': x['output'].copy()}
                   for x in example[s]] for s in example}
    corrupt['test'][0]['output'].flat[0] = (int(corrupt['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': len(signatures),
            'input_shapes': len({tuple(f['shape']) for f in features}),
            'output_shapes': len({tuple(f['output_shape']) for f in features}),
            'class_populations': len({tuple(f['classes']) for f in features}),
            'topology_changed_pairs': sum(f['classes'] != f['union_classes'] for f in features),
            'within_puzzle_population_and_fragment_variation': local_variation}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
