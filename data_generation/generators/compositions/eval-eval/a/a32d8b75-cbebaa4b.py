"""Assembled printing dies: exact docking creates the negative-space stencil.

The only glyph is a complete framed binary square. All other components are
rigid contour fragments. Translation-only pin superposition reconstructs a die
up to global translation. Its enclosed empty cells print the exchanged glyph;
its solid cells enlarge uniformly. No parent module or generation state is used
by solve. Standard library and NumPy only; no on-disk runtime state.
"""
from collections import Counter
import hashlib
import json
import numpy as np

N4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _components(cells):
    todo = set(cells)
    found = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    comp.add(q)
                    stack.append(q)
        found.append(comp)
    return found


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _normal(cells):
    t, l, _, _ = _bbox(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _enclosed(cells):
    """Four-neighbor background unable to reach a one-cell exterior collar."""
    t, l, b, r = _bbox(cells)
    start = (t - 1, l - 1)
    outside, stack = {start}, [start]
    while stack:
        a, c = stack.pop()
        for da, dc in N4:
            q = a + da, c + dc
            if (t - 1 <= q[0] <= b + 1 and l - 1 <= q[1] <= r + 1
                    and q not in cells and q not in outside):
                outside.add(q)
                stack.append(q)
    return {(a, c) for a in range(t, b + 1) for c in range(l, r + 1)
            if (a, c) not in cells and (a, c) not in outside}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _decode(value):
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError('the input needs a blank exterior border')
    comps = _components(map(tuple, np.argwhere(a != bg).tolist()))
    legends = [i for i, comp in enumerate(comps) if len({int(a[q]) for q in comp}) == 3]
    if len(legends) != 1:
        raise ValueError('expected one three-color framed glyph')
    li = legends[0]
    t, l, b, r = _bbox(comps[li])
    frame = a[t:b + 1, l:r + 1]
    if not (frame.shape[0] == frame.shape[1] and 4 <= len(frame) <= 5
            and len(comps[li]) == frame.size):
        raise ValueError('glyph frame must be a filled square of side 4 or 5')
    pin = int(frame[0, 0])
    if not (np.all(frame[0] == pin) and np.all(frame[-1] == pin)
            and np.all(frame[:, 0] == pin) and np.all(frame[:, -1] == pin)):
        raise ValueError('glyph frame must have a uniform pin-color border')
    tile = frame[1:-1, 1:-1].copy()
    colors = set(map(int, np.unique(tile)))
    if len(colors) != 2 or colors & {bg, pin}:
        raise ValueError('glyph must show two complete content roles')
    pieces = []
    for i, comp in enumerate(comps):
        if i == li:
            continue
        palette = {int(a[q]) for q in comp}
        if len(palette) != 2 or pin not in palette or palette & colors:
            raise ValueError('each fragment must have payload and pin colors')
        body = {q for q in comp if int(a[q]) != pin}
        pins = comp - body
        if len(_components(body)) != 1 or not pins:
            raise ValueError('disconnected payload or missing pins')
        top, left, _, _ = _bbox(comp)
        d = -top, -left
        pieces.append(dict(body=_shift(body, d), pins=_shift(pins, d),
                           color=int(a[next(iter(body))])))
    if not 3 <= len(pieces) <= 5:
        raise ValueError('expected 3..5 contour fragments')
    return bg, pin, tile, pieces


class _SearchLimit(ValueError):
    pass


def _assemble(pieces, bg, pin, limit=12000):
    """Exhaust all exact assemblies, comparing coarse grids up to translation.

    Individual pin cells are constraints, not presegmented keys. A placed pin
    needs exactly one second owner. Payload/payload and payload/pin overlaps and
    triple pin ownership are forbidden. Every insertion touches the assembly.
    """
    n = len(pieces)
    links = {}
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces):
            if i == j:
                continue
            matches = {}
            offsets = sorted({(r - s, c - t) for r, c in a['pins'] for s, t in b['pins']})
            for d in offsets:
                body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                if body & (a['body'] | a['pins']) or pins & a['body']:
                    continue
                for q in sorted(a['pins'] & pins):
                    matches.setdefault(q, []).append((j, d))
            links[i, j] = matches
    root = max(range(n), key=lambda i: (len(pieces[i]['body']), -i))
    solutions, visited, cache = {}, set(), {}
    nodes = 0

    def moved(i, d):
        k = i, d
        if k not in cache:
            cache[k] = _shift(pieces[i]['body'], d), _shift(pieces[i]['pins'], d)
        return cache[k]

    def visit(pos, bodies, singles, doubles, owners):
        nonlocal nodes
        state = tuple(sorted(pos.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise _SearchLimit('assembly uniqueness search exceeded its bounded budget')
        if not singles:
            if len(pos) != n:
                return
            allcells = bodies | doubles
            t, l, b, r = _bbox(allcells)
            coarse = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
            for i, d in pos.items():
                for a, c in moved(i, d)[0]:
                    coarse[a - t, c - l] = pieces[i]['color']
            for a, c in doubles:
                coarse[a - t, c - l] = pin
            normpos = {i: (d[0] - t, d[1] - l) for i, d in pos.items()}
            solutions.setdefault(_key(coarse), (coarse, normpos))
            return
        if len(pos) == n:
            return
        best = None
        for q in sorted(singles):
            i, di = owners[q], pos[owners[q]]
            local = q[0] - di[0], q[1] - di[1]
            choices = []
            for j in range(n):
                if j in pos:
                    continue
                for _, rel in links[i, j].get(local, []):
                    d = di[0] + rel[0], di[1] + rel[1]
                    body, pins = moved(j, d)
                    if body & (bodies | singles | doubles) or pins & (bodies | doubles):
                        continue
                    choices.append((j, d, body, pins, pins & singles))
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        for j, d, body, pins, overlap in best:
            pos[j] = d
            newowners = dict(owners)
            newowners.update({q: j for q in pins - overlap})
            visit(pos, bodies | body, (singles | pins) - overlap,
                  doubles | overlap, newowners)
            del pos[j]
            if len(solutions) > 1:
                return

    a = pieces[root]
    visit({root: (0, 0)}, a['body'], a['pins'], set(), {q: root for q in a['pins']})
    if len(solutions) != 1:
        raise ValueError('fragments do not determine one assembly up to translation')
    coarse, positions = next(iter(solutions.values()))
    return coarse, positions, nodes


def _print_die(coarse, tile, bg, *, exchange=True, fill_exterior=False, rotate_tile=False):
    occupied = set(map(tuple, np.argwhere(coarse != bg).tolist()))
    holes = _enclosed(occupied)
    if not holes:
        raise ValueError('die has no enclosed chamber')
    n = len(tile)
    out = np.repeat(np.repeat(coarse, n, axis=0), n, axis=1)
    if exchange:
        c = np.unique(tile).astype(int)
        stamp = (int(c[0]) + int(c[1]) - tile).astype(np.int8)
    else:
        stamp = tile.copy()
    if rotate_tile:
        stamp = np.rot90(stamp)
    targets = (set(map(tuple, np.argwhere(coarse == bg).tolist())) if fill_exterior else holes)
    for r, c in targets:
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, pin, tile, pieces = _decode(input_grid)
    coarse, _, _ = _assemble(pieces, bg, pin)
    out = _print_die(coarse, tile, bg)
    if any(d > 30 for d in out.shape):
        raise ValueError('the enlarged die exceeds the ARC size limit')
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _room_shell(rng, rooms, scale):
    """Connected, often concave room complexes on an uneven rectilinear mesh."""
    h = int(rng.integers(2, 4))
    w = int(rng.integers(2, 4))
    if h * w <= rooms:
        h = w = 3
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < rooms:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N4
                           if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
        cells.add(_pick(rng, frontier))
    stepmax = 4 if scale == 2 else 3
    yy = [0] + list(map(int, np.cumsum(rng.integers(2, stepmax + 1, size=h))))
    xx = [0] + list(map(int, np.cumsum(rng.integers(2, stepmax + 1, size=w))))
    shell = set()
    for r, c in cells:
        t, b, l, right = yy[r], yy[r + 1], xx[c], xx[c + 1]
        shell |= {(a, x) for a in range(t, b + 1) for x in (l, right)}
        shell |= {(a, x) for a in (t, b) for x in range(l, right + 1)}
    # Delete a shared wall to merge rectangular rooms into an irregular chamber.
    shared = []
    for r, c in sorted(cells):
        if (r + 1, c) in cells:
            shared.append({(yy[r + 1], x) for x in range(xx[c] + 1, xx[c + 1])})
        if (r, c + 1) in cells:
            shared.append({(a, xx[c + 1]) for a in range(yy[r] + 1, yy[r + 1])})
    if shared and rng.random() < .55:
        shell -= _pick(rng, shared)
    t, l, b, right = _bbox(shell)
    if max(b - t + 1, right - l + 1) * scale > 30:
        return None
    return _shift(shell, (-t, -l))


def _split_options(parts):
    choices = []
    for i, part in enumerate(parts):
        body, oldpins = part['body'], part['pins']
        t, l, b, r = _bbox(body)
        for axis, lo, hi in ((0, t, b), (1, l, r)):
            for cut_at in range(lo + 1, hi):
                cut = {q for q in body if q[axis] == cut_at}
                if not 2 <= len(cut) <= 5 or len(_components(cut)) != len(cut):
                    continue
                if any(max(abs(a - c), abs(b - d)) <= 1
                       for a, b in cut for c, d in oldpins):
                    continue
                comps = _components(body - cut)
                if len(comps) != 2 or min(map(len, comps)) < 3:
                    continue
                if any(not all(any((q[0] + dr, q[1] + dc) in comp for dr, dc in N4)
                                   for comp in comps) for q in cut):
                    continue
                children = [dict(body=comp, pins=set(cut)) for comp in comps]
                good = True
                for q in oldpins:
                    owners = [j for j, comp in enumerate(comps)
                              if any((q[0] + dr, q[1] + dc) in comp for dr, dc in N4)]
                    if len(owners) != 1:
                        good = False
                        break
                    children[owners[0]]['pins'].add(q)
                if good and all(len(_components(p['body'] | p['pins'])) == 1 for p in children):
                    choices.append((i, children))
    return choices


def _partition(rng, shell, count):
    parts = [dict(body=shell, pins=set())]
    for _ in range(count - 1):
        options = _split_options(parts)
        if not options:
            return None
        i, children = _pick(rng, options)
        parts = parts[:i] + children + parts[i + 1:]
    return parts


def _scatter(rng, pieces, tile, bg, pin):
    objects = []
    n = len(tile)
    legend = np.full((n + 2, n + 2), pin, dtype=np.int8)
    legend[1:-1, 1:-1] = tile
    objects.append({(r, c): int(legend[r, c]) for r in range(n + 2) for c in range(n + 2)})
    for p in pieces:
        t, l, _, _ = _bbox(p['body'] | p['pins'])
        obj = {(r - t, c - l): p['color'] for r, c in p['body']}
        obj.update({(r - t, c - l): pin for r, c in p['pins']})
        objects.append(obj)
    for restart in range(10):
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
        out = np.full((h, w), bg, dtype=np.int8)
        forbidden = set()
        order = list(map(int, rng.permutation(len(objects))))
        order.sort(key=lambda i: -len(objects[i]))
        success = True
        for i in order:
            obj = objects[i]
            _, _, b, r = _bbox(obj)
            for _ in range(240):
                d = int(rng.integers(1, h - b - 1)), int(rng.integers(1, w - r - 1))
                moved = _shift(obj, d)
                if moved & forbidden:
                    continue
                for a, c in obj:
                    out[a + d[0], c + d[1]] = obj[a, c]
                forbidden |= {(a + da, c + dc) for a, c in moved for da in (-1, 0, 1) for dc in (-1, 0, 1)}
                break
            else:
                success = False
                break
        if success:
            return out
    return None


def _joint_data(pieces, positions):
    pins = [_shift(p['pins'], positions[i]) for i, p in enumerate(pieces)]
    joints, degree = [], [0] * len(pieces)
    for i in range(len(pieces)):
        for j in range(i):
            q = pins[i] & pins[j]
            if q:
                joints.append(_normal(q))
                degree[i] += 1
                degree[j] += 1
    return joints, degree


def _make_pair(rng, count, rooms, scale, witness=False):
    for _attempt in range(300):
        shell = _room_shell(rng, rooms, scale)
        if shell is None:
            continue
        parts = _partition(rng, shell, count)
        if parts is None:
            continue
        colors = list(map(int, rng.permutation(10)))
        bg, pin, ca, cb = colors[:4]
        payloads = colors[4:4 + int(rng.integers(1, min(4, count) + 1))]
        for i, part in enumerate(parts):
            part['color'] = payloads[i % len(payloads)]
        # Apply a whole-die symmetry before scattering. Pieces never rotate in solving.
        k, reflect = int(rng.integers(4)), bool(rng.integers(2))
        def transform(cells):
            out = set()
            for r, c in cells:
                if reflect:
                    c = -c
                for _ in range(k):
                    r, c = c, -r
                out.add((r, c))
            return out
        for part in parts:
            part['body'], part['pins'] = transform(part['body']), transform(part['pins'])
        joints, _ = _joint_data(parts, {i: (0, 0) for i in range(count)})
        if witness and not any(len(a) == len(b) and a != b for i, a in enumerate(joints) for b in joints[:i]):
            continue
        bits = rng.random((scale, scale)) < rng.uniform(.2, .8)
        if bits.all() or not bits.any() or any(np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            continue
        if scale == 3 and any(np.array_equal(bits, np.rot90(bits[:, ::-1], k)) for k in range(4)):
            continue
        tile = np.where(bits, ca, cb).astype(np.int8)
        inp = _scatter(rng, parts, tile, bg, pin)
        if inp is None:
            continue
        try:
            dbg, dpin, dtile, decoded = _decode(inp)
            coarse, positions, _ = _assemble(decoded, dbg, dpin, limit=3500)
        except ValueError:
            continue
        occupied = set(map(tuple, np.argwhere(coarse != bg).tolist()))
        holes = _enclosed(occupied)
        inherited_holes = set().union(*[_enclosed(_shift(p['body'] | p['pins'], positions[i]))
                                       for i, p in enumerate(decoded)])
        outside = set(map(tuple, np.argwhere(coarse == bg).tolist())) - holes
        if not holes or not holes - inherited_holes or not outside:
            continue
        # Check against the construction, independently of the assembly search.
        full = set().union(*(p['body'] | p['pins'] for p in parts))
        t, l, b, r = _bbox(full)
        expected_coarse = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
        for p in parts:
            for a, c in p['body']:
                expected_coarse[a - t, c - l] = p['color']
            for a, c in p['pins']:
                expected_coarse[a - t, c - l] = pin
        if not np.array_equal(coarse, expected_coarse):
            continue
        out = _print_die(expected_coarse, tile, bg)
        if max(out.shape) <= 30:
            return {'input': inp, 'output': out}
    raise RuntimeError('bounded sampler could not construct a unique chamber die')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    shift = str(rng.choice(['fragments', 'chambers']))
    scales = list(map(int, rng.permutation([2, 3])))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if shift == 'fragments':
                pieces = (3 + i % 2) if split == 'train' else 5
                rooms = int(rng.integers(3, 7))
            else:
                pieces = (3 + i % 2) if split == 'train' else int(rng.integers(3, 6))
                rooms = 3 + i % 2 if split == 'train' else 6
            # Larger fragment sets need enough distinct wall crossings to cut.
            rooms = max(pieces, rooms)
            scale = scales[i] if split == 'train' and i < 2 else int(rng.choice([2, 3]))
            for _ in range(4):
                pair = _make_pair(rng, pieces, rooms, scale, witness=split == 'train' and i < 2)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('duplicate inputs after bounded retries')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    bg, pin, tile, pieces = _decode(pair['input'])
    coarse, positions, nodes = _assemble(pieces, bg, pin)
    cells = set(map(tuple, np.argwhere(coarse != bg).tolist()))
    holes = _enclosed(cells)
    allvoid = set(map(tuple, np.argwhere(coarse == bg).tolist()))
    inherited = set().union(*[_enclosed(_shift(p['body'] | p['pins'], positions[i]))
                             for i, p in enumerate(pieces)])
    joints, degrees = _joint_data(pieces, positions)
    return dict(piece_count=len(pieces), tile_size=len(tile), die_shape=coarse.shape,
                input_shape=np.asarray(pair['input']).shape, background=bg, pin=pin,
                chambers=len(_components(holes)), chamber_cells=len(holes),
                newly_enclosed=len(holes - inherited), exterior_cells=len(allvoid - holes),
                joints=joints, degrees=degrees, joint_sizes=[len(j) for j in joints],
                equal_area_different_keys=any(len(a) == len(b) and a != b for i, a in enumerate(joints) for b in joints[:i]),
                payload_colors=len({p['color'] for p in pieces}),
                body_shapes=[_normal(p['body']) for p in pieces], search_nodes=nodes)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, first = set(), []
    records = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            bg, pin, tile, pieces = _decode(inp)
            coarse, positions, _ = _assemble(pieces, bg, pin)
            cells = set(map(tuple, np.argwhere(coarse != bg).tolist()))
            holes = _enclosed(cells)
            assert holes
            multiplicity = Counter()
            occupied_bodies = set()
            for i, p in enumerate(pieces):
                body, pins = _shift(p['body'], positions[i]), _shift(p['pins'], positions[i])
                assert not body & occupied_bodies
                occupied_bodies |= body
                multiplicity.update(pins)
            assert set(multiplicity.values()) == {2}
            assert not set(multiplicity) & occupied_bodies
            assert len(cells) == len(occupied_bodies) + len(multiplicity)
            n = len(tile)
            assert out.shape == (coarse.shape[0] * n, coarse.shape[1] * n)
            assert np.array_equal(out, solve(inp))
            # Check output macro cells directly, including both colors of each stamp.
            vals = list(map(int, np.unique(tile)))
            exchanged = vals[0] + vals[1] - tile.astype(int)
            for r in range(len(coarse)):
                for c in range(coarse.shape[1]):
                    block = out[r * n:(r + 1) * n, c * n:(c + 1) * n]
                    if (r, c) in holes:
                        assert np.array_equal(block, exchanged)
                    else:
                        assert np.all(block == coarse[r, c])
            f = pair_features(pair)
            records[split].append(f)
            assert f['newly_enclosed'] > 0 and f['exterior_cells'] > 0
            # Cutting a fragment may distribute an older key over two children;
            # cycles as well as branched docking graphs are legitimate.
            joints, _ = _joint_data(pieces, positions)
            assert len(joints) >= len(pieces) - 1
            pinsets = [_shift(p['pins'], positions[i]) for i, p in enumerate(pieces)]
            reached, stack = {0}, [0]
            while stack:
                i = stack.pop()
                for j in range(len(pieces)):
                    if j not in reached and pinsets[i] & pinsets[j]:
                        reached.add(j)
                        stack.append(j)
            assert len(reached) == len(pieces)
            if split == 'train' and index < 2:
                assert f['equal_area_different_keys']
                assert not np.array_equal(out, _print_die(coarse, tile, bg, exchange=False))
                assert not np.array_equal(out, _print_die(coarse, tile, bg, fill_exterior=True))
                assert not np.array_equal(out, _print_die(coarse, tile, bg, rotate_tile=True))
                if n == 3:
                    for k in range(4):
                        assert not np.array_equal(out, _print_die(coarse, np.rot90(tile[:, ::-1], k), bg))
                first.append(f)
    assert {f['tile_size'] for f in first} == {2, 3}
    assert {f['piece_count'] for f in first} == {3, 4}
    assert (all(f['piece_count'] > max(t['piece_count'] for t in records['train']) for f in records['test'])
            or all(f['chambers'] > max(t['chambers'] for t in records['train']) for f in records['test']))


def self_test():
    """Stream 42 seeds and bounded count/metamorphic/negative checks; no files."""
    stats = {k: set() for k in ('input_shapes', 'die_shapes', 'body_shapes', 'keys', 'degrees', 'chambers', 'pin_colors', 'backgrounds', 'pieces', 'scales', 'palettes')}
    total, maxnodes, branched, reused, cyclic = 0, 0, 0, 0, 0
    hashes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                digest.update(np.asarray(pair['input'], dtype=np.uint8).tobytes())
                f = pair_features(pair)
                total += 1
                stats['input_shapes'].add(f['input_shape'])
                stats['die_shapes'].add(f['die_shape'])
                stats['body_shapes'].update(f['body_shapes'])
                stats['keys'].update(f['joints'])
                stats['degrees'].update(f['degrees'])
                stats['chambers'].add(f['chambers'])
                stats['pin_colors'].add(f['pin'])
                stats['backgrounds'].add(f['background'])
                stats['pieces'].add(f['piece_count'])
                stats['scales'].add(f['tile_size'])
                stats['palettes'].add(tuple(map(int, np.unique(pair['input']))))
                maxnodes = max(maxnodes, f['search_nodes'])
                branched += max(f['degrees']) >= 3
                reused += f['payload_colors'] < f['piece_count']
                cyclic += len(f['joints']) > f['piece_count'] - 1
        hashes.add(digest.hexdigest())
        if seed in (0, 17, 41):
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][i][k], repeat[s][i][k])
                       for s in puzzle for i in range(len(puzzle[s])) for k in ('input', 'output'))
    assert len(hashes) == 42
    assert stats['pieces'] == {3, 4, 5} and stats['scales'] == {2, 3}
    assert len(stats['input_shapes']) >= 30 and len(stats['die_shapes']) >= 20
    assert len(stats['body_shapes']) >= 100 and len(stats['keys']) >= 15
    assert len(stats['chambers']) >= 4 and stats['degrees'] >= {1, 2, 3}
    assert len(stats['pin_colors']) == 10 and len(stats['backgrounds']) == 10
    assert branched > 0 and reused > 50 and len(stats['palettes']) >= 50
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + 10 * nt + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for a, b in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Independent scatter cannot alter the cropped result; color renaming commutes.
    p = generate_puzzle(123, 2, 1)
    pair = p['test'][0]
    bg, pin, tile, pieces = _decode(pair['input'])
    inp = _scatter(np.random.default_rng(812), pieces, tile, bg, pin)
    assert inp is not None and np.array_equal(solve(inp), pair['output'])
    permutation = np.random.default_rng(91).permutation(10)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # A's binary exchange is answer-affecting, while the rigid shell is unchanged.
    changed = pair['input'].copy()
    a, b = map(int, np.unique(tile))
    amask, bmask = changed == a, changed == b
    changed[amask], changed[bmask] = b, a
    assert not np.array_equal(solve(changed), pair['output'])
    # Losing one pin makes exact double ownership impossible.
    erased = [dict(body=set(p['body']), pins=set(p['pins']), color=p['color']) for p in pieces]
    erased[0]['pins'].remove(min(erased[0]['pins']))
    try:
        _assemble(erased, bg, pin)
    except ValueError:
        pass
    else:
        raise AssertionError('unpaired pin accepted')
    ambiguous = [dict(body={(1, c) for c in range(5)}, pins={(0, 0), (0, 4)}, color=4),
                 dict(body={(0, 0)}, pins={(1, 0)}, color=2),
                 dict(body={(0, 0)}, pins={(1, 0)}, color=3)]
    try:
        _assemble(ambiguous, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('two distinct assembled outputs accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=total, variation={k: len(v) for k, v in stats.items()},
                branched_pairs=branched, cyclic_pairs=cyclic,
                repeated_payload_pairs=reused, max_search_nodes=maxnodes)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
