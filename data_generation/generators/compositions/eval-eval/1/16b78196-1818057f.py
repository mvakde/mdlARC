"""Cross-weld docking: geometry packs fragments; seam crosses select the packing.

Only standard-library modules and NumPy are used. No parent code or data is
imported. The solver searches visible objects and all boundary alignments.
"""
from __future__ import annotations

import itertools
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _dilate(mask):
    result = mask.copy()
    h, w = mask.shape
    for dr, dc in itertools.product((-1, 0, 1), repeat=2):
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        result[r0 + dr:r1 + dr, c0 + dc:c1 + dc] |= mask[r0:r1, c0:c1]
    return result


def _scene(grid):
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    objects = _components(grid != bg)
    bands = []
    for i, cells in enumerate(objects):
        for axis in (0, 1):
            if len(np.unique(cells[:, 1 - axis])) == grid.shape[1 - axis]:
                bands.append((i, axis))
    if len(bands) != 1:
        raise ValueError("expected one spanning connected scaffold")
    index, axis = bands[0]
    a = grid.copy() if axis == 0 else grid.T.copy()
    band_cells = objects[index] if axis == 0 else objects[index][:, ::-1]
    band_mask = np.zeros(a.shape, dtype=bool)
    band_mask[tuple(band_cells.T)] = True
    vals, freq = np.unique(a[band_mask], return_counts=True)
    if len(vals) != 2 or freq[0] == freq[1]:
        raise ValueError("scaffold must expose a majority carrier and minority ink")
    band, ink = int(vals[np.argmax(freq)]), int(vals[np.argmin(freq)])
    edges = []
    for c in range(a.shape[1]):
        rows = np.flatnonzero(band_mask[:, c])
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("scaffold columns must be solid intervals")
        edges.append((int(rows[0]), int(rows[-1]) + 1))
    pieces = []
    for i, cells in enumerate(objects):
        if i == index:
            continue
        cells = cells if axis == 0 else cells[:, ::-1]
        values = a[tuple(cells.T)].copy()
        carriers = set(map(int, values)) - {ink}
        if len(carriers) != 1 or band in carriers:
            raise ValueError("each detached fragment needs one non-scaffold carrier")
        pieces.append((cells, values, carriers.pop()))
    if not pieces:
        raise ValueError("no movable fragments")
    return dict(a=a, bg=bg, band=band, ink=ink, axis=axis,
                band_mask=band_mask, edges=np.array(edges), pieces=pieces)


def _profile_description(cells, sign):
    x = cells[:, 1] - cells[:, 1].min()
    y = sign * cells[:, 0]
    lo, hi = [], []
    for c in range(int(x.max()) + 1):
        rows = np.sort(y[x == c])
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("fragments must be full-width column-convex layers")
        lo.append(int(rows[0]))
        hi.append(int(rows[-1]) + 1)
    origin = min(lo)
    return (tuple(v - origin for v in lo), tuple(v - origin for v in hi),
            np.column_stack((y - origin, x)))


def _centers(mask):
    out = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        out[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                            & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    return np.argwhere(out)


def _welds(assembled, ink, owners, sequential=False):
    """Detect on the unchanged assembly; only multi-object crosses are welds."""
    mask = assembled == ink
    union = np.zeros(mask.shape, dtype=bool)
    seam_centers, internal = [], 0
    for r, c in _centers(mask):
        points = [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]
        identities = {int(owners[p]) for p in points}
        if len(identities) == 1:
            internal += 1
            continue
        if sequential and not all(mask[p] for p in points):
            continue
        seam_centers.append((int(r), int(c)))
        for p in points:
            union[p] = True
            if sequential:
                mask[p] = False
    return union, seam_centers, internal


def _assemble(info, chosen):
    a, bg = info['a'], info['bg']
    assembled = np.full_like(a, bg)
    assembled[info['band_mask']] = a[info['band_mask']]
    owners = np.full(a.shape, -1, dtype=int)
    owners[info['band_mask']] = 0
    for tab in chosen:
        for i, cells in tab['placements']:
            assembled[tuple(cells.T)] = info['pieces'][i][1]
            owners[tuple(cells.T)] = i + 1
    return assembled, owners


def _assemblies(grid, require_crosses=True, limit=2):
    """Search all full-width chains and a global exact cover of the components.

    There is no maximal-notch rule and no color-to-site dictionary. The family
    explicitly limits tabs to full-width column-convex layers. Distinct final
    colored outputs, not permutations of identical components, are counted.
    """
    info = _scene(grid)
    a, pieces = info['a'], info['pieces']
    height, width = a.shape
    candidates = []
    for sign in (-1, 1):
        descriptions = [_profile_description(p[0], sign) for p in pieces]
        lookup = {}
        for i, (lo, hi, local) in enumerate(descriptions):
            lookup.setdefault(lo, []).append(i)
        boundary = 1 - info['edges'][:, 0] if sign == -1 else info['edges'][:, 1]
        for span in sorted({len(d[0]) for d in descriptions}):
            for start in range(width - span + 1):
                initial = tuple(map(int, boundary[start:start + span]))

                def extend(profile, used, placements, occupied):
                    if used and len(set(profile)) == 1:
                        candidates.append(dict(bits=used, placements=placements,
                                               cells=occupied, side=sign, start=start,
                                               width=span, initial=initial, end=profile[0]))
                    offset = min(profile)
                    normalized = tuple(v - offset for v in profile)
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        lo, hi, local = descriptions[i]
                        target = np.column_stack((sign * (local[:, 0] + offset),
                                                  local[:, 1] + start))
                        if np.any(target[:, 0] < 0) or np.any(target[:, 0] >= height):
                            continue
                        if np.any(info['band_mask'][tuple(target.T)]):
                            continue
                        if set(map(tuple, target)) == set(map(tuple, pieces[i][0])):
                            continue
                        cells = frozenset(map(tuple, target))
                        if cells & occupied:
                            continue
                        extend(tuple(offset + v for v in hi), used | (1 << i),
                               placements + [(i, target)], occupied | cells)

                extend(initial, 0, [], frozenset())
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate['bits'] & (1 << i):
                by_piece[i].append(candidate)
    full = (1 << len(pieces)) - 1
    solutions = {}

    def cover(used, occupied, chosen):
        if len(solutions) >= limit:
            return
        if used == full:
            assembled, owners = _assemble(info, chosen)
            weld, centers, internal = _welds(assembled, info['ink'], owners)
            fixed_ink = info['band_mask'] & (a == info['ink'])
            if require_crosses and not np.all(weld[fixed_ink]):
                return
            answer = assembled.copy()
            if require_crosses:
                answer[weld] = info['band']
            solutions.setdefault(answer.tobytes(), dict(answer=answer, assembled=assembled,
                                  owners=owners, tabs=chosen, weld=weld,
                                  centers=centers, internal=internal))
            return
        options = [(i, [t for t in by_piece[i] if not t['bits'] & used
                        and not t['cells'] & occupied]) for i in range(len(pieces))
                   if not used & (1 << i)]
        _, available = min(options, key=lambda pair: len(pair[1]))
        for tab in available:
            cover(used | tab['bits'], occupied | tab['cells'], chosen + [tab])

    cover(0, frozenset(), [])
    return list(solutions.values()), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, uniquely assemble, and simultaneously highlight seam crosses."""
    grid = _grid(input_grid)
    answers, info = _assemblies(grid)
    if len(answers) != 1:
        raise ValueError("input must have exactly one cross-compatible colored answer")
    answer = answers[0]['answer']
    return (answer if info['axis'] == 0 else answer.T).copy()


def _random_profile(rng, width):
    amplitude = int(rng.integers(1, 4))
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(0, amplitude + 1, size=width)
    elif mode == 1:
        p = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.random() < .5:
            p = p[::-1]
    else:
        p = np.cumsum(rng.integers(-1, 2, size=width))
        p -= p.min()
        p = np.minimum(p, amplitude)
    p -= p.min()
    if not p.max():
        p[int(rng.integers(width))] = 1
    return p


def _palette(rng):
    bg = 0 if rng.random() < .5 else int(rng.integers(10))
    band, ink = map(int, rng.choice([c for c in range(10) if c != bg], 2, replace=False))
    payload = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, band, ink)])))
    return bg, band, ink, payload


def _scatter(rng, base, pieces, bg):
    h, w = base.shape
    for _ in range(30):
        result = base.copy()
        forbidden = _dilate(base != bg)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0]) + rng.random() * 10)
        success = True
        for i in order:
            cells, values = pieces[i]
            origin = cells.min(axis=0)
            local = cells - origin
            bh, bw = local.max(axis=0) + 1
            if bh > h or bw > w:
                success = False
                break
            valid = np.ones((h - bh + 1, w - bw + 1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r + h - bh + 1, c:c + w - bw + 1]
            if 0 <= origin[0] < valid.shape[0] and 0 <= origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            target = local + choices[int(rng.integers(len(choices)))]
            result[tuple(target.T)] = values
            placed = np.zeros(base.shape, dtype=bool)
            placed[tuple(target.T)] = True
            forbidden |= _dilate(placed)
        if success:
            return result
    raise ValueError("scatter clearance")


def _make_pair(rng, tabs, overlapping, sides, witness, palette, turn, join_witness=False):
    bg, band, ink, payload = palette
    common_width = int(rng.integers(6 if overlapping else 4, 9))
    widths = [common_width] * tabs
    if not witness and tabs > 1 and rng.random() < .25:
        widths = [int(rng.integers(6 if overlapping else 4, 10)) for _ in sides]
    roots, signatures = {}, [None] * tabs
    for width in set(widths):
        ids = [i for i, w in enumerate(widths) if w == width]
        pool = [(c, c + 1) for c in range(1, width - 2)] if overlapping else [(c,) for c in range(1, width - 1)]
        if len(pool) < len(ids):
            raise ValueError("not enough distinct docking signatures")
        selected = [pool[int(j)] for j in rng.permutation(len(pool))[:len(ids)]]
        p0 = np.zeros(width, dtype=int)
        p0[0], p0[-1] = map(int, rng.choice([1, 2, 3], 2, replace=False))
        protected = {c for signature in selected for c in signature}
        for c in range(1, width - 1):
            if c not in protected and rng.random() < .35:
                p0[c] = int(rng.integers(1, 3))
        roots[width] = p0
        for i, signature in zip(ids, selected):
            signatures[i] = signature
    counts = [2] * tabs if witness or join_witness else list(map(int, rng.integers(1, 4, size=tabs)))
    if sum(counts) > 7:
        raise ValueError("too many fragments for scattering")
    cuts = []
    for i, (width, count) in enumerate(zip(widths, counts)):
        sequence = [roots[width].copy()]
        if count == 1:
            sequence.append(np.full(width, max(6 if witness else 3,
                                                int(sequence[0].max()) + int(rng.integers(1, 4)))))
        else:
            first_depth = max(7 if witness else 0,
                              int(sequence[0].max()) + int(rng.integers(1, 4)))
            sequence.append(_random_profile(rng, width) + first_depth)
            for _ in range(count - 2):
                sequence.append(_random_profile(rng, width) + int(sequence[-1].max()) + int(rng.integers(1, 3)))
            sequence.append(np.full(width, int(sequence[-1].max()) + int(rng.integers(1, 4))))
        cuts.append(sequence)
    required_cross = max(16, max(sum(w for w, s in zip(widths, sides) if s == sign)
                                 + 2 * max(0, sides.count(sign) - 1) for sign in (-1, 1)))
    if required_cross > 30:
        raise ValueError("cross-axis clearance")
    width = int(rng.integers(required_cross, 31))
    starts, reserved = [], {-1: set(), 1: set()}
    for span, sign in zip(widths, sides):
        choices = [x for x in range(width - span + 1)
                   if not set(range(x - 1, x + span + 1)) & reserved[sign]]
        if not choices:
            raise ValueError("tab spacing")
        start = int(rng.choice(choices))
        starts.append(start)
        reserved[sign].update(range(start, start + span))
    edge = {sign: rng.integers(0, 3, size=width) for sign in (-1, 1)}
    for sign, start, span in zip(sides, starts, widths):
        edge[sign][start:start + span] = roots[span]
    core = int(rng.integers(3, 5))
    piece_depth = max(int(hi.max() - lo.min()) for sequence in cuts
                      for lo, hi in zip(sequence, sequence[1:]))
    extent = {sign: max(int(edge[sign].max()), piece_depth + 1,
                       max([int(seq[-1][0]) + 1 for s, seq in zip(sides, cuts) if s == sign] or [0]))
              for sign in (-1, 1)}
    minimum_height = extent[-1] + core + extent[1]
    if minimum_height > 30:
        raise ValueError("normal-axis clearance")
    height = int(rng.integers(max(18, minimum_height), 31))
    center = extent[-1] + int(rng.integers(height - minimum_height + 1))
    base = np.full((height, width), bg, dtype=np.int8)
    for c in range(width):
        base[center - edge[-1][c]:center + core + edge[1][c], c] = band

    def point(sign, start, c, u):
        return (center - 1 - u if sign == -1 else center + core + u, start + c)

    for sign, start, signature in zip(sides, starts, signatures):
        for c in signature:
            r, col = point(sign, start, c, -1)
            for p in ((r, col), (r - sign, col), (r, col - 1), (r, col + 1)):
                if base[p] not in (band, ink):
                    raise ValueError("fixed partial cross lies outside scaffold")
                base[p] = ink
    output, pieces = base.copy(), []
    join_carrier = int(rng.choice(payload)) if join_witness else None
    for tab_i, (sign, start, span, sequence, signature) in enumerate(zip(sides, starts, widths, cuts, signatures)):
        local_layers = [np.array([(u, c) for c in range(span)
                                  for u in range(int(lo[c]), int(hi[c]))], dtype=int)
                        for lo, hi in zip(sequence, sequence[1:])]
        layer_sets = [set(map(tuple, local)) for local in local_layers]
        join_ink = [set() for _ in local_layers]
        joined = False
        for j in range(len(local_layers) - 1):
            choices = []
            for c in range(1, span - 1):
                u = int(sequence[j + 1][c]) - 1
                upper_points = {(u, c), (u - 1, c), (u, c - 1), (u, c + 1)}
                if upper_points <= layer_sets[j] and (u + 1, c) in layer_sets[j + 1]:
                    choices.append((upper_points, (u + 1, c)))
            if choices and (join_witness or rng.random() < .6):
                upper_points, lower_point = choices[int(rng.integers(len(choices)))]
                join_ink[j].update(upper_points)
                join_ink[j + 1].add(lower_point)
                joined = True
        if join_witness and not joined:
            raise ValueError("no usable inter-fragment weld witness")
        for layer, (lo, hi) in enumerate(zip(sequence, sequence[1:])):
            local = local_layers[layer]
            carrier = (join_carrier if join_witness else
                       payload[len(pieces) % len(payload)] if rng.random() < .6 else int(rng.choice(payload)))
            values = np.full(len(local), carrier, dtype=np.int8)
            signal_cells = join_ink[layer].copy()
            if layer == 0:
                signal_cells.update((0, c) for c in signature)
                for c in signature:
                    if rng.random() < .75 or witness:
                        signal_cells.update((u, c) for u in range(1, min(int(hi[c]), int(rng.integers(2, 4)))))
                if witness or (hi.min() >= 6 and rng.random() < .45):
                    # A complete, connected interior cross is NOT a seam weld.
                    c = signature[0]
                    signal_cells.update((u, c) for u in range(5))
                    signal_cells.update(((3, c - 1), (3, c + 1)))
                    if rng.random() < .5:
                        signal_cells.add((4, c + 1))  # extra diagonal
            else:
                # Sparse ink camouflage away from the dock; accidental genuine
                # seam crosses are allowed and are detected on the final image.
                for _ in range(int(rng.integers(0, 5))):
                    signal_cells.add(tuple(local[int(rng.integers(len(local)))]))
            for j, p in enumerate(local):
                if tuple(p) in signal_cells:
                    values[j] = ink
            cells = np.array([point(sign, start, int(c), int(u)) for u, c in local])
            if np.any(output[tuple(cells.T)] != bg):
                raise ValueError("target overlap")
            output[tuple(cells.T)] = values
            pieces.append((cells, values))
    if np.count_nonzero(base != bg) + sum(len(p[0]) for p in pieces) >= base.size / 2:
        raise ValueError("background must be a visible majority")
    inp = _scatter(rng, base, pieces, bg)
    # Construct labels from the intended placement, then compare to unique
    # input-only search. Detection never reads the scattered camouflage first.
    owner = np.full(output.shape, -1, dtype=int)
    owner[base != bg] = 0
    for i, (cells, _) in enumerate(pieces):
        owner[tuple(cells.T)] = i + 1
    weld, _, _ = _welds(output, ink, owner)
    if not np.all(weld[(base != bg) & (base == ink)]):
        raise ValueError("uncompleted fixed ink")
    output[weld] = band
    inp, output = np.rot90(inp, turn).copy(), np.rot90(output, turn).copy()
    solutions, info = _assemblies(inp)
    canonical_output = output if info['axis'] == 0 else output.T
    if len(solutions) != 1 or not np.array_equal(solutions[0]['answer'], canonical_output):
        raise ValueError("ambiguous assembly or incorrect construction")
    return dict(input=inp, output=output)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # orientation transfer / more docks / mixed
    main_turn = int(rng.integers(4))
    dock_side = int(rng.choice((-1, 1)))
    stable = rng.random() < .2
    shared_palette = _palette(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train' and index == 0:
                tabs, overlap, sides, witness = 1, False, [-dock_side], False
            elif split == 'train' and index == 1:
                tabs, overlap, sides, witness = 2, True, [dock_side, dock_side], True
            else:
                tabs = 3 if split == 'test' and mode in (1, 2) else int(rng.integers(1, 4))
                overlap = bool(rng.integers(2))
                sides = list(map(int, rng.choice((-1, 1), tabs)))
                witness = False
            turn = main_turn
            if mode == 0 and split == 'test':
                turn = (main_turn + 1) % 4
            elif mode == 2:
                turn = (main_turn + (index % 2 if split == 'train' else int(rng.integers(2)))) % 4
            palette = shared_palette if stable else _palette(rng)
            for attempt in range(200):
                try:
                    pair = _make_pair(rng, tabs, overlap, sides, witness, palette, turn,
                                      join_witness=(split == 'train' and index == 0))
                except ValueError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _direct_weld_check(assembled, ink, owners):
    """A direct invariant check, not a second packing solver."""
    union = np.zeros(assembled.shape, dtype=bool)
    h, w = assembled.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            points = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if all(assembled[p] == ink for p in points) and len({int(owners[p]) for p in points}) > 1:
                for p in points:
                    union[p] = True
    return union


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    solutions, info = _assemblies(inp)
    assert len(solutions) == 1, "cross-compatible assembly must be unique"
    result = solutions[0]
    actual = out if info['axis'] == 0 else out.T
    assert np.array_equal(actual, result['answer']), "incorrect label"
    a, bg, band, ink = info['a'], info['bg'], info['band'], info['ink']
    weld = _direct_weld_check(result['assembled'], ink, result['owners'])
    assert np.array_equal(weld, result['weld'])
    assert np.all(weld[info['band_mask'] & (a == ink)])
    assert np.all(actual[weld] == band)
    assert np.array_equal(actual[~weld], result['assembled'][~weld])
    assert np.array_equal(actual[info['band_mask'] & (a == band)], a[info['band_mask'] & (a == band)])
    used = set()
    for tab in result['tabs']:
        for i, target in tab['placements']:
            assert i not in used
            used.add(i)
            source = info['pieces'][i][0]
            delta = target.min(axis=0) - source.min(axis=0)
            assert np.any(delta != 0)
            assert set(map(tuple, source + delta)) == set(map(tuple, target))
        u0, u1 = min(tab['initial']), tab['end']
        rows = [tab['side'] * u for u in range(u0, u1)]
        columns = range(tab['start'], tab['start'] + tab['width'])
        assert np.all(result['assembled'][np.ix_(rows, list(columns))] != bg), "tab has a hole"
    assert used == set(range(len(info['pieces'])))
    outside = result['owners'] == -1
    assert np.all(actual[outside] == bg), "source material was not erased"
    sequential, _, _ = _welds(result['assembled'], ink, result['owners'], sequential=True)
    fragment_join = any(0 not in {int(result['owners'][p]) for p in
                       [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]}
                        for r, c in result['centers'])
    same_carrier_join = False
    for r, c in result['centers']:
        identities = {int(result['owners'][p]) for p in
                      [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]}
        if 0 not in identities:
            same_carrier_join |= len({info['pieces'][i - 1][2] for i in identities}) == 1
    preserved_internal = False
    for r, c in _centers(result['assembled'] == ink):
        points = [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]
        preserved_internal |= (len({int(result['owners'][p]) for p in points}) == 1
                               and not any(weld[p] for p in points))
    return dict(info=info, result=result, overlap=not np.array_equal(weld, sequential),
                fragment_join=fragment_join, same_carrier_join=same_carrier_join,
                preserved_internal=preserved_internal)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            checked = _validate_pair(pair)
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            if split == 'train':
                training.append(checked)
    assert any(p['overlap'] for p in training), "training must teach simultaneous welds"
    assert any(p['preserved_internal'] for p in training), "training must preserve a whole interior cross"
    assert any(p['fragment_join'] for p in training), "training must teach inter-fragment welds"
    assert any(p['same_carrier_join'] for p in training), "training must distinguish source objects from colors"
    assert any(len(_assemblies(_grid(p['input']), require_crosses=False)[0]) > 1
               for p in puzzle['train']), "training must distinguish geometry-only routing"
    assert any(len(t['placements']) >= 2 for p in training for t in p['result']['tabs']), "training must teach interlocking"


def _fingerprint(puzzle):
    return tuple((_key(_grid(p['input'])), _key(_grid(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed diversity audit and task-specific adversarial checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, structures, widths, lengths, counts, palettes, orientations = set(), set(), set(), set(), set(), set(), set()
    opposite_origins = reused = internal = overlap = joins = same_carrier_joins = 0
    for puzzle in puzzles:
        local_shapes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                checked = _validate_pair(pair)
                info, result = checked['info'], checked['result']
                a = _grid(pair['input'])
                assert np.array_equal(solve(a), pair['output'])
                shapes.add(a.shape)
                local_shapes.add(a.shape)
                mapping = {}
                norm = tuple(mapping.setdefault(int(v), len(mapping)) for v in a.flat)
                structures.add((a.shape, norm))
                palettes.add((info['bg'], info['band'], info['ink']))
                orientations.add(info['axis'])
                counts.add(len(result['tabs']))
                reused += len({p[2] for p in info['pieces']}) < len(info['pieces'])
                internal += result['internal'] > 0
                overlap += checked['overlap']
                joins += checked['fragment_join']
                same_carrier_joins += checked['same_carrier_join']
                for tab in result['tabs']:
                    widths.add(tab['width'])
                    lengths.add(tab['end'] - min(tab['initial']))
                    for i, target in tab['placements']:
                        source = info['pieces'][i][0]
                        source_side = -1 if source[:, 0].max() < info['edges'][source[:, 1], 0].min() else 1
                        opposite_origins += source_side != tab['side']
        assert len(local_shapes) >= 2
    assert len(shapes) >= 45 and len(structures) >= 200
    assert widths >= {4, 5, 6, 7, 8} and len(lengths) >= 7
    assert counts == {1, 2, 3} and orientations == {0, 1}
    assert len(palettes) >= 50 and min(reused, internal, overlap, opposite_origins) >= 30
    # Two adjacent centers, different owners, simultaneous union 8 versus 5.
    block = np.full((3, 4), 7, dtype=np.int8)
    owners = np.zeros((3, 4), dtype=int)
    owners[2, :] = 1
    union, centers, _ = _welds(block, 7, owners)
    assert len(centers) == 2 and union.sum() == 8
    assert _welds(block, 7, owners, sequential=True)[0].sum() == 5
    assert not _welds(block, 7, np.zeros_like(owners))[0].any(), "internal crosses are not welds"
    clipped = np.array([[7, 7, 7], [0, 7, 0]], dtype=np.int8)
    assert not _welds(clipped, 7, np.zeros(clipped.shape, dtype=int))[0].any()
    extra = np.zeros((5, 6), dtype=np.int8)
    plus = {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
    for p in plus | {(1, 1), (2, 4)}:
        extra[p] = 7
    ownership = np.ones(extra.shape, dtype=int)
    ownership[:2, :] = 0
    marked = _welds(extra, 7, ownership)[0]
    assert set(map(tuple, np.argwhere(marked))) == plus, "diagonals and long arms must not change the motif"
    for bits in range(512):
        patch = np.array([(bits >> i) & 1 for i in range(9)], dtype=np.int8).reshape(3, 3) * 7
        ids = np.ones((3, 3), dtype=int)
        ids[0, :] = 0
        assert np.array_equal(_welds(patch, 7, ids)[0], _direct_weld_check(patch, 7, ids))
    # Labels that preserve all ink, recolor all ink, or leave sources must fail.
    for mutation in ('unchanged', 'all_ink', 'one_cell'):
        bad = {s: [{k: _grid(v) for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
        pair = (next(p for p in bad['train'] if _validate_pair(p)['preserved_internal'])
                if mutation == 'all_ink' else bad['test'][0])
        if mutation == 'unchanged':
            pair['output'] = pair['input'].copy()
        elif mutation == 'all_ink':
            info = _scene(pair['input'])
            pair['output'][pair['output'] == info['ink']] = info['band']
        else:
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted a wrong label")
    # An actual geometry-only packing that leaves a scaffold clue uncompleted
    # must fail even after its existing seam crosses have been highlighted.
    counterexample = None
    for index, pair in enumerate(puzzles[0]['train']):
        candidates, info = _assemblies(pair['input'], require_crosses=False)
        for candidate in candidates:
            fixed_ink = info['band_mask'] & (info['a'] == info['ink'])
            if not np.all(candidate['weld'][fixed_ink]):
                wrong = candidate['assembled'].copy()
                wrong[candidate['weld']] = info['band']
                counterexample = (index, wrong if info['axis'] == 0 else wrong.T)
                break
        if counterexample is not None:
            break
    assert counterexample is not None, "missing parent-B routing ablation"
    bad = {s: [{k: _grid(v) for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    bad['train'][counterexample[0]]['output'] = counterexample[1]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("geometry-only misrouting was accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(73 + a * 5 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), structures=len(structures),
                tab_widths=sorted(widths), tab_lengths=sorted(lengths),
                palettes=len(palettes), overlap_pairs=overlap, internal_cross_pairs=internal,
                reused_carrier_pairs=reused, opposite_origins=int(opposite_origins),
                inter_fragment_weld_pairs=joins, same_carrier_weld_pairs=same_carrier_joins)
