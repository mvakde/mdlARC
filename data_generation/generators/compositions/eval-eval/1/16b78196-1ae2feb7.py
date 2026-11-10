"""Echo-locked docking: complementary tiles and observed, routed pulse trains.

Standard library and NumPy only. No parent code, examples, or latent solving state.
The constructor uses proposed cuts; the solver searches the actual input masks.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, grid=None):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and (grid is None or grid[p] == grid[first]):
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _dilate(mask):
    out = mask.copy()
    for dr, dc in itertools.product((-1, 0, 1), repeat=2):
        a, b = max(0, -dr), min(mask.shape[0], mask.shape[0]-dr)
        c, d = max(0, -dc), min(mask.shape[1], mask.shape[1]-dc)
        out[a+dr:b+dr, c+dc:d+dc] |= mask[a:b, c:d]
    return out


def _restore(a, scene):
    if scene['flip']:
        a = a[:, ::-1]
    return (a.T if scene['transpose'] else a).copy()


def _scene(grid):
    """Recover the bus, detached components, and ordered receiver paths."""
    a = _grid(grid)
    counts = np.bincount(a.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(counts.argmax())
    comps = _components(a != bg, a)
    bands = []
    for cells in comps:
        for transpose in (False, True):
            x = cells[:, ::-1] if transpose else cells
            h = a.shape[1] if transpose else a.shape[0]
            if len(np.unique(x[:, 0])) == h:
                bands.append((int(a[tuple(cells[0])]), transpose))
    if len(bands) != 1:
        raise ValueError("expected exactly one monochrome spanning bus")
    bus, transpose = bands[0]
    canonical = a.T if transpose else a
    interpretations = []
    for flip in (False, True):
        x = canonical[:, ::-1] if flip else canonical
        spans = [np.flatnonzero(row == bus) for row in x]
        if any(not len(v) or not np.array_equal(v, np.arange(v[0], v[-1]+1)) for v in spans):
            continue
        right = [int(v[-1])+1 for v in spans]
        if len(set(right)) != 1:
            continue
        edge = right[0]
        if not 0 < edge < x.shape[1]:
            continue
        # All cells on the receiver side belong to induced simple paths.
        path_mask = x != bg
        path_mask[:, :edge] = False
        paths = []
        valid = True
        for cells in _components(path_mask):
            cellset = set(map(tuple, cells))
            roots = [p for p in cellset if p[1] == edge]
            neighbors = {p: [q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                                         (p[0], p[1]-1), (p[0], p[1]+1)) if q in cellset]
                         for p in cellset}
            if (len(roots) != 1 or len(cells) < 3
                    or len(neighbors[roots[0]]) != 1
                    or sorted(map(len, neighbors.values())).count(1) != 2
                    or any(len(v) not in (1, 2) for v in neighbors.values())):
                valid = False
                break
            path, previous, current = [], None, roots[0]
            while current is not None:
                path.append(current)
                nxt = [q for q in neighbors[current] if q != previous]
                previous, current = current, nxt[0] if nxt else None
            paths.append(tuple(path))
        if not valid or not paths:
            continue
        piece_mask = (x != bg) & (x != bus)
        piece_mask[:, edge:] = False
        pieces = _components(piece_mask, x)
        if not pieces:
            continue
        # Loose tiles are detached even when the same literal color is reused.
        forbidden = _dilate(x == bus)
        occupied = np.zeros(x.shape, dtype=bool)
        for cells in pieces:
            if (np.any(forbidden[cells[:, 0], cells[:, 1]])
                    or np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]])):
                valid = False
                break
            occupied[cells[:, 0], cells[:, 1]] = True
        if not valid:
            continue
        colors = {int(x[tuple(cells[0])]) for cells in pieces}
        path_colors = {int(x[p]) for path in paths for p in path}
        markers = path_colors - colors - {bg, bus}
        if len(markers) != 1:
            continue
        carrier = markers.pop()
        if any(carrier not in [int(x[p]) for p in path] for path in paths):
            continue
        # A solid bus core is its intersection across all rows.
        left = np.array([int(v[0]) for v in spans])
        core_start = int(left.max())
        if not 2 <= edge-core_start <= 4:
            continue
        interpretations.append({'grid': x.copy(), 'bg': bg, 'bus': bus,
                                'carrier': carrier, 'transpose': transpose, 'flip': flip,
                                'core_start': core_start, 'edge': edge,
                                'boundary': core_start-left,
                                'paths': sorted(paths),
                                'pieces': [(v, int(x[tuple(v[0])])) for v in pieces]})
    if len(interpretations) != 1:
        raise ValueError("ambiguous bus side or receiver geometry")
    return interpretations[0]


def _description(cells):
    rows = cells[:, 0]-cells[:, 0].min()
    outward = -cells[:, 1]
    lo, hi = [], []
    for row in range(int(rows.max())+1):
        values = np.sort(outward[rows == row])
        if not len(values) or not np.array_equal(values, np.arange(values[0], values[-1]+1)):
            raise ValueError("tiles must have a nonempty contiguous interval in each row")
        lo.append(int(values[0]))
        hi.append(int(values[-1])+1)
    offset = min(lo)
    return (tuple(v-offset for v in lo), tuple(v-offset for v in hi),
            np.column_stack((rows, outward-offset)))


def _runs(placements, pieces, root_row):
    """Read maximal color runs in the assembled transect, nearest first."""
    runs = []
    for i, cells in placements:
        length = int(np.count_nonzero(cells[:, 0] == root_row))
        color = pieces[i][1]
        if runs and runs[-1][0] == color:
            runs[-1] = (color, runs[-1][1]+length)
        else:
            runs.append((color, length))
    return tuple(runs)


def _wave(runs, length, carrier, priority='near', phase=0, period_delta=0):
    values = []
    for t in range(length):
        hit = [(i, color, size) for i, (color, size) in enumerate(runs)
               if (t-phase) % max(1, size+period_delta) == 0]
        if not hit:
            values.append(carrier)
        else:
            if priority == 'far':
                chosen = hit[-1]
            elif priority == 'short':
                chosen = min(hit, key=lambda v: (v[2], v[0]))
            elif priority == 'long':
                chosen = min(hit, key=lambda v: (-v[2], v[0]))
            else:
                chosen = hit[0]
            values.append(chosen[1])
    return tuple(values)


def _search(scene, respect_hints=True, limit=2):
    """All alignments and chain subsets, then a global tile/receiver exact cover.

    Positive observations filter each candidate before global assembly. We stop
    only once `limit` distinct complete colored answers have been found.
    """
    x, pieces, paths = scene['grid'], scene['pieces'], scene['paths']
    h, w = x.shape
    s, bg, bus, carrier = (scene[k] for k in ('core_start', 'bg', 'bus', 'carrier'))
    descriptions = [_description(cells) for cells, _ in pieces]
    lookup = {}
    for i, (lo, hi, local) in enumerate(descriptions):
        lookup.setdefault(lo, []).append(i)
    candidates = []
    for height in sorted({len(d[0]) for d in descriptions}):
        for start in range(h-height+1):
            eligible = [j for j, p in enumerate(paths) if start <= p[0][0] < start+height]
            if len(eligible) != 1:
                continue
            j = eligible[0]
            initial = tuple(map(int, scene['boundary'][start:start+height]))

            def extend(profile, used, placements):
                if used and len(set(profile)) == 1:
                    runs = _runs(placements, pieces, paths[j][0][0])
                    wave = _wave(runs, len(paths[j]), carrier)
                    if not respect_hints or all(x[p] == carrier or x[p] == value
                                                for p, value in zip(paths[j], wave)):
                        cells = frozenset((int(r), int(c)) for _, a in placements for r, c in a)
                        candidates.append({'used': used, 'path': j, 'placements': placements,
                                           'cells': cells, 'runs': runs, 'wave': wave,
                                           'start': start, 'height': height, 'depth': profile[0]})
                offset = min(profile)
                norm = tuple(v-offset for v in profile)
                for i in lookup.get(norm, ()):
                    if used & (1 << i):
                        continue
                    lo, hi, local = descriptions[i]
                    nxt = tuple(v+offset for v in hi)
                    if max(nxt) > s:
                        continue
                    cells = np.column_stack((local[:, 0]+start, s-1-local[:, 1]-offset))
                    if np.any(x[cells[:, 0], cells[:, 1]] == bus):
                        continue
                    extend(nxt, used | (1 << i), placements+[(i, cells)])

            extend(initial, 0, [])
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate['used'] & (1 << i):
                by_piece[i].append(candidate)
    solutions = {}
    full_pieces, full_paths = (1 << len(pieces))-1, (1 << len(paths))-1
    base = np.full_like(x, bg)
    base[x == bus] = bus
    for path in paths:
        for p in path:
            base[p] = carrier

    def cover(used, path_bits, occupied, chosen):
        if len(solutions) >= limit:
            return
        if used == full_pieces:
            if path_bits != full_paths:
                return
            answer = base.copy()
            for c in chosen:
                for i, cells in c['placements']:
                    answer[cells[:, 0], cells[:, 1]] = pieces[i][1]
                for p, color in zip(paths[c['path']], c['wave']):
                    answer[p] = color
            solutions.setdefault(answer.tobytes(), (answer, chosen))
            return
        available = []
        for i in range(len(pieces)):
            if not used & (1 << i):
                options = [c for c in by_piece[i] if not c['used'] & used
                           and not path_bits & (1 << c['path']) and not occupied & c['cells']]
                available.append((i, options))
        _, options = min(available, key=lambda v: len(v[1]))
        for c in options:
            cover(used | c['used'], path_bits | (1 << c['path']), occupied | c['cells'], chosen+[c])

    cover(0, 0, frozenset(), [])
    return list(solutions.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _scene(input_grid)
    solutions = _search(scene)
    if len(solutions) != 1:
        raise ValueError("pulse observations must select one complete colored assembly")
    return _restore(solutions[0][0], scene)


def _profile(rng, height):
    # Walks, slopes, plateaus, and teeth, with adjacent differences at most one.
    for _ in range(20):
        mode = int(rng.integers(3))
        if mode == 0:
            q = np.clip(np.cumsum(rng.integers(-1, 2, height)), -2, 2)
        elif mode == 1:
            q = np.floor(np.linspace(0, int(rng.integers(1, 4)), height)).astype(int)
            if rng.random() < .5:
                q = q[::-1]
        else:
            q = np.zeros(height, dtype=int)
            at = int(rng.integers(1, height))
            q[at:] = 1
            if at+1 < height and rng.random() < .5:
                q[at+1:] = 0
        q = q-q.min()
        if q.max() and np.max(np.abs(np.diff(q))) <= 1:
            return q
    return np.array([0, 1]+[0]*(height-2))


def _path(rng, row, edge, width, low, high, min_length, folded=False):
    """A monotone horizontal route with optional vertical detours; no chords."""
    for _ in range(40):
        if folded:
            # A return stroke is a test extension of the same path clock. The
            # parallel strokes are two rows apart, so they have no graph chords.
            sign = 1 if high-row >= row-low else -1
            room = high-row if sign == 1 else row-low
            if room < 2:
                raise ValueError("no room for a routed return stroke")
            far = int(rng.integers(max(edge+4, width-3), width))
            near = int(rng.integers(edge+2, far-1))
            cells = [(row, c) for c in range(edge, far+1)]
            cells += [(row+sign, far), (row+2*sign, far)]
            cells += [(row+2*sign, c) for c in range(far-1, near-1, -1)]
            if room >= 4 and rng.random() < .5:
                cells += [(row+3*sign, near), (row+4*sign, near)]
                end = int(rng.integers(near+2, width))
                cells += [(row+4*sign, c) for c in range(near+1, end+1)]
            if len(cells) >= min_length:
                return tuple(cells)
            continue
        cells = [(row, edge)]
        r, c = row, edge
        while c < width-1:
            steps = min(int(rng.integers(2, 5)), width-1-c)
            for _ in range(steps):
                c += 1
                cells.append((r, c))
            if c < width-1 and rng.random() < .85:
                choices = [v for v in range(low, high+1) if v != r and abs(v-r) <= 3]
                if choices:
                    target = int(rng.choice(choices))
                    sign = 1 if target > r else -1
                    while r != target:
                        r += sign
                        cells.append((r, c))
        if len(cells) >= min_length:
            return tuple(cells)
    raise ValueError("receiver too short for the required observations")


def _scatter(rng, base, tiles, bg, s):
    for _ in range(24):
        inp = base.copy()
        forbidden = _dilate(base != bg)
        order = sorted(range(len(tiles)), key=lambda i: -len(tiles[i][0])+rng.random()*5)
        success = True
        for i in order:
            target, color = tiles[i]
            shape = target-target.min(axis=0)
            height, width = shape.max(axis=0)+1
            valid = np.ones((base.shape[0]-height+1, s-width+1), dtype=bool)
            for r, c in shape:
                valid &= ~forbidden[r:r+valid.shape[0], c:c+valid.shape[1]]
            origin = target.min(axis=0)
            if origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            locations = np.argwhere(valid)
            if not len(locations):
                success = False
                break
            placed = shape+locations[int(rng.integers(len(locations)))]
            inp[placed[:, 0], placed[:, 1]] = color
            forbidden = _dilate(inp != bg)
        if success:
            return inp
    raise ValueError("insufficient tile scatter clearance")


def _build_pair(rng, palette, witness=None, shift=False):
    bg, bus, carrier, payload = palette
    tabs = int(rng.choice([1, 2], p=[.65, .35] if not shift else [.35, .65]))
    heights = [int(rng.integers(3, 7)) for _ in range(tabs)]
    h = int(rng.integers(max(17, sum(heights)+6), 31))
    s = int(rng.integers(13, 18))
    thickness = int(rng.integers(2, 4))
    w = int(rng.integers(max(s+thickness+7, 25), 31))
    if s+thickness+6 >= w:
        raise ValueError("no receiver room")
    starts = []
    # Disjoint row territories also keep routed receivers separate.
    if tabs == 1:
        starts = [int(rng.integers(1, h-heights[0]))]
    else:
        split = int(rng.integers(heights[0]+2, h-heights[1]-1))
        starts = [int(rng.integers(1, split-heights[0]+1)),
                  int(rng.integers(split+1, h-heights[1]))]
    q = np.zeros(h, dtype=int)
    cuts_by_tab, colors_by_tab, roots = [], [], []
    for j, height in enumerate(heights):
        profile = _profile(rng, height)
        start = starts[j]
        q[start:start+height] = profile
        special = witness is not None and j == 0
        if special:
            lengths, colors = witness
            cuts = [profile.copy(), profile+lengths[0], profile+sum(lengths)]
            colors = list(colors)+[next(c for c in payload if c not in colors)]
        else:
            keyed = rng.random() < .35
            count = int(rng.choice([2, 3], p=[.7, .3]))
            cuts = [profile.copy()]
            if keyed:
                next_shape = _profile(rng, height)
                offset = max(int(np.max(profile+1-next_shape)),
                             int(np.max(np.maximum(profile[:-1], profile[1:])+1-
                                        np.minimum(next_shape[:-1], next_shape[1:]))))
                cuts.append(next_shape+offset)
                count = 2  # key + two interchangeable layers + cap
            for _ in range(count):
                cuts.append(cuts[-1]+int(rng.integers(2, 5)))
            colors = list(map(int, rng.choice(payload, len(cuts), replace=False)))
            if len(colors) >= 3 and rng.random() < .25:
                colors[-1] = colors[0]  # nonadjacent repeated color
        end = int(cuts[-1].max())+int(rng.integers(1, 4))
        if end > s-1:
            raise ValueError("transmitter does not fit")
        cuts.append(np.full(height, end))
        root = start+int(rng.integers(height))
        cuts_by_tab.append(cuts)
        colors_by_tab.append(colors)
        roots.append(root)
    # Unused bus detail is permitted; receiver roots identify eligible tabs.
    unused = [r for r in range(h) if q[r] == 0 and all(not a-1 <= r <= a+b for a, b in zip(starts, heights))]
    if unused and rng.random() < .7:
        q[int(rng.choice(unused))] = 1
    base = np.full((h, w), bg, dtype=np.int8)
    for r in range(h):
        base[r, s-int(q[r]):s+thickness] = bus
    paths = []
    for j, root in enumerate(roots):
        low = 0 if j == 0 else (starts[j-1]+heights[j-1]+starts[j])//2+1
        high = h-1 if j == tabs-1 else (starts[j]+heights[j]+starts[j+1])//2-1
        required = math.lcm(*witness[0])+3 if witness is not None and j == 0 else 10
        path = _path(rng, root, s+thickness, w, low, high, required, folded=shift)
        paths.append(path)
        for p in path:
            base[p] = carrier
    intended = base.copy()
    tiles = []
    intended_waves = []
    for start, cuts, colors, path in zip(starts, cuts_by_tab, colors_by_tab, paths):
        runs = []
        root_row = path[0][0]-start
        for lo, hi, color in zip(cuts, cuts[1:], colors):
            cells = np.array([(start+r, s-1-d) for r in range(len(lo))
                              for d in range(int(lo[r]), int(hi[r]))])
            tiles.append((cells, color))
            intended[cells[:, 0], cells[:, 1]] = color
            length = int(hi[root_row]-lo[root_row])
            if runs and runs[-1][0] == color:
                runs[-1] = (color, runs[-1][1]+length)
            else:
                runs.append((color, length))
        # Construction uses slice-like overwrites, separately from _wave.
        wave = [carrier]*len(path)
        for color, period in reversed(runs):
            for t in range(0, len(path), period):
                wave[t] = color
        intended_waves.append(wave)
        for p, color in zip(path, wave):
            intended[p] = color
    if witness is not None:
        # Reject lessons whose output cannot show either of the added concepts.
        path, wave = paths[0], intended_waves[0]
        runs = [(color, int(hi[path[0][0]-starts[0]]-lo[path[0][0]-starts[0]]))
                for lo, hi, color in zip(cuts_by_tab[0], cuts_by_tab[0][1:], colors_by_tab[0])]
        distances = [abs(r-path[0][0])+abs(c-path[0][1]) for r, c in path]
        manhattan = [next((color for color, period in runs if d % period == 0), carrier)
                     for d in distances]
        bbox_runs = [(color, int(hi.max()-lo.min()))
                     for lo, hi, color in zip(cuts_by_tab[0], cuts_by_tab[0][1:], colors_by_tab[0])]
        if wave == manhattan or tuple(wave) == _wave(bbox_runs, len(path), carrier):
            raise ValueError("lesson hides path distance or slice length")
    inp = _scatter(rng, base, tiles, bg, s)
    scene = _scene(inp)
    alternatives = _search(scene, respect_hints=False, limit=129)
    if not 2 <= len(alternatives) < 129:
        raise ValueError("need a finite observable geometric ambiguity")
    if not any(np.array_equal(a, intended) for a, _ in alternatives):
        raise ValueError("construction disagrees with geometric search")
    remaining = [a for a, _ in alternatives if not np.array_equal(a, intended)]
    clues = []
    options = [p for path, wave in zip(paths, intended_waves) for t, (p, color) in enumerate(zip(path, wave))
               if t > 0 and color != carrier]
    # Positive observations must eliminate every different colored completion.
    while remaining:
        scores = [sum(a[p] != intended[p] for a in remaining) for p in options]
        best = max(scores, default=0)
        if not best:
            raise ValueError("available positive echoes cannot resolve the assembly")
        choices = [p for p, score in zip(options, scores) if score == best]
        p = choices[int(rng.integers(len(choices)))]
        clues.append(p)
        remaining = [a for a in remaining if a[p] == intended[p]]
        options.remove(p)
    # Redundant observations need not all encode the closest tile's color.
    if options:
        count = min(len(options), int(rng.integers(1, 4)))
        clues += [options[int(i)] for i in rng.choice(len(options), count, replace=False)]
    for p in clues:
        inp[p] = intended[p]
    if sum(inp[p] == carrier and intended[p] != carrier for path in paths for p in path) < 3:
        raise ValueError("too few pulses left to infer")
    # Both counts and the complete raw input must make the background unique.
    scene = _scene(inp)
    resolved = _search(scene)
    if len(resolved) != 1 or not np.array_equal(resolved[0][0], intended):
        raise ValueError("observations did not select exactly the intended answer")
    return {'input': inp, 'output': intended}


def _palette(rng, witness_colors, stable=None):
    if stable is not None:
        return stable
    other = [c for c in range(10) if c not in witness_colors]
    bg, bus, carrier = map(int, rng.choice(other, 3, replace=False))
    payload = list(witness_colors)+list(map(int, rng.permutation([c for c in other if c not in (bg, bus, carrier)])))
    return bg, bus, carrier, tuple(payload)


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    witness_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    periods = [(2, 3), (2, 5), (3, 4)][int(rng.integers(3))]
    stable = _palette(rng, witness_colors) if rng.random() < .30 else None
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    witness_map = {slots[0]: (periods, witness_colors),
                   slots[1]: (periods[::-1], witness_colors[::-1])}
    base_turn = int(rng.integers(4))
    # Controlled axis transfer, mixed training, or stable axes. Test routes also
    # have more docking groups on average; no rule parameter changes.
    mode = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = witness_map.get(index) if split == 'train' else None
            palette = _palette(rng, witness_colors, stable)
            for attempt in range(180):
                try:
                    pair = _build_pair(rng, palette, witness, split == 'test')
                except ValueError:
                    continue
                turn = ((base_turn+int(rng.integers(4))) % 4 if mode == 1 else
                        (base_turn+int(split == 'test')) % 4 if mode == 0 else base_turn)
                if rng.random() < .5:
                    pair = {k: a[::-1, :].copy() for k, a in pair.items()}
                pair = {k: np.rot90(a, turn).copy() for k, a in pair.items()}
                key = pair['input'].shape, pair['input'].tobytes()
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction could not find an observable docking scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(pair):
    scene = _scene(pair['input'])
    solutions = _search(scene)
    if len(solutions) != 1:
        raise AssertionError("nonunique echo-locked assembly")
    answer, tabs = solutions[0]
    actual = np.asarray(pair['output'])
    assert np.array_equal(actual, _restore(answer, scene)), "wrong completion"
    edges, short_near, long_near, seen_periods = set(), False, False, set()
    refuted = set()
    for tab in tabs:
        path, runs = scene['paths'][tab['path']], tab['runs']
        for name, kwargs in [('short', {'priority': 'short'}), ('long', {'priority': 'long'}),
                             ('far', {'priority': 'far'}), ('phase_one', {'phase': 1}),
                             ('period_plus_one', {'period_delta': 1}),
                             ('period_minus_one', {'period_delta': -1})]:
            if _wave(runs, len(path), scene['carrier'], **kwargs) != tab['wave']:
                refuted.add(name)
        for name, distances in [('manhattan', [abs(r-path[0][0])+abs(c-path[0][1]) for r, c in path]),
                                ('horizontal', [abs(c-path[0][1]) for r, c in path])]:
            rival = tuple(next((color for color, size in runs if d % size == 0), scene['carrier'])
                          for d in distances)
            if rival != tab['wave']:
                refuted.add(name)
        bounding_runs = []
        for i, cells in tab['placements']:
            color = scene['pieces'][i][1]
            depth = int(cells[:, 1].max()-cells[:, 1].min()+1)
            if bounding_runs and bounding_runs[-1][0] == color:
                bounding_runs[-1] = (color, bounding_runs[-1][1]+depth)
            else:
                bounding_runs.append((color, depth))
        if _wave(bounding_runs, len(path), scene['carrier']) != tab['wave']:
            refuted.add('bounding_box')
        for i, (color, period) in enumerate(runs):
            if any(t > 0 and t % period == 0 and tab['wave'][t] == color for t in range(len(path))):
                seen_periods.add(period)
            for far_color, far_period in runs[i+1:]:
                if color == far_color:
                    continue
                collision = any(t > 0 and t % period == t % far_period == 0 and tab['wave'][t] == color
                                for t in range(len(path)))
                survives = any(t % far_period == 0 and tab['wave'][t] == far_color
                               for t in range(len(path)))
                if collision and survives:
                    short_near |= period < far_period
                    long_near |= period > far_period
                    edges.add((color, far_color))
    return scene, tabs, {'short_near': short_near, 'long_near': long_near,
                         'edges': edges, 'periods': seen_periods, 'refuted': refuted}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            scene, tabs, lesson = _evidence(pair)
            canonical_out = out.T if scene['transpose'] else out
            if scene['flip']:
                canonical_out = canonical_out[:, ::-1]
            assert np.array_equal(scene['grid'] == scene['bus'], canonical_out == scene['bus'])
            assert len(_search(scene, respect_hints=False)) == 2, "no meaningful echo disambiguation"
            used = set()
            for tab in tabs:
                start, height, depth = (tab[k] for k in ('start', 'height', 'depth'))
                rect = canonical_out[start:start+height, scene['core_start']-depth:scene['core_start']]
                assert np.all(rect != scene['bg']), "tab has a hole"
                for i, target in tab['placements']:
                    assert i not in used
                    used.add(i)
                    source, color = scene['pieces'][i]
                    translated = source+target.min(axis=0)-source.min(axis=0)
                    assert set(map(tuple, translated)) == set(map(tuple, target)), "tile mask changed"
                    assert np.all(canonical_out[target[:, 0], target[:, 1]] == color)
                path = scene['paths'][tab['path']]
                for t, p in enumerate(path):
                    firing = [(i, color) for i, (color, size) in enumerate(tab['runs']) if t % size == 0]
                    expected = firing[0][1] if firing else scene['carrier']
                    assert canonical_out[p] == expected, "path phase or priority wrong"
                    assert scene['grid'][p] in (scene['carrier'], expected), "observation overwritten"
            assert used == set(range(len(scene['pieces'])))
            for path in scene['paths']:
                backtracks = any(b[1] < a[1] for a, b in zip(path, path[1:]))
                assert backtracks == (split == 'test'), "route shift missing"
            if split == 'train':
                teaching.append(lesson)
    assert any(e['short_near'] for e in teaching) and any(e['long_near'] for e in teaching)
    edges = set().union(*(e['edges'] for e in teaching))
    assert any((b, a) in edges for a, b in edges), "fixed color ranking not refuted"
    assert len(set().union(*(e['periods'] for e in teaching))) >= 2, "insufficient recurrence"
    required = {'short', 'long', 'far', 'phase_one', 'period_plus_one', 'period_minus_one',
                'manhattan', 'horizontal', 'bounding_box'}
    assert required <= set().union(*(e['refuted'] for e in teaching)), "a simpler rival fits training"


def _key(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k]).tobytes()) for s in ('train', 'test')
                 for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    # Fixed arithmetic diagnostics, not expectations generated by solve.
    assert _wave(((2, 2), (3, 3)), 13, 8) == tuple(map(int, '2823282823282'))
    assert _wave(((3, 3), (2, 2)), 13, 8) == tuple(map(int, '3823283823283'))
    assert _wave(((2, 4), (3, 2)), 9, 8) == tuple(map(int, '283828382'))
    # Hand-built joint fixture: q=(0,1,0), interchangeable depths 2 and 3,
    # and a cap ending at depth 7. A delayed collision observation selects order.
    fixture = np.zeros((15, 22), dtype=np.int8)
    fixture[:, 10:12] = 9
    fixture[5, 9] = 9
    fixture[0:3, 0:3] = np.array([[0, 2, 2], [2, 2, 0], [0, 2, 2]])
    fixture[8:11, 0:4] = np.array([[0, 3, 3, 3], [3, 3, 3, 0], [0, 3, 3, 3]])
    fixture[11:14, 5:7] = np.array([[4, 4], [4, 0], [4, 4]])
    fixture[4, 12:21] = 8
    fixture[4, 18] = 2
    expected = np.zeros_like(fixture)
    expected[:, 10:12] = 9
    expected[5, 9] = 9
    expected[4, 3:10] = list(map(int, '4433322'))
    expected[5, 3:10] = list(map(int, '4333229'))
    expected[6, 3:10] = list(map(int, '4433322'))
    expected[4, 12:21] = list(map(int, '282328282'))
    assert np.array_equal(solve(fixture), expected)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turn)), np.rot90(expected, turn))
    blank_echo = fixture.copy()
    blank_echo[4, 18] = 8
    try:
        solve(blank_echo)
    except ValueError:
        pass
    else:
        raise AssertionError("shape-only ambiguity was silently resolved")
    changed_echo = fixture.copy()
    changed_echo[4, 18] = 3
    swapped = expected.copy()
    swapped[4, 3:10] = swapped[6, 3:10] = list(map(int, '4422333'))
    swapped[5, 3:10] = list(map(int, '4223339'))
    swapped[4, 12:21] = list(map(int, '382328382'))
    assert np.array_equal(solve(changed_echo), swapped), "echo must change docking order"
    merged = fixture.copy()
    merged[merged == 3] = 2
    merged[4, 18] = 4
    merged_expected = expected.copy()
    merged_expected[merged_expected == 3] = 2
    merged_expected[4, 12:21] = list(map(int, '284842484'))
    assert np.array_equal(solve(merged), merged_expected), "touching colors must form period 5"
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, structures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene, tabs, _ = _evidence(pair)
                bends = 0
                for path in scene['paths']:
                    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
                    bends += sum(a != b for a, b in zip(steps, steps[1:]))
                features.append((pair['input'].shape, scene['transpose'], scene['flip'], len(tabs),
                                 tuple(len(c['placements']) for c in tabs),
                                 tuple(c['height'] for c in tabs), tuple(c['depth'] for c in tabs),
                                 tuple(len(p) for p in scene['paths']), bends,
                                 (scene['bg'], scene['bus'], scene['carrier'])))
                roles = {}
                structures.add((pair['input'].shape, tuple(roles.setdefault(int(v), len(roles))
                                                          for v in pair['input'].flat)))
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len({_key(p) for p in puzzles}) == 42
    assert len(structures) >= 200
    assert len({f[0] for f in features}) >= 30
    assert {f[1] for f in features} == {False, True}
    assert {f[2] for f in features} == {False, True}
    assert {f[3] for f in features} == {1, 2}
    assert {v for f in features for v in f[4]} >= {3, 4}
    assert len({v for f in features for v in f[5]}) >= 4
    assert len({v for f in features for v in f[6]}) >= 6
    assert len({v for f in features for v in f[7]}) >= 10
    assert sum(f[8] > 0 for f in features) >= 180
    assert len({f[9] for f in features}) >= 20
    assert _key(puzzles[0]) == _key(generate_puzzle(0))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(91000+10*train+test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Metamorphic geometry/palette checks must preserve the recovered answer.
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
        mapping = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8], dtype=np.int8)
        assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    corrupted = {s: [{k: a.copy() for k, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, arithmetic and geometric checks')
