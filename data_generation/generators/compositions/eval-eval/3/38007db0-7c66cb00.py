"""Recover overwritten material from exceptional copies of a tiled template.

Comparison creates a virtual source: mismatching cells, labeled by their OLD
(template) colors. Its monochromatic components move independently into the
matching reservoirs. Source tiles are repaired, rather than erased.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), "2D integer grid required")
    _require(all(1 <= n <= 30 for n in a.shape), "grid dimensions outside 1..30")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(labels, points, diagonal=False):
    """Deterministic monochromatic flood fill over ONLY the supplied cells."""
    remaining = set(points)
    groups = []
    neighbors = EIGHT if diagonal else FOUR
    while remaining:
        start = min(remaining)
        color = int(labels[start])
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining and int(labels[p]) == color:
                    remaining.remove(p)
                    stack.append(p)
        groups.append((color, tuple(sorted(cells))))
    return groups


def _uneven_mask(cells):
    bottoms = {}
    for r, c in cells:
        bottoms[c] = max(r, bottoms.get(c, r))
    return len(set(bottoms.values())) > 1


def _analyze(input_grid):
    """Parse all roles, panels, reservoirs and virtual pieces from pixels."""
    g = _grid(input_grid)
    h, w = g.shape
    lattice = int(g[0, 0])
    rs = np.flatnonzero(np.all(g == lattice, axis=1)).tolist()
    _require(len(rs) >= 2 and rs[0] == 0, "source needs complete lattice rows")
    ph = rs[1] - 1
    _require(ph >= 3 and rs == list(range(0, rs[-1] + 1, ph + 1)), "irregular source rows")
    end = rs[-1]
    source = g[:end + 1]
    cs = np.flatnonzero(np.all(source == lattice, axis=0)).tolist()
    _require(len(cs) >= 4 and cs[0] == 0 and cs[-1] == w - 1, "at least three equal source copies required")
    pw = cs[1] - 1
    _require(pw >= 3 and cs == list(range(0, w, pw + 1)), "irregular source columns")
    panels = [[g[r + 1:s, c + 1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _require(all(np.all(p != lattice) for row in panels for p in row), "lattice inside panel")
    counts = Counter(_key(p) for row in panels for p in row)
    winners = [k for k, n in counts.items() if n == max(counts.values())]
    _require(len(winners) == 1, "tied template")
    master = next(p for row in panels for p in row if _key(p) == winners[0]).copy()
    carrier = int(master[0, 0])
    routes = set(map(int, np.unique(master)))
    _require(2 <= len(routes) <= 4, "template needs two to four material colors")
    _require(end + 2 < h, "reservoir bank missing")
    background = int(g[end + 1, 0])
    _require(background not in routes | {lattice}, "background is a separate role")
    _require(np.all(g[end + 1] == background), "clear source-bank separator required")
    bands, r = [], end + 1
    while r < h:
        if np.all(g[r] == background):
            r += 1
            continue
        occupied = np.flatnonzero(g[r] != background)
        left, right = int(occupied[0]), int(occupied[-1])
        route, rail = int(g[r, left + 1]), int(g[r, left])
        _require(right - left >= 4, "reservoir too narrow")
        _require(route in routes and rail not in {background, lattice, route}, "invalid reservoir color roles")
        expected = np.full(w, background, dtype=np.int8)
        expected[left:right + 1] = route
        expected[left] = expected[right] = rail
        _require(np.array_equal(g[r], expected), "malformed reservoir row")
        top = r
        while r + 1 < h and np.array_equal(g[r + 1], expected):
            r += 1
        bands.append(dict(top=top, bottom=r, left=left, right=right, route=route, rail=rail))
        r += 1
        _require(r == h or np.all(g[r] == background), "reservoirs need a clear gap")
    _require(len(bands) == len(routes) and {b['route'] for b in bands} == routes, "one reservoir per template color required")
    repaired = g.copy()
    labels = np.full(g.shape, -1, dtype=np.int8)
    visible = np.full(g.shape, -1, dtype=np.int8)
    changes, columns, variants, balanced = [], [], [], []
    for bi, row in enumerate(panels):
        variants_here = [ci for ci, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(variants_here) == 1, "one visible exceptional tile per source band required")
        ci = variants_here[0]
        columns.append(ci)
        variant = row[ci]
        variants.append(_key(variant))
        balanced.append(Counter(map(int, variant.ravel())) == Counter(map(int, master.ravel())))
        r0, c0 = rs[bi] + 1, cs[ci] + 1
        for cj, panel in enumerate(row):
            _require(set(map(int, np.unique(panel))) <= routes, "unknown tile color")
            _require(np.all(panel[[0, -1], :] == carrier) and np.all(panel[:, [0, -1]] == carrier), "carrier margin altered")
            cc = cs[cj] + 1
            repaired[r0:r0 + ph, cc:cc + pw] = master
        for rr, cc in np.argwhere(master != variant):
            p = r0 + int(rr), c0 + int(cc)
            changes.append(p)
            labels[p] = master[rr, cc]
            visible[p] = variant[rr, cc]
    components = _components(labels, changes)
    bank = {b['route']: b for b in bands}
    targets, used = [], set()
    for color, cells in components:
        b = bank[color]
        shift = b['bottom'] - max(r for r, c in cells)
        target = {(r + shift, c) for r, c in cells}
        _require(all(b['top'] <= r <= b['bottom'] and b['left'] < c < b['right'] for r, c in target), "piece does not fit its reservoir")
        _require(not target & used, "recovered pieces collide")
        used.update(target)
        targets.append((b['rail'], target))
    expected = repaired.copy()
    for rail, target in targets:
        for p in target:
            expected[p] = rail
    return dict(grid=g, lattice=lattice, background=background, ph=ph, pw=pw,
                rows=rs, cols=cs, panels=panels, master=master, carrier=carrier,
                bands=bands, labels=labels, visible=visible, changes=changes,
                components=components, repaired=repaired, targets=targets,
                expected=expected, columns=tuple(columns), variants=tuple(variants),
                balanced=tuple(balanced))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only rule; no puzzle examples, random state or generation metadata."""
    return _analyze(input_grid)['expected'].copy()


def _assemble(rows, lattice):
    ph, pw = rows[0][0].shape
    result = np.full((len(rows) * (ph + 1) + 1, len(rows[0]) * (pw + 1) + 1), lattice, dtype=np.int8)
    for bi, row in enumerate(rows):
        for ci, p in enumerate(row):
            r, c = 1 + bi * (ph + 1), 1 + ci * (pw + 1)
            result[r:r + ph, c:c + pw] = p
    return result


def _polyomino(rng, height, width, count):
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    while len(cells) < count:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                           if 0 <= r + dr < height and 0 <= c + dc < width
                           and (r + dr, c + dc) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _template(rng, ph, pw, routes):
    h, w = ph - 2, pw - 2
    core = np.zeros((h, w), dtype=int)
    style = int(rng.integers(6))
    if style == 0:
        core = rng.choice(len(routes), size=(h, w), p=[.5] + [.5 / (len(routes) - 1)] * (len(routes) - 1))
    elif style == 1:  # Symmetric nested frames, sometimes interrupted later.
        rr, cc = np.indices((h, w))
        core = (np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc]) + 1) % len(routes)
    elif style == 2:  # Bands and periodic motifs.
        rr, cc = np.indices((h, w))
        core = ((rr if rng.random() < .5 else cc) + int(rng.integers(len(routes)))) % len(routes)
        if rng.random() < .4:
            core[(rr + cc) % 3 == 0] = 0
    elif style == 3:  # Branched silhouette.
        spine = int(rng.integers(w))
        core[:, spine] = 1
        for r in range(h):
            end = int(rng.integers(w))
            core[r, min(spine, end):max(spine, end) + 1] = 1 + r % (len(routes) - 1)
    elif style == 4:  # Blocks/islands at independently sampled locations.
        for color in range(1, len(routes)):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(1, h - r + 1)), int(rng.integers(1, w - c + 1))
            core[r:r + rh, c:c + cw] = color
    else:  # Organic region growth.
        for color in range(1, len(routes)):
            for p in _polyomino(rng, h, w, int(rng.integers(1, h * w + 1))):
                core[p] = color
    # Retain all material identities even in small structured motifs.
    available = list(map(int, rng.permutation(h * w)))
    for color in range(1, len(routes)):
        if not np.any(core == color):
            index = next(i for i in available if np.count_nonzero(core == core.flat[i]) > 1 or core.flat[i] == 0)
            core.flat[index] = color
            available.remove(index)
    master = np.full((ph, pw), routes[0], dtype=np.int8)
    master[1:-1, 1:-1] = np.asarray(routes, dtype=np.int8)[core]
    return master


def _witness(rng, master, routes, lesson):
    """Prepare VARIABLE local patches that visibly teach the shared rule."""
    h, w = master.shape[0] - 2, master.shape[1] - 2
    a, b = map(int, rng.choice(routes[:2], 2, replace=False))
    if lesson == 'independent':
        candidates = [(c, d) for c in range(1, w + 1) for d in range(c + 2, w + 1)]
        c, d = candidates[int(rng.integers(len(candidates)))]
        r, s = map(int, rng.choice(np.arange(1, h + 1), 2, replace=False))
        first = {(r, c), (s, d)}
        # An unchanged same-color path connects the two dots in the reference,
        # but cannot connect them in the disagreement-only virtual source.
        if rng.random() < .5:
            path = {(r, cc) for cc in range(c, d + 1)} | {(rr, d) for rr in range(min(r, s), max(r, s) + 1)}
        else:
            path = {(s, cc) for cc in range(c, d + 1)} | {(rr, c) for rr in range(min(r, s), max(r, s) + 1)}
        for p in path:
            master[p] = a
        dominos = [{(rr, cc), (rr, cc + 1)} for rr in range(1, h + 1) for cc in range(1, w)
                   if not {(rr, cc), (rr, cc + 1)} & path]
        second = dominos[int(rng.integers(len(dominos)))]
    else:
        cap = max(4, (h * w) // 2)
        # A sampled connected patch split across a coordinate cut. Its two
        # reference colors touch, but remain separate operational pieces.
        for _ in range(80):
            patch = {(r + 1, c + 1) for r, c in _polyomino(rng, h, w, int(rng.integers(4, cap + 1)))}
            axis = int(rng.integers(2))
            values = sorted({p[axis] for p in patch})
            if len(values) >= 2:
                cut = values[int(rng.integers(len(values) - 1))]
                first = {p for p in patch if p[axis] <= cut}
                second = patch - first
                mini = np.full(master.shape, -1, dtype=np.int8)
                for p in first:
                    mini[p] = a
                for p in second:
                    mini[p] = b
                if any(_uneven_mask(cells) for color, cells in _components(mini, patch)):
                    break
        else:
            r = int(rng.integers(1, h))
            c = int(rng.integers(1, w - 1))
            first, second = {(r, c), (r + 1, c), (r, c + 1)}, {(r, c + 2)}
    for p in first:
        master[p] = a
    for p in second:
        master[p] = b
    changed = first | second
    # Unused colors remain visibly present in the reference, outside the patch.
    free = [(r, c) for r in range(1, h + 1) for c in range(1, w + 1) if (r, c) not in changed]
    rng.shuffle(free)
    for color in routes:
        if not np.any(master == color):
            p = next(p for p in free if np.count_nonzero(master == master[p]) > 1)
            master[p] = color
            free.remove(p)
    replacements = {p: b for p in first} | {p: a for p in second}
    return replacements


def _random_changes(rng, master, routes, allowed):
    h, w = master.shape[0] - 2, master.shape[1] - 2
    cap = max(1, h * w // 2)
    kind = int(rng.integers(6))
    if kind == 0:
        points = {(int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1)))}
    elif kind == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if rng.random() < .5:
            length = int(rng.integers(1, min(w - c, cap) + 1))
            points = {(r + 1, c + j + 1) for j in range(length)}
        else:
            length = int(rng.integers(1, min(h - r, cap) + 1))
            points = {(r + j + 1, c + 1) for j in range(length)}
    elif kind in (2, 3):
        points = {(r + 1, c + 1) for r, c in _polyomino(rng, h, w, int(rng.integers(1, cap + 1)))}
    elif kind == 4:
        points = {(r + 1, c + 1) for r in range(h) for c in range(w) if rng.random() < .25}
    else:
        rh, cw = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
        r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - cw + 1))
        points = {(r + i + 1, c + j + 1) for i in range(rh) for j in range(cw)}
        if len(points) > cap:
            points = set(sorted(points)[:cap])
    points = {p for p in points if int(master[p]) in allowed}
    if not points:
        points = {next(p for p in [(r + 1, c + 1) for r in range(h) for c in range(w)] if int(master[p]) in allowed)}
    return {p: int(rng.choice([c for c in routes if c != int(master[p])])) for p in sorted(points)}


def _geometry_choices(policy, split, route_count, lesson):
    heights = range(7, 10) if split == 'test' and policy == 'taller' else range(5, 7) if policy == 'taller' else range(4, 9)
    band_counts = (3,) if split == 'test' and policy == 'more_bands' else (1, 2) if policy == 'more_bands' else (1, 2, 3)
    copies = (4, 5) if split == 'test' and policy == 'more_copies' else (3,) if policy == 'more_copies' else (3, 4, 5)
    result = []
    for ph in heights:
        for pw in range(3, 9):
            if (ph - 2) * (pw - 2) < route_count - 1:
                continue
            if lesson and (ph < 5 or pw < 5):
                continue
            for bands in band_counts:
                for count in copies:
                    if count * (pw + 1) + 1 > 30:
                        continue
                    # Reserve one destination row per route, and clear gaps.
                    minimum = bands * (ph + 1) + 1 + 2 * route_count
                    if minimum <= 30:
                        result.append((ph, pw, bands, count))
    return result


def _make_pair(rng, policy, split, lesson, reserve=False):
    n = 2 if lesson == 'independent' else int(rng.integers(3, 5)) if lesson == 'contact' else int(rng.integers(2, 5))
    geometry = _geometry_choices(policy, split, n, lesson)
    ph, pw, nb, nc = geometry[int(rng.integers(len(geometry)))]
    if reserve:
        n = 2 if lesson == 'independent' else 3
        ph, pw, nb, nc = (5, 5, 1, 3)
        if split == 'test':
            if policy == 'more_bands':
                ph, pw, nb, nc = 4, 5, 3, 3
            elif policy == 'more_copies':
                ph, pw, nb, nc = 5, 4, 1, 4
            else:
                ph, pw, nb, nc = 7, 5, 1, 3
    palette = list(map(int, rng.permutation(10)))
    background, lattice = palette[:2]
    routes = palette[2:2 + n]
    extras = palette[2 + n:]
    master = _template(rng, ph, pw, routes)
    teaching = _witness(rng, master, routes, lesson) if lesson else None
    if reserve and lesson == 'contact':
        for p in ((1, 1), (2, 1), (1, 2)):
            master[p] = routes[0]
        master[1, 3] = routes[1]
        master[3, 1] = routes[2]
        teaching = {(1, 1): routes[1], (2, 1): routes[1], (1, 2): routes[1], (1, 3): routes[0]}
    special_band = int(rng.integers(nb))
    rows, change_points, old_labels = [], [], {}
    occupied = set()
    allowed = set(routes[:2]) if lesson == 'contact' else set(routes)
    for bi in range(nb):
        accepted = None
        for _ in range(80):
            ci = bi % nc if reserve else int(rng.integers(nc))
            patch = teaching if bi == special_band and teaching else _random_changes(rng, master, routes, allowed)
            if reserve and not lesson:
                patch = {(1, 1): routes[1] if int(master[1, 1]) != routes[1] else routes[0]}
            r0, c0 = 1 + bi * (ph + 1), 1 + ci * (pw + 1)
            pts = [(r0 + r, c0 + c) for r, c in patch]
            labels = np.full((nb * (ph + 1) + 1, nc * (pw + 1) + 1), -1, dtype=np.int8)
            for (r, c), p in zip(patch, pts):
                labels[p] = master[r, c]
            comps = _components(labels, pts)
            # Positions relative to the far edge already reveal collisions;
            # reservoir depths and source-band gaps do not affect this test.
            targets = {(color, r - max(rr for rr, cc in cells), c)
                       for color, cells in comps for r, c in cells}
            if len(targets) != len(pts) or targets & occupied:
                continue
            # Avoid diagonal-only grouping ambiguity in the generated family.
            if len(_components(labels, pts, True)) != len(comps):
                continue
            accepted = ci, patch, pts, targets
            break
        if accepted is None:
            return None
        ci, patch, pts, targets = accepted
        occupied.update(targets)
        row = [master.copy() for _ in range(nc)]
        for p, color in patch.items():
            row[ci][p] = color
        rows.append(row)
        change_points.extend(pts)
        for p, q in zip(patch, pts):
            old_labels[q] = int(master[p])
    source = _assemble(rows, lattice)
    old = np.full(source.shape, -1, dtype=np.int8)
    for p, color in old_labels.items():
        old[p] = color
    components = _components(old, change_points)
    required = {c: 1 for c in routes}
    for color, cells in components:
        required[color] = max(required[color], max(r for r, c in cells) - min(r for r, c in cells) + 1)
    gap_count = n  # source-bank gap plus the n-1 reservoir gaps
    margin = 0 if reserve else int(rng.integers(3))
    free = 30 - source.shape[0] - sum(required.values()) - gap_count - margin
    if free < 0:
        return None
    gaps = [1] * n
    if not reserve:
        for i in range(n):
            if free and rng.random() < .35:
                gaps[i] += 1
                free -= 1
    depths = required.copy()
    if free:
        extra = int(rng.integers(free + 1))
        for color in rng.choice(routes, size=extra):
            depths[int(color)] += 1
    order = list(map(int, rng.permutation(routes)))
    common = int(rng.choice(extras))
    rails = {color: int(rng.choice([c for c in routes if c != color])) if rng.random() < .35 else
             common if rng.random() < .5 else int(rng.choice(extras)) for color in routes}
    height = source.shape[0] + sum(gaps) + sum(depths.values()) + margin
    grid = np.full((height, source.shape[1]), background, dtype=np.int8)
    grid[:source.shape[0]] = source
    bands, cursor = {}, source.shape[0]
    for color, gap in zip(order, gaps):
        cursor += gap
        xs = [c for route, cells in components if route == color for r, c in cells]
        left_limit = min(xs) - 1 if xs else grid.shape[1] // 4
        right_limit = max(xs) + 1 if xs else 3 * grid.shape[1] // 4
        inset = not reserve and rng.random() < .4
        left = int(rng.integers(min(left_limit, grid.shape[1] // 5) + 1)) if inset else 0
        right = int(rng.integers(max(right_limit, grid.shape[1] - 1 - grid.shape[1] // 5), grid.shape[1])) if inset else grid.shape[1] - 1
        grid[cursor:cursor + depths[color], left:right + 1] = color
        grid[cursor:cursor + depths[color], [left, right]] = rails[color]
        bands[color] = (cursor, cursor + depths[color] - 1)
        cursor += depths[color]
    # Labels use the constructed template and recovered old-color masks.
    # The public solver independently recovers the template and bank from input.
    output = grid.copy()
    output[:source.shape[0]] = _assemble([[master.copy() for _ in range(nc)] for _ in range(nb)], lattice)
    for color, cells in components:
        shift = bands[color][1] - max(r for r, c in cells)
        for r, c in cells:
            output[r + shift, c] = rails[color]
    pair = {'input': grid, 'output': output}
    try:
        a = _analyze(grid)
        if lesson == 'independent' and not _independence(a):
            return None
        if lesson == 'contact' and (not _contact(a) or not _unused(a) or
                                    not any(_uneven_mask(cells) for color, cells in a['components'])):
            return None
    except AssertionError:
        return None
    return pair


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    colors = sorted(map(int, np.unique(pair['input'])))
    # At most 13 inputs precede this one. Each distinct injective palette gives
    # a different grid, so this finite search needs at most 14 candidates.
    for chosen in permutations(range(10), len(colors)):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = chosen
        candidate = {k: lookup[v] for k, v in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('finite palette reserve exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    policy = str(rng.choice(['more_bands', 'more_copies', 'taller']))
    witnesses = list(map(int, rng.choice(nt, 2, replace=False)))
    lessons = {witnesses[0]: 'independent', witnesses[1]: 'contact'}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            lesson = lessons.get(index) if split == 'train' else None
            pair = None
            for _ in range(120):
                pair = _make_pair(rng, policy, split, lesson)
                if pair is not None:
                    break
            if pair is None:
                # A small construction of exactly the same rule and shift.
                for _ in range(80):
                    pair = _make_pair(rng, policy, split, lesson, reserve=True)
                    if pair is not None:
                        break
            _require(pair is not None, 'bounded construction failed')
            pair = _unique(pair, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _independence(a):
    return any(x == y and max(r for r, c in p) != max(r for r, c in q)
               for i, (x, p) in enumerate(a['components']) for y, q in a['components'][i + 1:])


def _contact(a):
    pts = set(a['changes'])
    return any((r + dr, c + dc) in pts and a['labels'][r, c] != a['labels'][r + dr, c + dc]
               for r, c in pts for dr, dc in FOUR)


def _unused(a):
    return len(a['bands']) - len({c for c, cells in a['components']})


def _masked_bridge(a):
    end = a['rows'][-1]
    points = [tuple(map(int, p)) for p in np.argwhere(a['repaired'][:end + 1] != a['lattice'])]
    groups = _components(a['repaired'], points)
    membership = {p: i for i, (color, cells) in enumerate(groups) for p in cells}
    return any(color == other and membership[cells[0]] == membership[part[0]]
               and max(r for r, c in cells) != max(r for r, c in part)
               for i, (color, cells) in enumerate(a['components'])
               for other, part in a['components'][i + 1:])


def _alternative(a, visible=False, group_color=False):
    labels = a['visible'] if visible else a['labels']
    comps = _components(labels, a['changes'])
    if group_color:
        comps = [(color, tuple(p for c, cells in comps if c == color for p in cells))
                 for color in sorted({c for c, cells in comps})]
    bank = {b['route']: b for b in a['bands']}
    out = a['repaired'].copy()
    used = set()
    for color, cells in comps:
        b = bank[color]
        shift = b['bottom'] - max(r for r, c in cells)
        target = {(r + shift, c) for r, c in cells}
        if target & used or any(not (b['top'] <= r <= b['bottom'] and b['left'] < c < b['right']) for r, c in target):
            return None
        used.update(target)
        for p in target:
            out[p] = b['rail']
    return out


def _column_alternative(a):
    bank = {b['route']: b for b in a['bands']}
    out = a['repaired'].copy()
    for color, cells in a['components']:
        b = bank[color]
        bottoms = {c: max(rr for rr, cc in cells if cc == c) for r, c in cells}
        for r, c in cells:
            out[r + b['bottom'] - bottoms[c], c] = b['rail']
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
             len(puzzle['train']) > len(puzzle['test']), 'illegal counts')
    records, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas must be preserved')
            _require(_key(inp) not in seen, 'duplicate inputs')
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, a['expected']), 'incorrect recovery or source repair')
            _require(len(_components(a['labels'], a['changes'], True)) == len(a['components']), 'generated diagonal ambiguity')
            # Nonstamp pixels in the bank, all separators and all source lattice
            # cells remain unchanged. Source panels become exact master copies.
            stamped = {p for rail, target in a['targets'] for p in target}
            end = a['rows'][-1]
            _require(all(out[r, c] == inp[r, c] for r in range(end + 1, inp.shape[0])
                         for c in range(inp.shape[1]) if (r, c) not in stamped), 'unoccupied bank altered')
            _require(sum(len(t) for rail, t in a['targets']) == len(a['changes']), 'lost or duplicated material cells')
            _require(np.count_nonzero(out[:end + 1] != inp[:end + 1]) == len(a['changes']), 'wrong repair cell count')
            _require(np.count_nonzero(out[end + 1:] != inp[end + 1:]) == len(a['changes']), 'wrong recovered cell count')
            _require(not np.array_equal(out, a['repaired']), 'B contribution omitted: no recovery stamps')
            _require(not np.array_equal(inp, out), 'identity pair')
            records[split].append(a)
    train = records['train']
    _require(any(any(a['balanced']) for a in train), 'training must distinguish bitmap comparison from color counts')
    _require(any(_independence(a) for a in train), 'training must expose independent component alignment')
    _require(any(_masked_bridge(a) for a in train), 'training must show unchanged cells do not connect recovery pieces')
    _require(any(_contact(a) for a in train), 'training must expose multicolor component splitting')
    _require(any(not np.array_equal(_column_alternative(a), a['expected']) for a in train), 'columnwise falling shortcut survives')
    _require(any(not np.array_equal(_alternative(a, visible=True), a['expected']) for a in train), 'visible-color routing shortcut survives')
    _require(any(not np.array_equal(_alternative(a, group_color=True), a['expected']) for a in train), 'one-motion-per-color shortcut survives')
    _require(any(_unused(a) for a in train), 'training must demonstrate unused reservoirs')
    # A visible structural extrapolation always occurs along at least one axis.
    axes = (lambda a: len(a['panels']), lambda a: len(a['panels'][0]), lambda a: a['ph'])
    _require(any(all(axis(a) > max(axis(t) for t in train) for a in records['test']) for axis in axes), 'missing structural test shift')


def _regression_pair():
    master = np.array([[int(c) for c in row] for row in ('11111', '12221', '11121', '13121', '11111')], dtype=np.int8)
    variant = master.copy()
    variant[1, 1] = variant[3, 3] = 1
    variant[2, 1] = variant[2, 2] = 2
    source = _assemble([[master, variant, master]], 9)
    inp = np.zeros((16, 19), dtype=np.int8)
    inp[:7] = source
    for top, bottom, route, rail in ((8, 8, 1, 4), (10, 12, 2, 5), (14, 14, 3, 6)):
        inp[top:bottom + 1] = route
        inp[top:bottom + 1, [0, -1]] = rail
    out = inp.copy()
    out[:7] = _assemble([[master, master, master]], 9)
    out[8, 8] = out[8, 9] = 4
    out[12, 8] = out[12, 10] = 5
    return {'input': inp, 'output': out}


def self_test():
    """Bounded semantic regressions, mutations, count checks and 42-seed audit."""
    fixture = _regression_pair()
    _require(np.array_equal(solve(fixture['input']), fixture['output']), 'old-color/negative-space regression failed')
    a = _analyze(fixture['input'])
    _require(_masked_bridge(a), 'regression must expose disagreement-only connectivity')
    _require(not np.array_equal(_alternative(a, visible=True), fixture['output']), 'regression does not defeat visible colors')
    _require(not np.array_equal(_alternative(a, group_color=True), fixture['output']), 'regression does not defeat grouped movement')
    lookup = np.random.default_rng(91).permutation(10).astype(np.int8)
    _require(np.array_equal(solve(lookup[fixture['input']]), lookup[fixture['output']]), 'palette equivariance failed')
    too_tall = fixture['input'].copy()
    too_tall[4, 9] = 2  # Joins the recovered 1-domino to a second source row.
    collision_master = a['master'].copy()
    collision_master[3, 1], collision_master[3, 3] = 2, 3
    collision_variant = collision_master.copy()
    collision_variant[1, 1] = collision_variant[3, 1] = 1
    collision_variant[2, 1] = collision_variant[2, 2] = 2
    collision = fixture['input'].copy()
    collision[:7] = _assemble([[collision_master, collision_variant, collision_master]], 9)
    for bad in (too_tall, collision):
        try:
            solve(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('boundary crossing or true collision accepted')
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        _require(signature == tuple(_key(p[k]) for s in ('train', 'test') for p in repeat[s] for k in ('input', 'output')), 'nondeterminism')
        signatures.add(signature)
        for split in ('train', 'test'):
            records.extend((split, _analyze(p['input'])) for p in puzzle[split])
    _require(len(signatures) == 42, 'cross-seed duplicates')
    all_a = [a for split, a in records]
    _require(len({a['grid'].shape for a in all_a}) >= 35, 'narrow canvas geometry')
    _require(len({_key(a['master']) for a in all_a}) >= 100, 'narrow templates')
    _require({a['lattice'] for a in all_a} == set(range(10)), 'missing lattice palette roles')
    _require({a['background'] for a in all_a} == set(range(10)), 'missing background palette roles')
    _require({len(a['bands']) for a in all_a} == {2, 3, 4}, 'narrow route counts')
    _require({len(a['panels']) for a in all_a} >= {1, 2, 3}, 'narrow source counts')
    _require({len(a['panels'][0]) for a in all_a} >= {3, 4, 5}, 'narrow copy counts')
    masks = set()
    for a in all_a:
        for color, cells in a['components']:
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            masks.add(tuple((r - r0, c - c0) for r, c in cells))
    _require(len(masks) >= 20, 'narrow recovered mask geometry')
    _require(any(a['ph'] != a['pw'] for a in all_a), 'no rectangular tiles')
    _require(any(len(set(a['columns'])) < len(a['columns']) for a in all_a), 'no repeated exception column')
    _require(any(b['left'] or b['right'] < a['grid'].shape[1] - 1 for a in all_a for b in a['bands']), 'no inset reservoir')
    _require(any(b['rail'] in {c['route'] for c in a['bands']} for a in all_a for b in a['bands']), 'no cross-role color reuse')
    for split in ('train', 'test'):
        group = [a for s, a in records if s == split]
        _require(any(_contact(a) for a in group), 'missing split contact variation')
        _require(any(_independence(a) for a in group), 'missing split independent variation')
        _require(max(len(a['changes']) for a in group) >= 8, 'only tiny anomalies')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + nt * 10 + ne, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), 'counts ignored')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Deliberate label, template, bank and margin corruptions must be rejected.
    base = generate_puzzle(987, 2, 1)
    for defect in ('label', 'missing_exception', 'gap', 'margin'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = bad['train'][0]
        a = _analyze(pair['input'])
        if defect == 'label':
            pair['output'][a['bands'][0]['top'], a['bands'][0]['left']] = a['background']
        elif defect == 'missing_exception':
            r, c = a['rows'][0] + 1, a['cols'][a['columns'][0]] + 1
            pair['input'][r:r + a['ph'], c:c + a['pw']] = a['master']
        elif defect == 'gap':
            pair['input'][a['rows'][-1] + 1, 0] = a['lattice']
        else:
            pair['input'][1, 1] = a['lattice']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('mutation accepted: ' + defect)
    # Exercise the compatible bounded reserve, even when random attempts succeed.
    rng = np.random.default_rng(765)
    for policy in ('more_bands', 'more_copies', 'taller'):
        pair1 = _make_pair(rng, policy, 'train', 'independent', reserve=True)
        pair2 = _make_pair(rng, policy, 'train', 'contact', reserve=True)
        pair3 = _make_pair(rng, policy, 'test', None, reserve=True)
        _require(all(p is not None for p in (pair1, pair2, pair3)), 'reserve failed')
        validate_generated_puzzle({'train': [pair1, pair2], 'test': [pair3]})
