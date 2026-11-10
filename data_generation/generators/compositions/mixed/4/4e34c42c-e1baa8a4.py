"""Register overlapping rectangular views, then collapse equality runs.

Only Python/NumPy are required. There is no parent import, seed lookup, learned
glyph dictionary, or persistent solver cache. Sampling never writes to disk.
"""
from __future__ import annotations

import hashlib
import itertools
import numbers

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be two dimensional with sides 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _groups(points):
    remaining = set(points)
    groups = []
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
        groups.append(group)
    return groups


def _normalize(cells):
    top = min(r for r, c in cells)
    left = min(c for r, c in cells)
    return {(r-top, c-left): int(v) for (r, c), v in cells.items()}


def _render(cells, bg):
    cells = _normalize(cells)
    shape = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    a = np.full(shape, bg, dtype=np.int8)
    for p, v in cells.items():
        a[p] = v
    return a


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(counts.max()*2 > a.size and len(colors) > 1,
             "input must have a strict-majority background")
    return int(colors[counts.argmax()])


def _components(a, bg):
    return [_normalize({p: int(a[p]) for p in group})
            for group in _groups(zip(*np.nonzero(a != bg)))]


def _features(cells):
    """Observed whole color regions and complete, possibly gapped color layers."""
    result = set()
    for color in sorted(set(cells.values())):
        layer = {p for p, v in cells.items() if v == color}
        result.add(frozenset(layer))
        result.update(frozenset(g) for g in _groups(layer))
    return result


def _registrations(a, b, bg):
    """Exact-context, feature-supported translations; no scalar overlap score.

    A signature includes >=2 foreground colors, variation along both axes,
    and an entire observed color region/layer on at least one side. Discard a
    clipped candidate only if BOTH matching footprints are
    contained in a fuller candidate; incomparable occurrences remain ambiguous.
    """
    aa, bb = _render(a, bg), _render(b, bg)
    ah, aw = aa.shape
    bh, bw = bb.shape
    fa, fb = _features(a), _features(b)
    shifts = {(r-u, c-v) for (r, c), x in a.items()
              for (u, v), y in b.items() if x == y}
    found = []
    for dr, dc in sorted(shifts):
        top, bottom = max(0, dr), min(ah, dr+bh)
        left, right = max(0, dc), min(aw, dc+bw)
        av = aa[top:bottom, left:right]
        bv = bb[top-dr:bottom-dr, left-dc:right-dc]
        if not np.array_equal(av, bv):
            continue
        points = np.argwhere(av != bg)
        if len(points) < 2:
            continue
        if (len(np.unique(av[av != bg])) < 2
                or not np.any(av[1:] != av[:-1])
                or not np.any(av[:, 1:] != av[:, :-1])):
            continue
        hit_a = frozenset((int(r)+top, int(c)+left) for r, c in points)
        hit_b = frozenset((r-dr, c-dc) for r, c in hit_a)
        if not any(f <= hit
                   for features, hit in ((fa, hit_a), (fb, hit_b)) for f in features):
            continue
        found.append(((dr, dc), hit_a, hit_b))
    return [m for m in found if not any(
        m[1] <= q[1] and m[2] <= q[2] and (m[1] != q[1] or m[2] != q[2])
        for q in found)]


def _assemble(parts, bg):
    """Solve all observed signature equations, requiring one placement system."""
    n = len(parts)
    _require(2 <= n <= 5, "expected two to five separated fragments")
    adjacency = [[] for _ in parts]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            matches = _registrations(parts[i], parts[j], bg)
            if matches:
                deltas = [m[0] for m in matches]
                edges.append((i, j, matches))
                adjacency[i].append((j, deltas))
                adjacency[j].append((i, [(-r, -c) for r, c in deltas]))
    offsets = {0: (0, 0)}
    solutions = []
    visits = 0
    views = [_render(p, bg) for p in parts]

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 2000, "registration search bound exceeded")
        if len(solutions) > 1:
            return
        if len(offsets) == n:
            union = {}
            for i, cells in enumerate(parts):
                dr, dc = offsets[i]
                for (r, c), v in cells.items():
                    p = r+dr, c+dc
                    if p in union and union[p] != v:
                        return
                    union[p] = v
            top = min(r for r, c in union)
            left = min(c for r, c in union)
            bottom = max(r for r, c in union)+1
            right = max(c for r, c in union)+1
            if max(bottom-top, right-left) > 30:
                return
            picture = _render(union, bg)
            # Every fragment is an observed rectangular window: background
            # holes and open negative space inside its tight box also agree.
            for i, view in enumerate(views):
                r, c = offsets[i][0]-top, offsets[i][1]-left
                h, w = view.shape
                if not np.array_equal(picture[r:r+h, c:c+w], view):
                    return
            solutions.append((picture, dict(offsets)))
            return
        choices = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, deltas in adjacency[j]:
                if i in offsets:
                    possible = {(offsets[i][0]-dr, offsets[i][1]-dc) for dr, dc in deltas}
                    domain = possible if domain is None else domain & possible
            if domain is not None:
                if not domain:
                    return
                choices.append((len(domain), j, domain))
        if not choices:
            return
        _, j, domain = min(choices, key=lambda q: (q[0], q[1]))
        for position in sorted(domain):
            offsets[j] = position
            search()
            del offsets[j]

    search()
    _require(len(solutions) == 1, "disconnected, inconsistent, or ambiguous signatures")
    picture, offsets = solutions[0]
    return picture, offsets, edges


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return [0] + [i for i in range(1, len(lines))
                  if not np.array_equal(lines[i], lines[i-1])]


def _runs(a, axis):
    edges = _starts(a, axis) + [a.shape[axis]]
    return tuple(b-a for a, b in zip(edges, edges[1:]))


def _collapse(a):
    # Compare both axes in the SAME complete union, not in individual pieces.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    bg = _background(a)
    union, _, _ = _assemble(_components(a, bg), bg)
    return _collapse(union)


def _key(a):
    return (tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes())


def _repeats(a, axis):
    lines = a if axis == 0 else a.T
    return any(np.array_equal(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i+2, len(lines)))


def _local_traps(parts, offsets, bg, union):
    """Local equal neighbors that are unequal as complete union vectors."""
    top = min(offsets[i][0]+r for i, p in enumerate(parts) for r, c in p)
    left = min(offsets[i][1]+c for i, p in enumerate(parts) for r, c in p)
    result = [0, 0]
    for i, p in enumerate(parts):
        view = _render(p, bg)
        start = offsets[i][0]-top, offsets[i][1]-left
        for axis in (0, 1):
            lines = view if axis == 0 else view.T
            whole = union if axis == 0 else union.T
            for k in range(1, len(lines)):
                if (np.array_equal(lines[k], lines[k-1])
                        and not np.array_equal(whole[start[axis]+k], whole[start[axis]+k-1])):
                    result[axis] += 1
    return tuple(result)


def _analyze(pair):
    a, answer = _grid(pair['input']), _grid(pair['output'])
    bg = _background(a)
    parts = _components(a, bg)
    union, offsets, edges = _assemble(parts, bg)
    expected = _collapse(union)
    _require(np.array_equal(answer, expected), "incorrect registered-then-collapsed output")
    rows, cols = _runs(union, 0), _runs(union, 1)
    _require(np.array_equal(np.repeat(np.repeat(answer, rows, axis=0), cols, axis=1), union),
             "compression lost a value or a boundary")
    _require(set(np.unique(a))-{bg} == set(np.unique(answer))-{bg}, "foreground colors lost")
    _require(all(np.any(answer.take(k, axis=axis) != bg)
                 for axis in (0, 1) for k in (0, answer.shape[axis]-1)), "crop is not tight")
    placed = [{(r+offsets[i][0], c+offsets[i][1]) for r, c in p}
              for i, p in enumerate(parts)]
    contributions = [len(p-set().union(*(q for j, q in enumerate(placed) if j != i)))
                     for i, p in enumerate(placed)]
    return {'background': bg, 'parts': len(parts), 'rows': rows, 'cols': cols,
            'union_shape': union.shape, 'output_shape': answer.shape,
            'traps': _local_traps(parts, offsets, bg, union),
            'repeats': (_repeats(answer, 0), _repeats(answer, 1)),
            'contributions': contributions, 'edge_count': len(edges),
            'holes': _hole_count(answer, bg),
            'raw_compression_differs': not np.array_equal(_collapse(a), answer)}


def _hole_count(a, bg):
    return sum(all(0 < r < a.shape[0]-1 and 0 < c < a.shape[1]-1 for r, c in g)
               for g in _groups(zip(*np.nonzero(a == bg))))


def _premature_differs(a, answer):
    """Ablate operation order using pixels, counting unresolved keys as failure."""
    bg = _background(a)
    early = [_normalize({(int(r), int(c)): int(view[r, c])
                         for r, c in zip(*np.nonzero(view != bg))})
             for view in (_collapse(_render(part, bg)) for part in _components(a, bg))]
    try:
        return not np.array_equal(_collapse(_assemble(early, bg)[0]), answer)
    except ValueError:
        return True


def _sample_runs(rng, n, cap):
    for _ in range(40):
        values = rng.integers(1, 4, size=n)
        values[int(rng.integers(n))] = 1
        values[int(rng.integers(n))] = 2
        if values.sum() <= cap and len(set(map(int, values))) > 1:
            return tuple(map(int, values))
    return (2,) + (1,)*(n-1)


def _latent(rng, h, w, bg, colors, kind):
    z = rng.choice(colors, size=(h, w)).astype(np.int8)
    rr, cc = np.indices((h, w))
    if kind == 0:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = rng.random((h-2, w-2)) > .18
    elif kind == 1:
        mask = ((rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
                | (rr == h//2) | (cc == w//2) | (rng.random((h, w)) < .2))
    elif kind == 2:
        mask = rng.random((h, w)) < .78
        mask[h//2, :] = True
        mask[:, w//2] = True
    else:
        mask = ((rr == h//2) | (cc == w//2) | (rr == 1) | (cc == 1))
        mask |= rng.random((h, w)) < .25
        mask[0, :w//2+1] = True
        mask[:h//2+1, 0] = True
    z[~mask] = bg
    # A partial equality is misleading until the right-hand view is merged.
    z[2, :w-2] = z[1, :w-2]
    z[1, w-2], z[2, w-2] = colors[0], colors[1]
    z[0, 0] = colors[-1]
    # Separated identical WHOLE vectors survive the final collapse.
    z[-1] = z[0]
    z[:, -1] = z[:, 0]
    if h >= 7 and rng.random() < .4:
        z[h-3] = z[0]
    if w >= 7 and rng.random() < .4:
        z[:, w-3] = z[:, 0]
    _require(len(_groups(zip(*np.nonzero(z != bg)))) == 1, "disconnected latent picture")
    _require(len(_starts(z, 0)) == h and len(_starts(z, 1)) == w,
             "latent boundary unobservable")
    return z


def _boxes(rng, h, w, target):
    # First two views expose a hidden row distinction; later cuts may use
    # either axis and need not produce a chain or equal-sized rectangles.
    boxes = [(0, h, 0, w-2), (0, h, w-4, w)]
    if target >= 3:
        cut = int(rng.integers(2, h-1))
        first = int(rng.integers(2))
        # A shared two-row central strip keeps real signature evidence between
        # every pair of views. Which side is subdivided is independently drawn.
        indices = {first} if target == 3 else {0, 1}
        boxes = [child for i, (r0, r1, c0, c1) in enumerate(boxes)
                 for child in ([(r0, cut+1, c0, c1), (cut-1, r1, c0, c1)]
                               if i in indices else [(r0, r1, c0, c1)])]
    return boxes


def _pack(parts, bg, rng):
    """Bounded shelf packing; input positions never encode solved adjacency."""
    views = [_render(p, bg) for p in parts]
    total = sum(len(p) for p in parts)
    n = len(parts)
    layouts = []
    for order in itertools.permutations(range(n)):
        for cuts in range(1 << (n-1)):
            shelves = [[order[0]]]
            for k in range(n-1):
                if cuts & (1 << k):
                    shelves.append([])
                shelves[-1].append(order[k+1])
            heights = [max(views[i].shape[0] for i in shelf) for shelf in shelves]
            widths = [sum(views[i].shape[1] for i in shelf)+len(shelf)-1 for shelf in shelves]
            h, w = sum(heights)+len(shelves)-1, max(widths)
            if h <= 30 and w <= 30 and 2*total < 900:
                layouts.append((shelves, heights, widths, h, w))
    _require(bool(layouts), "fragments cannot fit a 30 by 30 canvas")
    shelves, heights, widths, base_h, base_w = layouts[int(rng.integers(len(layouts)))]
    canvases = [(h, w) for h in range(base_h, 31) for w in range(base_w, 31)
                if h*w > 2*total]
    _require(bool(canvases), "no strict-majority background packing")
    h, w = canvases[int(rng.integers(len(canvases)))]
    a = np.full((h, w), bg, dtype=np.int8)
    extra_h = h-base_h
    gaps = rng.multinomial(extra_h, np.full(len(shelves)+1, 1/(len(shelves)+1)))
    top = int(gaps[0])
    for k, shelf in enumerate(shelves):
        free_w = w-widths[k]
        horizontal = rng.multinomial(free_w, np.full(len(shelf)+1, 1/(len(shelf)+1)))
        left = int(horizontal[0])
        for j, i in enumerate(shelf):
            view = views[i]
            ph, pw = view.shape
            r = top+int(rng.integers(heights[k]-ph+1))
            a[r:r+ph, left:left+pw] = view
            left += pw+1+int(horizontal[j+1])
        top += heights[k]+1+int(gaps[k+1])
    return a


def _make_pair(rng, target, transpose, is_test):
    for _ in range(180):
        try:
            h = int(rng.integers(6, 9) if is_test else rng.integers(5, 8))
            w = int(rng.integers(6, 9) if is_test else rng.integers(5, 8))
            palette = rng.permutation(10)
            bg = int(palette[0])
            colors = palette[1:int(rng.integers(3, 8))]
            kind = int(rng.integers(4))
            z = _latent(rng, h, w, bg, colors, kind)
            cap = 17 if is_test else 15
            rows, cols = _sample_runs(rng, h, cap), _sample_runs(rng, w, cap)
            union = np.repeat(np.repeat(z, rows, axis=0), cols, axis=1)
            rb, cb = np.r_[0, np.cumsum(rows)], np.r_[0, np.cumsum(cols)]
            views = [union[rb[r0]:rb[r1], cb[c0]:cb[c1]].copy()
                     for r0, r1, c0, c1 in _boxes(rng, h, w, target)]
            flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))

            def transform(a):
                if flip_r:
                    a = a[::-1]
                if flip_c:
                    a = a[:, ::-1]
                return (a.T if transpose else a).copy()

            union = transform(union)
            parts = []
            for view in views:
                view = transform(view)
                groups = _components(view, bg)
                _require(len(groups) == 1, "a view is disconnected")
                parts.append(groups[0])
            reconstructed, offsets, _ = _assemble(parts, bg)
            _require(np.array_equal(reconstructed, union), "ambiguous intended assembly")
            placed = [{(r+offsets[i][0], c+offsets[i][1]) for r, c in p}
                      for i, p in enumerate(parts)]
            _require(all(p-set().union(*(q for j, q in enumerate(placed) if j != i))
                         for i, p in enumerate(placed)), "a view has no exclusive material")
            _require(any(_local_traps(parts, offsets, bg, union)), "no registration-dependent boundary")
            answer = _collapse(union)
            _require(_repeats(answer, 0) and _repeats(answer, 1), "separated repetitions absent")
            inp = _pack(parts, bg, rng)
            _require(not np.array_equal(_collapse(inp), answer), "registration ablation is ineffective")
            return {'input': inp, 'output': answer}
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback is emitted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))
                 and lo <= int(value) <= hi, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_axis = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            test = split == 'test'
            target = 4 if test else (2 if index == 0 else 3 if index == 1 else int(rng.integers(2, 4)))
            transpose = first_axis if index == 0 else not first_axis if index == 1 and not test else bool(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, target, transpose, test)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a = _grid(pair['input'])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(pair)
            _require(max(info['rows']) > 1 and max(info['cols']) > 1, "both-axis collapse must matter")
            _require(all(x > 0 for x in info['contributions']), "generated views must contribute material")
            _require(info['raw_compression_differs'], "raw-canvas compression is a shortcut")
            if split == 'train':
                evidence.append(info)
    _require(any(len(set(i['rows'])) > 1 and len(set(i['cols'])) > 1 for i in evidence),
             "training must show independent nonuniform runs")
    _require(all(any(i['repeats'][axis] for i in evidence) for axis in (0, 1)),
             "training must preserve separated whole-line repetitions on both axes")
    _require(all(any(i['traps'][axis] for i in evidence) for axis in (0, 1)),
             "training must expose why registration precedes compression on both axes")


def self_test():
    """Stream 42 seeds, retain bounded statistics, and exercise semantic failures."""
    shapes, input_shapes, palettes, patterns, structures, backgrounds, counts = (set() for _ in range(7))
    puzzle_hashes = set()
    holes = traps = shifted = 0
    order_witnesses = early_differences = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, out = pair['input'], pair['output']
                info = _analyze(pair)
                _require(np.array_equal(solve(a), out), "input-only solver mismatch")
                shapes.add(out.shape)
                input_shapes.add(a.shape)
                backgrounds.add(info['background'])
                palettes.add(tuple(map(int, np.unique(out))))
                patterns.add((info['rows'], info['cols']))
                counts.add(info['parts'])
                holes += info['holes'] > 0
                traps += any(info['traps'])
                shifted += split == 'test' and info['parts'] > max(
                    len(_components(p['input'], _background(p['input']))) for p in puzzle['train'])
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in out.flat)
                structures.add(hashlib.sha256(bytes(out.shape)+normalized).hexdigest())
                digest.update(bytes(a.shape)+a.tobytes()+bytes(out.shape)+out.tobytes())
        puzzle_hashes.add(digest.hexdigest())
        ablations = [_premature_differs(p['input'], p['output']) for p in puzzle['train'][:2]]
        order_witnesses += any(ablations)
        early_differences += sum(ablations)
        # Determinism is checked per seed without retaining previous puzzles.
        again = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                     for p, q in zip(puzzle[s], again[s]) for k in ('input', 'output')),
                 "seed did not reproduce complete puzzle")
    _require(len(shapes) >= 10 and len(input_shapes) >= 35, "insufficient dimension variation")
    _require(len(puzzle_hashes) == 42, "different seeds duplicated a puzzle")
    _require(len(structures) >= 100 and len(patterns) >= 100, "insufficient structural variation")
    _require(len(palettes) >= 30 and len(backgrounds) == 10, "insufficient color-role variation")
    _require(counts == {2, 3, 4} and shifted == 42, "fragment-count shift missing")
    _require(holes >= 20 and traps == 210, "holes or causal equality examples missing")
    _require(order_witnesses == 42, "a minimum-size training set failed to distinguish operation order")
    # Input-only metamorphisms: orientation, role-neutral colors, and repacking
    # cannot change the inferred operation or depend on latent generation data.
    sample = generate_puzzle(0, 2, 1)
    for pair in sample['train'] + sample['test']:
        a, out = pair['input'], pair['output']
        _require(np.array_equal(solve(a.T), out.T), "transpose convention leaked")
        mapping = np.array([7, 2, 8, 0, 9, 1, 6, 4, 5, 3], dtype=np.int8)
        _require(np.array_equal(solve(mapping[a]), mapping[out]), "literal color role leaked")
        bg = _background(a)
        repacked = _pack(_components(a, bg), bg, np.random.default_rng(314159))
        _require(np.array_equal(solve(repacked), out), "input packing encoded placement")
        union, _, _ = _assemble(_components(a, bg), bg)
        _require(union.shape[0] > out.shape[0] and union.shape[1] > out.shape[1],
                 "omitting either equality axis did not change dimensions")
    bad = {k: v.copy() for k, v in sample['train'][0].items()}
    bad['output'][0, 0] = (int(bad['output'][0, 0])+1) % 10
    try:
        _analyze(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("changed output cell accepted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(20261023+nt*10+nq, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), "wrong explicit counts")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Separated rows/columns, background holes, and literal crossing values.
    base = np.array([[1, 2, 1], [3, 0, 3], [1, 2, 1]], dtype=np.int8)
    expanded = np.repeat(np.repeat(base, [2, 1, 3], axis=0), [1, 3, 2], axis=1)
    _require(np.array_equal(_collapse(expanded), base), "global deduplication or wrong crossing")
    _require(np.array_equal(_collapse(expanded.T), base.T), "axis asymmetry")
    # Partial equalities must not erase a full-vector distinction.
    local = np.array([[1, 1, 2], [1, 1, 3], [1, 1, 2]], dtype=np.int8)
    _require(np.array_equal(_collapse(local), local[:, [0, 2]]), "cellwise run collapse")
    # Two equal bars have unresolved placements; do not invent a tie-break.
    try:
        _assemble([{(0, c): 1 for c in range(5)}, {(0, c): 1 for c in range(3)}], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous bars accepted")
    return {'seeds': 42, 'pairs': 210, 'output_shapes': len(shapes),
            'distinct_puzzles': len(puzzle_hashes),
            'input_shapes': len(input_shapes), 'structures': len(structures),
            'band_schedules': len(patterns), 'palettes': len(palettes),
            'backgrounds': len(backgrounds), 'hole_pairs': holes,
            'local_equality_trap_pairs': traps, 'shifted_tests': shifted,
            'order_witness_puzzles': order_witnesses,
            'early_compression_differences_in_first_two_train': early_differences}
