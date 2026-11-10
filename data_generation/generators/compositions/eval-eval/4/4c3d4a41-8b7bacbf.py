"""Comb-operated paint switchboard: mechanics selects route source colors.

The solver reads only pixels. Construction metadata is used only by the
generator's direct output constructor, never by solve().
"""
from collections import deque
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D8):
    unseen = set(cells)
    groups = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        group, todo = {root}, [root]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _near(cells, steps=D8):
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _runs(mask):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, tuple(int(x) for x in a.flat)


def _hardware(f, shifted=False):
    """Direct construction of the lower hardware, with explicit destinations."""
    a = np.full(f['shape'], f['bg'], dtype=int)
    t, b, left, right = f['top'], f['base'], f['left'], f['right']
    s = f['structure']
    a[t, left:right + 1] = s
    a[t:b + 1, left] = a[t:b + 1, right] = s
    kb = b if shifted else f['key_base']
    a[kb, f['lanes'][0]:f['lanes'][-1] + 1] = s
    for c, h, seq in zip(f['lanes'], f['heights'], f['stacks']):
        a[kb - h:kb, c] = s
        start = b - h - len(seq) if shifted else t + 1
        for j, color in enumerate(seq):
            if start + j > t:
                a[start + j, c] = color
    for c, r in f['ports']:
        a[t - 1:r + 2, c + 2] = f['wire']
        a[r, c + 1] = f['marker']
    return a


def _read_hardware(a, bg, s, kb, first, last):
    """Recognize a comb and its dock; verify every pixel below the roof."""
    teeth = np.flatnonzero(a[kb - 1, first:last + 1] == s) + first
    assert 2 <= len(teeth) <= 5 and teeth[0] == first and teeth[-1] == last
    spacing = int(teeth[1] - teeth[0])
    assert 4 <= spacing <= 6 and np.all(np.diff(teeth) == spacing)
    hs = []
    for c in teeth:
        r = kb - 1
        while r >= 0 and a[r, c] == s:
            r -= 1
        hs.append(kb - r - 1)
    left, right = first - 2, last + 3
    assert 0 <= left < right < a.shape[1]
    candidates = []
    common = (a[:, left] == s) & (a[:, right] == s)
    for t, b in _runs(common):
        if not (b - t >= 4 and b < kb - max(hs) and t >= 3):
            continue
        try:
            span = b - t
            assert all(1 <= h <= span - 2 for h in hs)
            assert np.all(a[t, left:right + 1] != bg)
            seqs, ports, wire, marker = [], [], None, None
            for c, h in zip(teeth, hs):
                seq = a[t + 1:b, c]
                occupied = np.flatnonzero(seq != bg)
                assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
                seq = tuple(int(v) for v in seq[:len(occupied)])
                assert len(seq) >= span - h - 1
                p = next((i for i, v in enumerate(seq) if v != s), len(seq))
                assert p < len(seq) and all(v not in (bg, s) for v in seq[p:])
                seqs.append(seq)
                rows = np.flatnonzero(a[t + 1:b, c + 1] != bg) + t + 1
                if len(rows):
                    assert len(rows) == 1
                    r = int(rows[0])
                    assert t + 2 <= r <= b - 2
                    w, m = int(a[t, c + 2]), int(a[r, c + 1])
                    assert len({bg, s, w, m}) == 4
                    assert wire is None or (wire, marker) == (w, m)
                    wire, marker = w, m
                    ports.append((int(c), r))
                else:
                    assert a[t, c + 2] == s
            assert 1 <= len(ports) <= 3
            assert all(v not in (wire, marker) for seq in seqs for v in seq)
            f = dict(shape=a.shape, bg=bg, structure=s, top=t, base=b,
                     left=left, right=right, key_base=kb,
                     lanes=tuple(int(c) for c in teeth), heights=tuple(hs),
                     stacks=tuple(seqs), ports=tuple(ports), wire=wire, marker=marker)
            # Only the above-roof network is unrestricted here. The route's
            # straight feedthrough and three-neighbor reading cup are visible.
            assert np.array_equal(a[t:], _hardware(f)[t:])
            assert np.all(a[t - 1, [left, right]] == bg)
            assert set(map(tuple, np.argwhere(a == marker))) == {(r, c + 1) for c, r in ports}
            candidates.append(f)
        except (AssertionError, IndexError):
            continue
    assert len(candidates) == 1
    return candidates[0]


def _network(a, f):
    """Separate actual wire attachments from incidental contour contact."""
    wires = _components(map(tuple, np.argwhere(a == f['wire'])))
    owners = {}
    for i, cells in enumerate(wires):
        ports = [p for p in f['ports'] if (p[1], p[0] + 2) in cells]
        assert len(ports) == 1, "Each route must have exactly one reading socket"
        owners[i] = ports[0]
    assert len(wires) == len(f['ports'])
    # Fully closed targets live above the roof. Hardware hollows are not
    # targets: their boundaries contain pins, wires and the open dock.
    upper = a[:f['top'] - 1]
    height, width = upper.shape
    holes = []
    for cells in _components(map(tuple, np.argwhere(upper == f['bg'])), D4):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in cells):
            continue
        shell = _near(cells, D4)
        assert {int(a[p]) for p in shell} == {f['structure']}, "Unintended enclosure"
        attached = [i for i, route in enumerate(wires) if route & _near(shell)]
        assert len(attached) <= 1, "Conflicting socket supplies"
        holes.append((cells, shell, attached[0] if attached else None))
    assert holes and any(owner is not None for _, _, owner in holes)
    # Contours are precisely thin four-neighbor shells. This rules out noise
    # and arbitrary open curves without requiring a particular cavity shape.
    all_shells = set().union(*(shell for _, shell, _ in holes))
    assert set(map(tuple, np.argwhere(upper == f['structure']))) == all_shells
    assert np.all((upper == f['bg']) | (upper == f['structure']) | (upper == f['wire']))
    return wires, owners, holes


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    candidates = []
    for code in range(8):
        g = _unorient(a, code)
        for s in values:
            s = int(s)
            if s == bg:
                continue
            for kb in range(2, g.shape[0]):
                for first, last in _runs(g[kb] == s):
                    if last - first < 4:
                        continue
                    try:
                        f = _read_hardware(g, bg, s, kb, first, last)
                        network = _network(g, f)
                    except (AssertionError, IndexError):
                        continue
                    candidates.append((g, f, network, code))
    assert len(candidates) == 1, "Input needs exactly one geometric interpretation"
    return candidates[0]


def _move_pixels(a, f):
    """Insert the actual key, then push each pin one cell at a time."""
    out = a.copy()
    t, b, kb, s, bg = f['top'], f['base'], f['key_base'], f['structure'], f['bg']
    first, last = f['lanes'][0], f['lanes'][-1]
    key_cells = [(r, c) for r in range(kb - max(f['heights']), kb + 1)
                 for c in range(first, last + 1) if a[r, c] == s]
    for r, c in key_cells:
        out[r, c] = bg
    for c in f['lanes']:
        out[t + 1:b, c] = bg
    for r, c in key_cells:
        out[r - (kb - b), c] = s
    for c, h in zip(f['lanes'], f['heights']):
        cells = [(r, int(a[r, c])) for r in range(t + 1, b) if a[r, c] != bg]
        while cells[-1][0] >= b - h:
            cells = [(r - 1, v) for r, v in cells if r - 1 > t]
            assert cells
        for r, v in cells:
            out[r, c] = v
    return out


def _paint(out, f, network, source_grid):
    _, owners, holes = network
    for cells, _, owner in holes:
        if owner is None:
            continue
        c, r = owners[owner]
        paint = int(source_grid[r, c])
        if paint not in (f['bg'], f['structure']):
            for p in cells:
                out[p] = paint
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the answer from input geometry, ordered pixels and connectivity."""
    a, f, network, code = _parse(_grid(input_grid))
    moved = _move_pixels(a, f)
    out = _paint(moved.copy(), f, network, moved)
    return _orient(out, code)


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _shape(rng):
    """Connected, asymmetric row intervals; the shell follows the cavity."""
    h, w = _ri(rng, 1, 3), _ri(rng, 1, 4)
    kind = int(rng.integers(5))
    rows = []
    left = 0
    for r in range(h):
        if kind == 1 and r:
            left += int(rng.choice((-1, 0, 1))) if w > 1 else 0
        width = w - 1 if kind == 2 and h > 2 and r == 1 and w > 1 else w
        if kind == 3:
            width = max(1, w - abs(r - h // 2))
        rows.extend((r, c) for c in range(left, left + width))
    cells = set(rows)
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    shell = _near(cells, D4)
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return {(r - r0, c - c0) for r, c in cells}, {(r - r0, c - c0) for r, c in shell}


def _path(rng, starts, goals, allowed):
    todo = deque(sorted(starts))
    previous = {p: None for p in starts}
    dirs = [D8[i] for i in rng.permutation(8)]
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in dirs:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in previous:
                previous[q] = p
                todo.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _gallery(rng, f, fanout=False, sparse=False, lesson=False):
    """Free-place target shapes, then route branches from fixed socket exits."""
    height, width = f['top'] - 1, f['shape'][1]
    canvas = {(r, c) for r in range(height) for c in range(width)}
    nroute = len(f['ports'])
    wires = [{(height - 1, c + 2)} for c, _ in f['ports']]
    counts = [_ri(rng, 1, 2) for _ in range(nroute)]
    if fanout:
        counts[int(rng.integers(min(2, nroute)))] = 2
    if sparse:
        counts = [2 if fanout and i == 0 else 1 for i in range(nroute)]
    detached = 1 if lesson else _ri(rng, 0, 2)
    owner_list = [i for i, n in enumerate(counts) for _ in range(n)] + [None] * detached
    rng.shuffle(owner_list)
    regions, occupied = [], set()
    roots = set().union(*wires)
    for owner in owner_list:
        inner, shell = ({(1, 1)}, {(0, 1), (1, 0), (1, 2), (2, 1)}) if sparse else _shape(rng)
        hh, ww = max(r for r, c in shell) + 1, max(c for r, c in shell) + 1
        positions = [(r, c) for r in range(height - hh) for c in range(width - ww + 1)]
        rng.shuffle(positions)
        found = False
        for rr, cc in positions:
            ins = {(r + rr, c + cc) for r, c in inner}
            edge = {(r + rr, c + cc) for r, c in shell}
            full = ins | edge
            if full & occupied or _near(full) & roots:
                continue
            # Detached contours may touch a supplied contour's shell, but
            # cavities stay distinct and sources never transfer across shells.
            contact = _near(full) & occupied
            if contact and (owner is not None or not contact <= set().union(*(x[1] for x in regions))):
                continue
            if _near(ins, D4) & set().union(*(x[0] for x in regions)):
                continue
            regions.append((ins, edge, owner))
            occupied |= full
            found = True
            break
        if not found:
            return None
    all_loops = set().union(*(ins | edge for ins, edge, _ in regions))
    for i in rng.permutation(nroute):
        i = int(i)
        for ins, edge, owner in regions:
            if owner != i:
                continue
            other_wires = set().union(*(cells for j, cells in enumerate(wires) if j != i))
            other_loops = all_loops - (ins | edge)
            blocked = all_loops | _near(other_loops) | other_wires | _near(other_wires)
            goals = (_near(edge) & canvas) - blocked
            path = _path(rng, wires[i], goals, (canvas - blocked) | wires[i])
            if path is None:
                return None
            wires[i] |= path
    a = _hardware(f)
    for _, shell, _ in regions:
        for p in shell:
            a[p] = f['structure']
    for route in wires:
        for p in route:
            a[p] = f['wire']
    b = _hardware(f, shifted=True)
    b[:f['top'] - 1] = a[:f['top'] - 1]
    for cells, _, owner in regions:
        if owner is not None:
            c, r = f['ports'][owner]
            paint = int(b[r, c])
            if paint not in (f['bg'], f['structure']):
                for p in cells:
                    b[p] = paint
    try:
        info = _network(a, f)
        assert len(info[2]) == len(regions)
        for cells, _, owner in regions:
            actual = next(o for ins, _, o in info[2] if ins == cells)
            assert (actual is None) == (owner is None)
            if actual is not None:
                assert info[1][actual] == f['ports'][owner]
        values, ns = np.unique(a, return_counts=True)
        assert np.count_nonzero(ns == ns.max()) == 1 and int(values[np.argmax(ns)]) == f['bg']
        assert np.any(b[:f['top'] - 1] != a[:f['top'] - 1])
    except (AssertionError, StopIteration):
        return None
    return a, b


def _sample_hardware(rng, palette, lesson, larger=False, allow_flat=True):
    bg, structure, wire, marker = palette[:4]
    paints = palette[4:]
    n = 5 if larger else _ri(rng, 2, 4)
    spacing = _ri(rng, 4, min(6, 23 // (n - 1)))
    fw = spacing * (n - 1) + 6
    width = _ri(rng, max(18, fw), 30)
    left = _ri(rng, 0, width - fw)
    lanes = tuple(left + 2 + spacing * i for i in range(n))
    top, span = _ri(rng, 10, 13), _ri(rng, 5, 8)
    base = top + span
    heights = rng.integers(1, min(4, span - 2) + 1, size=n).tolist()
    if rng.random() < .2 and lesson < 0:
        heights = [int(rng.choice(heights))] * n
    if not allow_flat and len(set(heights)) == 1:
        i = int(rng.integers(n))
        heights[i] = 1 if heights[i] > 1 else 2
    shifts = [_ri(rng, 0, h) for h in heights]
    ports_idx = sorted(int(i) for i in rng.choice(n, size=min(n, _ri(rng, 2, 3)), replace=False))
    if lesson < 0 and rng.random() < .25:
        ports_idx = [int(rng.choice(ports_idx))]
    stacks = []
    for i, h in enumerate(heights):
        length = span - h - 1 + shifts[i]
        prefix = _ri(rng, 0, length - 1)
        stacks.append([structure] * prefix + [int(rng.choice(paints[:_ri(rng, 2, len(paints))]))
                                              for _ in range(length - prefix)])
    ports = [(lanes[i], _ri(rng, top + 2, base - 2)) for i in ports_idx]
    if lesson >= 0:
        # Lesson 0: a changed paint source and a paint-to-null socket.
        # Lesson 1: two independent selected paints, fanout, and a stationary
        # pin. None of these lane positions is tied to a profile extremum.
        i, j = ports_idx[:2]
        heights[i], shifts[i] = (1 if lesson == 0 else 2), 1
        li = span - heights[i] - 1 + 1
        stacks[i] = [structure] + [paints[0]] * (li - 1)
        ri = top + 2
        stacks[i][ri - top - 1] = paints[1]
        stacks[i][ri - top] = paints[0]
        ports[0] = (lanes[i], ri)
        if lesson == 0:
            stacks[i][3:] = [paints[1]] * (li - 3)
            heights[j], shifts[j] = 3, 3
            lj = span - heights[j] - 1 + shifts[j]
            stacks[j] = [paints[1]] * lj
            ports[1] = (lanes[j], base - 2)
        else:
            # One prefix cell survives and visibly moves. The reader changes
            # from structural gray to paint, instead of treating gray as paint.
            stacks[i] = [structure, structure] + [paints[0]] * (li - 2)
            heights[j], shifts[j] = 1, 0
            stacks[j] = [structure] + [paints[1]] * (span - 3)
            ports[1] = (lanes[j], top + 2)
    gap = _ri(rng, 1, 2)
    kb = base + gap + max(heights) + 1
    height = kb + 1
    assert height <= 30
    height = _ri(rng, height, min(30, height + 1))
    return dict(shape=(height, width), bg=bg, structure=structure, wire=wire,
                marker=marker, top=top, base=base, left=left,
                right=lanes[-1] + 3, key_base=kb, lanes=lanes,
                heights=tuple(heights), stacks=tuple(tuple(seq) for seq in stacks), ports=tuple(ports))


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.permutation(10))
    phase = int(rng.integers(8))
    shift = int(rng.integers(3))  # size, new profile, or reflection transfer
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else -1
            for attempt in range(160):
                f = _sample_hardware(rng, palette, lesson,
                                     split == 'test' and shift == 0,
                                     not (split == 'train' and shift == 1))
                if split == 'test' and shift == 1:
                    # Hold flat profiles out of demonstrations; the same
                    # height/contact formula still handles equal teeth.
                    h = max(f['heights'])
                    f['heights'] = (h,) * len(f['lanes'])
                candidate = _gallery(rng, f, fanout=lesson == 1, sparse=attempt >= 100,
                                     lesson=lesson >= 0)
                if candidate is None:
                    continue
                a, b = candidate
                code = phase if index < 2 or shift == 2 else (phase + int(rng.integers(4))) % 8
                if split == 'test' and shift == 2:
                    code = (phase + 4) % 8
                pair = dict(input=_orient(a, code), output=_orient(b, code))
                if _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded switchboard construction exhausted")
    return puzzle


def _facts(a, f, network):
    moved = _move_pixels(a, f)
    shifts = tuple(len(seq) - (f['base'] - f['top'] - h - 1)
                   for h, seq in zip(f['heights'], f['stacks']))
    reads = tuple(int(moved[r, c]) for c, r in f['ports'])
    before = tuple(int(a[r, c]) for c, r in f['ports'])
    wires, owners, holes = network
    active = [i for i in owners if moved[owners[i][1], owners[i][0]] not in (f['bg'], f['structure'])]
    counts = [sum(o == i for _, _, o in holes) for i in owners]
    prefixes = [next((i for i, v in enumerate(seq) if v != f['structure']), len(seq))
                for seq in f['stacks']]
    return dict(lanes=len(f['lanes']), spacing=f['lanes'][1] - f['lanes'][0],
                span=f['base'] - f['top'], heights=f['heights'], shifts=shifts,
                reads=reads, before=before, active=len(active), ports=len(owners),
                loops=len(holes), detached=sum(o is None for _, _, o in holes),
                fanout=max(counts), cavity_shapes=tuple(tuple(sorted((r - min(x[0] for x in ins),
                                                                    c - min(x[1] for x in ins))
                                                                   for r, c in ins))
                                                       for ins, _, _ in holes),
                wire_steps=sum(len(cells) for cells in wires),
                productive_colors=len({int(moved[owners[i][1], owners[i][0]]) for i in active}),
                productive_fanout=max((counts[i] for i in active), default=0),
                moving_prefix=any(0 < d < p for d, p in zip(shifts, prefixes)),
                payload_clip=any(d > p for d, p in zip(shifts, prefixes)),
                ablate_A=not np.array_equal(_paint(moved.copy(), f, network, a),
                                           _paint(moved.copy(), f, network, moved)),
                ablate_B=bool(active and any(o in active for _, _, o in holes)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a, f, network, code = _parse(inp)
            moved = _move_pixels(a, f)
            expected = _paint(moved.copy(), f, network, moved)
            assert np.array_equal(_orient(expected, code), out)
            # Pins preserve exact suffix order; cells on/above the roof vanish.
            for c, h, seq in zip(f['lanes'], f['heights'], f['stacks']):
                capacity = f['base'] - f['top'] - h - 1
                assert tuple(moved[f['top'] + 1:f['base'] - h, c]) == seq[-capacity:]
                assert moved[f['top'], c] == f['structure']
            # No socket/cup/route/outline cell is recolored by paint propagation.
            changed = set(map(tuple, np.argwhere(expected != moved)))
            holes = network[2]
            assert changed <= set().union(*(ins for ins, _, _ in holes))
            for ins, _, owner in holes:
                vals = {int(expected[p]) for p in ins}
                assert len(vals) == 1
                if owner is None:
                    assert vals == {f['bg']}
            facts = _facts(a, f, network)
            assert facts['ablate_B']
            if split == 'train':
                training.append(facts)
    assert any(f['ablate_A'] for f in training)
    assert any(f['active'] < f['ports'] for f in training)
    assert any(f['productive_colors'] >= 2 and f['productive_fanout'] >= 2 for f in training)
    assert any(0 in f['shifts'] for f in training)
    assert any(f['moving_prefix'] for f in training)
    assert any(f['payload_clip'] for f in training)
    assert sum(f['detached'] for f in training) >= 2


def self_test():
    """Bounded semantic, count, determinism, corruption and diversity checks."""
    signatures, shapes, profiles, cavities = set(), set(), set(), set()
    facts, orientations, palettes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                a, f, network, code = _parse(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                facts.append(_facts(a, f, network))
                shapes.add(a.shape)
                profiles.add(f['heights'])
                cavities.update(_facts(a, f, network)['cavity_shapes'])
                orientations.add(code)
                palettes.add((f['bg'], f['structure'], f['wire'], f['marker']))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(profiles) >= 50
    assert len(cavities) >= 15 and len(palettes) >= 30 and len(orientations) == 8
    assert {f['lanes'] for f in facts} >= {2, 3, 4, 5}
    assert {f['spacing'] for f in facts} >= {4, 5, 6}
    assert {f['ports'] for f in facts} >= {1, 2, 3}
    assert any(len(set(f['heights'])) == 1 for f in facts)
    assert any(max(f['shifts']) >= 3 for f in facts)
    assert any(f['active'] < f['ports'] for f in facts)
    assert any(f['productive_colors'] >= 2 for f in facts)
    again = generate_puzzle(0)
    assert tuple(_key(p['input']) for s in again for p in again[s]) in signatures
    repeat = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in again
               for p, q in zip(again[s], repeat[s]) for k in ('input', 'output'))
    # The semantic operation is equivariant under all eight orientations and
    # arbitrary role recoloring, including a black payload or nonzero exterior.
    for code in range(8):
        mapping = np.array([(3 * i + code) % 10 for i in range(10)])
        for p in again['train'][:2]:
            assert np.array_equal(solve(_orient(mapping[p['input']], code)),
                                  _orient(mapping[p['output']], code))
    # A hand-built feasible member outside the sampler's ranges: a narrow
    # canvas, a lower roof, a ten-cell span and a six-cell tooth. Expected pin
    # suffixes and two source colors are explicit, not learned from examples.
    f = dict(shape=(25, 14), bg=0, structure=5, wire=1, marker=8,
             top=6, base=16, left=0, right=9, key_base=24,
             lanes=(2, 6), heights=(6, 3),
             stacks=((5, 5, 2, 3, 2, 3, 2, 3, 4), (5, 5, 5, 4, 4, 2, 4)),
             ports=((2, 8), (6, 11)))
    a, expected = _hardware(f), _hardware(f, shifted=True)
    assert tuple(expected[7:10, 2]) == (2, 3, 4)
    assert tuple(expected[7:13, 6]) == (5, 5, 4, 4, 2, 4)
    for cell in ((1, 1), (1, 8)):
        for p in _near({cell}, D4):
            a[p] = expected[p] = 5
    for p in ((2, 3), (3, 4), (4, 4), (3, 8), (4, 8)):
        a[p] = expected[p] = 1
    expected[1, 1], expected[1, 8] = 3, 2
    for code in range(8):
        assert np.array_equal(solve(_orient(a, code)), _orient(expected, code))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(91 + nt * 5 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in again[s]] for s in again}
    p = bad['train'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
