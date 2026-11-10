"""Parity selects a paint source; its string route supplies closed loop cavities.

No files, parent modules, examples, or generation state are used by solve.
All diagnostics in self_test are bounded and remain in memory.
"""
from collections import deque
import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return a


def _components(cells, steps=D8):
    left = set(cells)
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group, todo = {root}, [root]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _neighbors(cells, n, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < n and 0 <= c + dc < n} - set(cells)


def _shift(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _key_color(block, bg):
    colors = {int(x) for x in block.flat if x != bg}
    if len(colors) != 1:
        raise ValueError("Each corner swatch must have exactly one non-background color")
    return colors.pop()


def _analyze(input_grid):
    """Reconstruct cavities, cups, and the wire/contour graph from input only."""
    a = _grid(input_grid)
    n = a.shape[0]
    if a.shape != (n, n) or n < 8:
        raise ValueError("The family requires a square canvas of side at least eight")
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("Background must occupy strictly more than half the canvas")
    left = _key_color(a[:2, :2], bg)
    right = _key_color(a[:2, -2:], bg)
    selected = left if n % 2 else right
    # The two swatches are controls, not route terminals or contour pixels.
    body = a.copy()
    body[:2, :2] = bg
    body[:2, -2:] = bg
    loops = []
    for region in _components(zip(*np.where(body == bg)), D4):
        if any(r in (0, n - 1) or c in (0, n - 1) for r, c in region):
            continue
        shell = _neighbors(region, n, D4)
        colors = {int(body[p]) for p in shell}
        if len(colors) != 1 or bg in colors:
            raise ValueError("A cavity needs one unbroken monochrome shell")
        loops.append((region, shell, colors.pop()))
    shells = set().union(*(shell for _, shell, _ in loops))
    sources = []
    for color in range(10):
        if color == bg or np.count_nonzero(body == color) != 1:
            continue
        for group in _components(zip(*np.where(body == color))):
            if len(group) != 1 or group & shells:
                continue
            p = next(iter(group))
            around = {q for q in _neighbors(group, n) if body[q] != bg}
            inks = {int(body[q]) for q in around}
            if len(around) in (3, 5) and len(inks) == 1:
                sources.append((p, color, inks.pop()))
    markers = {p for p, _, _ in sources}
    wires = []
    for color in range(10):
        if color != bg:
            cells = set(zip(*np.where(body == color))) - shells - markers
            wires.extend((group, color) for group in _components(cells))
    k = len(loops)
    graph = [set() for _ in range(k + len(wires))]
    # Contours are nodes, not conduits between merely touching contours.
    for i, (_, shell, _) in enumerate(loops):
        near = _neighbors(shell, n)
        for j, (wire, _) in enumerate(wires, k):
            if near & wire:
                graph[i].add(j)
                graph[j].add(i)
    owners = [set() for _ in loops]
    for s, (p, color, ink) in enumerate(sources):
        near = _neighbors({p}, n)
        todo = [k + j for j, (wire, col) in enumerate(wires) if col == ink and wire & near]
        seen = set(todo)
        for node in todo:
            for nxt in sorted(graph[node] - seen):
                seen.add(nxt)
                todo.append(nxt)
        for i in seen:
            if i < k:
                owners[i].add(s)
    if any(len(owner) > 1 for owner in owners):
        raise ValueError("A contour cannot be supplied by multiple terminals")
    return dict(bg=bg, left=left, right=right, selected=selected, loops=loops,
                sources=sources, wires=wires, graph=graph, owners=owners)


def _paint(a, info, color):
    out = np.array(a, copy=True)
    for (region, _, _), owners in zip(info['loops'], info['owners']):
        if owners and info['sources'][next(iter(owners))][1] == color:
            for p in region:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Keep the whole canvas, filling only cavities supplied by the selected color."""
    a = _grid(input_grid)
    info = _analyze(a)
    return _paint(a, info, info['selected'])


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _contour(rng, small):
    if small == 2:
        inner = {(0, c) for c in range(_ri(rng, 1, 2))}
        shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
        return _shift(inner, 1, 1), _shift(shell, 1, 1)
    h, w = _ri(rng, 1, 2 if small else 4), _ri(rng, 1, 3 if small else 5)
    mode = _ri(rng, 0, 5)
    if mode == 0:  # independently sized rectangle
        inner = {(r, c) for r in range(h) for c in range(w)}
    elif mode == 1:
        radius = 1 if small else _ri(rng, 1, 2)
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    elif mode == 2:  # concave elbow
        h, w = max(2, h), max(2, w)
        inner = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif mode == 3:  # drifting, four-connected strip
        inner = {(r, c) for r in range(h + 1) for c in range(r, r + max(2, w))}
    elif mode == 4:  # asymmetric terrace
        inner = {(r, c) for r in range(h + 1) for c in range(w + r // 2)}
    else:
        inner = {(0, c) for c in range(_ri(rng, 2, 4 if small else 8))}
    if rng.random() < .5:
        inner = {(c, r) for r, c in inner}
    if rng.random() < .5:
        inner = {(r, -c) for r, c in inner}
    shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
    y, x = min(r for r, c in shell), min(c for r, c in shell)
    return _shift(inner, -y, -x), _shift(shell, -y, -x)


def _cup(style, rotation):
    if style == 3:
        source, wire = {(0, 0)}, {(0, 1), (1, 0), (1, 1)}
    else:
        source, wire = {(0, 1)}, {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
    for _ in range(rotation):
        source = {(-c, r) for r, c in source}
        wire = {(-c, r) for r, c in wire}
    y, x = min(r for r, c in source | wire), min(c for r, c in source | wire)
    return _shift(source, -y, -x), _shift(wire, -y, -x)


def _path(rng, starts, goals, allowed):
    todo = sorted(starts)
    rng.shuffle(todo)
    queue = deque(todo)
    prev = {p: None for p in todo}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    while queue:
        p = queue.popleft()
        if p in goals:
            route = []
            while p is not None:
                route.append(p)
                p = prev[p]
            return set(route)
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in prev:
                prev[q] = p
                queue.append(q)
    return None


def _scene(rng, n, palette, style, loop_counts, detached, selected_owner, small):
    bg, ink, edge = palette[:3]
    paints = palette[3:3 + len(loop_counts)]
    canvas = {(r, c) for r in range(3, n - 1) for c in range(1, n - 1)}
    # Randomly oriented spatial neighborhoods shorten competing routes without
    # putting every scene in a fixed row/column strip layout. Routing itself
    # remains free over the canvas, and route bounds can overlap.
    phase = float(rng.uniform(0, 2 * np.pi))
    hubs = [(n * .55 + n * .25 * np.sin(phase + 2 * np.pi * j / len(loop_counts)),
             n * .5 + n * .25 * np.cos(phase + 2 * np.pi * j / len(loop_counts)))
            for j in range(len(loop_counts))]
    def territory(r, c):
        return min(range(len(hubs)), key=lambda j: (r-hubs[j][0])**2 + (c-hubs[j][1])**2)
    occupied, regions = set(), []
    owners = [i for i, count in enumerate(loop_counts) for _ in range(count)] + [None] * detached
    rng.shuffle(owners)
    for owner in owners:
        inner, shell = _contour(rng, small)
        height = max(r for r, c in shell) + 1
        width = max(c for r, c in shell) + 1
        placed = False
        if height > n - 4 or width > n - 2:
            return None
        for _ in range(70):
            r, c = _ri(rng, 3, n - height - 1), _ri(rng, 1, n - width - 1)
            if owner is not None and territory(r + height / 2, c + width / 2) != owner:
                continue
            ins, rim = _shift(inner, r, c), _shift(shell, r, c)
            cells = ins | rim
            if cells & occupied or _neighbors(cells, n) & occupied:
                continue
            occupied |= cells
            regions.append((ins, rim, owner))
            placed = True
            break
        if not placed:
            return None
    wire_sets, source_sets = [], []
    for owner in range(len(loop_counts)):
        source, cup = _cup(style, _ri(rng, 0, 3))
        height = max(r for r, c in source | cup) + 1
        width = max(c for r, c in source | cup) + 1
        placed = False
        for _ in range(100):
            r, c = _ri(rng, 3, n - height - 1), _ri(rng, 1, n - width - 1)
            if territory(r + height / 2, c + width / 2) != owner:
                continue
            src, wire = _shift(source, r, c), _shift(cup, r, c)
            if (src | wire) & occupied or _neighbors(src | wire, n) & occupied:
                continue
            source_sets.append(src)
            wire_sets.append(wire)
            occupied |= src | wire
            placed = True
            break
        if not placed:
            return None
    all_loops = set().union(*(ins | rim for ins, rim, _ in regions))
    all_sources = set().union(*source_sets)
    for owner in rng.permutation(len(loop_counts)):
        owner = int(owner)
        own = wire_sets[owner]
        others = set().union(*(wire_sets[j] for j in range(len(loop_counts)) if j != owner))
        targets = [i for i, (_, _, who) in enumerate(regions) if who == owner]
        rng.shuffle(targets)
        previous = None
        serial = rng.random() < .5
        for i in targets:
            ins, rim, _ = regions[i]
            other_loops = all_loops - ins - rim
            if previous is not None and serial:
                other_loops -= regions[previous][0] | regions[previous][1]
            blocked = (all_loops | _neighbors(other_loops, n) | all_sources
                       | _neighbors(all_sources, n) | others | _neighbors(others, n))
            starts = regions[previous][1] if previous is not None and serial else own
            allowed = (canvas - blocked) | own | starts
            goals = (_neighbors(rim, n) & canvas) - blocked
            if not goals:
                return None
            # Vary the attachment side independently of the source's position.
            axis, high = _ri(rng, 0, 1), _ri(rng, 0, 1)
            extreme = (max if high else min)(p[axis] for p in goals)
            preferred = {p for p in goals if p[axis] == extreme}
            path = _path(rng, starts, preferred, allowed)
            if path is None:
                path = _path(rng, starts, goals, allowed)
            if path is None:
                return None
            own |= path - all_loops
            previous = i
    a = np.full((n, n), bg, dtype=np.int8)
    for _, rim, _ in regions:
        for p in rim:
            a[p] = edge
    for own in wire_sets:
        for p in own:
            a[p] = ink
    for owner, src in enumerate(source_sets):
        for p in src:
            a[p] = paints[owner]
    selected = paints[selected_owner]
    decoy = paints[(selected_owner + _ri(rng, 1, len(paints) - 1)) % len(paints)]
    corner_colors = (selected, decoy) if n % 2 else (decoy, selected)
    for start, color in zip((0, n - 2), corner_colors):
        positions = rng.choice(4, size=_ri(rng, 1, 4), replace=False)
        for pos in positions:
            a[int(pos) // 2, start + int(pos) % 2] = color
    b = a.copy()
    for inner, _, owner in regions:
        if owner == selected_owner:
            for p in inner:
                b[p] = selected
    try:
        info = _analyze(a)
        if (len(info['loops']) != len(regions) or len(info['sources']) != len(paints)
                or sorted(len(x) for x in info['owners']) != sorted(0 if o is None else 1 for o in owners)
                or not np.array_equal(_paint(a, info, selected), b)):
            return None
        # Check every route, including the unselected ones, against construction.
        for owner, paint in enumerate(paints):
            expected = set().union(*(ins for ins, _, who in regions if who == owner))
            actual = set().union(*(ins for (ins, _, _), who in zip(info['loops'], info['owners'])
                                  if who and info['sources'][next(iter(who))][1] == paint))
            if actual != expected:
                return None
    except ValueError:
        return None
    return {'input': a, 'output': b}


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _fallback(rng, n, palette, style, counts, detached, selected_owner):
    """A compact, rotated serial construction if free routing exhausts its budget.

    This changes only the sampled layout, never the rule, source/loop counts,
    or demonstration requirements. All cavities have one cell in this fallback.
    """
    bg, ink, edge = palette[:3]
    paints = palette[3:3 + len(counts)]
    side = n - 4
    a = np.full((side, side), bg, dtype=np.int8)
    b = a.copy()
    order = [int(x) for x in rng.permutation(len(counts))]
    for slot, owner in enumerate(order):
        r = 4 * slot
        a[r + 1, 0] = paints[owner]
        a[r + 1, 1] = ink
        a[r + 2, :2] = ink
        if style == 5:
            a[r, :2] = ink
        last = 5 + 4 * (counts[owner] - 1)
        a[r + 1, 2:last + 2] = ink
        for j in range(counts[owner]):
            center = r + 1, 5 + 4 * j
            a[center] = bg
            for dr, dc in D4:
                a[center[0] + dr, center[1] + dc] = edge
    if detached:
        center = side - 2, side - 2
        for dr, dc in D4:
            a[center[0] + dr, center[1] + dc] = edge
    b[:] = a
    for slot, owner in enumerate(order):
        if owner == selected_owner:
            for j in range(counts[owner]):
                b[4 * slot + 1, 5 + 4 * j] = paints[owner]
    rotation = _ri(rng, 0, 3)
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    full = np.full((n, n), bg, dtype=np.int8)
    out = full.copy()
    full[3:n - 1, 1:n - 3], out[3:n - 1, 1:n - 3] = a, b
    selected = paints[selected_owner]
    other = paints[(selected_owner + 1) % len(paints)]
    colors = (selected, other) if n % 2 else (other, selected)
    for start, color in zip((0, n - 2), colors):
        for pos in rng.choice(4, size=_ri(rng, 1, 4), replace=False):
            full[int(pos) // 2, start + int(pos) % 2] = color
    out[:2] = full[:2]
    _require(np.array_equal(solve(full), out), 'Fallback ownership')
    return {'input': full, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("Require integer counts: train 2..10, test 1..4, train > test")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    style = int(rng.choice([3, 5]))
    parity = _ri(rng, 0, 1)
    parities = [parity, 1 - parity, 1 - parity, parity]
    sizes = []
    for i in range(num_train):
        p = parities[i % 4]
        choices = [n for n in range(18, 25) if n % 2 == p and n not in sizes]
        if not choices:
            choices = [n for n in range(18, 25) if n % 2 == p]
        sizes.append(int(rng.choice(choices)))
    test_parity = _ri(rng, 0, 1)
    test_sizes = [int(rng.choice([n for n in range(25, 31) if n % 2 == (test_parity + i) % 2]))
                  for i in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, sides in (('train', sizes), ('test', test_sizes)):
        for i, n in enumerate(sides):
            source_count = 2 if split == 'train' and (i < 2 or n <= 20) else _ri(rng, 2, 3)
            selected_owner = i % 2 if split == 'train' else _ri(rng, 0, source_count - 1)
            loop_counts = [_ri(rng, 1, 2) for _ in range(source_count)]
            loop_counts[selected_owner] = 1 + (i % 2) if split == 'train' else _ri(rng, 2, 3)
            detached = _ri(rng, 0, 1) if i >= 2 or split == 'test' else 0
            if split == 'train' and n <= 20:
                loop_counts[1 - selected_owner] = 1
                detached = 0
            # Background, ink, and shell roles remain visible; their literal
            # values can change between examples without changing the rule.
            local = palette.copy()
            if rng.random() < .5:
                local[0], local[6] = local[6], local[0]
            if rng.random() < .5:
                local[1], local[2] = local[2], local[1]
            pair = None
            for attempt in range(100):
                candidate = _scene(rng, n, local, style, loop_counts, detached,
                                   selected_owner, small=2 if attempt >= 60 else int(attempt >= 15 or n <= 20))
                if candidate is not None and _signature(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                for _ in range(32):
                    candidate = _fallback(rng, n, local, style, loop_counts, detached, selected_owner)
                    if _signature(candidate['input']) not in seen:
                        pair = candidate
                        break
                if pair is None:
                    raise RuntimeError("Bounded uniqueness attempts exhausted")
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    # Shuffle only after the two parity witnesses have been constructed.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'Split schema')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Pair counts')
    seen, training = set(), []
    degrees = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Canvas dimensions must be preserved')
            info = _analyze(a)
            bg, n = info['bg'], a.shape[0]
            _require(info['left'] != info['right'], 'Corner colors must differ')
            _require(np.all(a[:2, 2:-2] == bg), 'Only corner controls belong in the top two rows')
            _require(np.all(a[2] == bg), 'Swatches need an empty moat')
            _require(np.all(a[3:, 0] == bg) and np.all(a[3:, -1] == bg)
                     and np.all(a[-1] == bg), 'Body must retain its exterior border')
            _require(len(info['sources']) in (2, 3), 'Need competing terminals')
            paints = [color for _, color, _ in info['sources']]
            _require(len(set(paints)) == len(paints), 'Terminal colors must be distinct')
            _require({info['left'], info['right']} <= set(paints), 'Each corner must key a source')
            _require(all(any(owner == {j} for owner in info['owners'])
                         for j in range(len(paints))), 'Every terminal needs a cavity')
            source_degrees = {sum(a[q] == ink for q in _neighbors({p}, n))
                              for p, _, ink in info['sources']}
            _require(len(source_degrees) == 1 and source_degrees <= {3, 5}, 'Cup geometry')
            if split == 'train':
                degrees |= source_degrees
            else:
                _require(source_degrees <= degrees, 'Untaught cup style')
            expected = _paint(a, info, info['selected'])
            _require(np.array_equal(b, expected), 'Input-derived cavity ownership disagrees')
            changed = a != b
            _require(np.any(changed) and np.all(a[changed] == bg)
                     and np.all(b[changed] == info['selected']), 'Only selected cavities change')
            decoy = info['right'] if n % 2 else info['left']
            _require(not np.array_equal(b, _paint(a, info, decoy)), 'Crop ablation has no effect')
            all_paint = a.copy()
            for (inner, _, _), owner in zip(info['loops'], info['owners']):
                if owner:
                    color = paints[next(iter(owner))]
                    for p in inner:
                        all_paint[p] = color
            _require(not np.array_equal(b, all_paint), 'Ignoring selection has no effect')
            key = _signature(a)
            _require(key not in seen, 'Duplicate input')
            seen.add(key)
            if split == 'train':
                training.append((n, info, int(changed.sum())))
    _require(len(degrees) == 1, 'Cup convention must be consistent within a puzzle')
    _require({n % 2 for n, _, _ in training} == {0, 1}, 'Both parity branches need demonstrations')
    _require(len({info['selected'] for _, info, _ in training}) >= 2, 'Demonstrate two paint colors')
    _require(any(sum(bool(owner) and info['sources'][next(iter(owner))][1] == info['selected']
                         for owner in info['owners']) >= 2 for _, info, _ in training),
             'Demonstrate filling every supplied loop')
    if nt >= 4:
        _require(all(len({n for n, _, _ in training if n % 2 == p}) >= 2 for p in (0, 1)),
                 'Four demonstrations need two sizes of each parity')


def _parity_twin(a, info):
    """Change only canvas parity and corner placement; preserve the body."""
    n = a.shape[0]
    side = n + 1 if n < 30 else n - 1
    twin = np.full((side, side), info['bg'], dtype=a.dtype)
    k = min(side, n)
    twin[:k, :k] = a[:k, :k]
    twin[:2] = info['bg']
    twin[:2, :2] = a[:2, :2]
    twin[:2, -2:] = a[:2, -2:]
    return twin


def self_test():
    """42 streaming seeds; geometry, causal ablations, count bounds, and mutation tests."""
    stats = dict(puzzles=0, pairs=0, parity_flips=0)
    sizes, normalized, shapes, backgrounds, paints, styles, source_counts = (set() for _ in range(7))
    areas, loop_counts, test_parities, fingerprints = set(), set(), set(), set()
    detached = serial = branches = diagonal = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                before = a.copy()
                _require(np.array_equal(solve(a), b) and np.array_equal(a, before), 'Pure solve')
                info = _analyze(a)
                sizes.add(a.shape[0]); backgrounds.add(info['bg']); paints.add(info['selected'])
                source_counts.add(len(info['sources'])); loop_counts.add(len(info['loops']))
                digest.update(bytes(a.shape)); digest.update(a.tobytes()); digest.update(b.tobytes())
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add(hashlib.sha256(bytes(a.shape) + norm).hexdigest())
                for inner, _, _ in info['loops']:
                    y, x = min(r for r, c in inner), min(c for r, c in inner)
                    shapes.add(tuple(sorted((r-y, c-x) for r, c in inner)))
                    areas.add(len(inner))
                detached += sum(not owner for owner in info['owners'])
                k = len(info['loops'])
                serial += sum(len(info['graph'][i]) >= 2 for i in range(k))
                branches += sum(len(neighbors) >= 3 for neighbors in info['graph'][k:])
                for wire, _ in info['wires']:
                    diagonal += any((r+1, c+1) in wire and (r+1, c) not in wire
                                    and (r, c+1) not in wire for r, c in wire)
                styles.update(int(sum(a[q] == ink for q in _neighbors({p}, a.shape[0])))
                              for p, _, ink in info['sources'])
                if split == 'test':
                    test_parities.add(a.shape[0] % 2)
                twin = _parity_twin(a, info)
                other = info['right'] if a.shape[0] % 2 else info['left']
                twin_expected = _paint(twin, info, other)
                _require(np.array_equal(solve(twin), twin_expected), 'Canvas-parity counterfactual')
                _require(not np.array_equal(solve(twin)[3:a.shape[0]-1, 1:a.shape[0]-1],
                                           b[3:a.shape[0]-1, 1:a.shape[0]-1]), 'Parity must affect body')
                # Hold corner control and geometry fixed but swap the two
                # competing terminal colors. Ownership must follow the cups.
                swapped = a.copy()
                for p, color, _ in info['sources']:
                    if color in (info['selected'], other):
                        swapped[p] = other if color == info['selected'] else info['selected']
                expected = swapped.copy()
                for (inner, _, _), owner in zip(info['loops'], info['owners']):
                    if owner and info['sources'][next(iter(owner))][1] == other:
                        for p in inner:
                            expected[p] = info['selected']
                _require(np.array_equal(solve(swapped), expected), 'Terminal-to-route counterfactual')
                _require(not np.array_equal(swapped != expected, a != b), 'Terminal ownership must matter')
                stats['pairs'] += 1
                stats['parity_flips'] += 1
        fingerprint = digest.hexdigest()
        _require(fingerprint not in fingerprints, 'Repeated puzzle across seeds')
        fingerprints.add(fingerprint)
        stats['puzzles'] += 1
    _require(len(sizes) >= 10 and len(normalized) >= 200 and len(shapes) >= 35, 'Structural variation')
    _require(len(backgrounds) >= 8 and len(paints) >= 8 and styles == {3, 5}, 'Palette and cup variation')
    _require(source_counts == {2, 3} and len(loop_counts) >= 4 and len(areas) >= 12, 'Count and cavity variation')
    _require(test_parities == {0, 1} and detached and serial and branches and diagonal, 'Mechanism coverage')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, 'Explicit counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    original, repeat = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[role], y[role]) for s in original
                 for x, y in zip(original[s], repeat[s]) for role in ('input', 'output')), 'Determinism')
    bad = {s: [{role: a.copy() for role, a in p.items()} for p in original[s]] for s in original}
    pair = bad['test'][0]
    r, c = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][r, c] = pair['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial fill was accepted')
    return dict(stats, sizes=sorted(sizes), normalized_inputs=len(normalized), cavity_shapes=len(shapes),
                backgrounds=sorted(backgrounds), paint_colors=sorted(paints), cup_styles=sorted(styles),
                source_counts=sorted(source_counts), loop_counts=sorted(loop_counts), cavity_areas=sorted(areas),
                detached_cavities=detached, serial_contours=serial, branch_nodes=branches,
                wires_with_diagonal_steps=diagonal)
