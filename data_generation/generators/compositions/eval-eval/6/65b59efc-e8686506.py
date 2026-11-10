"""Complement-seam assembly: infer missing stamp codes by exact fine-piece fit.

The solver reads only the framed library, coarse map and loose inventory.  The
constructor's completed map and placements are never supplied to inference.
"""
from itertools import product
from functools import lru_cache
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "Invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(cells):
    cells = set(cells)
    groups = []
    while cells:
        start = min(cells)
        cells.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in cells:
                    cells.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _frame_width(a, r, c, frame):
    end = c
    while end < a.shape[1] and a[r, end] == frame:
        end += 1
    return end-c


def _parse(input_grid):
    a = _grid(input_grid)
    _require(min(a.shape) >= 12, "Input is too small for the framed encoding")
    bg, frame = int(a[0, 0]), int(a[1, 1])
    _require(bg != frame, "Missing frame")
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])), "Missing background border")
    width = _frame_width(a, 1, 1, frame)
    lines = [r for r in range(2, min(9, a.shape[0])) if np.all(a[r, 1:1+width] == frame)]
    _require(bool(lines), "Missing library divider")
    n = lines[0]-2
    _require(2 <= n <= 5 and (width-1) % (n+1) == 0, "Invalid library scale")
    k = (width-1)//(n+1)
    _require(2 <= k <= 4 and 1+width < a.shape[1], "Invalid library count")
    bottom = n+4
    _require(np.all(a[bottom, 1:1+width] == frame), "Missing library footer")
    labels, templates, targets = [], {}, {}
    expected = np.full((n+4, width), bg, dtype=np.int8)
    expected[[0, n+1, n+3], :] = frame
    for j in range(k+1):
        expected[:, j*(n+1)] = frame
    for j in range(k):
        c = 2+j*(n+1)
        panel = a[2:2+n, c:c+n]
        colors = set(map(int, np.unique(panel))) - {bg}
        _require(len(colors) == 1, "Stamp must have one source identity")
        label = colors.pop()
        _require(label != frame and label not in labels, "Duplicate or structural source identity")
        mask = panel == label
        _require(mask.any() and not mask.all(), "Stamp needs foreground and holes")
        labels.append(label)
        templates[label] = mask
        swatch = a[n+3, c:c+n]
        _require(swatch[n//2] != bg, "Invisible key")
        target = int(swatch[n//2])
        targets[label] = target
        expected[1:n+1, c-1:c-1+n] = panel
        expected[n+2, c-1+n//2] = target
    _require(np.array_equal(a[1:bottom+1, 1:1+width], expected), "Malformed library")
    areas = {int((~m).sum()) for m in templates.values()}
    _require(len(areas) == 1, "The family uses equal-area hole masks")
    _require(len({m.tobytes() for m in templates.values()}) == k, "Stamp geometries must differ")

    mr = n+6
    mw = _frame_width(a, mr, 1, frame)-2
    _require(2 <= mw <= 6, "Missing coarse-map frame")
    ends = [r for r in range(mr+2, min(mr+8, a.shape[0]-1))
            if np.all(a[r, 1:mw+3] == frame)]
    _require(bool(ends), "Missing coarse-map bottom")
    end = ends[0]
    mh = end-mr-1
    _require(2 <= mh <= 6 and mh*n <= 30 and mw*n <= 30, "Invalid board dimensions")
    _require(np.all(a[mr:end+1, 1] == frame) and np.all(a[mr:end+1, mw+2] == frame), "Broken map frame")
    macro = a[mr+1:end, 2:mw+2].copy()
    _require(set(map(int, np.unique(macro))) <= {bg, *labels}, "Unknown map identity")
    unknown = list(map(tuple, np.argwhere(macro == bg)))
    _require(1 <= len(unknown) <= 4, "Need one to four missing map labels")
    outside = a != bg
    outside[1:bottom+1, 1:1+width] = False
    outside[mr:end+1, 1:mw+3] = False
    pieces, exterior = [], _components(map(tuple, np.argwhere(outside)))
    for cells in exterior:
        colors = {int(a[p]) for p in cells}
        _require(len(colors) == 1, "Loose pieces must be separated and monochrome")
        color = colors.pop()
        _require(color not in {bg, frame, *labels, *targets.values()}, "Inventory color overlaps another role")
        pieces.append((color, _normalize(cells)))
    _require(2 <= len(pieces) <= 16, "Invalid inventory count")
    _require(sum(len(s) for color, s in pieces) == mh*mw*next(iter(areas)), "Inventory area mismatch")
    return dict(bg=bg, frame=frame, n=n, k=k, labels=labels,
                templates=templates, targets=targets, macro=macro,
                unknown=unknown, pieces=pieces, exterior=exterior,
                map_origin=(mr+1, 2), library_width=width)


def _expand(p, macro, use_key=True, turn=0):
    n, bg = p['n'], p['bg']
    h, w = macro.shape
    out = np.full((h*n, w*n), bg, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            label = int(macro[r, c])
            mask = np.rot90(p['templates'][label], turn)
            color = p['targets'][label] if use_key else label
            out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(mask, color, bg)
    return out


class _SearchLimit(ValueError):
    pass


def _fill(board, bg, pieces, node_limit=6000):
    """Enumerate up to two distinct colored exact covers, counting each item.

    Identical pieces are multiplicities. Different decompositions with identical
    pixel colors count as one answer. No rotations or reflections are proposed.
    """
    holes = sorted(map(tuple, np.argwhere(board == bg)))
    if len(holes) != sum(len(s) for color, s in pieces):
        return [], 0, []
    ids = {p: i for i, p in enumerate(holes)}
    inventory = {}
    for item in pieces:
        inventory[item] = inventory.get(item, 0)+1
    types = sorted(inventory)
    by_cell = [[] for p in holes]
    initial_counts = []
    for t, (color, shape) in enumerate(types):
        sh = max(r for r, c in shape)+1
        sw = max(c for r, c in shape)+1
        moves = []
        for r in range(board.shape[0]-sh+1):
            for c in range(board.shape[1]-sw+1):
                cells = [(r+dr, c+dc) for dr, dc in shape]
                if not all(cell in ids for cell in cells):
                    continue
                indices = [ids[cell] for cell in cells]
                bits = sum(1 << i for i in indices)
                paint = sum((color+1) << (4*i) for i in indices)
                move = (t, bits, paint, r, c)
                moves.append(move)
                for i in indices:
                    by_cell[i].append(move)
        initial_counts.append(len(moves))
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise _SearchLimit("Exact-cover proof exceeded its node bound")
        if not remaining:
            return ((0, ()),) if not any(counts) else ()
        choices = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            options = [m for m in by_cell[i] if counts[m[0]] and (m[1] & remaining) == m[1]]
            if not options:
                return ()
            if choices is None or len(options) < len(choices):
                choices = options
                if len(choices) == 1:
                    break
            bits ^= bit
        answers = {}
        for t, mask, paint, r, c in choices:
            rest = list(counts)
            rest[t] -= 1
            for suffix, path in search(remaining ^ mask, tuple(rest)):
                code = paint | suffix
                answers.setdefault(code, ((t, r, c),)+path)
                if len(answers) >= 2:
                    return tuple(answers.items())
        return tuple(answers.items())

    answers = []
    for code, path in search((1 << len(holes))-1, tuple(inventory[t] for t in types)):
        out = board.copy()
        for i, cell in enumerate(holes):
            out[cell] = ((code >> (4*i)) & 15)-1
        placements = [(types[t][0], types[t][1], r, c) for t, r, c in path]
        answers.append((out, placements))
    return answers, nodes, initial_counts


def _infer(input_grid):
    p = _parse(input_grid)
    feasible = []
    nodes = 0
    # This search does not consult the construction's true map. In particular,
    # ALL equally sized stamp alternatives are considered for every blank.
    for values in product(p['labels'], repeat=len(p['unknown'])):
        macro = p['macro'].copy()
        for cell, label in zip(p['unknown'], values):
            macro[cell] = label
        fixed = _expand(p, macro)
        answers, count, choices = _fill(fixed, p['bg'], p['pieces'])
        nodes += count
        if len(answers) > 1:
            raise ValueError("Inventory permits multiple colored completions")
        if answers:
            feasible.append((macro, fixed, answers[0][0], answers[0][1], choices))
            if len(feasible) > 1:
                raise ValueError("Missing map labels are ambiguous")
    _require(len(feasible) == 1, "No exact inventory completion")
    macro, fixed, output, placements, choices = feasible[0]
    p.update(completed_macro=macro, fixed=fixed, output=output,
             placements=placements, initial_choices=choices, solver_nodes=nodes)
    return p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique completed fine board using only the supplied input."""
    return _infer(input_grid)['output'].copy()


def _hole_masks(rng, n, k):
    q = int(rng.integers(n+1, min(2*n, n*n-1)+1))
    # A random edge-to-edge channel, with branches, makes real seam crossings.
    row = int(rng.integers(n))
    first = {(row, c) for c in range(n)}
    masks, signatures = [], set()
    for j in range(k):
        for attempt in range(100):
            cells = first.copy() if j == 0 else {(int(rng.integers(n)), int(rng.integers(n)))}
            while len(cells) < q:
                neighbors = sorted({(r+dr, c+dc) for r, c in cells
                                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                                    if 0 <= r+dr < n and 0 <= c+dc < n}-cells)
                cells.add(neighbors[int(rng.integers(len(neighbors)))])
            mask = np.zeros((n, n), dtype=bool)
            for cell in cells:
                mask[cell] = True
            if mask.tobytes() not in signatures:
                masks.append(mask)
                signatures.add(mask.tobytes())
                break
        else:
            raise ValueError("Could not obtain distinct motifs")
    return masks


def _partition(rng, holes):
    remaining = set(holes)
    pieces = []
    max_area = int(rng.integers(6, 14))
    while remaining:
        comp = max(_components(remaining), key=lambda x: (len(x), _normalize(x)))
        # Some complete components coexist with split, interlocking channels.
        if len(comp) <= max_area and rng.random() < .45:
            cells = comp
        else:
            target = min(len(comp), int(rng.integers(2, max_area+1)))
            ordered = sorted(comp)
            cells = {ordered[int(rng.integers(len(ordered)))]}
            while len(cells) < target:
                options = sorted({(r+dr, c+dc) for r, c in cells
                                  for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} & (comp-cells))
                cells.add(options[int(rng.integers(len(options)))])
        pieces.append(cells)
        remaining -= cells
    return pieces


def _build_input(rng, p, macro, pieces):
    n, k, bg, frame = p['n'], p['k'], p['bg'], p['frame']
    width = k*(n+1)+1
    for attempt in range(5):
        h = int(rng.integers(23, 31)) if attempt < 4 else 30
        w = int(rng.integers(max(23, width+2), 31)) if attempt < 4 else 30
        a = np.full((h, w), bg, dtype=np.int8)
        a[1, 1:width+1] = a[n+2, 1:width+1] = a[n+4, 1:width+1] = frame
        for j in range(k+1):
            a[1:n+5, 1+j*(n+1)] = frame
        order = [int(x) for x in rng.permutation(p['labels'])]
        for j, label in enumerate(order):
            c = 2+j*(n+1)
            a[2:n+2, c:c+n] = np.where(p['templates'][label], label, bg)
            a[n+3, c+n//2] = p['targets'][label]
        mr = n+6
        mh, mw = macro.shape
        a[mr, 1:mw+3] = a[mr+mh+1, 1:mw+3] = frame
        a[mr:mr+mh+2, 1] = a[mr:mr+mh+2, mw+2] = frame
        a[mr+1:mr+mh+1, 2:mw+2] = macro
        # The tray is unframed. Its empty space is not another macro panel.
        blocked = np.ones(a.shape, dtype=bool)
        blocked[mr:h-1, mw+5:w-1] = False
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), i))
        for i in order:
            color, shape = pieces[i]
            sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            candidates = [(r, c) for r in range(mr, h-sh) for c in range(mw+5, w-sw)
                          if all(not blocked[r+dr, c+dc] for dr, dc in shape)]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            for dr, dc in shape:
                rr, cc = r+dr, c+dc
                a[rr, cc] = color
                for ar, ac in ((rr, cc), (rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    blocked[ar, ac] = True
        else:
            return a
    return None


def _metrics(p):
    n, bg = p['n'], p['bg']
    cross = sum(len({((r+dr)//n, (c+dc)//n) for dr, dc in shape}) > 1
                for color, shape, r, c in p['placements'])
    gaps = _components(map(tuple, np.argwhere(p['fixed'] == bg)))
    shared = sum(len({int(p['output'][cell]) for cell in gap}) > 1 for gap in gaps)
    repetitions = len(p['pieces'])-len(set(p['pieces']))
    return dict(n=n, k=p['k'], macro_shape=list(p['macro'].shape),
                unknowns=len(p['unknown']), pieces=len(p['pieces']),
                gap_components=len(gaps), shared_gaps=shared, crossing_pieces=cross,
                repeated_items=repetitions, nodes=p['solver_nodes'],
                max_initial_choices=max(p['initial_choices'], default=0),
                singleton=any(len(shape) == 1 for color, shape in p['pieces']),
                nonfirst_unknown=any(p['completed_macro'][cell] != p['labels'][0]
                                     for cell in p['unknown']))


def _make_pair(rng, unknowns, force_shared=False):
    for attempt in range(240):
        n = int(rng.choice([3, 4]))
        k = int(rng.choice([2, 3, 4]))
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        if h*w < k+1:
            continue
        palette = [int(x) for x in rng.permutation(10)]
        bg, frame = palette[:2]
        labels = palette[2:2+k]
        payload = palette[2+k:4+k]
        target_pool = [x for x in range(10) if x not in {bg, *payload}]
        targets = [int(x) for x in rng.choice(target_pool, k, replace=False)]
        if sum(a != b for a, b in zip(labels, targets)) < 2:
            continue
        masks = _hole_masks(rng, n, k)
        p = dict(n=n, k=k, bg=bg, frame=frame, labels=labels,
                 templates={x: ~m for x, m in zip(labels, masks)},
                 targets=dict(zip(labels, targets)))
        macro = np.array(rng.choice(labels, size=(h, w)), dtype=np.int8)
        macro[0, :2] = labels[0]
        # Exercise every template somewhere; identities are not tied to columns.
        available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in ((0, 0), (0, 1))]
        rng.shuffle(available)
        for cell, label in zip(available, labels[1:]):
            macro[cell] = label
        fixed = _expand(p, macro)
        cells = _partition(rng, map(tuple, np.argwhere(fixed == bg)))
        if not 2 <= len(cells) <= 12:
            continue
        signatures = sorted({_normalize(x) for x in cells})
        if len(signatures) < 2:
            continue
        colors = {s: payload[int(rng.integers(2))] for s in signatures}
        if len(set(colors.values())) < 2:
            colors[signatures[0]], colors[signatures[-1]] = payload
        pieces = [(colors[_normalize(x)], _normalize(x)) for x in cells]
        output = fixed.copy()
        for group, (color, shape) in zip(cells, pieces):
            for cell in group:
                output[cell] = color
        visible = macro.copy()
        hidden = rng.choice(h*w, unknowns, replace=False)
        visible.flat[hidden] = bg
        # Two labeled cells are useful anchors, even in the three-blank tests.
        if len(set(map(int, visible[visible != bg]))) < 2:
            continue
        try:
            answers, _, _ = _fill(fixed, bg, pieces)
        except _SearchLimit:
            continue
        if len(answers) != 1 or not np.array_equal(answers[0][0], output):
            continue
        inp = _build_input(rng, p, visible, pieces)
        if inp is None:
            continue
        try:
            inferred = _infer(inp)
        except ValueError:
            continue
        if not np.array_equal(inferred['output'], output):
            continue
        metrics = _metrics(inferred)
        if metrics['crossing_pieces'] < 1 or (force_shared and
                (metrics['shared_gaps'] < 1 or not metrics['singleton']
                 or not metrics['nonfirst_unknown'])):
            continue
        return {'input': inp, 'output': output}
    raise RuntimeError("Bounded complement-seam construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "Seed must be a nonnegative integer")
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _require(nt > ne, "num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    # One and two unknowns are both demonstrated; tests combine three unknowns.
    training_unknowns = [1, 2]+[int(rng.integers(1, 3)) for _ in range(nt-2)]
    rng.shuffle(training_unknowns)
    for split, counts in (('train', training_unknowns), ('test', [3]*ne)):
        for i, u in enumerate(counts):
            for retry in range(8):
                pair = _make_pair(rng, u, force_shared=(split == 'train' and i == 1))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "Invalid pair counts")
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            p = _infer(inp)
            _require(np.array_equal(p['output'], out), "Incorrect unique completion")
            _require(not np.any(out == p['bg']), "Board is not completely filled")
            fixed = p['fixed'] != p['bg']
            _require(np.array_equal(out[fixed], p['fixed'][fixed]), "Keyed scaffold changed")
            for color in {x for x, shape in p['pieces']}:
                expected = sum(len(shape) for x, shape in p['pieces'] if x == color)
                _require(np.count_nonzero(out[~fixed] == color) == expected, "Inventory color area not conserved")
            wrong_key = _expand(p, p['completed_macro'], use_key=False)
            _require(np.any(out[fixed] != wrong_key[fixed]), "No meaningful key use")
            _require(set(map(int, p['completed_macro'].flat)) == set(p['labels']), "Unused library identity")
            m = _metrics(p)
            _require(m['crossing_pieces'] > 0, "Inventory never crosses a stamp seam")
            (training if split == 'train' else testing).append(m)
    _require({m['unknowns'] for m in training} >= {1, 2}, "Missing uncertainty demonstrations")
    _require(any(m['shared_gaps'] for m in training), "No demonstration of multiple colors in one channel")
    _require(any(m['singleton'] for m in training), "No finite-inventory witness against reusable dot filling")
    _require(any(m['nonfirst_unknown'] for m in training), "First-library-stamp shortcut is not refuted")
    _require(all(m['unknowns'] == 3 for m in testing), "Missing three-label test shift")


def self_test():
    """Bounded checks of 42 seeds, actual geometry, counterexamples and errors."""
    shapes, motifs, maps, pieces, palettes = set(), set(), set(), set(), set()
    counts, scales, records = set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _infer(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), "Input-only solve disagrees")
                shapes.add(pair['output'].shape)
                scales.add(p['n'])
                counts.add(p['k'])
                palettes.add((p['bg'], p['frame']))
                motifs.update((p['n'], m.tobytes()) for m in p['templates'].values())
                normal = np.array([[p['labels'].index(int(x)) for x in row] for row in p['completed_macro']], dtype=np.int8)
                maps.add(_key(normal))
                pieces.update(shape for color, shape in p['pieces'])
                records.append(_metrics(p))
                # Omitting B leaves real holes. Omitting A's key changes fixed
                # pixels. Equal hole counts hold for EVERY missing-label choice.
                _require(np.any(p['fixed'] == p['bg']), "Omitting assembly has no effect")
                wrong_key = _expand(p, p['completed_macro'], use_key=False)
                fixed = p['fixed'] != p['bg']
                _require(np.any(wrong_key[fixed] != pair['output'][fixed]), "Key shortcut survived")
                cell = p['unknown'][0]
                alternative = p['completed_macro'].copy()
                alternative[cell] = next(x for x in p['labels'] if x != alternative[cell])
                changed = _expand(p, alternative)
                _require(np.count_nonzero(changed == p['bg']) == np.count_nonzero(p['fixed'] == p['bg']), "Unequal-area stamp shortcut")
                _require(not _fill(changed, p['bg'], p['pieces'])[0], "An incorrect stamp fits the inventory")
                if split == 'train' and any(len(shape) == 1 for color, shape in p['pieces']):
                    dot_color = next(color for color, shape in p['pieces'] if len(shape) == 1)
                    reused = p['fixed'].copy()
                    reused[~fixed] = dot_color
                    _require(not np.array_equal(reused, pair['output']), "Unlimited singleton copying reproduces the answer")
    _require(scales == {3, 4} and counts == {2, 3, 4}, "Scale or identity diversity collapsed")
    _require(len(shapes) >= 12 and len(motifs) >= 90 and len(maps) >= 120
             and len(pieces) >= 100 and len(palettes) >= 35, "Insufficient structural diversity")
    _require(any(m['shared_gaps'] >= 2 for m in records), "No richer shared channels")
    _require(any(m['repeated_items'] for m in records), "No repeated inventory items")
    _require(any(m['max_initial_choices'] > 1 for m in records), "No globally constrained placements")
    first = generate_puzzle(991, 2, 1)
    generate_puzzle(992, 2, 1)
    repeated = generate_puzzle(991, 2, 1)
    _require(all(np.array_equal(a[g], b[g]) for s in first for a, b in zip(first[s], repeated[s])
                 for g in ('input', 'output')), "Determinism failed after another call")
    for nt, ne in ((2, 1), (10, 4)):
        puzzle = generate_puzzle(123+nt, nt, ne)
        _require(len(puzzle['train']) == nt and len(puzzle['test']) == ne, "Counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Identical-item swaps are harmless; different-color swaps are ambiguous.
    slots = np.array([[0, 9, 0]], dtype=np.int8)
    _require(len(_fill(slots, 0, [(1, ((0, 0),)), (2, ((0, 0),))])[0]) == 2, "Ambiguous color swap missed")
    _require(len(_fill(slots, 0, [(1, ((0, 0),)), (1, ((0, 0),))])[0]) == 1, "Identical items treated as different answers")
    tee = np.array([[0, 0, 0], [9, 0, 9]], dtype=np.int8)
    _require(not _fill(tee, 0, [(1, ((0, 1), (1, 0), (1, 1), (1, 2)))])[0], "Piece illegally reflected")
    for kind in ('output', 'inventory'):
        bad = {s: [{g: a.copy() for g, a in pair.items()} for pair in first[s]] for s in first}
        pair = bad['test'][0]
        if kind == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        else:
            p = _parse(pair['input'])
            pair['input'][min(p['exterior'][0])] = p['bg']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid completion accepted")
    return dict(seeds=42, pairs=len(records), output_shapes=len(shapes),
                motifs=len(motifs), macro_maps=len(maps), piece_shapes=len(pieces),
                palettes=len(palettes), crossing_pairs=sum(m['crossing_pieces'] > 0 for m in records),
                shared_gap_pairs=sum(m['shared_gaps'] > 0 for m in records))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
