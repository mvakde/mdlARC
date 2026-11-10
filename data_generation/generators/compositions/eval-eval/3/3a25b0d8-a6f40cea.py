"""Reconstruct nested rooms beneath a window and transfer their topological colors.

Only standard-library Python and NumPy are used. The reader knows no generation
indices: it completes visible rectangular loops, obtains containment trees, and
matches rooms by their rooted tree addresses, including symmetric branch orbits.
"""
from itertools import product
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                    (num_test, 'num_test', 1, 4)):
        if value is not None:
            value = _integer(value, name)
            if not low <= value <= high:
                raise ValueError(name + ' is outside its legal range')
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.add(p)
        groups.append(group)
    return groups


def _points(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _size(box):
    t, b, l, r = box
    return (b - t + 1) * (r - l + 1)


def _contains(a, b):
    return a[0] < b[0] <= b[1] < a[1] and a[2] < b[2] <= b[3] < a[3]


def _tree(boxes):
    """Return containment parents, signatures, and symmetry-invariant addresses."""
    parents, children = [], [[] for _ in boxes]
    for i, box in enumerate(boxes):
        containing = [j for j, other in enumerate(boxes) if _contains(other, box)]
        parents.append(min(containing, key=lambda j: _size(boxes[j])) if containing else -1)
    roots = [i for i, p in enumerate(parents) if p == -1]
    assert len(roots) == 1, 'one enclosing root is required'
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    # Crossing loops, touching loops, and partial overlaps do not form rooms.
    areas = [_area(b) for b in boxes]
    borders = [_perimeter(b) for b in boxes]
    halos = [{(y + dy, x + dx) for y, x in border
              for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))}
             for border in borders]
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            if areas[i] & areas[j]:
                assert _contains(a, b) or _contains(b, a), 'non-laminar loops'
            assert not (borders[i] & halos[j]), 'touching loops'
    signatures = [None] * len(boxes)

    def signature(i):
        if signatures[i] is None:
            signatures[i] = tuple(sorted(signature(j) for j in children[i]))
        return signatures[i]

    root = roots[0]
    signature(root)
    addresses = [None] * len(boxes)

    def visit(i, address):
        addresses[i] = address
        for j in children[i]:
            visit(j, address + (signatures[j],))

    visit(root, ())
    return parents, children, signatures, addresses, root


def _room_cells(boxes, children, i):
    t, b, l, r = boxes[i]
    cells = _area((t + 1, b - 1, l + 1, r - 1))
    for j in children[i]:
        cells -= _area(boxes[j])
    assert cells and len(_components(cells)) == 1, 'room must be connected'
    return cells


def _donor_and_window(grid):
    # A guaranteed one-cell exterior border makes the background explicit.
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    candidates = []
    for group in _components(_points(grid != bg)):
        box = _bbox(group)
        if group == _area(box):
            border_values = {int(grid[p]) for p in _perimeter(box)}
            if len(border_values) == 1:
                wall = next(iter(border_values))
                inside = grid[box[0] + 1:box[1], box[2] + 1:box[3]]
                # An unconnected opaque card has one uniform fill; a donor has
                # a nested wall and at least two literal compartment colors.
                if wall in inside and len(set(inside.flat) - {wall}) >= 2:
                    candidates.append((box, wall))
    assert len(candidates) == 1, 'donor not uniquely visible'
    donor_box, wall = candidates[0]
    donor_area = _area(donor_box)
    wall_points = {p for p in donor_area if int(grid[p]) == wall}
    donor_boxes = []
    for group in _components(wall_points):
        box = _bbox(group)
        assert box[1] - box[0] >= 2 and box[3] - box[2] >= 2
        assert group == _perimeter(box), 'donor walls must be rectangular loops'
        donor_boxes.append(box)
    dt = _tree(donor_boxes)
    colors = {}
    for i, address in enumerate(dt[3]):
        cells = _room_cells(donor_boxes, dt[1], i)
        values = {int(grid[p]) for p in cells}
        assert len(values) == 1 and bg not in values and wall not in values
        color = next(iter(values))
        assert address not in colors or colors[address] == color, 'symmetric donor rooms disagree'
        colors[address] = color
    residual = _points(grid != bg) - donor_area
    window_candidates = []
    for color in sorted({int(grid[p]) for p in residual} - {wall}):
        group = {p for p in residual if int(grid[p]) == color}
        box = _bbox(group)
        if box[1] - box[0] < 4 or box[3] - box[2] < 4:
            continue
        if group != _perimeter(box):
            continue
        t, b, l, r = box
        inside = grid[t + 1:b, l + 1:r]
        fill = int(inside[0, 0])
        if np.all(inside == fill) and fill not in (wall, color) and fill not in colors.values():
            window_candidates.append((box, color, fill))
    assert len(window_candidates) == 1, 'one uniform framed window is required'
    window, frame, fill = window_candidates[0]
    assert not (donor_area & _area(window))
    visible = residual - _area(window)
    assert visible and all(int(grid[p]) == wall for p in visible)
    return bg, wall, donor_box, donor_boxes, dt, colors, window, frame, fill, visible


def _complete_loops(visible, mask, donor_signature):
    """Complete up to two fragments per rectangle, then require a unique cover.

    Visible extrema are retained for every loop. Exact visible pixels, rather
    than a guessed line direction or a generation descriptor, constrain the
    candidate geometry. Containment is checked while evaluating full covers.
    """
    parts = _components(visible)
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            group = first if i == j else first | parts[j]
            box = _bbox(group)
            if box[1] - box[0] >= 2 and box[3] - box[2] >= 2:
                if _perimeter(box) - mask == group:
                    candidates.append((frozenset((i, j)), box))
    answers = []

    def search(remaining, boxes):
        if len(answers) >= 2:
            return
        if not remaining:
            try:
                tree = _tree(boxes)
                if tree[2][tree[4]] == donor_signature:
                    answers.append((boxes, tree))
            except AssertionError:
                pass
            return
        first = min(remaining)
        for used, box in candidates:
            if first in used and used <= remaining:
                if any(_perimeter(box) & _perimeter(old) for old in boxes):
                    continue
                search(remaining - used, boxes + [box])

    search(set(range(len(parts))), [])
    assert len(answers) == 1, 'target completion is not unique'
    return answers[0]


def _read_input(input_grid):
    grid = _grid(input_grid)
    info = _donor_and_window(grid)
    bg, wall, db, donor, dt, colors, window, frame, fill, visible = info
    boxes, tree = _complete_loops(visible, _area(window), dt[2][dt[4]])
    t, b, l, r = window
    output = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for i, address in enumerate(tree[3]):
        assert address in colors
        for y, x in _room_cells(boxes, tree[1], i):
            if t < y < b and l < x < r:
                output[y - t - 1, x - l - 1] = colors[address]
        for y, x in _perimeter(boxes[i]):
            if t < y < b and l < x < r:
                output[y - t - 1, x - l - 1] = wall
    return output, {'background': bg, 'wall': wall, 'donor_box': db,
                    'donor_boxes': donor, 'donor_tree': dt, 'colors': colors,
                    'window': window, 'frame': frame, 'fill': fill,
                    'target_boxes': boxes, 'target_tree': tree}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible pixels alone; do not mutate the input."""
    return _read_input(input_grid)[0]


def _pack(rng, branches, reorder=False, detail=0, root_axis=0):
    """Independent rectangular embeddings of one shallow abstract tree.

    Root children are stars. The input-only solver supports deeper trees too;
    the bounded sampler uses depth at most two to fit both figures in 30 cells.
    """
    def leaf(minimum=3):
        h = minimum + int(rng.integers(detail + 1))
        w = minimum + int(rng.integers(detail + 1))
        return [(0, h - 1, 0, w - 1)], h, w

    def combine(items, axis):
        gap = 1 + int(rng.integers(detail + 1))
        pads = [1 + int(rng.integers(detail + 1)) for _ in range(4)]
        h = (sum(item[1] for item in items) + gap * (len(items) - 1)
             if axis == 0 else max(item[1] for item in items)) + pads[0] + pads[1] + 2
        w = (sum(item[2] for item in items) + gap * (len(items) - 1)
             if axis == 1 else max(item[2] for item in items)) + pads[2] + pads[3] + 2
        boxes = [(0, h - 1, 0, w - 1)]
        offset = 0
        for child, ch, cw in items:
            y = 1 + pads[0] + (offset if axis == 0 else int(rng.integers(h - 2 - pads[0] - pads[1] - ch + 1)))
            x = 1 + pads[2] + (offset if axis == 1 else int(rng.integers(w - 2 - pads[2] - pads[3] - cw + 1)))
            boxes += [(t + y, b + y, l + x, r + x) for t, b, l, r in child]
            offset += (ch if axis == 0 else cw) + gap
        return boxes, h, w

    items = []
    for count in branches:
        if count == 0:
            # A 1x1 interior cannot be shown in a strict window crop without
            # hiding the entire loop. Enlarge target root leaves to 2x2.
            items.append(leaf(4 if reorder else 3))
        else:
            axis = 0 if detail == 0 else int(rng.integers(2))
            items.append(combine([leaf() for _ in range(count)], axis))
    if reorder:
        items.reverse()
    return combine(items, root_axis)


def _color_map(rng, tree, palette, reuse):
    addresses = sorted(set(tree[3]), key=repr)
    assert len(addresses) <= len(palette)
    colors = {address: int(palette[i]) for i, address in enumerate(addresses)}
    if reuse and len(addresses) >= 4:
        # Reuse labels across unrelated orbits without ever inventing a match.
        colors[addresses[-1]] = colors[addresses[0]]
    return colors


def _paint(shape, boxes, bg, wall, colors=None):
    grid = np.full(shape, bg, dtype=np.int8)
    tree = _tree(boxes)
    if colors is not None:
        for i in range(len(boxes)):
            for p in _room_cells(boxes, tree[1], i):
                grid[p] = colors[tree[3][i]]
    for box in boxes:
        for p in _perimeter(box):
            grid[p] = wall
    return grid


def _window_options(rng, boxes, shape, bg, wall, colors, demonstration):
    h, w = shape
    tree = _tree(boxes)
    target = _paint(shape, boxes, bg, wall, colors)
    perimeters = set().union(*(_perimeter(box) for box in boxes))
    children = tree[1][tree[4]]
    child_rooms = {i: _room_cells(boxes, tree[1], i) for i in children}
    borders = [_perimeter(b) for b in boxes]
    options = []
    # Bounded randomized search; the first acceptable choices are sampled again
    # by the pair loop if fitting or exact-cover ambiguity prevents their use.
    for _ in range(420):
        hh = int(rng.integers(5, min(18, h + 3) + 1))
        ww = int(rng.integers(5, min(12, w + 3) + 1))
        t = int(rng.integers(-2, max(-1, h - hh + 3) + 1))
        l = int(rng.integers(-2, max(-1, w - ww + 3) + 1))
        box = (t, t + hh - 1, l, l + ww - 1)
        mask = _area(box)
        if any(not (p - mask) or _bbox(p - mask) != b for b, p in zip(boxes, borders)):
            continue
        inside = _area((t + 1, t + hh - 2, l + 1, l + ww - 2))
        if len(inside & perimeters) < 3:
            continue
        painted = {int(target[p]) for p in inside if 0 <= p[0] < h and 0 <= p[1] < w} - {wall, bg}
        if len(painted) < (3 if demonstration else 2):
            continue
        if demonstration:
            hit = [i for i in children if child_rooms[i] & inside]
            if len({colors[tree[3][i]] for i in hit}) < 2:
                continue
        options.append(box)
        if len(options) == 8:
            break
    return options


def _translate_boxes(boxes, dy, dx):
    return [(t + dy, b + dy, l + dx, r + dx) for t, b, l, r in boxes]


def _make_pair(rng, branches, demonstration, roles, detail, reuse):
    bg, wall, frame, fill, palette = roles
    for attempt in range(90):
        # Detail only affects geometry, never the topology or transformation.
        effective_detail = detail if attempt < 65 else 0
        donor, dh, dw = _pack(rng, branches, detail=0)
        target, th, tw = _pack(rng, branches, reorder=True,
                               detail=effective_detail, root_axis=0)
        if max(dh, dw, th, tw) > 28:
            continue
        dt = _tree(donor)
        colors = _color_map(rng, dt, palette, reuse and not demonstration)
        options = _window_options(rng, target, (th, tw), bg, wall, colors, demonstration)
        for win in options:
            sy, sx = min(0, win[0]), min(0, win[2])
            eh, ew = max(th, win[1] + 1) - sy, max(tw, win[3] + 1) - sx
            gap = 1 + int(rng.integers(2))
            # Narrow independent embeddings share a canvas, with freely offset
            # vertical positions. Common rotation later includes stacked scenes.
            height, width = max(dh, eh) + 2, dw + ew + gap + 2
            if height > 30 or width > 30:
                continue
            dy = 1 + int(rng.integers(height - dh - 1))
            ty = 1 + int(rng.integers(height - eh - 1))
            dx, tx = 1, 1 + dw + gap
            if bool(rng.integers(2)):
                tx, dx = 1, 1 + ew + gap
            dboxes = _translate_boxes(donor, dy, dx)
            tboxes = _translate_boxes(target, ty - sy, tx - sx)
            wbox = _translate_boxes([win], ty - sy, tx - sx)[0]
            inp = np.full((height, width), bg, dtype=np.int8)
            dgrid = _paint((dh, dw), donor, bg, wall, colors)
            inp[dy:dy + dh, dx:dx + dw] = dgrid
            for box in tboxes:
                for p in _perimeter(box):
                    inp[p] = wall
            t, b, l, r = wbox
            inp[t + 1:b, l + 1:r] = fill
            for p in _perimeter(wbox):
                inp[p] = frame
            complete = _paint((height, width), tboxes, bg, wall, colors)
            expected = complete[t + 1:b, l + 1:r].copy()
            expected[expected == bg] = fill
            # All eight common scene orientations change placement and shape,
            # but the rule itself uses no directional matching convention.
            turns, flip = int(rng.integers(4)), bool(rng.integers(2))
            inp, expected = np.rot90(inp, turns), np.rot90(expected, turns)
            if flip:
                inp, expected = inp[:, ::-1], expected[:, ::-1]
            inp, expected = inp.copy(), expected.copy()
            try:
                predicted, _ = _read_input(inp)
                assert np.array_equal(predicted, expected)
            except AssertionError:
                continue
            return {'input': inp, 'output': expected}
    raise RuntimeError('bounded geometry search found no inferable scene')


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    puzzle, seen = {'train': [], 'test': []}, set()
    # Sampling intensity and reuse preference vary by puzzle; roles vary by pair.
    detail = int(rng.integers(2))
    backgrounds = list(map(int, rng.permutation(10)))
    previous_wall = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demonstration = split == 'train' and index < 2
            if demonstration:
                branches = (0, index + 1)
            else:
                low, high = (3, 6) if split == 'train' else (6, 7)
                choices = [tuple(a) for k in range(1, 4)
                           for a in product(range(4), repeat=k)
                           if low <= 1 + k + sum(a) <= high
                           and max(a, default=0) >= 1
                           and (split == 'train' or (len(set(a)) >= 2
                                and 4 * (1 + k + sum(a)) - 1 + a.count(0) <= 28))]
                branches = choices[int(rng.integers(len(choices)))]
                branches = tuple(branches[i] for i in rng.permutation(len(branches)))
            bg = backgrounds[len(seen) % 10]
            palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
            wall = next(c for c in palette if c != previous_wall)
            palette.remove(wall)
            frame = palette.pop()
            fill = bg if len(seen) % 2 == 0 else palette.pop()
            roles = bg, wall, frame, fill, palette
            for _ in range(8):
                pair = _make_pair(rng, branches, demonstration or split == 'test', roles, detail,
                                  bool(rng.integers(2)))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique scene search exhausted')
            previous_wall = wall
    return puzzle


def _measure(pair):
    output, scene = _read_input(pair['input'])
    assert np.array_equal(output, np.asarray(pair['output'])), 'wrong reconstructed colors'
    boxes, tree = scene['target_boxes'], scene['target_tree']
    colors, wall = scene['colors'], scene['wall']
    t, b, l, r = scene['window']
    mask = _area((t + 1, b - 1, l + 1, r - 1))
    walls = set().union(*(_perimeter(box) for box in boxes)) & mask
    assert len(walls) >= 3
    assert sum(int(output[y - t - 1, x - l - 1]) == wall for y, x in walls) == len(walls)
    outside = mask - _area(boxes[tree[4]])
    assert all(int(output[y - t - 1, x - l - 1]) == scene['fill'] for y, x in outside)
    painted_roles = []
    for i, address in enumerate(tree[3]):
        cells = _room_cells(boxes, tree[1], i) & mask
        assert all(int(output[y - t - 1, x - l - 1]) == colors[address] for y, x in cells)
        if cells:
            painted_roles.append(address)
    assert len({colors[a] for a in painted_roles}) >= 2, 'A must change the answer'
    # Removing B's completion and coloring the intact input crop cannot retain
    # these wall pixels: every one of them was overwritten by the opaque fill.
    assert np.all(pair['input'][t + 1:b, l + 1:r] == scene['fill'])
    assert scene['fill'] != wall and scene['fill'] not in colors.values()
    branches = tuple(sorted(len(tree[1][i]) for i in tree[1][tree[4]]))
    same_depth_colors = len({colors[tree[3][i]] for i in tree[1][tree[4]]
                             if _room_cells(boxes, tree[1], i) & mask})
    # An area-only rule cannot pair the explicitly rearranged structural roles.
    donor_sizes = {a: sorted(len(_room_cells(scene['donor_boxes'], scene['donor_tree'][1], i))
                            for i, a0 in enumerate(scene['donor_tree'][3]) if a0 == a)
                   for a in colors}
    target_sizes = {a: sorted(len(_room_cells(boxes, tree[1], i))
                             for i, a0 in enumerate(tree[3]) if a0 == a)
                    for a in colors}
    # Read the order along the axis that separates root children. It is a
    # diagnostic against spatial matching, never a key used by solve.
    def sibling_order(bs, tr):
        indices = tr[1][tr[4]]
        if len(indices) != 2:
            return None
        a, z = (bs[i] for i in indices)
        axis = 0 if a[1] < z[0] or z[1] < a[0] else 2
        return tuple(tr[3][i] for i in sorted(indices, key=lambda i: bs[i][axis]))
    dorder = sibling_order(scene['donor_boxes'], scene['donor_tree'])
    torder = sibling_order(boxes, tree)
    return {'rooms': len(boxes), 'branches': branches,
            'roles': len(colors), 'visible_roles': len(set(painted_roles)),
            'same_depth_colors': same_depth_colors,
            'wall_cells': len(walls), 'outside_cells': len(outside),
            'background': scene['background'], 'wall': wall, 'fill_matches': scene['fill'] == scene['background'],
            'reused_colors': len(set(colors.values())) < len(colors),
            'geometry_changes': donor_sizes != target_sizes,
            'sibling_order_reversed': dorder is not None and dorder != torder,
            'structure': (tuple(sorted(scene['target_boxes'])), scene['window'])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            metrics[split].append(_measure({'input': inp, 'output': out}))
    assert metrics['train'][0]['branches'] == (0, 1)
    assert metrics['train'][1]['branches'] == (0, 2)
    assert all(m['same_depth_colors'] >= 2 and m['visible_roles'] >= 3 for m in metrics['train'][:2])
    assert all(m['sibling_order_reversed'] and m['geometry_changes'] for m in metrics['train'][:2])
    assert metrics['train'][0]['background'] != metrics['train'][1]['background']
    assert metrics['train'][0]['wall'] != metrics['train'][1]['wall']
    assert metrics['train'][0]['fill_matches'] and not metrics['train'][1]['fill_matches']
    assert all(m['rooms'] in (6, 7) for m in metrics['test'])
    assert all(m['same_depth_colors'] >= 2 and m['visible_roles'] >= 3 for m in metrics['test'])


def self_test():
    """Bounded 42-seed semantic/variation audit plus contract boundary checks."""
    structures, input_shapes, output_shapes, rooms, branches, metrics = set(), set(), set(), set(), set(), []
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input'])
                m = _measure(pair)
                metrics.append(m)
                structures.add(m['structure'])
                input_shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                rooms.add(m['rooms'])
                branches.add(m['branches'])
    assert len(puzzle_keys) == 42 and len(structures) >= 180
    assert len(input_shapes) >= 25 and len(output_shapes) >= 25
    assert rooms == {3, 4, 5, 6, 7} and len(branches) >= 10
    assert {m['background'] for m in metrics} == set(range(10))
    assert {m['wall'] for m in metrics} == set(range(10))
    assert sum(m['geometry_changes'] for m in metrics) >= 40
    assert any(m['reused_colors'] for m in metrics) and any(m['outside_cells'] for m in metrics)
    for a, b in ((2, 1), (10, 4), (None, 4), (2, None)):
        one, two = generate_puzzle(781, a, b), generate_puzzle(781, a, b)
        validate_generated_puzzle(one)
        assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                   for p, q in zip(one[s], two[s]) for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts were accepted')
    bad = generate_puzzle(90, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer was accepted')
