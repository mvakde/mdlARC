import itertools
import numpy as np


# A rod is (orientation, top row, left column, length, color).
# Horizontal rods are drawn over vertical rods. Only intersections already
# visible in the input are sliding joints; new contacts are solid supports.
def _cells(p):
    o, r, c, n, _ = p
    return frozenset((r + (k if o == 'V' else 0),
                      c + (k if o == 'H' else 0)) for k in range(n))


def _shift(p, distance):
    return p[0], p[1] + distance, p[2], p[3], p[4]


def _joints(pieces):
    cells = [_cells(p) for p in pieces]
    return {(i, j) for i in range(len(pieces)) for j in range(i)
            if cells[i] & cells[j]}


def _settle(pieces, height, joints=None):
    result = list(pieces)
    if joints is None:
        joints = _joints(pieces)
    for _ in range(height):
        cells = [_cells(p) for p in result]
        targets = [frozenset((r + 1, c) for r, c in s) for s in cells]
        moving = {i for i, s in enumerate(targets) if all(r < height for r, c in s)}
        while True:
            blocked = {i for i in moving for j in range(len(result))
                       if j not in moving and (max(i, j), min(i, j)) not in joints
                       and targets[i] & cells[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError('settling exceeded the canvas height')


def _settle_direct(pieces, height):
    """Separate, event-distance implementation used to cross-check gravity."""
    result = list(pieces)
    joints = _joints(pieces)
    for _ in range(height * (len(result) + 1)):
        changed = False
        # The generated support graph has no cycles. Move lowest rods first.
        for i in sorted(range(len(result)), key=lambda k: max(r for r, c in _cells(result[k])), reverse=True):
            p = result[i]
            for d in range(1, height):
                target = _cells(_shift(p, d))
                if any(r >= height for r, c in target) or any(
                    target & _cells(q) for j, q in enumerate(result)
                    if j != i and (max(i, j), min(i, j)) not in joints
                ):
                    d -= 1
                    break
            if d:
                result[i] = _shift(p, d)
                changed = True
        if not changed:
            return result
    raise AssertionError('cyclic or unresolved support graph')


def _render(shape, background, band, pieces):
    grid = np.full(shape, background, dtype=np.int8)
    grid[0] = band
    for p in sorted(pieces, key=lambda p: p[0] == 'H'):
        for cell in _cells(p):
            grid[cell] = p[4]
    return grid


def _controls(grid):
    assert grid.shape[0] >= 4
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg)
    controls = [int(x) for x in grid[0] if x != bg]
    assert 1 <= len(controls) <= 3 and len(controls) == len(set(controls))
    return bg, controls


def _parse(grid):
    """Recover rods from the drawing, including bracketed hidden continuations.

    A vertical continuation must have two visible cells outside horizontal runs.
    Hidden endpoints are deliberately not guessed. Every foreground cell must
    be explained, and every overlap must join perpendicular rods.
    """
    bg, controls = _controls(grid)
    h, w = grid.shape
    horizontal = np.zeros(grid.shape, dtype=bool)
    pieces = []
    for r in range(2, h):
        c = 0
        while c < w:
            end = c + 1
            while end < w and grid[r, end] == grid[r, c]:
                end += 1
            if grid[r, c] != bg and end - c >= 2:
                pieces.append(('H', r, c, end - c, int(grid[r, c])))
                horizontal[r, c:end] = True
            c = end
    for c in range(w):
        for color in range(10):
            if color == bg:
                continue
            rows = [r for r in range(2, h) if grid[r, c] == color]
            groups = []
            for r in rows:
                if groups and (r == groups[-1][-1] + 1 or
                               (r == groups[-1][-1] + 2 and horizontal[r - 1, c])):
                    groups[-1].append(r)
                else:
                    groups.append([r])
            for group in groups:
                if sum(not horizontal[r, c] for r in group) >= 2:
                    first, last = group[0], group[-1]
                    pieces.append(('V', first, c, last - first + 1, color))
    assert pieces
    joints = _joints(pieces)
    for i, j in joints:
        assert pieces[i][0] != pieces[j][0]
    assert np.array_equal(_render(grid.shape, bg, grid[0], pieces), grid), 'unresolved rod decomposition'
    return bg, controls, pieces


def _prediction(grid):
    bg, controls, pieces = _parse(grid)
    if not any(p[4] in controls for p in pieces):
        return grid.copy()
    survivors = [p for p in pieces if p[4] not in controls]
    settled = _settle(survivors, grid.shape[0])
    return _render(grid.shape, bg, grid[0], settled)


def solve(train_pairs, test_input):
    """Prediction uses only the visible demonstrations and queried input."""
    assert train_pairs
    for pair in train_pairs:
        assert np.array_equal(_prediction(pair['input']), pair['output'])
    return _prediction(test_input)


def _drop(pieces, orientation, col, length, height):
    p = (orientation, 2, col, length, 0)
    if any(_cells(p) & _cells(q) for q in pieces):
        return False
    for d in range(height):
        target = _cells(_shift(p, d + 1))
        if any(r >= height for r, c in target) or any(target & _cells(q) for q in pieces):
            pieces.append(_shift(p, d))
            return True
    return False


def _geometry(rng, kind, large=False):
    if kind == 'compact':
        h, w = int(rng.integers(6, 15)), int(rng.integers(5, 15))
        n = int(rng.integers(2, h - 2))
        left = int(rng.integers(0, w - 2))
        right = int(rng.integers(left + 2, w))
        x = int(rng.integers(left, right + 1))
        return (h, w), [('V', h - n, x, n, 0), ('H', h - n - 1, left, right - left + 1, 0)], None
    if rng.random() < .09:
        h = w = 12
    else:
        h = int(rng.integers(8 if not large else 15, 31))
        w = int(rng.integers(6 if not large else 13, 31))
    left = int(rng.integers(0, min(4, w - 4) + 1))
    right = int(rng.integers(max(left + 3, w - 5), w))
    pieces = []
    linked = None
    if kind.startswith('joint'):
        h = max(h, 12)
        w = max(w, 10)
        left = int(rng.integers(0, w - 6))
        right = int(rng.integers(left + 5, w))
        x = int(rng.integers(left + 2, right - 1))
        length = int(rng.integers(7, min(16, h - 2) + 1))
        top = h - length
        # Crosses and top-corner contacts have visibly determined endpoints.
        row = top if kind == 'joint_corner' else int(rng.integers(top + 2, h - 3))
        if kind == 'joint_corner':
            x = left
        leg_x = right if kind == 'joint_corner' else left
        pieces = [('V', top, x, length, 0),
                  ('V', row + 1, leg_x, h - row - 1, 0),
                  ('H', row, left, right - left + 1, 0)]
        linked = (0, 2)
        # A second support is optional; deleting both can release the slider.
        if rng.random() < .35:
            second_x = (left + right) // 2 if kind == 'joint_corner' else right
            pieces.append(('V', row + 1, second_x, h - row - 1, 0))
        # A broad shelf can pass through several independently visible rails.
        # Each is floor-supported and has an explicit, recoverable endpoint.
        rail_candidates = [c for c in range(left + 2, right - 1)
                           if all(abs(c - p[2]) >= 2 for p in pieces if p[0] == 'V')]
        rng.shuffle(rail_candidates)
        for c in rail_candidates[:int(rng.choice([0, 0, 1, 2]))]:
            if any(abs(c - p[2]) < 2 for p in pieces if p[0] == 'V'):
                continue
            rail_top = row if kind == 'joint_corner' else int(rng.integers(max(2, h - 16), row - 1))
            pieces.append(('V', rail_top, c, h - rail_top, 0))
    elif kind == 'bridge':
        length = int(rng.integers(3, min(13, h - 4) + 1))
        xs = sorted(rng.choice(np.arange(left, right + 1), size=min(4, right - left + 1), replace=False).tolist())
        # Two equally tall load-bearing legs, plus optional shorter landings.
        for j, x in enumerate(xs):
            if j < 2:
                n = length
            elif rng.random() < .65:
                n = int(rng.integers(2, length))
            else:
                continue
            pieces.append(('V', h - n, int(x), n, 0))
        pieces.append(('H', h - length - 1, left, right - left + 1, 0))
    else:
        # Independent lengths and insertion locations build a support DAG,
        # without assigning constructions to equal-width slots.
        count = int(rng.integers(2, 7 if h < 13 else 19))
        for _ in range(count * 4):
            if len(pieces) >= count:
                break
            o = 'H' if rng.random() < .48 else 'V'
            n = int(rng.integers(2, min(right - left + 1, 22) + 1)) if o == 'H' else int(rng.integers(2, min(14, h - 3) + 1))
            x = int(rng.integers(left, right - n + 2)) if o == 'H' else int(rng.integers(left, right + 1))
            _drop(pieces, o, x, n, h)
    # Add further shelves/uprights across existing supports, independently of
    # the base network. A new beam can connect several previously separate piles.
    extras = int(rng.integers(0, 10 if large else 6))
    for _ in range(extras):
        o = 'H' if rng.random() < .60 else 'V'
        n = int(rng.integers(2, min(right - left + 1, 22) + 1)) if o == 'H' else int(rng.integers(2, min(10, h - 3) + 1))
        x = int(rng.integers(left, right - n + 2)) if o == 'H' else int(rng.integers(left, right + 1))
        _drop(pieces, o, x, n, h)
    return (h, w), pieces, linked


def _color(rng, pieces, bg, linked, kind):
    palette = rng.choice([c for c in range(10) if c != bg], size=int(rng.integers(2, 10)), replace=False).tolist()
    cells = [_cells(p) for p in pieces]
    adjacent = []
    for i, s in enumerate(cells):
        neighbors = s | {(r + dr, c + dc) for r, c in s for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))}
        for j in range(i):
            if neighbors & cells[j]:
                adjacent.append((i, j))
    same_joint = linked is not None and kind != 'joint_cross'
    contacts = _joints(pieces) if same_joint else set()
    assigned = {}
    for i in range(len(pieces)):
        forbidden = {assigned[j] for a, j in adjacent if a == i and (i, j) not in contacts}
        forced = {assigned[j] for a, j in contacts if a == i}
        if forced:
            if len(forced) != 1:
                return None
            value = next(iter(forced))
            if value in forbidden:
                return None
        else:
            choices = [c for c in palette if c not in forbidden]
            if not choices:
                return None
            value = int(rng.choice(choices))
        assigned[i] = value
    return [(*p[:4], assigned[i]) for i, p in enumerate(pieces)]


def _band(rng, width, bg, controls):
    band = np.full(width, bg, dtype=np.int8)
    positions = sorted(rng.choice(width, size=len(controls), replace=False).tolist())
    values = list(controls)
    rng.shuffle(values)
    for c, value in zip(positions, values):
        band[c] = value
    return band


def _effects(pieces, height, controls):
    survivors = [p for p in pieces if p[4] not in controls]
    settled = _settle(survivors, height)
    distances = [q[1] - p[1] for p, q in zip(survivors, settled)]
    present = set(controls) & {p[4] for p in pieces}
    return survivors, settled, distances, present


def _command_options(rng, pieces, height, bg, max_commands):
    colors = sorted({p[4] for p in pieces})
    absent = [c for c in range(10) if c != bg and c not in colors]
    commands = [(c,) for c in colors + absent[:1]]
    for n in range(2, max_commands + 1):
        choices = list(itertools.combinations(colors + absent[:1], n))
        if len(choices) > 14:
            choices = [choices[int(i)] for i in rng.choice(len(choices), size=14, replace=False)]
        commands.extend(choices)
    result = []
    for cmd in commands:
        survivors, settled, distances, present = _effects(pieces, height, cmd)
        effect = 'absent' if not present else 'all' if not survivors else 'collapse' if any(distances) else 'harmless'
        result.append((cmd, effect, len(set(distances) - {0}), settled))
    return result


def _make_pair(rng, kind, max_commands, requested=None, background=None, large=False):
    if requested is None and max_commands >= 2 and rng.random() < .08:
        kind, requested = 'compact', 'all'
    # Rejections enforce observable decomposition, not a favored layout or effect.
    for attempt in range(60):
        shape, geometry, linked = _geometry(rng, kind, large)
        bg = int(rng.integers(10)) if background is None else background
        for _ in range(12):
            pieces = _color(rng, geometry, bg, linked, kind)
            if pieces is None:
                continue
            probe = _render(shape, bg, _band(rng, shape[1], bg, [pieces[0][4]]), pieces)
            try:
                _, _, parsed = _parse(probe)
                assert set(parsed) == set(pieces)
                assert _settle(pieces, shape[0]) == pieces
                assert _settle_direct(pieces, shape[0]) == pieces
            except AssertionError:
                continue
            options = _command_options(rng, pieces, shape[0], bg, max_commands)
            if requested == 'joint_fall':
                eligible = [x for x in options if x[1] == 'collapse' and all(pieces[i][4] not in x[0] for i in linked)
                            and any(p == pieces[linked[1]] and q[1] > p[1]
                                    for p, q in zip([p for p in pieces if p[4] not in x[0]], x[3]))]
            elif requested:
                eligible = [x for x in options if x[1] == requested]
            else:
                # Sample consequences before commands, avoiding the combinatorial
                # bias toward multi-color deletion of most of the construction.
                effects = sorted({x[1] for x in options})
                weights = np.array([{'absent': .12, 'all': .06, 'harmless': .28, 'collapse': .54}[x] for x in effects])
                chosen = str(rng.choice(effects, p=weights / weights.sum()))
                eligible = [x for x in options if x[1] == chosen]
            if not eligible:
                break
            option = eligible[int(rng.integers(len(eligible)))]
            controls, _, _, settled = option
            band = _band(rng, shape[1], bg, controls)
            pair = {'input': _render(shape, bg, band, pieces),
                    'output': _render(shape, bg, band, settled)}
            if _source_copy(pair['input']):
                continue
            return pair
    # A small deterministic family remains available even for unlucky palettes.
    # It has a sole support, an independent landing, and a sliding cross when needed.
    bg = int(rng.integers(10)) if background is None else background
    cols = [c for c in range(10) if c != bg]
    rng.shuffle(cols)
    shape = (12, 10)
    if kind.startswith('joint'):
        row = 3 if kind == 'joint_corner' else 6
        beam_color = cols[2] if kind == 'joint_cross' else cols[0]
        main_x, leg_x = (1, 8) if kind == 'joint_corner' else (4, 1)
        pieces = [('V', 3, main_x, 9, cols[0]), ('V', row + 1, leg_x, 11 - row, cols[1]), ('H', row, 1, 8, beam_color)]
    else:
        pieces = [('V', 6, 2, 6, cols[1]), ('H', 5, 1, 7, cols[0])]
    controls = [cols[3]] if requested == 'absent' else [pieces[-1][4]] if requested == 'harmless' else [cols[1]]
    band = _band(rng, shape[1], bg, controls)
    return {'input': _render(shape, bg, band, pieces),
            'output': _render(shape, bg, band, _settle([p for p in pieces if p[4] not in controls], shape[0]))}


def _counterfactual(rng, pair, max_commands):
    bg, old, pieces = _parse(pair['input'])
    options = _command_options(rng, pieces, pair['input'].shape[0], bg, max_commands)
    options = [x for x in options if set(x[0]) != set(old)]
    if not options:
        return None
    # Favor true joint release after a harmless single-leg demonstration.
    interactions = []
    for option in options:
        cmd = option[0]
        if len(cmd) == 2 and option[1] == 'collapse':
            if all(not any(_effects(pieces, pair['input'].shape[0], [c])[2]) for c in cmd):
                interactions.append(option)
    option = (interactions or options)[int(rng.integers(len(interactions or options)))]
    band = _band(rng, pair['input'].shape[1], bg, option[0])
    return {'input': _render(pair['input'].shape, bg, band, pieces),
            'output': _render(pair['input'].shape, bg, band, option[3])}


def _check_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    return int(seed)


def _count(value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError('invalid pair count')
    return int(value)


def _choose_counts(seed, rng, num_train, num_test):
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('training count must exceed test count')
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


# Exact original inputs, not a ban on their dimensions or color palettes.
_SOURCE_SCENES = (
    ('000051111000','000050001000','000050001000','006333055700','006003050700','006003050700','042222222270','040000000070','040000000070','040000000070'),
    ('000002555800','000002000800','000002000800','000002000800','009444702330','009000702030','009000702030','065501777740','060501000040','060501000040'),
    ('004330000000','004030877700','004030800700','004030800700','006555550700','006000050700','006000050700','031110222290','030010200090','030010200090'),
)
_SOURCE_BANDS = (('900000000000','400000000000','700000000000'), ('400000000000','600000000000'), ('700000000000','210000000000'))


def _source_copy(grid):
    if grid.shape != (12, 12):
        return False
    rows = tuple(''.join(str(int(x)) for x in row) for row in grid)
    return any(rows[1] == '000000000000' and rows[2:] == scene and rows[0] in bands
               for scene, bands in zip(_SOURCE_SCENES, _SOURCE_BANDS))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _check_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(seed, rng, num_train, num_test)
    # The physics, marker equality and rendering order are universal. These
    # puzzle-level choices control teaching coverage and the supported split shift.
    junctions = bool(rng.random() < .65)
    # A test may recombine the demonstrated junction category, but may not
    # introduce a hidden-endpoint or visibility convention absent from training.
    joint_kind = str(rng.choice(['joint_corner', 'joint_same', 'joint_cross']))
    source_shift = bool(rng.random() < .35)
    train_max = 1 if source_shift else int(rng.integers(2, 4))
    test_max = 2 if source_shift else train_max
    first_bg = int(rng.integers(10))
    train = [_make_pair(rng, joint_kind if junctions else str(rng.choice(['pile', 'bridge'])),
                        train_max, 'joint_fall' if junctions else 'collapse', first_bg)]
    train.append(_make_pair(rng, 'bridge', train_max, 'harmless', (first_bg + int(rng.integers(1, 10))) % 10))
    for _ in range(num_train - 2):
        if rng.random() < .30:
            pair = _counterfactual(rng, train[int(rng.integers(len(train)))], train_max)
        else:
            pair = None
        if pair is None or any(_grid_key(pair['input']) == _grid_key(p['input']) for p in train):
            pair = _make_pair(rng, str(rng.choice(['pile', 'bridge', joint_kind] if junctions else ['pile', 'bridge'])), train_max)
        train.append(pair)
    # If triple commands are allowed in tests, a compositional command is shown.
    if not source_shift and not any(len(_controls(p['input'])[1]) >= 2 for p in train):
        replacement = _counterfactual(rng, train[1], train_max)
        if replacement is not None and len(_controls(replacement['input'])[1]) >= 2:
            train[1] = replacement
        else:
            # A present-plus-absent pair establishes union without new mechanics.
            bg, controls, rods = _parse(train[1]['input'])
            extra = next((c for c in range(10) if c != bg and c not in {p[4] for p in rods}), None)
            if extra is not None:
                band = _band(rng, train[1]['input'].shape[1], bg, controls + [extra])
                train[1]['input'][0] = band
                train[1]['output'][0] = band
            else:
                test_max = 2
    tests = []
    seen = {_grid_key(p['input']) for p in train}
    for _ in range(num_test):
        for attempt in range(30):
            kind = str(rng.choice(['pile', 'bridge', joint_kind] if junctions else ['pile', 'bridge']))
            pair = _make_pair(rng, kind, test_max, large=bool(rng.random() < .50))
            if _grid_key(pair['input']) not in seen:
                seen.add(_grid_key(pair['input']))
                tests.append(pair)
                break
        else:
            raise AssertionError('duplicate inputs after bounded construction')
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {'train': train, 'test': tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _support_edges(pieces):
    joints = _joints(pieces)
    return [(i, j) for i, p in enumerate(pieces) for j, q in enumerate(pieces)
            if i != j and (max(i, j), min(i, j)) not in joints
            and _cells(_shift(p, 1)) & _cells(q)]


def features(pair):
    bg, controls, pieces = _parse(pair['input'])
    h, w = pair['input'].shape
    survivors, settled, distances, present = _effects(pieces, h, controls)
    edges = _support_edges(pieces)
    out_edges = _support_edges(settled)
    joints = _joints(pieces)
    joint_types = []
    joint_slips = 0
    survivor_positions = {p: q[1] for p, q in zip(survivors, settled)}
    for i, j in joints:
        a, b = pieces[i], pieces[j]
        joint_types.append(('same_' if a[4] == b[4] else 'different_') +
                           ('corner' if a[1] == b[1] else 'cross'))
        if a in survivor_positions and b in survivor_positions:
            joint_slips += survivor_positions[a] - a[1] != survivor_positions[b] - b[1]
    support = {i: [j for a, j in edges if a == i] for i in range(len(pieces))}
    def depth(i, path=()):
        assert i not in path, 'cyclic support graph'
        return 1 + max((depth(j, path + (i,)) for j in support[i]), default=0)
    components = [{i} for i in range(len(pieces))]
    for i, j in edges + list(joints):
        a = next(s for s in components if i in s)
        b = next(s for s in components if j in s)
        if a is not b:
            a.update(b)
            components.remove(b)
    floor_landings = sum(d > 0 and max(r for r, c in _cells(p)) == h - 1 for d, p in zip(distances, settled))
    post_landings = sum(distances[i] > 0 and settled[j][0] == 'V' for i, j in out_edges)
    shelf_landings = sum(distances[i] > 0 and settled[j][0] == 'H' for i, j in out_edges)
    short_post_landings = sum(distances[i] > 0 and survivors[i][0] == 'H'
                             and survivors[j][0] == 'V' and distances[j] == 0
                             and survivors[j][1] > survivors[i][1] + 1 for i, j in out_edges)
    multiple_shelf_stops = len({p[1] for p, d in zip(settled, distances) if p[0] == 'H' and d}) >= 2
    overhangs = 0
    max_posts = 0
    for i, p in enumerate(pieces):
        posts = [pieces[j][2] for a, j in edges if a == i and pieces[j][0] == 'V']
        if p[0] == 'H' and posts:
            max_posts = max(max_posts, len(posts))
            overhangs += p[2] < min(posts) or p[2] + p[3] - 1 > max(posts)
    # A genuine interaction needs the SAME surviving rod to fall farther
    # because of at least two removals. An absent selector or an unrelated
    # cap deletion must not count as interacting support removal.
    survivor_ids = [i for i, p in enumerate(pieces) if p[4] not in controls]
    necessary = {i: 0 for i in survivor_ids}
    if len(controls) > 1:
        for c in controls:
            other_commands = set(controls) - {c}
            alternative_ids = [i for i, p in enumerate(pieces) if p[4] not in other_commands]
            alternative = _settle([pieces[i] for i in alternative_ids], h)
            rows = {i: p[1] for i, p in zip(alternative_ids, alternative)}
            for i, p in zip(survivor_ids, settled):
                necessary[i] += p[1] > rows[i]
    interaction = any(n >= 2 for n in necessary.values())
    two_leg_release = len(controls) == 2 and any(distances) and all(
        not any(_effects(pieces, h, [c])[2]) for c in controls)
    redundant = any(len(support[i]) >= 2 for i, p in enumerate(pieces) if p[0] == 'H')
    redundant_harmless = any(pieces[i][0] == 'H' and len(support[i]) >= 2
                            and any(pieces[j][4] in controls for j in support[i])
                            and i in survivor_ids and settled[survivor_ids.index(i)][1] == pieces[i][1]
                            for i in range(len(pieces)))
    affected_components = sum(any(pieces[i][4] in controls for i in s) for s in components)
    load = {pieces[j][4] for i, j in edges}
    harmless = {p[4] for j, p in enumerate(pieces) if not any(b == j for a, b in edges)}
    positions = np.flatnonzero(pair['input'][0] != bg).tolist()
    cells = set().union(*[_cells(p) for p in pieces])
    invisible = sum(len(_cells(p)) for p in pieces) - len(cells)
    gains = max([int(np.sum(pair['output'][2:] == c) - np.sum(pair['input'][2:] == c)) for c in range(10) if c != bg] or [0])
    return {'background': bg, 'palette_size': len({p[4] for p in pieces}),
            'commands': len(controls), 'effect': 'absent' if not present else 'all' if not survivors else 'collapse' if any(distances) else 'harmless',
            'present_absent': bool(present) and len(present) < len(controls),
            'pieces': len(pieces), 'constructions': len(components),
            'h_lengths': [p[3] for p in pieces if p[0] == 'H'], 'v_lengths': [p[3] for p in pieces if p[0] == 'V'],
            'edges': len(edges), 'depth': max(depth(i) for i in range(len(pieces))),
            'redundant_support': redundant, 'redundant_harmless': redundant_harmless,
            'interaction': interaction, 'independent_commands': len(controls) > 1 and affected_components >= 2,
            'two_leg_release': two_leg_release, 'short_post_landings': short_post_landings,
            'multiple_shelf_stops': multiple_shelf_stops, 'overhangs': overhangs, 'max_posts': max_posts,
            'load_and_harmless_color': bool(load & harmless),
            'selected_multiple_roles': bool(load & harmless & set(controls)),
            'repeated_colors': len(pieces) > len({p[4] for p in pieces}),
            'fall_distances': sorted(set(distances) - {0}), 'moved': sum(d > 0 for d in distances),
            'floor_landings': floor_landings, 'post_landings': post_landings, 'shelf_landings': shelf_landings,
            'joints': len(joints), 'same_color_joints': sum(pieces[i][4] == pieces[j][4] for i, j in joints),
            'joint_types': sorted(set(joint_types)), 'joint_slips': joint_slips,
            'corner_joints': sum(pieces[i][1] == pieces[j][1] for i, j in joints),
            'hidden_cells': invisible, 'visible_gain': gains,
            'left_margin': min(c for r, c in cells), 'right_margin': w - 1 - max(c for r, c in cells),
            'scene_depth': h - min(r for r, c in cells), 'density': len(cells) / ((h - 2) * w),
            'swatch_offset': min(positions), 'swatch_gaps': [b - a for a, b in zip(positions, positions[1:])]}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    train_features = []
    test_features = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape) and np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            key = _grid_key(inp)
            assert key not in seen and not _source_copy(inp)
            seen.add(key)
            bg, controls, pieces = _parse(inp)
            assert _settle(pieces, inp.shape[0]) == pieces
            surviving, settled, distances, _ = _effects(pieces, inp.shape[0], controls)
            assert all(d >= 0 for d in distances)
            assert np.array_equal(out, _render(inp.shape, bg, inp[0], settled))
            (train_features if split == 'train' else test_features).append(features(pair))
    assert any(f['moved'] for f in train_features), 'training must teach gravity'
    assert len({f['background'] for f in train_features}) >= 2
    if any(f['joints'] for f in test_features):
        assert any(f['joint_slips'] for f in train_features)
        demonstrated = {t for f in train_features for t in f['joint_types']}
        assert {t for f in test_features for t in f['joint_types']} <= demonstrated
    if any(f['commands'] == 3 for f in test_features):
        assert any(f['commands'] >= 2 for f in train_features)


def _fingerprint(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _mechanics_test():
    # Source train 5: the gray arm changes its position along the upright.
    # A free joint must not be treated as a rigid monochrome component.
    rods = [('V', 4, 3, 5, 5), ('H', 4, 3, 4, 5), ('V', 5, 6, 4, 7)]
    assert _settle(rods, 9) == rods
    after = _settle(rods[:2], 9)
    assert after[0] == rods[0] and after[1][1] == 8
    # Source train 2's newly visible red cell motivates a fully bracketed,
    # recoverable occlusion diagnostic rather than guessed hidden endpoints.
    rods = [('V', 2, 4, 9, 2), ('H', 5, 1, 7, 4), ('V', 6, 1, 5, 8)]
    band = np.array([4] + [0] * 8, dtype=np.int8)
    inp = _render((11, 9), 0, band, rods)
    assert set(_parse(inp)[2]) == set(rods)
    out = _prediction(inp)
    assert np.sum(out[2:] == 2) == np.sum(inp[2:] == 2) + 1
    # Independent stacks: four-row and two-row falls in the same scene.
    rods = [('V', 5, 1, 4, 1), ('H', 4, 0, 3, 2), ('V', 7, 6, 2, 1), ('H', 6, 5, 3, 3)]
    assert [q[1] - p[1] for p, q in zip([rods[1], rods[3]], _settle([rods[1], rods[3]], 9))] == [4, 2]
    # Two differently colored redundant legs, and a shorter landing post.
    rods = [('V', 4, 1, 6, 1), ('V', 4, 7, 6, 2), ('V', 7, 4, 3, 3), ('H', 3, 0, 9, 4)]
    assert all(not any(_effects(rods, 10, [c])[2]) for c in (1, 2))
    assert _effects(rods, 10, [1, 2])[2] == [0, 3]


def self_test() -> None:
    _mechanics_test()
    for scene, bands in zip(_SOURCE_SCENES, _SOURCE_BANDS):
        for band in bands:
            source_input = np.array([[int(c) for c in row] for row in
                                     (band, '000000000000') + scene], dtype=np.int8)
            assert _source_copy(source_input)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) == 42
    records = []
    shapes = set()
    structures = set()
    repeated = 0
    mixed_train = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        layouts = []
        for pair in puzzle['train'] + puzzle['test']:
            f = features(pair)
            records.append(f)
            shapes.add(pair['input'].shape)
            bg, controls, pieces = _parse(pair['input'])
            survivors = [p for p in pieces if p[4] not in controls]
            assert _settle(survivors, pair['input'].shape[0]) == _settle_direct(survivors, pair['input'].shape[0])
            layout = (pair['input'].shape, (pair['input'][2:] != bg).tobytes())
            layouts.append(layout)
            structures.add(layout)
        repeated += len(set(layouts)) < len(layouts)
        mixed_train += len({len(_controls(p['input'])[1]) for p in puzzle['train']}) > 1
        # Hold labels outside the prediction interface.
        for pair in puzzle['test']:
            prediction = solve([{'input': p['input'].copy(), 'output': p['output'].copy()} for p in puzzle['train']], pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
    assert len(shapes) >= 60 and len(structures) >= 180
    assert len({f['background'] for f in records}) == 10
    assert len({f['pieces'] for f in records}) >= 12
    assert len({f['palette_size'] for f in records}) >= 6
    assert len({f['depth'] for f in records}) >= 5
    assert len({f['constructions'] for f in records}) >= 4
    assert len({f['left_margin'] for f in records}) >= 5
    assert min(h for h, w in shapes) <= 10 and max(h for h, w in shapes) == 30
    assert (12, 12) in shapes
    assert max(n for f in records for n in f['v_lengths']) >= 12
    assert max(n for f in records for n in f['h_lengths']) >= 20
    assert {f['effect'] for f in records} >= {'absent', 'all', 'harmless', 'collapse'}
    for name in ('interaction', 'two_leg_release', 'independent_commands', 'redundant_support', 'redundant_harmless', 'present_absent', 'load_and_harmless_color', 'selected_multiple_roles', 'same_color_joints', 'corner_joints', 'visible_gain', 'floor_landings', 'post_landings', 'shelf_landings', 'short_post_landings', 'multiple_shelf_stops', 'overhangs'):
        assert sum(bool(f[name]) for f in records) >= 3, name
    assert sum(len(f['fall_distances']) > 1 for f in records) >= 12
    assert sum(f['joints'] and f['repeated_colors'] and len(f['fall_distances']) > 1 for f in records) >= 3
    assert repeated >= 5 and mixed_train >= 10
    assert _fingerprint(generate_puzzle(12345, 7, 3)) == _fingerprint(generate_puzzle(12345, 7, 3))
    assert _fingerprint(generate_puzzle(12345, 7, 3)) != _fingerprint(generate_puzzle(12346, 7, 3))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(8877, a, b)
        assert a is None or len(p['train']) == a
        assert b is None or len(p['test']) == b
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(5, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
