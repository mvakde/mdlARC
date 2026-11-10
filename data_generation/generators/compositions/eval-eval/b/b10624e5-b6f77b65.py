"""Gravity chooses between the two anchored mirror poses of a rigid load.

All output-affecting conventions are fixed here and described in task_spec.md.
Generation metadata is never consulted by solve or the validator.
"""
from collections import Counter
from math import gcd
import numpy as np


HOST_SEEDS = (0, 286664462, 438243343, 60173703, 281898920, 28116421,
              459226329, 980544154, 983603529, 441505999, 486183200,
              499306465, 648146624, 545861153, 634570620, 806818978,
              889791948, 123931765, 100609852, 705795692, 662535156,
              587912467, 523714621, 74864041, 413086961, 111150186,
              518672990, 819520997, 755091085, 253281346, 339878188,
              852875946, 718717442, 4815314, 127986771, 831095325,
              154691766, 589774426, 970676264, 577450071, 159573864,
              937762013)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'illegal {name}')
    return value


def _counts(train, test):
    train = _integer(train, 'num_train', 2, 10)
    test = _integer(test, 'num_test', 1, 4)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _square(array, color):
    points = np.argwhere(array == color)
    assert len(points), 'missing square'
    r, c = points.min(0).tolist()
    bottom, right = (points.max(0) + 1).tolist()
    side = bottom - r
    assert right - c == side and len(points) == side * side, 'anchor is not one square'
    return r, c, side


def _connected(points):
    remaining = set(points)
    assert remaining
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    h, w = grid.shape
    assert h >= 8 and w >= 5
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg), 'row 1 must be a blank separator'
    controls = [int(x) for x in grid[0] if x != bg]
    assert 1 <= len(controls) <= 3 and len(set(controls)) == len(controls)
    dividers = [r for r in range(3, h-3)
                if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    assert len(dividers) == 1, 'expected one horizontal blueprint divider'
    divider = dividers[0]
    wall = int(grid[divider, 0])
    source = grid[2:divider]
    scene = grid[divider+1:]
    common = (set(map(int, np.unique(source))) & set(map(int, np.unique(scene)))) - {bg, wall}
    anchors = []
    for color in sorted(common):
        try:
            sa = _square(source, color)
            da = _square(scene, color)
        except AssertionError:
            continue
        # All non-marker foreground in the lower area must be rooted posts.
        terrain = scene.copy()
        terrain[scene == color] = bg
        try:
            posts = _posts(terrain, bg)
        except AssertionError:
            continue
        anchors.append((color, sa, da, posts))
    assert len(anchors) == 1, 'anchor identity must be uniquely recoverable'
    anchor, sa, da, posts = anchors[0]
    assert wall not in controls and anchor not in controls
    assert not np.any(source == wall) and not np.any(scene == wall)
    motif = {(int(r)-sa[0], int(c)-sa[1]): int(source[r, c])
             for r, c in np.argwhere(source != bg)}
    assert len(set(motif.values())) >= 2 and _connected(motif)
    return dict(grid=grid, bg=bg, controls=set(controls), divider=divider,
                wall=wall, anchor=anchor, source=source, motif=motif,
                source_anchor=sa, marker=da, scene=scene, posts=posts)


def _posts(terrain, bg):
    """Each occupied column is a fully visible one-cell-wide floor-rooted post."""
    result = []
    h, w = terrain.shape
    for c in range(w):
        rows = np.flatnonzero(terrain[:, c] != bg)
        if not len(rows):
            continue
        top = int(rows[0])
        color = int(terrain[top, c])
        assert len(rows) >= 2 and np.array_equal(rows, np.arange(top, h))
        assert np.all(terrain[top:, c] == color), 'posts must be monochrome'
        result.append((top, c, color))
    assert result, 'at least one support post is required in the input'
    return result


def _scaled(data, destination_side=None):
    """Exact rational raster transfer, including fine details and anchor offsets.

    Work in an integer common resolution. A destination cell is valid only if
    all its source subcells have one color; interpolation is never used.
    """
    side = data['source_anchor'][2]
    target = data['marker'][2] if destination_side is None else destination_side
    factor = gcd(side, target)
    numerator, denominator = target // factor, side // factor
    motif = data['motif']
    rs, cs = zip(*motif)
    top = min(rs) // denominator * denominator
    left = min(cs) // denominator * denominator
    bottom = ((max(rs) + denominator) // denominator) * denominator
    right = ((max(cs) + denominator) // denominator) * denominator
    raw = np.full((bottom-top, right-left), data['bg'], dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r-top, c-left] = color
    expanded = np.repeat(np.repeat(raw, numerator, 0), numerator, 1)
    hh, ww = expanded.shape[0] // denominator, expanded.shape[1] // denominator
    blocks = expanded.reshape(hh, denominator, ww, denominator)
    low, high = blocks.min((1, 3)), blocks.max((1, 3))
    assert np.array_equal(low, high), 'requested scale mixes different colors'
    ar, ac = -top * numerator // denominator, -left * numerator // denominator
    original = {(int(r)-ar, int(c)-ac): int(low[r, c])
                for r, c in np.argwhere(low != data['bg'])}
    reflected = {(r, target-1-c): color for (r, c), color in original.items()}
    return original, reflected


def _descent(motif, marker, posts, height):
    """First collision with rooted columns or the floor, using the whole load."""
    y, x, _ = marker
    tops = {c: top for top, c, _ in posts}
    bottom = {}
    for r, c in motif:
        bottom[x+c] = max(bottom.get(x+c, -100), y+r)
    distance = min(tops.get(c, height) - 1 - r for c, r in bottom.items())
    assert distance >= 0, 'load initially intersects a support or floor'
    return distance


def _result(data, *, delete=True, reflect=True, gravity=True, anchor_only=False):
    original, mirrored = _scaled(data)
    poses = (original, mirrored) if reflect else (original,)
    posts = [p for p in data['posts'] if not delete or p[2] not in data['controls']]
    distances = []
    for pose in poses:
        proxy = ({(r, c): data['anchor'] for r in range(data['marker'][2])
                  for c in range(data['marker'][2])} if anchor_only else pose)
        distances.append(_descent(proxy, data['marker'], posts, data['scene'].shape[0]))
    winner = int(len(distances) == 2 and distances[1] > distances[0])
    fall = distances[winner] if gravity else 0
    out = data['grid'].copy()
    scene = out[data['divider']+1:]
    scene[:] = data['bg']
    for top, c, color in posts:
        scene[top:, c] = color
    y, x, _ = data['marker']
    for (r, c), color in poses[winner].items():
        rr, cc = y + r + fall, x + c
        assert 0 <= rr < scene.shape[0] and 0 <= cc < scene.shape[1], 'no clipping is allowed'
        assert scene[rr, cc] == data['bg'], 'load overlaps a post'
        scene[rr, cc] = color
    return out, winner, tuple(distances), fall


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the chosen reflection, scale and landing using only input pixels."""
    return _result(_decode(input_grid))[0]


def _motif(rng, anchor, details, simple=False):
    a = 1 if simple else int(rng.choice([1, 1, 2]))
    cells = {(r, c): anchor for r in range(a) for c in range(a)}
    if simple:
        cells.update({(0, -2): details[0], (0, -1): details[0],
                      (1, -2): details[-1], (0, 1): details[-1], (0, 2): details[-1]})
        return cells, a
    # Unequal arms with independently sized fingers and later branches.
    for index, sign in enumerate((-1, 1)):
        row = int(rng.integers(a))
        n = int(rng.integers(1, 4))
        tip = -n if sign < 0 else a+n-1
        for k in range(1, n+1):
            c = -k if sign < 0 else a+k-1
            cells[row, c] = details[index % len(details)]
        for r in range(row+1, row+int(rng.integers(1, 4))):
            cells[r, tip] = details[(index+1) % len(details)]
    for index in range(int(rng.integers(0, 7))):
        choices = sorted(cells)
        r, c = choices[int(rng.integers(len(choices)))]
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
        for k in range(1, int(rng.integers(2, 4))):
            p = r+dr*k, c+dc*k
            if not (-2 <= p[0] <= a+2 and -4 <= p[1] <= a+3):
                break
            if cells.get(p) != anchor:
                cells[p] = details[index % len(details)]
    for _ in range(int(rng.integers(4))):
        cells = {(c, a-1-r): color for (r, c), color in cells.items()}
    if rng.random() < .5:
        cells = {(r, a-1-c): color for (r, c), color in cells.items()}
    return cells, a


def _build(rng, units, extra_posts=0, simple=False):
    colors = [int(x) for x in rng.permutation(10)]
    bg, wall, anchor = colors[:3]
    details = colors[3:3+int(rng.integers(1, 4))]
    support_colors = colors[6:10]
    motif, a = _motif(rng, anchor, details, simple)
    u, v = units
    rs, cs = zip(*motif)
    mh, mw = max(rs)-min(rs)+1, max(cs)-min(cs)+1
    source_h, source_w = mh*u, mw*u
    template = np.full((source_h, source_w), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        rr, cc = (r-min(rs))*u, (c-min(cs))*u
        template[rr:rr+u, cc:cc+u] = color
    # These construction rectangles supply labels. solve separately rasterizes
    # the returned source pixels at their exact rational scale.
    poses = []
    base = {}
    for (r, c), color in motif.items():
        for dr in range(v):
            for dc in range(v):
                base[r*v+dr, c*v+dc] = color
    poses.append(base)
    poses.append({(r, a*v-1-c): color for (r, c), color in base.items()})
    bottoms = [{c: max(r for r, cc in pose if cc == c) for c in {cc for _, cc in pose}}
               for pose in poses]
    columns = sorted(set(bottoms[0]) | set(bottoms[1]))
    plus = [c for c in columns if bottoms[0].get(c, -100) > bottoms[1].get(c, -100)]
    minus = [c for c in columns if bottoms[1].get(c, -100) > bottoms[0].get(c, -100)]
    if not plus or not minus:
        return None
    minr, maxr = min(r for r, c in base), max(r for r, c in base)
    minc, maxc = min(columns), max(columns)
    source_padding = int(rng.integers(0, 3))
    divider = source_h + 2 + source_padding
    y = -minr + int(rng.integers(0, 3))
    tallest_top = y + maxr + int(rng.integers(2, 5))
    minimum_scene_h = tallest_top + max(3, maxr-minr+1)
    minimum_width = max(source_w, maxc-minc+3, 8, extra_posts+4)
    if minimum_scene_h + divider + 1 > 30 or minimum_width > 30:
        return None
    scene_h = int(rng.integers(minimum_scene_h, 30-divider))
    width = int(rng.integers(minimum_width, 31))
    x = int(rng.integers(-minc, width-maxc))
    primary_columns = [x+int(rng.choice(plus)), x+int(rng.choice(minus))]
    posts = [(tallest_top, c, support_colors[i]) for i, c in enumerate(primary_columns)]
    relevant = [x+c for c in columns if x+c not in primary_columns]
    outside = [c for c in range(width) if c not in primary_columns and c not in relevant]
    rng.shuffle(relevant)
    rng.shuffle(outside)
    available = relevant + outside
    for i, c in enumerate(available[:extra_posts]):
        # Additional columns lie under the candidate silhouettes first. They
        # may be equally tall constraints, or shorter intermediate landings.
        # The second added post is shorter, making the test height shift visible.
        top = int(rng.integers(tallest_top+(i == 1), scene_h-1))
        # Repeated identities interact: deleting a color removes every such post.
        color = support_colors[i % len(support_colors)]
        posts.append((top, c, color))
    grid = np.full((divider+1+scene_h, width), bg, dtype=np.int8)
    left = int(rng.integers(width-source_w+1))
    top = 2 + int(rng.integers(source_padding+1))
    grid[top:top+source_h, left:left+source_w] = template
    grid[divider] = wall
    scene = grid[divider+1:]
    scene[y:y+a*v, x:x+a*v] = anchor
    for post_top, c, color in posts:
        scene[post_top:, c] = color
    return dict(grid=grid, bg=bg, wall=wall, anchor=anchor,
                marker=(y, x, a*v), poses=poses, posts=posts,
                scene=scene, divider=divider, support_colors=support_colors)


def _command_pair(rng, built, controls):
    grid = built['grid'].copy()
    columns = rng.choice(grid.shape[1], len(controls), replace=False).tolist()
    for c, color in zip(columns, rng.permutation(sorted(controls))):
        grid[0, c] = int(color)
    posts = [p for p in built['posts'] if p[2] not in controls]
    distances = [_descent(pose, built['marker'], posts, built['scene'].shape[0])
                 for pose in built['poses']]
    winner = int(distances[1] > distances[0])
    out = grid.copy()
    scene = out[built['divider']+1:]
    scene[:] = built['bg']
    for top, c, color in posts:
        scene[top:, c] = color
    y, x, _ = built['marker']
    for (r, c), color in built['poses'][winner].items():
        scene[y+r+distances[winner], x+c] = color
    return {'input': grid, 'output': out}


def _witnesses(rng):
    for attempt in range(160):
        units = ((2, 1) if attempt >= 128 else
                 ((2, 1), (1, 2))[int(rng.integers(2))])
        built = _build(rng, units, simple=attempt >= 128)
        if built is None:
            continue
        # Delete the post worse for each pose in turn. Both directions must be
        # uniquely selected, genuinely fall, and depend on more than the square.
        pairs = [_command_pair(rng, built, {color}) for color in built['support_colors'][:2]]
        try:
            analyses = [_analyze(p) for p in pairs]
            assert [f['winner'] for f in analyses] == [0, 1]
            assert all(f['gap'] > 0 and f['fall'] > 0 and f['anchor_proxy_differs']
                       and f['deletion_differs'] for f in analyses)
        except AssertionError:
            continue
        return pairs, units
    raise AssertionError('could not build the bounded counterfactual witnesses')


def _new_pair(rng, units_choices, extra, max_commands, forced_commands=None,
              allow_tie=True):
    for attempt in range(128):
        units = units_choices[int(rng.integers(len(units_choices)))]
        built = _build(rng, units, extra_posts=extra, simple=attempt >= 100)
        if built is None:
            continue
        present = sorted({p[2] for p in built['posts']})
        absent = [c for c in range(10) if c not in present
                  and c not in (built['bg'], built['wall'], built['anchor'])]
        count = forced_commands or int(rng.integers(1, max_commands+1))
        pool = present + absent[:1]
        controls = set(map(int, rng.choice(pool, size=count, replace=False)))
        pair = _command_pair(rng, built, controls)
        try:
            features = _analyze(pair)
            if not allow_tie and features['gap'] == 0:
                continue
        except AssertionError:
            continue
        return pair
    raise AssertionError('could not create a bounded scene')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(num_train, num_test)
    train, _ = _witnesses(rng)
    seen = {_key(p['input']) for p in train}
    choices = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2)]
    for index in range(2, num_train):
        for _ in range(64):
            pair = _new_pair(rng, choices, int(rng.integers(1, 7)), 3,
                             forced_commands=2 if index == 2 else None)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('duplicate training input after bounded sampling')
        train.append(pair)
        seen.add(_key(pair['input']))
    observed_ratios = {_analyze(p)['ratio'] for p in train}
    test_choices = [uv for uv in choices if (uv[1]//gcd(*uv), uv[0]//gcd(*uv)) in observed_ratios]
    max_commands = max(_analyze(p)['commands'] for p in train)
    tie_shown = any(_analyze(p)['gap'] == 0 for p in train)
    tests = []
    for _ in range(num_test):
        for _ in range(64):
            pair = _new_pair(rng, test_choices, int(rng.integers(2, 7)), max_commands,
                             allow_tie=tie_shown)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('duplicate test input after bounded sampling')
        tests.append(pair)
        seen.add(_key(pair['input']))
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {'train': train, 'test': tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
    data = _decode(inp)
    poses = _scaled(data)
    y, x, side = data['marker']
    for pose in poses:
        assert all(0 <= y+r < data['scene'].shape[0] and
                   0 <= x+c < data['scene'].shape[1] for r, c in pose), 'initial pose is clipped'
        _descent(pose, data['marker'], data['posts'], data['scene'].shape[0])
    expected, winner, distances, fall = _result(data)
    assert out.shape == inp.shape and np.array_equal(out, expected), 'incorrect landing or pose'
    assert np.array_equal(out[:data['divider']+1], inp[:data['divider']+1])
    # The load's material is conserved exactly by its one rigid translation.
    settled_scene = out[data['divider']+1:].copy()
    surviving = [p for p in data['posts'] if p[2] not in data['controls']]
    for top, c, color in surviving:
        assert np.all(settled_scene[top:, c] == color)
        settled_scene[top:, c] = data['bg']
    assert Counter(map(int, settled_scene[settled_scene != data['bg']])) == Counter(poses[winner].values())
    assert _descent(poses[winner], (y+fall, x, side), surviving, data['scene'].shape[0]) == 0
    deletion_differs = not np.array_equal(_result(data, delete=False)[0], out)
    # Compare anchor proxy distances without painting the resulting colliding
    # body: a square-only gravity calculation may put the load through a post.
    proxy = {(r, c): data['anchor'] for r in range(side) for c in range(side)}
    proxy_distance = _descent(proxy, data['marker'], surviving, data['scene'].shape[0])
    source_side = data['source_anchor'][2]
    divisor = gcd(side, source_side)
    boundaries = []
    for (r, c), color in data['motif'].items():
        for dr, dc, offset in ((-1, 0, r), (1, 0, r+1),
                               (0, -1, c), (0, 1, c+1)):
            if data['motif'].get((r+dr, c+dc), data['bg']) != color:
                boundaries.append(offset)
    quantum = source_side
    for coordinate in boundaries:
        quantum = gcd(quantum, coordinate)
    reduced = {(r//quantum, c//quantum): color for (r, c), color in data['motif'].items()}
    normalized = []
    palette = {data['anchor']: 0}
    for (r, c), color in sorted(reduced.items()):
        palette.setdefault(color, len(palette))
        normalized.append((r, c, palette[color]))
    occupied_columns = {x+c for pose in poses for r, c in pose}
    landing_columns = {x+c for (r, c) in poses[winner]
                       if any(top == y+r+fall+1 and pc == x+c for top, pc, _ in surviving)}
    shortest_top = min((p[0] for p in data['posts']), default=0)
    short_landing = any(c in landing_columns and top > shortest_top for top, c, _ in surviving)
    return dict(winner=winner, distances=distances, fall=fall,
                gap=abs(distances[0]-distances[1]), posts=len(data['posts']),
                heights=tuple(sorted({data['scene'].shape[0]-p[0] for p in data['posts']})),
                commands=len(data['controls']), removed=len(data['posts'])-len(surviving),
                repeated=len({p[2] for p in data['posts']}) < len(data['posts']),
                ratio=(side//divisor, source_side//divisor), bg=data['bg'],
                anchor_side=side, source_side=source_side,
                anchor_proxy_differs=proxy_distance != fall, deletion_differs=deletion_differs,
                mirror_differs=not np.array_equal(_result(data, reflect=False)[0], out),
                shape=inp.shape, motif_signature=tuple(normalized),
                relevant_posts=sum(c in occupied_columns for top, c, _ in data['posts']),
                short_landing=short_landing,
                detail_command=bool(data['controls'] & (set(data['motif'].values())-{data['anchor']})),
                absent_command=bool(data['controls']-{p[2] for p in data['posts']}),
                floor_landing=any(y+r+fall == data['scene'].shape[0]-1 for r, c in poses[winner]),
                fine=any(offset % source_side for offset in boundaries))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                grid = np.asarray(value)
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape) and np.all((0 <= grid) & (grid <= 9))
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate inputs'
            seen.add(key)
            features[split].append(_analyze(pair))
    witnesses = [f for f in features['train'] if f['gap'] and f['fall'] and
                 f['ratio'] != (1, 1) and f['anchor_proxy_differs'] and f['deletion_differs']]
    assert {f['winner'] for f in witnesses} == {0, 1}, 'both poses need decisive scaled fall demonstrations'
    assert any(f['mirror_differs'] for f in witnesses)
    scaffold_witnesses = []
    for pair, feature in zip(puzzle['train'], features['train']):
        if feature['gap'] and feature['fall'] and feature['ratio'] != (1, 1) and \
                feature['anchor_proxy_differs'] and feature['deletion_differs']:
            scaffold_witnesses.append((_key(np.asarray(pair['input'])[1:]), feature['winner']))
    assert any(scaffold == other and winner != other_winner
               for scaffold, winner in scaffold_witnesses
               for other, other_winner in scaffold_witnesses), 'missing counterfactual mirror-switch scaffold'
    ratios = {f['ratio'] for f in features['train']}
    assert all(f['ratio'] in ratios for f in features['test']), 'test scales must be demonstrated'
    assert max(f['commands'] for f in features['test']) <= max(f['commands'] for f in features['train'])
    if any(f['gap'] == 0 for f in features['test']):
        assert any(f['gap'] == 0 for f in features['train']), 'test tie convention must be demonstrated'
    assert all(f['posts'] >= 4 and len(f['heights']) >= 2 and f['repeated']
               for f in features['test']), 'test must recombine unequal, repeated supports'


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def _physics_checks():
    # An overhanging foot, not the anchor, strikes a post. Removing that post
    # changes the favored pose. On empty terrain the two depths tie.
    original = {(0, 0): 1, (0, -1): 2, (1, -1): 2, (0, 1): 3}
    mirror = {(r, -c): color for (r, c), color in original.items()}
    marker = (1, 3, 1)
    posts = [(6, 2, 4), (6, 4, 5)]
    assert [_descent(p, marker, posts[:1], 12) for p in (original, mirror)] == [3, 4]
    assert [_descent(p, marker, posts[1:], 12) for p in (original, mirror)] == [4, 3]
    assert [_descent(p, marker, [], 12) for p in (original, mirror)] == [9, 9]
    # Hand-specified input-only fixture: an absent command leaves equal-depth
    # supports, and deleting both identities gives an empty-floor tie. In both
    # cases the colored original orientation must be drawn, not its mirror.
    grid = np.zeros((18, 8), dtype=np.int8)
    grid[0, 0] = 9
    grid[5] = 8
    for (r, c), color in original.items():
        grid[2+r, 3+c] = color
    grid[7, 3] = 1
    grid[12:, 2] = 4
    grid[12:, 4] = 5
    for delete_all, distance in ((False, 3), (True, 9)):
        query = grid.copy()
        if delete_all:
            query[0] = 0
            query[0, :2] = [4, 5]
        expected = query.copy()
        expected[6:] = 0
        if not delete_all:
            expected[12:, 2] = 4
            expected[12:, 4] = 5
        for (r, c), color in original.items():
            expected[6+1+r+distance, 3+c] = color
        frozen = query.copy()
        assert np.array_equal(solve(query), expected)
        assert np.array_equal(query, frozen), 'solve must not mutate its input'
    # Exact reduction is allowed for a coarse source; a fine boundary crossing
    # a destination cell must be rejected, even if its shape would fit.
    data = dict(source_anchor=(0, 0, 2), marker=(0, 0, 1), bg=0,
                motif={(r, c): 1 for r in range(2) for c in range(2)})
    assert _scaled(data)[0] == {(0, 0): 1}
    data['motif'][-1, 0] = 2
    try:
        _scaled(data)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous reduction accepted')


def _stepwise_descent(pose, marker, posts, shape):
    """Diagnostic only: verify the profile formula against literal cell moves."""
    y, x, _ = marker
    terrain = {(r, c) for top, c, _ in posts for r in range(top, shape[0])}
    for distance in range(shape[0]+1):
        next_cells = {(y+r+distance+1, x+c) for r, c in pose}
        if any(r >= shape[0] for r, c in next_cells) or next_cells & terrain:
            return distance
    raise AssertionError('no floor collision')


def self_test():
    """Bounded 42-seed structural and mechanism checks; no independent judge."""
    _physics_checks()
    fingerprints, records = set(), []
    for index, seed in enumerate(HOST_SEEDS):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if index < 3:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
                data = _decode(pair['input'])
                survivors = [p for p in data['posts'] if p[2] not in data['controls']]
                for pose in _scaled(data):
                    assert _descent(pose, data['marker'], survivors, data['scene'].shape[0]) == \
                        _stepwise_descent(pose, data['marker'], survivors, data['scene'].shape)
                records.append(_analyze(pair))
    assert len(fingerprints) == 42
    assert len({f['shape'] for f in records}) >= 35
    assert len({f['motif_signature'] for f in records}) >= 60
    assert {f['bg'] for f in records} == set(range(10))
    assert {f['winner'] for f in records} == {0, 1}
    assert len({f['posts'] for f in records}) >= 5
    assert len({f['fall'] for f in records}) >= 8
    assert len({f['ratio'] for f in records}) >= 4
    assert any(f['gap'] == 0 for f in records), 'ties need coverage'
    assert any(f['commands'] >= 2 and f['removed'] >= 2 for f in records)
    assert sum(f['fine'] for f in records) >= 30
    assert sum(f['short_landing'] for f in records) >= 15
    assert sum(f['detail_command'] for f in records) >= 10
    assert sum(f['floor_landing'] for f in records) >= 15
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(79, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), ('4', 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(7, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]]
              for s in puzzle}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
