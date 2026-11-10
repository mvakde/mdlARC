"""Synthetic ARC tasks: recover rectangles from a disposable periodic tile."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = {
    '6c1e3be0bc7678bf3092b28258011f87af3320dc9becac6782a24ee105a46016',
    '39d245b9bb4531d868664ae8d3908fb4183e20adeaa1f2560d98353393bdc20f',
    '6cad8ce74e3008babd975d464704c65e241e96e134874e2eb71cde0803e5d5d1',
    '1511ea5a45711d04a226ffb4f81397dd01cbdf6241280ca338882e9dc42ab75d',
}


def _paint(grid, rect):
    y, x, h, w, color, solid = rect
    if solid:
        grid[y:y+h, x:x+w] = color
    else:
        grid[y, x:x+w] = color
        grid[y+h-1, x:x+w] = color
        grid[y:y+h, x] = color
        grid[y:y+h, x+w-1] = color


def _assemble(rects):
    out = np.zeros((rects[0][2], rects[0][3]), dtype=np.int8)
    bottom, left = out.shape[0] - 1, 0
    for index, (_, _, height, width, color, solid) in enumerate(rects):
        if index and not rects[index-1][5]:
            bottom -= 1
            left += 1
        top = bottom-height+1
        if top < 0 or left+width > out.shape[1]:
            raise ValueError('layer does not fit')
        _paint(out, (top, left, height, width, color, solid))
    return out


def _wallpaper_candidates(grid):
    rows, cols = np.indices(grid.shape)
    candidates = []
    for th in range(1, 5):
        for tw in range(1, 5):
            tile = np.empty((th, tw), dtype=np.int8)
            for y in range(th):
                for x in range(tw):
                    cell = grid[(rows % th == y) & (cols % tw == x)]
                    tile[y, x] = np.bincount(cell, minlength=10).argmax()
            if len(np.unique(tile)) < 2:
                continue
            wallpaper = tile[rows % th, cols % tw]
            errors = int(np.count_nonzero(grid != wallpaper))
            candidates.append((errors, th*tw, th, tw, tile, wallpaper))
    candidates.sort(key=lambda item: item[:4])
    return candidates


def _recover(grid):
    solutions = []
    for _, area, th, tw, tile, wallpaper in _wallpaper_candidates(grid):
        colors = tuple(int(c) for c in np.unique(grid[grid != wallpaper]))
        if not 3 <= len(colors) <= 5 or set(colors) & set(tile.flat):
            continue
        rects = []
        for color in colors:
            ys, xs = np.where(grid == color)
            y, x = int(ys.min()), int(xs.min())
            h, w = int(ys.max()-y+1), int(xs.max()-x+1)
            if h < 2 or w < 2:
                break
            solid = (h == 2 and w == 2) or bool(np.any(
                grid[y+1:y+h-1, x+1:x+w-1] == color))
            rects.append((y, x, h, w, color, solid))
        if len(rects) != len(colors):
            continue
        rects.sort(key=lambda rect: (-rect[2], -rect[3]))
        if rects[0][5] or any(
            rects[i][2] <= rects[i+1][2] or rects[i][3] <= rects[i+1][3]
            for i in range(len(rects)-1)
        ):
            continue
        rebuilt = wallpaper.copy()
        for rect in rects:
            _paint(rebuilt, rect)
        if not np.array_equal(rebuilt, grid):
            continue
        try:
            answer = _assemble(rects)
        except ValueError:
            continue
        solutions.append((area, tuple(rects), tile.copy(), answer))
    if not solutions:
        raise AssertionError('wallpaper and rectangles cannot be recovered')
    best_area = min(s[0] for s in solutions)
    best = [s for s in solutions if s[0] == best_area]
    if any(not np.array_equal(s[3], best[0][3]) for s in solutions):
        raise AssertionError('ambiguous output')
    _, rects, tile, answer = best[0]
    return {'rects': rects, 'tile': tile, 'answer': answer}


def _profile(rng, first=False):
    if first:
        choices = ((False, False, True), (False, True, True),
                   (False, False, True, True), (False, True, False, True, True),
                   (False, False, True, True, True), (False, False, False, True, True))
        return choices[int(rng.integers(len(choices)))]
    count = int(rng.integers(3, 6))
    return (False,) + tuple(bool(rng.integers(2)) for _ in range(count-2)) + (True,)


def _sizes(rng, outer, profile):
    minima = [2 if not role else 1 for role in profile[:-1]]
    maximum_last = outer - sum(minima)
    if maximum_last < 2:
        return None
    if maximum_last == 2 or rng.random() < .32:
        last = 2
    else:
        last = int(rng.integers(3, maximum_last+1))
    extra = maximum_last-last
    if extra:
        weights = rng.dirichlet(np.full(len(minima), .65))
        gaps = np.asarray(minima) + rng.multinomial(extra, weights)
    else:
        gaps = np.asarray(minima)
    sizes = [outer]
    for gap in gaps:
        sizes.append(sizes[-1]-int(gap))
    return sizes


def _tile(rng, object_colors, include_zero):
    th = int(rng.integers(1, 5))
    tw = int(rng.integers(2, 5))
    if th == 1 and rng.random() < .7:
        th = int(rng.integers(2, 5))
    choices = [c for c in range(10) if c not in object_colors and (include_zero or c != 0)]
    if include_zero and 0 not in choices:
        return None
    count = min(len(choices), int(rng.integers(2, 5)))
    if count < 2:
        return None
    if include_zero:
        palette = [0] + [int(c) for c in rng.choice([c for c in choices if c], count-1, replace=False)]
    else:
        palette = [int(c) for c in rng.choice(choices, count, replace=False)]
    mode = int(rng.integers(3))
    if mode == 0:
        base = rng.permutation(palette)
        tile = np.array([[base[(x+y) % len(base)] for x in range(tw)] for y in range(th)], dtype=np.int8)
    elif mode == 1:
        base = np.asarray([palette[x % len(palette)] for x in range(tw)], dtype=np.int8)
        shifts = rng.integers(0, tw, size=th)
        tile = np.array([np.roll(base, int(shift)) for shift in shifts], dtype=np.int8)
    else:
        tile = np.asarray(rng.choice(palette, size=(th, tw)), dtype=np.int8)
    if include_zero and 0 not in tile:
        tile[0, 0] = 0
    if len(np.unique(tile)) < 2:
        tile[0, 0] = palette[0]
        tile[-1, -1] = palette[1]
    return tile


def _intersection(a, b):
    y, x, h, w = a[:4]
    Y, X, H, W = b[:4]
    return max(0, min(y+h, Y+H)-max(y, Y))*max(0, min(x+w, X+W)-max(x, X))


def _position(rng, parent, h, w, H, W, mode):
    py, px, ph, pw = parent[:4]
    possible = []
    for y in range(H-h+1):
        for x in range(W-w+1):
            candidate = (y, x, h, w)
            overlap = _intersection(candidate, parent)
            inside = y >= py and x >= px and y+h <= py+ph and x+w <= px+pw
            if mode == 'separate':
                good = overlap == 0
            elif mode == 'nested':
                good = inside and (y > py or x > px)
            elif mode == 'corner':
                corners = ((py, px), (py, px+pw-1), (py+ph-1, px), (py+ph-1, px+pw-1))
                good = any(y <= cy < y+h and x <= cx < x+w for cy, cx in corners) and overlap < ph*pw
            elif mode == 'heavy':
                good = overlap >= min(h*w, ph*pw)//2 and overlap > 0
            else:
                good = overlap > 0 and not inside and overlap < h*w
            if good:
                possible.append((y, x))
    if not possible:
        return None
    return possible[int(rng.integers(len(possible)))]


def _hidden_corners(grid, rects):
    count = 0
    for rect in rects[:-1]:
        y, x, h, w, color, _ = rect
        if any(grid[cy, cx] != color for cy, cx in
               ((y, x), (y, x+w-1), (y+h-1, x), (y+h-1, x+w-1))):
            count += 1
    return count


def _make_pair(rng, regime, profile, zero_mode, required_wallpaper=None, require_corner=False,
               require_blank=False):
    n = len(profile)
    object_pool = [c for c in range(1, 10) if c != required_wallpaper]
    for attempt in range(320):
        colors = tuple(int(c) for c in rng.choice(object_pool, n, replace=False))
        lower = max(8, 2+sum(2 if not role else 1 for role in profile[:-1]))
        upper = 20
        if regime == 'target_test':
            scale = rng.random()
            if scale < .28:
                upper = 13
            elif scale < .60:
                lower, upper = max(lower, 19), 23
        outer_h = int(rng.integers(lower, max(lower, upper)+1))
        outer_w = int(rng.integers(lower, max(lower, upper)+1))
        heights, widths = _sizes(rng, outer_h, profile), _sizes(rng, outer_w, profile)
        if heights is None or widths is None:
            continue
        H = int(rng.integers(outer_h+2, min(30, outer_h+11)+1))
        W = int(rng.integers(outer_w+2, min(30, outer_w+11)+1))
        y0 = int(rng.integers(1, H-outer_h))
        x0 = int(rng.integers(1, W-outer_w))
        rects = [(y0, x0, outer_h, outer_w, colors[0], False)]
        modes = ('separate', 'nested', 'cross', 'corner', 'heavy')
        for i in range(1, n):
            parent = rects[int(rng.integers(i))]
            if require_corner and i == 1:
                mode = 'corner'
            else:
                mode = modes[int(rng.integers(len(modes)))]
            pos = _position(rng, parent, heights[i], widths[i], H, W, mode)
            if pos is None:
                pos = _position(rng, parent, heights[i], widths[i], H, W, 'cross')
            if pos is None:
                break
            rects.append((pos[0], pos[1], heights[i], widths[i], colors[i], profile[i]))
        if len(rects) != n:
            continue
        tile = _tile(rng, colors, zero_mode)
        if tile is None or (required_wallpaper is not None and required_wallpaper not in tile):
            if required_wallpaper is not None:
                # Replace a nonzero tile entry to exhibit cross-pair role exchange.
                tile = _tile(rng, colors, zero_mode)
                if tile is None:
                    continue
                tile[-1, -1] = required_wallpaper
            else:
                continue
        rr, cc = np.indices((H, W))
        inp = tile[rr % tile.shape[0], cc % tile.shape[1]].copy()
        for rect in rects:
            _paint(inp, rect)
        if require_corner and not _hidden_corners(inp, rects):
            continue
        answer = _assemble(rects)
        if require_blank and 0 not in answer:
            continue
        if zero_mode and 0 not in inp:
            continue
        if not zero_mode and 0 in inp:
            continue
        try:
            recovered = _recover(inp)
        except AssertionError:
            continue
        if not np.array_equal(recovered['answer'], answer):
            continue
        if recovered['rects'] != tuple(rects):
            continue
        return {'input': inp, 'output': answer}
    raise RuntimeError('unable to construct recoverable pair')


def _check_count(value, low, high, name):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name + ' is out of range')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, 'num_train')
    _check_count(num_test, 1, 4, 'num_test')
    if num_train is None and num_test is None:
        num_test = 1+int(seed) % 4
        low = max(2, num_test+1)
        num_train = low+(int(seed)//4) % (11-low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test)+1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    puzzle = {'train': [], 'test': []}
    seen_inputs, seen_outputs = set(), set()
    first_outer_color = None
    first_profile = None
    train_zero_first = bool(rng.integers(2))
    for group, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(24):
                child = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                profile = _profile(child, first=(group == 'train' and index == 0))
                if group == 'train' and index == 1:
                    has_outlines = any(not first_profile[i] and not first_profile[i+1]
                                       for i in range(len(first_profile)-1))
                    has_solids = any(first_profile[i] and first_profile[i+1]
                                     for i in range(len(first_profile)-1))
                    if not has_outlines:
                        profile = ((False, False, True), (False, False, True, True))[
                            int(child.integers(2))]
                    elif not has_solids:
                        profile = ((False, True, True), (False, False, True, True))[
                            int(child.integers(2))]
                if group == 'train' and index < 2:
                    zero_mode = train_zero_first if index == 0 else not train_zero_first
                else:
                    zero_mode = bool(child.integers(2))
                required = first_outer_color if group == 'train' and index == 1 else None
                pair = _make_pair(child, 'source_train' if group == 'train' else 'target_test',
                                  profile, zero_mode, required,
                                  require_corner=(group == 'test' and index == 0) or (group == 'train' and index == 1),
                                  require_blank=(group == 'train' and index < 2 and not zero_mode))
                ikey = (pair['input'].shape, pair['input'].tobytes())
                okey = (pair['output'].shape, pair['output'].tobytes())
                if ikey in seen_inputs or (group == 'train' and okey in seen_outputs):
                    continue
                seen_inputs.add(ikey)
                if group == 'train':
                    seen_outputs.add(okey)
                puzzle[group].append(pair)
                if group == 'train' and index == 0:
                    first_outer_color = _recover(pair['input'])['rects'][0][4]
                    first_profile = profile
                break
            else:
                raise RuntimeError('unable to create distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[g], list) for g in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs, train_outputs = set(), set(), set()
    meta = {'train': [], 'test': []}
    for group in ('train', 'test'):
        for pair in puzzle[group]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in 'iu' and all(1 <= z <= 30 for z in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.shape, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            digest = hashlib.sha256(bytes(inp.shape)+np.asarray(inp, dtype=np.uint8).tobytes()).hexdigest()
            assert digest not in _SOURCE_INPUT_DIGESTS
            recovered = _recover(inp)
            rects = recovered['rects']
            assert np.array_equal(recovered['answer'], out)
            assert out.shape == (rects[0][2], rects[0][3])
            assert inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]
            assert len(rects) in (3, 4, 5) and not rects[0][5] and rects[-1][5]
            assert len({r[4] for r in rects}) == len(rects)
            assert set(r[4] for r in rects).isdisjoint(set(recovered['tile'].flat))
            assert all(rects[i][2]-rects[i+1][2] >= (1 if rects[i][5] else 2) and
                       rects[i][3]-rects[i+1][3] >= (1 if rects[i][5] else 2)
                       for i in range(len(rects)-1))
            if group == 'train':
                okey = (out.shape, out.tobytes())
                assert okey not in train_outputs
                train_outputs.add(okey)
            meta[group].append(recovered)
    train = meta['train']
    assert (0 in train[0]['tile']) != (0 in train[1]['tile'])
    assert all(0 in m['answer'] for m in train[:2] if 0 not in m['tile'])
    assert any(not r[5] and not m['rects'][i+1][5] for m in train for i, r in enumerate(m['rects'][:-1]))
    assert any(r[5] and m['rects'][i+1][5] for m in train for i, r in enumerate(m['rects'][:-1]))
    assert train[0]['rects'][0][4] in set(train[1]['tile'].flat)
    assert _hidden_corners(puzzle['train'][1]['input'], train[1]['rects']) >= 1
    assert _hidden_corners(puzzle['test'][0]['input'], meta['test'][0]['rects']) >= 1


def _same_puzzle(a, b):
    return all(len(a[g]) == len(b[g]) and all(np.array_equal(x[k], y[k])
        for x, y in zip(a[g], b[g]) for k in ('input', 'output')) for g in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, shapes, profiles, tiles, layouts = set(), set(), set(), set(), set(), set()
    hidden, terminal, zero_test, smaller_test = 0, 0, 0, 0
    larger_test, detached, repeated_tiles, non_diagonal_tiles = 0, 0, 0, 0
    both_periods, role_switch_puzzles, gap_pairs, outer_positions = 0, 0, set(), set()
    placements = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        max_train = max(p['output'].shape[0]*p['output'].shape[1] for p in puzzle['train'])
        roles_by_color = {c: set() for c in range(10)}
        puzzle_zero_modes, puzzle_profiles = set(), set()
        for group in ('train', 'test'):
            for pair in puzzle[group]:
                m = _recover(pair['input'])
                assert np.array_equal(_assemble(m['rects']), pair['output'])
                shapes.add(pair['output'].shape)
                profile = tuple(r[5] for r in m['rects'])
                profiles.add(profile)
                puzzle_profiles.add(profile)
                tiles.add((m['tile'].shape, tuple(int(c) for c in m['tile'].flat)))
                layouts.add((pair['input'].shape, tuple(r[:4] for r in m['rects'])))
                hidden += _hidden_corners(pair['input'], m['rects'])
                terminal += m['rects'][-1][2:4] == (2, 2)
                detached += _intersection(m['rects'][-1], m['rects'][-2]) == 0
                repeated_tiles += len(set(m['tile'].flat)) < m['tile'].size
                both_periods += m['tile'].shape[0] > 1 and m['tile'].shape[1] > 1
                non_diagonal_tiles += m['tile'].shape[0] > 1 and any(
                    not all(m['tile'][y, x] == m['tile'][(y-1) % m['tile'].shape[0],
                                                   (x+1) % m['tile'].shape[1]]
                            for x in range(m['tile'].shape[1]))
                    for y in range(1, m['tile'].shape[0]))
                outer_positions.add(m['rects'][0][1])
                puzzle_zero_modes.add(0 in m['tile'])
                for rect in m['rects']:
                    roles_by_color[rect[4]].add(rect[5])
                for a, b in zip(m['rects'], m['rects'][1:]):
                    gap_pairs.add((a[2]-b[2], a[3]-b[3]))
                    area = _intersection(a, b)
                    inside = (b[0] >= a[0] and b[1] >= a[1] and
                              b[0]+b[2] <= a[0]+a[2] and b[1]+b[3] <= a[1]+a[3])
                    placements.add('separate' if area == 0 else 'nested' if inside else 'crossing')
                if group == 'test':
                    zero_test += int(0 in m['tile'])
                    smaller_test += int(pair['output'].size <= max_train)
                    larger_test += int(pair['output'].size > max_train)
        role_switch_puzzles += any(len(v) > 1 for v in roles_by_color.values())
        assert len(puzzle_zero_modes) == 2 and len(puzzle_profiles) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(profiles) >= 6 and len(tiles) >= 30 and len(layouts) >= 100
    assert hidden >= 30 and terminal >= 10 and zero_test >= 5 and smaller_test >= 5
    assert larger_test >= 5 and detached >= 20 and repeated_tiles >= 50
    assert both_periods >= 30 and non_diagonal_tiles >= 20
    assert role_switch_puzzles >= 10 and len(gap_pairs) >= 25 and len(outer_positions) >= 7
    assert placements == {'separate', 'nested', 'crossing'}
    for seed, counts in ((101, (2, 1)), (102, (10, 1)), (103, (10, 4)),
                         (104, (None, 1)), (105, (None, 4)), (106, (2, None)), (107, (10, None))):
        a = generate_puzzle(seed, *counts)
        assert _same_puzzle(a, generate_puzzle(seed, *counts))
        if counts[0] is not None:
            assert len(a['train']) == counts[0]
        if counts[1] is not None:
            assert len(a['test']) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
