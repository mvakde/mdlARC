"""Reroute a damaged dividing seam, then dock its two cargo regions.

Only standard-library modules and NumPy are used. Public solve is input-only.
Construction renders labels from clean geometry; damage is accepted only when
the input reconstruction recovers that geometry exactly.
"""
from collections import deque
import json
import numpy as np

REPAIR = 2
DIRECTION = 4
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
AVAILABLE = tuple(c for c in range(10) if c not in (REPAIR, DIRECTION))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _side(p, shape):
    sides = [i for i, yes in enumerate((p[0] == 0, p[1] == shape[1]-1,
                                        p[0] == shape[0]-1, p[1] == 0)) if yes]
    _require(len(sides) == 1, "route endpoints must avoid canvas corners")
    return sides[0]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal line")
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    n = max(abs(b[0]-a[0]), abs(b[1]-a[1]))
    return [(a[0]+i*d[0], a[1]+i*d[1]) for i in range(n+1)]


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _components(cells):
    todo, result = set(cells), []
    while todo:
        first = min(todo)
        todo.remove(first)
        component, stack = {first}, [first]
        while stack:
            for q in _neighbors(stack.pop(), todo):
                todo.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _simple_path(cells, shape):
    if len(cells) < 4 or len(_components(cells)) != 1:
        return False
    ends = [p for p in cells if len(_neighbors(p, cells)) == 1]
    return (len(ends) == 2 and all(_boundary(p, shape) for p in ends)
            and all(len(_neighbors(p, cells)) == 2 for p in cells if p not in ends)
            and {p for p in cells if _boundary(p, shape)} == set(ends)
            and all(sum((p[0] == 0, p[0] == shape[0]-1,
                         p[1] == 0, p[1] == shape[1]-1)) == 1 for p in ends))


def _corners(side, shape):
    h, w = shape
    return (((0, 0), (0, w-1)), ((0, w-1), (h-1, w-1)),
            ((h-1, w-1), (h-1, 0)), ((h-1, 0), (0, 0)))[side]


def _regions(shape, seam, exit_side):
    """The seam is impermeable; cargo and annotations do not block flood fill."""
    labels = np.full(shape, -1, dtype=np.int16)
    number = 0
    for r in range(shape[0]):
        for c in range(shape[1]):
            p = (r, c)
            if p in seam or labels[p] >= 0:
                continue
            labels[p] = number
            queue = deque([p])
            while queue:
                current = queue.popleft()
                for d in DIRS:
                    q = _add(current, d)
                    if _inside(q, shape) and q not in seam and labels[q] < 0:
                        labels[q] = number
                        queue.append(q)
            number += 1
    _require(number == 2, "seam must divide the canvas into exactly two regions")
    corners = _corners(exit_side, shape)
    ids = [int(labels[p]) for p in corners]
    _require(set(ids) == {0, 1}, "exit-side corners must lie in different regions")
    return labels, dict(zip(ids, corners))


def _reconstruct(inp, route, background):
    visible = {tuple(map(int, p)) for p in np.argwhere((inp == route) | (inp == REPAIR))}
    additions = set()
    # A short gap is an all-background straight run, bounded by route waypoints.
    # All candidates are computed from the unchanged visible cells, not iterated.
    for p in sorted(visible):
        for d in ((1, 0), (0, 1)):
            for distance in (2, 3):
                q = (p[0]+distance*d[0], p[1]+distance*d[1])
                if q in visible:
                    between = _line(p, q)[1:-1]
                    if all(int(inp[x]) == background for x in between):
                        additions.update(between)
    cells = visible | additions
    _require(_simple_path(cells, inp.shape), "damaged route has no permitted simple completion")
    return cells, additions


def _walk_branch(pivot, start, path, shape):
    previous, current, result = pivot, start, []
    while True:
        _require(current not in result, "route cycle")
        result.append(current)
        choices = [p for p in _neighbors(current, path) if p != previous]
        if not choices:
            _require(_boundary(current, shape), "terminal branch does not reach boundary")
            return result
        _require(len(choices) == 1, "branched seam")
        previous, current = current, choices[0]


def _analyze(input_grid):
    inp = _array(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(counts.argmax())
    _require(counts[background] > inp.size // 2, "background must occupy a strict majority")
    _require(background not in (REPAIR, DIRECTION), "reserved annotation used as background")
    rim = np.zeros(inp.shape, dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    border_colors = set(map(int, np.unique(inp[rim]))) - {background, REPAIR, DIRECTION}
    _require(len(border_colors) == 1, "border must identify one route color")
    route = border_colors.pop()
    _require(not np.any((inp == REPAIR) & rim), "repair waypoints must be interior")
    qs = [tuple(map(int, p)) for p in np.argwhere(inp == DIRECTION)]
    _require(len(qs) == 1, "exactly one directional marker required")
    q = qs[0]
    path, gaps = _reconstruct(inp, route, background)
    pivots = _neighbors(q, path)
    _require(len(pivots) == 1, "direction marker must have one route neighbor")
    pivot = pivots[0]
    d = (q[0]-pivot[0], q[1]-pivot[1])
    old_start = (pivot[0]-d[0], pivot[1]-d[1])
    _require(old_start in path and len(_neighbors(pivot, path)) == 2, "invalid directional pivot")
    other = next(p for p in _neighbors(pivot, path) if p != old_start)
    _require((other[0]-pivot[0])*d[0] + (other[1]-pivot[1])*d[1] == 0,
             "retained continuation must initially be perpendicular")
    old_walk = _walk_branch(pivot, old_start, path, inp.shape)
    kept_walk = _walk_branch(pivot, other, path, inp.shape)
    ray = _ray(pivot, d, inp.shape)
    _require(ray, "empty new ray")
    kept = path - set(old_walk)
    _require(not (set(ray) & kept), "new ray intersects retained continuation")
    new_path = kept | set(ray)
    _require(_simple_path(new_path, inp.shape), "rerouted seam is not simple")
    exit_side = _side(ray[-1], inp.shape)
    _require(_side(kept_walk[-1], inp.shape) != exit_side, "both new endpoints on same side")
    _require(_side(old_walk[-1], inp.shape) != _side(kept_walk[-1], inp.shape),
             "both original endpoints on same side")
    payload = {tuple(map(int, p)): int(inp[tuple(p)]) for p in np.argwhere(
        ~np.isin(inp, [background, route, REPAIR, DIRECTION]))}
    _require(payload and all(not _boundary(p, inp.shape) for p in payload), "cargo must be interior")
    _require(not (set(payload) & path), "cargo overlaps original route")
    labels, corners = _regions(inp.shape, new_path, exit_side)
    groups = {i: {p: color for p, color in payload.items() if labels[p] == i} for i in (0, 1)}
    _require(all(groups.values()), "both new regions require cargo")
    pins = {p: color for p, color in payload.items() if p in new_path}
    return dict(inp=inp, background=background, route=route, path=path, gaps=gaps,
                pivot=pivot, direction=d, old_walk=old_walk, kept_walk=kept_walk,
                new_path=new_path, ray=set(ray), exit_side=exit_side,
                labels=labels, corners=corners, groups=groups, pins=pins, payload=payload)


def _translation(cells, corner, shape):
    rs, cs = zip(*cells)
    return ((-min(rs) if corner[0] == 0 else shape[0]-1-max(rs)),
            (-min(cs) if corner[1] == 0 else shape[1]-1-max(cs)))


def _render(shape, background, route, seam, groups, corners, pins):
    out = np.full(shape, background, dtype=np.int8)
    for p in seam:
        out[p] = route
    for p, color in pins.items():
        out[p] = color
    occupied = set(seam)
    for i in sorted(groups):
        group = groups[i]
        if not group:
            continue
        shift = _translation(group, corners[i], shape)
        targets = {_add(p, shift): color for p, color in group.items()}
        _require(all(_inside(p, shape) for p in targets), "cargo clipped")
        _require(not (set(targets) & occupied), "docked cargo collision")
        occupied.update(targets)
        for p, color in targets.items():
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _render(info['inp'].shape, info['background'], info['route'], info['new_path'],
                   info['groups'], info['corners'], info['pins'])


def _without_rerouting(info):
    side = _side(info['old_walk'][-1], info['inp'].shape)
    labels, corners = _regions(info['inp'].shape, info['path'], side)
    groups = {i: {p: c for p, c in info['payload'].items() if labels[p] == i} for i in (0, 1)}
    try:
        return _render(info['inp'].shape, info['background'], info['route'], info['path'], groups, corners, {})
    except ValueError:
        # Overlap itself disproves applying corner docking to the unrevised cut.
        return None


def _straight_partition(info):
    # Treat the full axis through the new ray as A's divider, ignoring the bend.
    pivot, d, shape = info['pivot'], info['direction'], info['inp'].shape
    corners = _corners(info['exit_side'], shape)
    groups = {0: {}, 1: {}}
    for p, color in info['payload'].items():
        if p in info['pins']:
            continue
        coordinate = p[1] if d[0] else p[0]
        origin = pivot[1] if d[0] else pivot[0]
        if coordinate == origin:
            return None  # An off-ray axial cell already distinguishes the rules.
        nearest = 0 if (corners[0][1] if d[0] else corners[0][0]) == 0 else 1
        groups[nearest if coordinate < origin else 1-nearest][p] = color
    try:
        return _render(shape, info['background'], info['route'], info['new_path'],
                       groups, dict(enumerate(corners)), info['pins'])
    except ValueError:
        return None


def _separately_docked(info, by_color=False):
    """Incorrect alternative: dock every component or color independently."""
    shape = info['inp'].shape
    out = np.full(shape, info['background'], dtype=np.int8)
    for p in info['new_path']:
        out[p] = info['route']
    for p, color in info['pins'].items():
        out[p] = color
    for i, group in info['groups'].items():
        parts = ([{p for p, c in group.items() if c == color} for color in sorted(set(group.values()))]
                 if by_color else _components(group))
        for part in parts:
            shift = _translation(part, info['corners'][i], shape)
            for p in sorted(part):
                out[_add(p, shift)] = group[p]
    return out


def _turns(walk):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _random_branch(rng, pivot, first, shape, segments):
    """Sample independently spaced rectilinear segments, ending at an edge."""
    walk = [pivot]
    current, direction = pivot, first
    for _ in range(segments):
        possible = [n for n in range(3, 9) if _inside(
            (current[0]+n*direction[0], current[1]+n*direction[1]), shape)
            and not _boundary((current[0]+n*direction[0], current[1]+n*direction[1]), shape)]
        if not possible:
            return None
        n = int(rng.choice(possible))
        end = (current[0]+n*direction[0], current[1]+n*direction[1])
        walk.extend(_line(current, end)[1:])
        current = end
        direction = (-direction[1], direction[0]) if rng.integers(2) else (direction[1], -direction[0])
    tail = _ray(current, direction, shape)
    if not tail:
        return None
    walk += tail
    return walk


def _geometry(rng, shape, rich=False):
    h, w = shape
    for _ in range(80):
        pivot = (int(rng.integers(6, h-5)), int(rng.integers(6, w-5)))
        old = _random_branch(rng, pivot, (1, 0), shape, int(rng.integers(1, 4 if rich else 3)))
        kept = _random_branch(rng, pivot, (0, 1 if rng.integers(2) else -1), shape,
                              int(rng.integers(0, 4 if rich else 3)))
        if old is None or kept is None:
            continue
        path = set(old + kept)
        new_ray = set(_ray(pivot, (-1, 0), shape))
        new_path = set(kept) | new_ray
        if (len(path) != len(old)+len(kept)-1 or not _simple_path(path, shape)
                or not _simple_path(new_path, shape) or set(kept) & new_ray
                or _neighbors(_add(pivot, (-1, 0)), path) != [pivot]
                or _side(kept[-1], shape) == 0 or _side(old[-1], shape) == _side(kept[-1], shape)):
            continue
        try:
            _regions(shape, new_path, 0)
        except ValueError:
            continue
        return pivot, old, kept
    # A varied guaranteed geometry; this is also a legitimate family member.
    pivot = (h//2, w//2)
    r, c = pivot
    old_points = [pivot, (r+3, c), (r+3, c-3), (h-1, c-3)]
    kept_points = [pivot, (r, c+3), (r+3, c+3), (r+3, w-1)]
    def polyline(points):
        return [points[0]] + [p for a, b in zip(points, points[1:]) for p in _line(a, b)[1:]]
    return pivot, polyline(old_points), polyline(kept_points)


def _cargo(rng, shape, pivot, path, kept, count, colors, witness):
    r, c = pivot
    new_path = set(kept) | set(_ray(pivot, (-1, 0), shape))
    labels, corners = _regions(shape, new_path, 0)
    free = {(rr, cc) for rr in range(max(2, r-4), min(shape[0]-2, r+5))
            for cc in range(max(2, c-4), min(shape[1]-2, c+5))
            if (rr, cc) not in path and (rr, cc) != (r-1, c)}
    for _ in range(30):
        cargo = {}
        clusters = []
        if witness:
            bar = {(r-2, cc) for cc in range(c-2, c+3)}
            if not bar <= free:
                return None
            clusters.append(bar)
            for p in bar:
                cargo[p] = colors[0]
        for index in range(count-len(clusters)):
            candidates = sorted(p for p in free if p not in cargo and not _neighbors(p, cargo))
            if not candidates:
                break
            if witness and index < 2:
                # Remote satellites on each side expose collective docking.
                chosen_label = index
                candidates = [p for p in candidates if labels[p] == chosen_label]
                if not candidates:
                    break
            center = candidates[int(rng.integers(len(candidates)))]
            cells = {center}
            target_size = 1 if witness and index < 2 else int(rng.integers(1, 7))
            for _ in range(target_size-1):
                frontier = sorted({q for p in cells for q in _neighbors(p, free-cells)
                                   if q not in cargo and not _neighbors(q, cargo)})
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(len(frontier)))])
            color = colors[(index+1) % len(colors)] if witness else int(rng.choice(colors))
            clusters.append(cells)
            cargo.update({p: color for p in cells})
        if len(clusters) != count:
            continue
        groups = {i: {p: color for p, color in cargo.items() if labels[p] == i} for i in (0, 1)}
        pins = {p: color for p, color in cargo.items() if p in new_path}
        if not all(groups.values()) or (witness and (not pins or not all(
                len(_components(group)) >= 2 for group in groups.values()))):
            continue
        if witness and not any(len(set(group.values())) > 1 for group in groups.values()):
            continue
        try:
            _render(shape, 0, 1, new_path, groups, corners, pins)
        except ValueError:
            continue
        return cargo, groups, corners, pins
    return None


def _damage(rng, inp, path, old, kept, pivot, background, route, witness):
    corners = [p for p in sorted(path) if not _boundary(p, inp.shape)
               and len(_neighbors(p, path)) == 2
               and (_neighbors(p, path)[0][0]+_neighbors(p, path)[1][0],
                    _neighbors(p, path)[0][1]+_neighbors(p, path)[1][1]) != (2*p[0], 2*p[1])]
    mandatory = [pivot] if witness else []
    mandatory += [p for p in old[1:-1] if p in corners][:1]
    candidates = sorted(p for p in path if not _boundary(p, inp.shape))
    rng.shuffle(candidates)
    marks = list(dict.fromkeys(mandatory + candidates[:int(rng.integers(1, 6))]))
    for p in marks:
        inp[p] = REPAIR
    # A two-cell surviving-branch hole in the integrated demonstration teaches
    # interpolation beyond the immediate-neighbor shortcut.
    gap_count = 0
    if witness:
        windows = [kept[j:j+2] for j in range(2, len(kept)-3)
                   if all(inp[p] == route and p not in corners for p in kept[j:j+2])]
        rng.shuffle(windows)
        for window in windows:
            for p in window:
                inp[p] = background
            try:
                recovered, _ = _reconstruct(inp, route, background)
                _require(recovered == path, "incorrect two-cell hole completion")
            except ValueError:
                for p in window:
                    inp[p] = route
            else:
                gap_count = 2
                break
        if not gap_count:
            return False
    # One surviving-branch gap is required, not just damage on a deleted branch.
    gap_candidates = [p for p in kept[2:-2] if inp[p] == route and p not in corners
                      and all(inp[q] == route for q in _neighbors(p, path))]
    rng.shuffle(gap_candidates)
    desired = int(rng.integers(1, 4))
    for p in gap_candidates:
        if gap_count >= desired:
            break
        inp[p] = background
        try:
            recovered, _ = _reconstruct(inp, route, background)
            _require(recovered == path, "incorrect reconstructed geometry")
        except ValueError:
            inp[p] = route
        else:
            gap_count += 1
            if gap_count >= desired:
                break
    if not gap_count:
        return False
    try:
        recovered, _ = _reconstruct(inp, route, background)
        return recovered == path
    except ValueError:
        return False


def _make_pair(rng, rotation, colors, count, witness, rich):
    for attempt in range(120):
        shape = (int(rng.integers(17, 31)), int(rng.integers(17, 31)))
        pivot, old, kept = _geometry(rng, shape, rich)
        path = set(old + kept)
        cargo_result = _cargo(rng, shape, pivot, path, kept, count, colors[2:], witness)
        if cargo_result is None:
            continue
        cargo, groups, corners, pins = cargo_result
        background, route = colors[:2]
        inp = np.full(shape, background, dtype=np.int8)
        for p in path:
            inp[p] = route
        for p, color in cargo.items():
            inp[p] = color
        inp[_add(pivot, (-1, 0))] = DIRECTION
        if not _damage(rng, inp, path, old, kept, pivot, background, route, witness):
            continue
        new_path = set(kept) | set(_ray(pivot, (-1, 0), shape))
        out = _render(shape, background, route, new_path, groups, corners, pins)
        pair = {k: np.rot90(a, -rotation).copy() for k, a in (('input', inp), ('output', out))}
        if rng.integers(2):
            # Reflect parallel to the exit side: the exit direction stays fixed.
            axis = 1 if rotation % 2 == 0 else 0
            pair = {k: np.flip(a, axis=axis).copy() for k, a in pair.items()}
        try:
            info = _analyze(pair['input'])
            _require(np.array_equal(solve(pair['input']), pair['output']), "latent label disagreement")
            # The payload answer itself must depend on rerouting.
            mutant = _without_rerouting(info)
            if mutant is not None and np.array_equal(np.isin(mutant, colors[2:]), np.isin(pair['output'], colors[2:])):
                continue
            if witness and _turns([info['pivot']] + info['old_walk']) < 1:
                continue
            if witness:
                straight = _straight_partition(info)
                if straight is not None and np.array_equal(straight, pair['output']):
                    continue
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded seam/cargo construction exhausted")


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _integer(num_train, 'num_train', 2, 10)
    ntest = _integer(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    held_side = int(rng.integers(4))
    mode = int(rng.integers(3))
    pool = [i for i in range(4) if i != held_side] if mode == 1 else list(range(4))
    if mode == 2:
        pool = [int(rng.integers(4))]
    train_dirs = [int(rng.choice(pool)) for _ in range(ntrain)]
    if len(pool) > 1:
        train_dirs[1] = int(rng.choice([i for i in pool if (i-train_dirs[0]) % 4 in (1, 3)]))
    test_dirs = [held_side if mode == 1 else int(rng.choice(train_dirs)) for _ in range(ntest)]
    variable_palette = bool(rng.random() < .7)
    palette = [int(c) for c in rng.choice(AVAILABLE, 5, replace=False)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, directions in (('train', train_dirs), ('test', test_dirs)):
        for index, rotation in enumerate(directions):
            witness = split == 'train' and index == 0
            count = int(rng.integers(4 if witness else 3, 7)) if split == 'train' else int(rng.integers(8, 12))
            if variable_palette:
                palette = [int(c) for c in rng.choice(AVAILABLE, 5, replace=False)]
            payload_count = 2 if witness else int(rng.integers(1, 4))
            roles = palette[:2] + palette[2:2+payload_count]
            for _ in range(12):
                pair = _make_pair(rng, rotation, roles, count, witness, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _integer(len(puzzle['train']), 'num_train', 2, 10)
    _integer(len(puzzle['test']), 'num_test', 1, 4)
    _require(len(puzzle['train']) > len(puzzle['test']), "invalid split counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _array(pair['input']), _array(pair['output'])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = _render(inp.shape, info['background'], info['route'], info['new_path'],
                               info['groups'], info['corners'], info['pins'])
            _require(np.array_equal(out, expected), "output violates seam/group transformation")
            _require(not np.any(np.isin(out, [REPAIR, DIRECTION])), "annotations survived")
            cargo_colors = set(info['payload'].values())
            for color in cargo_colors:
                _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color), "cargo color not conserved")
            for p, color in info['pins'].items():
                _require(int(out[p]) == color, "on-seam cargo moved")
            _require(info['gaps'] and bool(set(info['gaps']) & set(info['kept_walk'])), "missing retained-gap witness")
            ablation = _without_rerouting(info)
            _require(ablation is None or not np.array_equal(np.isin(ablation, list(cargo_colors)),
                                                            np.isin(out, list(cargo_colors))),
                     "cargo answer does not depend on rerouting")
            unchanged_cargo = np.full_like(inp, info['background'])
            for p, color in info['payload'].items():
                unchanged_cargo[p] = color
            _require(not np.array_equal(np.isin(out, list(cargo_colors)),
                                        np.isin(unchanged_cargo, list(cargo_colors))), "cargo docking has no effect")
            features[split].append((info, len(_components(info['payload']))))
    train = [f[0] for f in features['train']]
    _require(any(f['pins'] and all(len(_components(g)) >= 2 for g in f['groups'].values())
                 and any(len(set(g.values())) > 1 for g in f['groups'].values())
                 and int(f['inp'][f['pivot']]) == REPAIR
                 and any(_neighbors(p, f['gaps']) for p in f['gaps'])
                 and _turns([f['pivot']]+f['old_walk']) >= 1 for f in train),
             "missing integrated training witness")
    _require(any(_straight_partition(f) is None or not np.array_equal(_straight_partition(f), solve(f['inp']))
                 for f in train), "training does not distinguish topological from axial grouping")
    for by_color in (False, True):
        _require(any(not np.array_equal(_separately_docked(f, by_color), solve(f['inp'])) for f in train),
                 "training does not distinguish collective docking from component/color docking")
    _require(min(f[1] for f in features['test']) > max(f[1] for f in features['train']),
             "missing fragment-count train/test shift")
    return True


def self_test():
    """Bounded 42-seed audit, structural diversity, equivariance, and negatives."""
    shapes, structures, palettes, sides, island_counts, turn_counts = set(), set(), set(), set(), set(), set()
    signatures, pins, fragmented, topological, old_markers = set(), 0, 0, 0, 0
    component_shortcuts, color_shortcuts = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                f = _analyze(inp)
                shapes.add(inp.shape)
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                palettes.add((f['background'], f['route'], tuple(sorted(set(f['payload'].values())))))
                sides.add(f['exit_side'])
                island_counts.add(len(_components(f['payload'])))
                turn_counts.add(_turns([f['pivot']]+f['old_walk']) + _turns([f['pivot']]+f['kept_walk']))
                pins += bool(f['pins'])
                fragmented += any(len(_components(g)) > 1 for g in f['groups'].values())
                shortcut = _straight_partition(f)
                topological += shortcut is None or not np.array_equal(shortcut, out)
                component_shortcuts += not np.array_equal(_separately_docked(f), out)
                color_shortcuts += not np.array_equal(_separately_docked(f, True), out)
                old_markers += sum(inp[p] == REPAIR for p in f['old_walk'])
                _require(np.array_equal(solve(inp.copy()), out), "solver disagreement")
                # Transform the entire scene; individual cargo pieces never rotate.
                if seed < 3:
                    for k in (1, 2, 3):
                        transformed = np.rot90(np.fliplr(inp), k).copy()
                        target = np.rot90(np.fliplr(out), k).copy()
                        _require(np.array_equal(solve(transformed), target), "symmetry failure")
        if seed < 3:
            pair = puzzle['test'][0]
            f = _analyze(pair['input'])
            sites = [min(f['new_path']), min(f['old_walk']), f['pivot']]
            sites += [min(f['pins'])] if f['pins'] else []
            sites += [min(f['groups'][0])]
            for p in sites:
                mutated = {'train': puzzle['train'], 'test': [dict(pair)]}
                bad = pair['output'].copy()
                bad[p] = (int(bad[p])+1) % 10
                mutated['test'][0]['output'] = bad
                try:
                    validate_generated_puzzle(mutated)
                except ValueError:
                    pass
                else:
                    raise AssertionError("corrupt output accepted")
    assert len(signatures) == 42 and len(structures) >= 200 and len(shapes) >= 70
    assert len(palettes) >= 80 and sides == {0, 1, 2, 3}
    assert {3, 4, 5, 6, 8, 9, 10, 11} <= island_counts
    assert len(turn_counts) >= 4 and max(turn_counts) >= 4
    assert min(pins, fragmented, topological, old_markers) >= 42
    assert component_shortcuts >= 42 and color_shortcuts >= 42
    # Hand-written 9x12 fixture: a one-cell new ray, a marked pivot, deletion
    # through two bends, a retained hole, and cargo docking at both top corners.
    # The sampler uses wider margins, but the input-only solver has no such limit.
    def digit_grid(rows):
        return np.array([[int(c) for c in row] for row in rows.split('/')], dtype=np.int8)
    short_input = digit_grid('000040000000/000028888000/000080008070/005080000070/'
                             '000080008000/088820008888/080000000000/080000000000/080000000000')
    short_output = digit_grid('500080000007/000088888007/000000008000/000000008000/'
                              '000000008000/000000008888/000000000000/000000000000/000000000000')
    assert np.array_equal(solve(short_input), short_output)
    a, b = generate_puzzle(94179426, 2, 1), generate_puzzle(94179426, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test')
               for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            validate_generated_puzzle(generate_puzzle(100+ntrain*10+ntest, ntrain, ntest))
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {'seeds': 42, 'pairs': 210, 'distinct_shapes': len(shapes),
            'normalized_structures': len(structures), 'role_palettes': len(palettes),
            'exit_sides': sorted(sides), 'cargo_islands': sorted(island_counts),
            'total_branch_turns': sorted(turn_counts), 'pinned_pairs': pins,
            'fragmented_groups': fragmented, 'topology_shortcut_failures': topological,
            'component_shortcut_failures': component_shortcuts, 'color_shortcut_failures': color_shortcuts,
            'old_branch_repair_markers': int(old_markers)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
