"""A growing pointed carrier collects fragments by feature-registered landings.

Only Python/NumPy are used. Generation coordinates are never supplied to solve.
The two patch colors, carrier color, and cargo color are read from geometry.
"""
from collections import deque
from itertools import combinations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "bad grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((g >= 0) & (g <= 9)), "bad color")
    return g.astype(np.int8, copy=False)


def _rot(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _move(cells, turns, delta=(0, 0), pivot=(0, 0)):
    result = {}
    for (r, c), color in cells.items():
        rr, cc = _rot((r - pivot[0], c - pivot[1]), turns)
        result[rr + delta[0], cc + delta[1]] = color
    return result


def _components(points, diagonal=False):
    unseen, groups = set(points), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _signature(cells):
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0, int(v)) for (r, c), v in cells.items()))


def _orbit(cells):
    return min(_signature(_move(cells, t)) for t in range(4))


def _ports(cells, colors):
    points = {p for p, v in cells.items() if v in colors}
    result = []
    for group in _components(points):
        patch = {p: cells[p] for p in group}
        r0, c0, r1, c1 = _bbox(group)
        _require(r1 > r0 and c1 > c0 and len(group) == (r1-r0+1)*(c1-c0+1),
                 "registration patches must be filled rectangles at least 2 by 2")
        _require(set(patch.values()) == set(colors), "patch must use both registration colors")
        _require(len({_signature(_move(patch, t)) for t in range(4)}) == 4,
                 "a registration patch must distinguish all quarter turns")
        result.append(patch)
    _require(bool(result), "missing registration patch")
    return result


def _pointed(layer):
    """Read a symmetric axis and a single projecting end from the bitmap."""
    found = []
    for tip in sorted(layer):
        for direction in range(4):
            offsets = {_rot((r-tip[0], c-tip[1]), 1-direction) for r, c in layer}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {(-r, c) for r, c in offsets} == offsets:
                found.append((tip, direction))
    return found[0] if len(found) == 1 else None


def _decode(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    _require(counts.max()*2 > g.size, "background must occupy more than half the board")
    bg = int(colors[counts.argmax()])
    comps = [{p: int(g[p]) for p in group}
             for group in _components(zip(*np.where(g != bg)), diagonal=True)]
    _require(len(comps) >= 3, "need a carrier and at least two stations")
    occurrences = {int(v): sum(v in set(c.values()) for c in comps) for v in colors if v != bg}
    candidates = []
    for index, comp in enumerate(comps):
        for ink in sorted(set(comp.values())):
            if occurrences[ink] != 1:
                continue
            layer = {p for p, v in comp.items() if v == ink}
            pointed = _pointed(layer)
            if pointed is None:
                continue
            rest = sorted(set(comp.values()) - {ink})
            if len(rest) != 3:
                continue
            for keys in combinations(rest, 2):
                cargo = next(v for v in rest if v not in keys)
                try:
                    patches = [_ports(c, keys) for c in comps]
                    _require(len(patches[index]) == 1, "carrier starts with one patch")
                    _require(all(set(c.values()) <= set(keys) | {cargo}
                                 for j, c in enumerate(comps) if j != index), "unknown station color")
                    _require(all(len(ps) in (1, 2) for j, ps in enumerate(patches) if j != index),
                             "station must contain one or two patches")
                    # IDs distinguish bitmap geometry/colors, never literal color commands.
                    per_id = {}
                    for ps in patches:
                        for patch in ps:
                            per_id.setdefault(_orbit(patch), []).append(patch)
                    _require(len(per_id[_orbit(patches[index][0])]) >= 2, "initial patch has no copy")
                    _require(all(len({_orbit(p) for p in ps}) == len(ps) for ps in patches),
                             "two patches in one fragment cannot have the same ID")
                    candidates.append((index, ink, keys, cargo, pointed, patches))
                except ValueError:
                    continue
    _require(len(candidates) == 1, "carrier or registration color roles are ambiguous")
    index, ink, keys, cargo, (tip, facing), patches = candidates[0]
    return g, bg, comps, index, ink, keys, cargo, tip, facing, patches


def _fits(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _landings(body, tip, facing, stations, keys, shape, *, geometric_only=False):
    """Enumerate complete feature-supported translations on cardinal rays."""
    candidates = {}
    for direction, (vr, vc) in enumerate(DIRS):
        rotated = _move(body, direction-facing, pivot=tip)
        carried_ports = _ports(rotated, keys)
        for index, station in stations.items():
            for carried in carried_ports:
                sig = _signature(carried)
                ar, ac, _, _ = _bbox(carried)
                for dock in _ports(station, keys):
                    if _signature(dock) != sig:
                        continue
                    br, bc, _, _ = _bbox(dock)
                    landing = (br-ar, bc-ac)  # the rotated tip is at (0,0)
                    dr, dc = landing[0]-tip[0], landing[1]-tip[1]
                    distance = dr*vr + dc*vc
                    if distance <= 0 or (dr, dc) != (distance*vr, distance*vc):
                        continue
                    moved = _move(rotated, 0, landing)
                    if not geometric_only and not _fits(moved, shape):
                        continue
                    if not geometric_only and any(moved[p] != station[p] for p in moved.keys() & station.keys()):
                        continue
                    candidates[(index, direction, landing)] = (distance, moved)
    return candidates


def _run(input_grid):
    g, bg, comps, initial, ink, keys, cargo, tip, facing, patches = _decode(input_grid)
    body = dict(comps[initial])
    stations = {i: c for i, c in enumerate(comps) if i != initial}
    history, growth, states = [], [], [(dict(body), tip, facing)]
    key_depths = {_orbit(p): 0 for p in _ports(body, keys)}
    stop_depths, matched_ids = [], []
    while stations:
        options = _landings(body, tip, facing, stations, keys, g.shape)
        if not options:
            break
        rays = {d for i, d, p in options}
        _require(len(rays) == 1, "several eligible rays: no direction priority is defined")
        if not history:
            _require(next(iter(rays)) == facing, "first landing must follow the visible arrow")
        distance = min(value[0] for value in options.values())
        nearest = [k for k, v in options.items() if v[0] == distance]
        _require(len(nearest) == 1, "tied nearest registration landing")
        index, direction, landing = nearest[0]
        moved = options[nearest[0]][1]
        overlap = moved.keys() & stations[index].keys()
        added = len(stations[index].keys() - moved.keys())
        docks = _ports(stations[index], keys)
        matches = [_orbit(p) for p in docks if all(moved.get(q) == v for q, v in p.items())]
        _require(bool(matches), "landing has no complete coincident patch")
        depth = 1 + min(key_depths[k] for k in matches)
        for patch in docks:
            key_depths.setdefault(_orbit(patch), depth)
        matched_ids.append(tuple(matches))
        stop_depths.append(depth)
        moved.update(stations.pop(index))
        _require(_fits(moved, g.shape), "merged carrier leaves the board")
        body, tip, facing = moved, landing, direction
        growth.append(added)
        history.append((index, direction, landing, distance, len(overlap)))
        states.append((dict(body), tip, facing))
    _require(len(history) >= 2, "family members have at least two collected stops")
    out = np.full(g.shape, bg, dtype=np.int8)
    for p, color in body.items():
        out[p] = color
    return out, dict(bg=bg, ink=ink, keys=keys, cargo=cargo, tip=tip, facing=facing,
                     initial=initial, components=comps, patches=patches, states=states,
                     history=history, growth=growth, unvisited=sorted(stations),
                     matched_ids=matched_ids, dependency_depth=max(stop_depths))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, match features, and move/merge using only this input."""
    return _run(input_grid)[0]


def _core(rng, ink):
    span = int(rng.integers(1, 3))
    extension = int(rng.integers(1, 3))
    cells = {(0, -i): ink for i in range(extension)}
    for depth in range(1, span+1):
        for r in range(-depth, depth+1):
            cells[r, -extension+1-depth] = ink
    return cells


def _new_patch(rng, colors, used):
    for _ in range(80):
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        patch = {(r, c): int(rng.choice(colors)) for r in range(h) for c in range(w)}
        if set(patch.values()) != set(colors):
            continue
        if len({_signature(_move(patch, t)) for t in range(4)}) != 4:
            continue
        key = _orbit(patch)
        if key not in used:
            used.add(key)
            return patch
    raise ValueError("could not make a new asymmetric patch")


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _bridge(a, b, forbidden, rng):
    starts = sorted(a)
    rng.shuffle(starts)
    queue, previous = deque(starts), {p: None for p in starts}
    steps = list(DIRS)
    rng.shuffle(steps)
    end = None
    while queue:
        p = queue.popleft()
        if p in b:
            end = p
            break
        for dr, dc in steps:
            q = p[0]+dr, p[1]+dc
            if not (-7 <= q[0] <= 7 and -14 <= q[1] <= -2):
                continue
            if q in previous or (q in forbidden and q not in b):
                continue
            previous[q] = p
            queue.append(q)
    _require(end is not None, "no connecting path")
    path = []
    while end is not None:
        path.append(end)
        end = previous[end]
    _require(len(path) <= 11, "connector too long")
    return path


def _assembly(rng, n, palette, branch):
    bg, ink, ka, kb, cargo = palette
    core = _core(rng, ink)
    depth = -min(c for r, c in core)
    used = set()
    p0 = _new_patch(rng, (ka, kb), used)
    h, w = max(r for r, c in p0)+1, max(c for r, c in p0)+1
    top = -int(rng.integers(h))
    p0 = _move(p0, 0, (top, -depth-w-1))
    body = dict(core)
    body.update(p0)
    body[0, -depth-1] = cargo
    initial = dict(body)
    ports, fragments, prefixes = [p0], [], [dict(body)]
    parent_ids = []
    for step in range(n):
        # The second stop needs the first newly acquired patch. Subsequent
        # branches can return to an older port instead of discarding old keys.
        parent = (int(rng.integers(len(ports)-1)) if branch and step > 1 else len(ports)-1)
        incoming = ports[parent]
        # A terminal fragment may contain just its incoming patch and an extension.
        terminal_single = step == n-1 and rng.random() < .30
        fragment = dict(incoming)
        if terminal_single:
            occupied = set(body)
            tips = [q for p in incoming for dr, dc in DIRS
                    if (q := (p[0]+dr, p[1]+dc)) not in occupied]
            _require(bool(tips), "terminal extension has no room")
            q = tips[int(rng.integers(len(tips)))]
            fragment[q] = cargo
            for _ in range(int(rng.integers(1, 5))):
                choices = [q for p in fragment for dr, dc in DIRS
                           if (q := (p[0]+dr, p[1]+dc)) not in occupied
                           and q not in fragment and -7 <= q[0] <= 7 and -14 <= q[1] <= -2]
                if choices:
                    fragment[choices[int(rng.integers(len(choices)))]] = cargo
        else:
            fresh = _new_patch(rng, (ka, kb), used)
            ph, pw = max(r for r, c in fresh)+1, max(c for r, c in fresh)+1
            ar, ac, br, bc = _bbox(incoming)
            placements = []
            for rr in range(-6, 7-ph):
                for cc in range(-13, -depth-pw):
                    proposed = _move(fresh, 0, (rr, cc))
                    if set(proposed) & _halo(set().union(*(set(p) for p in ports), set(core))):
                        continue
                    if set(proposed) & set(body):
                        continue
                    box = _bbox(set(incoming) | set(proposed))
                    if box[2]-box[0] > 8 or box[3]-box[1] > 8:
                        continue
                    gap = min(abs(r-u)+abs(c-v) for r, c in incoming for u, v in proposed)
                    if 2 <= gap <= 5:
                        placements.append(proposed)
            _require(bool(placements), "assembly has no free patch position")
            outgoing = placements[int(rng.integers(len(placements)))]
            forbidden = set(core) | set().union(*(set(p) for p in ports))
            path = _bridge(incoming, outgoing, forbidden, rng)
            fragment.update(outgoing)
            for p in path:
                if p not in fragment:
                    fragment[p] = cargo
            # Short irregular branches vary the contribution independently of its key.
            for _ in range(int(rng.integers(0, 4))):
                choices = [q for p, v in fragment.items() if v == cargo for dr, dc in DIRS
                           if (q := (p[0]+dr, p[1]+dc)) not in body
                           and q not in fragment and q not in _halo(outgoing)
                           and -7 <= q[0] <= 7 and -14 <= q[1] <= -2]
                if choices:
                    fragment[choices[int(rng.integers(len(choices)))]] = cargo
            ports.append(outgoing)
        _require(not any(body[p] != fragment[p] for p in body.keys() & fragment.keys()), "assembly conflict")
        body.update(fragment)
        fragments.append(fragment)
        prefixes.append(dict(body))
        parent_ids.append(parent)
    _require(len(_components(initial, diagonal=True)) == 1, "initial carrier disconnected")
    _require(all(len(_components(p, diagonal=True)) == 1 for p in fragments), "station disconnected")
    return initial, fragments, prefixes, ports, parent_ids


def _place_journey(rng, initial, fragments, prefixes, shape, need_turn):
    """Bounded geometric backtracking, with no route bank or coordinate lattice."""
    n = len(fragments)
    rotated_parts = [[_move(part, d-1) for d in range(4)] for part in [initial]+fragments]
    rotated_bodies = [[_move(part, d-1) for d in range(4)] for part in prefixes]
    budget = [1400]

    def search(step, tip, facing, fixed, input_halo, past_halos, route):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if step == n:
            if need_turn and all(d == route[0][1] for p, d in route):
                return None
            return fixed, route
        options = []
        ds = [facing] if step == 0 else list(range(4))
        for d in ds:
            vr, vc = DIRS[d]
            for distance in range(4, max(shape)):
                q = tip[0]+vr*distance, tip[1]+vc*distance
                if any(q == earlier for earlier, old_direction in route):
                    continue
                station = _move(rotated_parts[step+1][d], 0, q)
                carried = _move(rotated_bodies[step+1][d], 0, q)
                if not _fits(station, shape) or not _fits(carried, shape):
                    continue
                if set(station) & input_halo or any(set(station) & halo for halo in past_halos):
                    continue
                options.append((q, d, station, carried))
        rng.shuffle(options)
        for q, d, station, carried in options:
            result = search(step+1, q, d, fixed+[station], input_halo | _halo(station),
                            past_halos+[_halo(carried)], route+[(q, d)])
            if result is not None:
                return result
        return None

    for _ in range(20):
        if budget[0] < 0:
            break
        facing = int(rng.integers(4))
        possible = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                placed = _move(rotated_parts[0][facing], 0, (r, c))
                if _fits(placed, shape):
                    possible.append((r, c))
        if not possible:
            continue
        tip = possible[int(rng.integers(len(possible)))]
        first = _move(rotated_parts[0][facing], 0, tip)
        result = search(0, tip, facing, [first], _halo(first), [_halo(first)], [(tip, facing)])
        if result is not None:
            return result
    raise ValueError("could not place journey")


def _draw(shape, bg, parts):
    g = np.full(shape, bg, dtype=np.int8)
    for part in parts:
        for p, v in part.items():
            _require(g[p] == bg or g[p] == v, "rendering conflict")
            g[p] = v
    return g


def _add_unreachable(rng, g, ports, canonical_body, cargo, keys, required):
    occupied = set(zip(*np.where(g != _background(g))))
    blocked = _halo(occupied)
    _, original = _run(g)
    state_halos = [_halo(body) for body, tip, facing in original['states']]
    # A genuine alternative fragment: an existing patch with material absent
    # from the intended assembly. Its translation must not lie on a usable ray.
    order = list(range(len(ports)))
    rng.shuffle(order)
    for index in order:
        patch = ports[index]
        fragment = dict(patch)
        choices = [q for p in patch for dr, dc in DIRS
                   if (q := (p[0]+dr, p[1]+dc)) not in canonical_body]
        rng.shuffle(choices)
        if not choices:
            continue
        fragment[choices[0]] = cargo
        for _ in range(int(rng.integers(1, 4))):
            options = [q for p in fragment for dr, dc in DIRS
                       if (q := (p[0]+dr, p[1]+dc)) not in canonical_body and q not in fragment]
            if options:
                fragment[options[int(rng.integers(len(options)))]] = cargo
        for _ in range(90):
            d = int(rng.integers(4))
            rotated = _move(fragment, d-1)
            r0, c0, r1, c1 = _bbox(rotated)
            if r1-r0 >= g.shape[0] or c1-c0 >= g.shape[1]:
                continue
            delta = (int(rng.integers(-r0, g.shape[0]-r1)),
                     int(rng.integers(-c0, g.shape[1]-c1)))
            placed = _move(rotated, 0, delta)
            if set(placed) & blocked or any(set(placed) & halo for halo in state_halos):
                continue
            if any(_landings(body, tip, facing, {0: placed}, keys, g.shape, geometric_only=True)
                   for body, tip, facing in original['states']):
                continue
            trial = g.copy()
            for p, v in placed.items():
                trial[p] = v
            try:
                out, info = _run(trial)
                if len(info['unvisited']) == 1:
                    return trial
            except ValueError:
                continue
    _require(not required, "could not place an unreachable station")
    return g


def _background(g):
    cs, counts = np.unique(g, return_counts=True)
    return int(cs[counts.argmax()])


def _make_pair(rng, n, palette, excluded, branch, teach, avoid_core=None):
    last_error = None
    for attempt in range(90):
        try:
            initial, fragments, prefixes, ports, parent_ids = _assembly(rng, n, palette, branch)
            core_id = _signature({p: 1 for p, v in initial.items() if v == palette[1]})
            _require(core_id != avoid_core, "repeat teaching core")
            low = 18 if n == 2 else 22 if n == 3 else 25
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
            if attempt >= 35:
                shape = (30, 30)
            fixed, route = _place_journey(rng, initial, fragments, prefixes, shape, need_turn=teach)
            g = _draw(shape, palette[0], fixed)
            expected = _draw(shape, palette[0], [_move(prefixes[-1], route[-1][1]-1, route[-1][0])])
            out, info = _run(g)
            _require(len(info['history']) == n and not info['unvisited'], "intended stations not all collected")
            _require(np.array_equal(out, expected), "pixel registration differs from construction")
            if excluded:
                g = _add_unreachable(rng, g, ports, prefixes[-1], palette[4], palette[2:4], True)
                out, info = _run(g)
                _require(np.array_equal(out, expected), "extra station changed the journey")
            _require(min(info['growth']) > 0, "each generated collected station contributes cells")
            if teach:
                _require(len({step[1] for step in info['history']}) >= 2, "teaching journey needs a turn")
            return {'input': g, 'output': expected}
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"bounded generation exhausted: {last_error}")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _fingerprint(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .25)
    palette = tuple(int(v) for v in rng.choice(10, 5, replace=False))
    longer = bool(rng.random() < .65)
    puzzle, seen, first_core = {'train': [], 'test': []}, set(), None
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            n = (2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4))) if split == 'train' else (4 if longer else int(rng.integers(3, 5)))
            if not shared_palette:
                palette = tuple(int(v) for v in rng.choice(10, 5, replace=False))
            excluded = (split == 'train' and i == 0) or bool(rng.random() < .50)
            if longer:
                branch = (split == 'train' and i == 1) or bool(rng.random() < .5)
            else:
                branch = split == 'train' and n >= 3
            for _ in range(8):
                pair = _make_pair(rng, n, palette, excluded, branch, teach=split == 'train' and i < 2,
                                  avoid_core=first_core if split == 'train' and i == 1 else None)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        _, info = _run(pair['input'])
                        body, tip, facing = info['states'][0]
                        core = {p: 1 for p, v in body.items() if v == info['ink']}
                        first_core = _signature(_move(core, 1-facing, pivot=tip))
                    break
            else:
                raise RuntimeError('could not make distinct inputs')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'wrong example counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            g, out = _grid(pair['input']), _grid(pair['output'])
            key = _fingerprint(g)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            predicted, info = _run(g)
            _require(g.shape == out.shape and np.array_equal(predicted, out), 'incorrect terminal carrier')
            _require(all(v > 0 for v in info['growth']), 'generated stops must grow the carrier')
            # Ink is a persistent core, never enlarged, recolored, or substituted.
            _require(np.count_nonzero(g == info['ink']) == np.count_nonzero(out == info['ink']), 'core changed area')
            _require(sum(out[p] == info['ink'] for p in info['states'][-1][0]) > 0, 'core lost')
            initial_area = len(info['states'][0][0])
            _require(np.count_nonzero(out != info['bg']) == initial_area + sum(info['growth']), 'overlap did not collapse')
            _require(info['tip'] in info['states'][-1][0], 'tip is not part of the final carrier')
            for body, tip, facing in info['states']:
                core = {p for p, v in body.items() if v == info['ink']}
                _require(_pointed(core) == (tip, facing), 'core geometry or reference changed')
            _require(not _landings(info['states'][-1][0], info['tip'], info['facing'],
                                  {i: info['components'][i] for i in info['unvisited']}, info['keys'], g.shape),
                     'output stops before an eligible station')
            # No future station obscures a supplied or simulated carrier state.
            visited = set()
            for state_index, (body, tip, facing) in enumerate(info['states']):
                if state_index:
                    visited.add(info['history'][state_index-1][0])
                future = [comp for i, comp in enumerate(info['components'])
                          if i != info['initial'] and i not in visited]
                _require(all(not (set(comp) & _halo(body)) for comp in future), 'future station touches carrier')
            analyses[split].append(info)
    train = analyses['train']
    _require(any(i['unvisited'] for i in train), 'training must show excluded matching stations')
    _require(any(len({s[1] for s in i['history']}) > 1 for i in train), 'training must show rotation')
    _require({len(i['history']) for i in train} >= {2, 3}, 'training must show short and longer growth')
    core_ids = set()
    for info in train:
        body, tip, facing = info['states'][0]
        core = {p: 1 for p, v in body.items() if v == info['ink']}
        core_ids.add(_signature(_move(core, 1-facing, pivot=tip)))
        _require(info['dependency_depth'] >= 2, 'training must show a newly acquired key enabling a stop')
    _require(len(core_ids) >= 2, 'training must show different visible core geometries')
    _require(any(len(i['history']) >= 3 and
                 set(i['matched_ids'][2]) & {_orbit(p) for p in _ports(i['states'][1][0], i['keys'])}
                 for i in train), 'training must show use of a retained older key')
    max_stops = max(len(i['history']) for i in train)
    max_depth = max(i['dependency_depth'] for i in train)
    _require(all(len(i['history']) > max_stops or i['dependency_depth'] > max_depth
                 for i in analyses['test']), 'missing structural train/test shift')
    return None


def self_test():
    """Bounded 42-seed invariants and diversity checks; no independent solver claim."""
    metrics, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_fingerprint(pair['input']) for s in p for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                out, info = _run(pair['input'])
                core = {q for q, v in info['states'][0][0].items() if v == info['ink']}
                tip0, dir0 = info['states'][0][1:]
                core_sig = _signature(_move({q: 1 for q in core}, 1-dir0, pivot=tip0))
                metrics.append((pair['input'].shape, info, core_sig, split))
                # Remove A's accretion: transport just the initial bitmap to
                # the true terminal tip. Every sampled answer must change.
                initial_only = _move(info['states'][0][0], info['facing']-dir0, info['tip'], pivot=tip0)
                ablated = _draw(out.shape, info['bg'], [initial_only])
                assert not np.array_equal(ablated, out)
                # Remove B's travel: retain the growing union at the initial
                # tip/orientation. This must change its position or facing.
                stationary = _move(info['states'][-1][0], dir0-info['facing'], tip0, pivot=info['tip'])
                assert stationary != info['states'][-1][0]
                # A's match/merge without B's ray constraint would also retain
                # the tempting alternative fragment. Align that fragment into
                # the final carrier frame and prove it adds absent material.
                final_body = info['states'][-1][0]
                for index in info['unvisited']:
                    extra_cells = set()
                    for turns in range(4):
                        rotated = _move(info['components'][index], turns)
                        for dock in _ports(rotated, info['keys']):
                            for carried in _ports(final_body, info['keys']):
                                if _signature(dock) != _signature(carried):
                                    continue
                                ar, ac, _, _ = _bbox(dock)
                                br, bc, _, _ = _bbox(carried)
                                aligned = _move(rotated, 0, (br-ar, bc-ac))
                                assert all(aligned[p] == final_body[p]
                                           for p in aligned.keys() & final_body.keys())
                                extra_cells.update(aligned.keys() - final_body.keys())
                    assert extra_cells, 'unvisited station is not a meaningful alternative'
    assert len(signatures) == 42
    assert len({shape for shape, info, core, split in metrics}) >= 20
    assert len({core for shape, info, core, split in metrics}) >= 4
    assert {info['bg'] for shape, info, core, split in metrics} == set(range(10))
    assert {info['states'][0][2] for shape, info, core, split in metrics} == set(range(4))
    assert {info['facing'] for shape, info, core, split in metrics} == set(range(4))
    assert {len(info['history']) for shape, info, core, split in metrics} >= {2, 3, 4}
    assert any(info['unvisited'] for shape, info, core, split in metrics)
    assert len({_orbit(patch) for shape, info, core, split in metrics
                for ps in info['patches'] for patch in ps}) >= 60
    route_shapes = set()
    for shape, info, core, split in metrics:
        p0, d0 = info['states'][0][1:]
        route_shapes.add(tuple(_rot((s[2][0]-p0[0], s[2][1]-p0[1]), 1-d0) for s in info['history']))
    assert len(route_shapes) >= 100
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(pa[k], pb[k]) for s in a for pa, pb in zip(a[s], b[s]) for k in pa)
    example = a['train'][0]
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(example['input'], turns)), np.rot90(example['output'], turns))
    palette_map = np.array([8, 4, 7, 1, 9, 0, 2, 6, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(palette_map[example['input']]), palette_map[example['output']])
    # A hand-constructed junction has two valid rays. Even one straight-ahead
    # candidate cannot silently take priority over the other.
    patch = {(r-1, c-4): v for r, row in enumerate(((2, 2), (2, 3))) for c, v in enumerate(row)}
    leader = {(0, 0): 1, (-1, -1): 1, (0, -1): 1, (1, -1): 1, (0, -2): 4}
    leader.update(patch)
    station = dict(patch)
    station[1, -4] = 4
    junction = _draw((30, 30), 0, [_move(leader, 0, (8, 12)),
                                 _move(station, 0, (8, 23)), _move(station, 1, (22, 12))])
    try:
        solve(junction)
    except ValueError as exc:
        assert 'several eligible rays' in str(exc)
    else:
        raise AssertionError('forward-priority junction accepted')
    altered = example['input'].copy()
    _, decoded = _run(altered)
    first_station = decoded['history'][0][0]
    initial_id = _orbit(_ports(decoded['states'][0][0], decoded['keys'])[0])
    incoming = next(p for p in decoded['patches'][first_station] if _orbit(p) == initial_id)
    q = min(incoming)
    altered[q] = next(c for c in decoded['keys'] if c != int(altered[q]))
    try:
        changed = solve(altered)
    except ValueError:
        pass
    else:
        assert not np.array_equal(changed, example['output']), 'incoming patch mismatch ignored'
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in a[s]] for s in a}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
