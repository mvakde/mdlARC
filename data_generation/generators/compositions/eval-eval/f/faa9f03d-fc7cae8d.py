"""Unfold two repaired and redirected bead tapes into an oriented loom.

Only NumPy and the standard library are used. No generation state is consulted
by solve; the colored frame, heads, damage, and beads all occur in the input.
"""

import copy
import json

import numpy as np


BACKGROUND, REPAIR, REDIRECT, HEAD = 0, 5, 8, 9
ROLE_COLORS = (1, 2, 3, 4, 6, 7)
SIDES = ("T", "R", "B", "L")
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 37217464, 364806172, 312847814, 638509837, 459990030,
              953659048, 348948174, 456774730, 306179041, 985234477,
              548532618, 747346415, 603120829, 574935170, 232028307,
              284943711, 199706534, 121960895, 742039045, 362811429,
              691680428, 20283075, 299299209, 536061068, 511743968,
              988182223, 624064157, 412512879, 520707838, 257154203,
              352607394, 471148720, 846129796, 376606013, 935547348,
              216100036, 492395541, 44341901, 952070202, 683031000,
              429807558)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)), name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal example counts")
    return nt, nq


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "invalid grid cells")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "diagonal segment")
    n = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [(a[0] + i*d[0], a[1] + i*d[1]) for i in range(n + 1)]


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _side(grid, s):
    return {"T": grid[0, :], "R": grid[:, -1],
            "B": grid[-1, :], "L": grid[:, 0]}[s]


def _controls(grid):
    controls = {}
    for s in SIDES:
        core = _side(grid, s)[1:-1]
        _require(len(core) >= 1 and np.all(core == core[0]), "broken outer control")
        if int(core[0]) != BACKGROUND:
            controls[s] = int(core[0])
    _require(len(controls) == 2 and len(set(controls.values())) == 2,
             "need two distinct controls")
    _require(any(set(controls) == {SIDES[i], SIDES[(i+1) % 4]} for i in range(4)),
             "controls must be adjacent")
    _require(set(controls.values()) <= set(ROLE_COLORS), "reserved control color")
    _require(all(grid[p] == BACKGROUND for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))),
             "frame corners must be blank")
    return controls


def _skeleton(arena):
    """Single-cell holes are bridged simultaneously; markers/beads are waypoints."""
    observed = {tuple(map(int, p)) for p in np.argwhere(
        (arena != BACKGROUND) & (arena != REDIRECT))}
    filled = set()
    for p in observed:
        for d in ((1, 0), (0, 1)):
            q = _add(p, d)
            end = _add(q, d)
            if end in observed and arena[q] == BACKGROUND:
                filled.add(q)
    return observed | filled, filled


def _walk_component(component, start):
    path, previous, current = [], None, start
    while True:
        _require(current not in path, "cyclic tape")
        path.append(current)
        following = [p for p in _neighbors(current, component) if p != previous]
        if not following:
            break
        _require(len(following) == 1, "branching tape")
        previous, current = current, following[0]
    _require(set(path) == component, "head must be a tape endpoint")
    return path


def _decode(input_grid):
    grid = _grid(input_grid)
    _require(min(grid.shape) >= 5, "canvas too small")
    controls = _controls(grid)
    arena = grid[1:-1, 1:-1].copy()
    cells, holes = _skeleton(arena)
    components, pending = [], set(cells)
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = {start}, [start]
        while stack:
            for p in _neighbors(stack.pop(), pending):
                pending.remove(p)
                comp.add(p)
                stack.append(p)
        components.append(comp)
    _require(len(components) == 2, "need exactly two recoverable tapes")
    records = []
    for comp in components:
        heads = [p for p in comp if int(arena[p]) == HEAD]
        wires = set(int(arena[p]) for p in comp) & set(controls.values())
        _require(len(heads) == len(wires) == 1, "ambiguous head or tape identity")
        color = wires.pop()
        path = _walk_component(comp, heads[0])
        _require(len(path) >= 5 and len(path) <= 30, "invalid original tape length")
        _require(not _boundary(path[0], arena.shape) and _boundary(path[-1], arena.shape),
                 "tape must run from an interior head to the arena perimeter")
        _require(not any(_boundary(p, arena.shape) for p in path[:-1]),
                 "premature perimeter contact")
        records.append(dict(color=color, path=path, head=heads[0],
                            holes=holes & comp, component=comp, edit=None))
    _require({r['color'] for r in records} == set(controls.values()), "missing wire identity")
    for q0 in np.argwhere(arena == REDIRECT):
        q = tuple(map(int, q0))
        adjacent = [(r, p) for r in records for p in _neighbors(q, r['component'])]
        _require(len(adjacent) == 1, "redirect must touch exactly one tape pivot")
        record, pivot = adjacent[0]
        _require(record['edit'] is None, "at most one redirect per tape")
        path = record['path']
        k = path.index(pivot)
        d = (q[0] - pivot[0], q[1] - pivot[1])
        _require(0 < k < len(path)-1 and path[k+1] == _add(pivot, (-d[0], -d[1])),
                 "old terminal branch must leave opposite the redirect")
        incoming = (path[k-1][0]-pivot[0], path[k-1][1]-pivot[1])
        _require(incoming[0]*d[0] + incoming[1]*d[1] == 0, "pivot must be a bend")
        new = _ray(pivot, d, arena.shape)
        _require(new and not set(new) & set(path[:k+1]), "new ray meets its own retained prefix")
        effective = path[:k+1] + new
        _require(len(effective) <= 30, "unfolded tape too long")
        record['edit'] = dict(pivot=pivot, index=k, direction=d,
                              old=path[k+1:], new=new)
    # No crossing convention is needed: new rays are separated from the other
    # original tape and each other. Every step has one visible identity.
    for r in records:
        edit = r['edit']
        if edit:
            other = next(x for x in records if x is not r)
            _require(not set(edit['new']) & other['component'], "ray crosses the other tape")
    if all(r['edit'] for r in records):
        _require(not set(records[0]['edit']['new']) & set(records[1]['edit']['new']),
                 "new rays intersect")
    for r in records:
        prefix = r['path'] if not r['edit'] else r['path'][:r['edit']['index']+1]
        samples = []
        for p in prefix[1:]:
            value = int(arena[p])
            samples.append(r['color'] if value in (BACKGROUND, REPAIR) else value)
        if r['edit']:
            samples.extend([r['color']] * len(r['edit']['new']))
        _require(1 <= len(samples) <= 29, "invalid unfolded extent")
        _require(HEAD not in samples and REDIRECT not in samples and REPAIR not in samples,
                 "annotation in samples")
        r['samples'] = samples
        r['target'] = next(s for s, c in controls.items() if c == r['color'])
        r['beads'] = set(samples) - set(controls.values())
    return dict(grid=grid, arena=arena, controls=controls, records=records,
                beads=set(np.unique(arena)) - set(controls.values()) -
                      {BACKGROUND, REPAIR, REDIRECT, HEAD})


def _weave(sequences, targets):
    """The shared corner is the origin; distance from it is the tape index."""
    horizontal = next(c for c, side in targets.items() if side in ("T", "B"))
    vertical = next(c for c, side in targets.items() if side in ("L", "R"))
    x, y = sequences[horizontal], sequences[vertical]
    out = np.zeros((len(y)+1, len(x)+1), dtype=np.int8)
    out[0, 0] = HEAD
    out[0, 1:] = x
    out[1:, 0] = y
    for row, a in enumerate(y, 1):
        for col, b in enumerate(x, 1):
            if a == b and a not in targets:
                out[row, col] = a
    if targets[horizontal] == "B":
        out = np.flipud(out)
    if targets[vertical] == "R":
        out = np.fliplr(out)
    return out.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input alone, repair/edit its tapes, then unfold and weave."""
    info = _decode(input_grid)
    return _weave({r['color']: r['samples'] for r in info['records']},
                  {r['color']: r['target'] for r in info['records']})


def _simple(path):
    s = set(path)
    return len(s) == len(path) and all(len(_neighbors(p, s)) ==
        (1 if i in (0, len(path)-1) else 2) for i, p in enumerate(path))


def _random_path(rng, shape):
    head = (int(rng.integers(1, shape[0]-1)), int(rng.integers(1, shape[1]-1)))
    axis = int(rng.integers(2))
    path = [head]
    for _ in range(int(rng.integers(1, 5))):
        p = path[-1]
        positions = [v for v in range(1, shape[axis]-1) if 2 <= abs(v-p[axis]) <= 7]
        rng.shuffle(positions)
        found = False
        for v in positions:
            q = list(p)
            q[axis] = v
            candidate = path + _line(p, tuple(q))[1:]
            if len(candidate) <= 24 and _simple(candidate):
                path, found = candidate, True
                break
        if not found:
            return None
        axis = 1-axis
    side_order = list((0, shape[axis]-1))
    rng.shuffle(side_order)
    for v in side_order:
        q = list(path[-1])
        q[axis] = v
        candidate = path + _line(path[-1], tuple(q))[1:]
        if 5 <= len(candidate) <= 30 and _simple(candidate):
            return candidate
    return None


def _bends(path):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _edit_options(path, shape, core=False, bent_tail=False):
    options = []
    for k in range(5 if core else 3, len(path)-2):
        p, following, previous = path[k], path[k+1], path[k-1]
        old_d = (following[0]-p[0], following[1]-p[1])
        incoming = (previous[0]-p[0], previous[1]-p[1])
        if incoming[0]*old_d[0] + incoming[1]*old_d[1] != 0:
            continue
        d = (-old_d[0], -old_d[1])
        new = _ray(p, d, shape)
        if not new or len(path[:k+1])+len(new) > 30:
            continue
        if set(new) & set(path[:k+1]) or _neighbors(new[0], set(path)) != [p]:
            continue
        if bent_tail and _bends(path[k:]) == 0:
            continue
        options.append(dict(index=k, pivot=p, direction=d, new=new, old=path[k+1:]))
    return options


def _scene(rng, shape, n_redirect, core):
    # Random walks, not a fixed T or L layout bank. Every rejected proposal is
    # discarded before any public pair is formed. The bound is deliberately high
    # enough for the two interacting teaching tapes, yet finite.
    for _ in range(2400):
        paths = [_random_path(rng, shape), _random_path(rng, shape)]
        if any(p is None for p in paths):
            continue
        sets = [set(p) for p in paths]
        if sets[0] & sets[1] or any(_neighbors(p, sets[1]) for p in sets[0]):
            continue
        edits = []
        success = True
        for i, path in enumerate(paths):
            if i >= n_redirect:
                edits.append(None)
                continue
            options = _edit_options(path, shape, core=core, bent_tail=(core and i == 0))
            options = [e for e in options if not set(e['new']) & sets[1-i]]
            if i == 1 and edits[0]:
                options = [e for e in options if not set(e['new']) & set(edits[0]['new'])]
            if not options:
                success = False
                break
            edits.append(options[int(rng.integers(len(options)))])
        if not success:
            continue
        lengths = [len(p) if not e else e['index']+1+len(e['new']) for p, e in zip(paths, edits)]
        if core and lengths[0] == lengths[1]:
            continue
        return paths, edits
    return None


def _paint_frame(grid, corner, wires, reverse):
    sides = (SIDES[corner], SIDES[(corner+1) % 4])
    colors = wires[::-1] if reverse else wires
    controls = dict(zip(sides, colors))
    for side, color in controls.items():
        _side(grid, side)[1:-1] = color
    return {color: side for side, color in controls.items()}


def _make_pair(rng, corner, reverse, palette=None, core=False, size_mode=0, test=False):
    for _ in range(40):
        if size_mode == 1:
            low, high = ((12, 19) if not test else (17, 29))
        elif size_mode == 2:
            low, high = ((17, 27) if not test else (11, 20))
        else:
            low, high = (12, 29) if core else (9, 29)
        shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
        n_redirect = 2 if core else int(rng.integers(3))
        scene = _scene(rng, shape, n_redirect, core)
        if scene is None:
            continue
        paths, edits = scene
        colors = list(map(int, rng.permutation(ROLE_COLORS))) if palette is None else list(palette)
        wires = colors[:2]
        bead_count = 2 if core else int(rng.integers(1, 4))
        bead_colors = colors[2:2+bead_count]
        arena = np.zeros(shape, dtype=np.int8)
        bead_sites, reserved_gaps = [], []
        good = True
        for i, (path, edit) in enumerate(zip(paths, edits)):
            for p in path:
                arena[p] = wires[i]
            arena[path[0]] = HEAD
            last = edit['index'] if edit else len(path)-1
            straight = [k for k in range(1, last) if
                        path[k-1][0]+path[k+1][0] == 2*path[k][0] and
                        path[k-1][1]+path[k+1][1] == 2*path[k][1]]
            if not straight:
                good = False
                break
            gap_index = straight[int(rng.integers(len(straight)))]
            reserved_gaps.append(path[gap_index])
            available = [k for k in range(1, last) if k != gap_index]
            if core and len(available) < 3:
                good = False
                break
            rng.shuffle(available)
            desired = min(len(available), 3 if core else int(rng.integers(1, 7)))
            sites = available[:desired]
            for j, k in enumerate(sites):
                arena[path[k]] = bead_colors[j % len(bead_colors)]
            bead_sites.append({path[k] for k in sites})
            # At least one deleted matching bead makes ignoring redirection
            # visibly change both the header and the matching-cell pattern.
            if edit:
                old_site = edit['old'][int(rng.integers(len(edit['old'])))]
                arena[old_site] = bead_colors[0]
                bead_sites[-1].add(old_site)
        if not good:
            continue
        for i, (path, edit) in enumerate(zip(paths, edits)):
            arena[reserved_gaps[i]] = BACKGROUND
            marker_sites = [p for p in path[1:] if p not in bead_sites[i] and p != reserved_gaps[i]]
            rng.shuffle(marker_sites)
            if edit:
                marker_sites = list(dict.fromkeys([edit['pivot']] + marker_sites))
            for p in marker_sites[:int(rng.integers(1, 4))]:
                arena[p] = REPAIR
            if edit:
                arena[edit['new'][0]] = REDIRECT
        recovered, holes = _skeleton(arena)
        if recovered != set(paths[0]) | set(paths[1]):
            continue
        grid = np.zeros((shape[0]+2, shape[1]+2), dtype=np.int8)
        grid[1:-1, 1:-1] = arena
        targets = _paint_frame(grid, corner, wires, reverse)
        sequences = {}
        for color, path, edit in zip(wires, paths, edits):
            prefix = path if not edit else path[:edit['index']+1]
            seq = [color if arena[p] in (BACKGROUND, REPAIR) else int(arena[p]) for p in prefix[1:]]
            if edit:
                seq += [color] * len(edit['new'])
            sequences[color] = seq
        pair = {'input': grid, 'output': _weave(sequences, targets)}
        try:
            info = _decode(grid)
        except ValueError:
            continue
        if any(r['path'] != next(p for p in paths if p[0] == r['head']) for r in info['records']):
            continue
        if core:
            if any(set(r['samples']) - set(wires) != set(bead_colors) for r in info['records']):
                continue
            if all(len(r['samples']) == len(r['path'])-1 for r in info['records']):
                continue
        _require(np.array_equal(solve(grid), pair['output']), "construction/decoding mismatch")
        return pair, colors
    raise RuntimeError("could not construct separated, uniquely recoverable tapes")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    _require(seed >= 0, "seed must be nonnegative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    corner = int(rng.integers(4))
    held = int(rng.choice([i for i in range(4) if i != corner]))
    reverse = bool(rng.integers(2))
    size_mode = int(rng.integers(3))
    stable = bool(rng.random() < .3)
    first, palette = _make_pair(rng, corner, reverse, core=True, size_mode=size_mode)
    second = {k: v.copy() for k, v in first.items()}
    controls = _controls(first['input'])
    sides = list(controls)
    for s, color in zip(sides, [controls[sides[1]], controls[sides[0]]]):
        _side(second['input'], s)[1:-1] = color
    info = _decode(second['input'])
    second['output'] = _weave({r['color']: r['samples'] for r in info['records']},
                              {r['color']: r['target'] for r in info['records']})
    train, test = [first, second], []
    seen = {_key(first['input']), _key(second['input'])}
    for split, count in ((train, nt-2), (test, nq)):
        for _ in range(count):
            for attempt in range(30):
                c = held if split is test else int(rng.choice([i for i in range(4) if i != held]))
                pair, _ = _make_pair(rng, c, bool(rng.integers(2)),
                                     palette=palette if stable else None,
                                     size_mode=size_mode, test=(split is test))
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    split.append(pair)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
    # Keep the direct correspondence contrast together, with variable order.
    if bool(rng.integers(2)):
        train[0], train[1] = train[1], train[0]
    return {'train': train, 'test': test}


def _measure(info):
    return dict(canvas=tuple(info['grid'].shape),
                corner=tuple(s for s in SIDES if s in info['controls']),
                correspondence=tuple(info['controls'].get(s, 0) for s in SIDES),
                original_lengths=tuple(sorted(len(r['path'])-1 for r in info['records'])),
                lengths=tuple(sorted(len(r['samples']) for r in info['records'])),
                turns=tuple(sorted(_bends(r['path']) for r in info['records'])),
                old_bends=tuple(_bends([r['edit']['pivot']]+r['edit']['old'])
                                for r in info['records'] if r['edit']),
                directions=tuple(r['edit']['direction'] for r in info['records'] if r['edit']),
                redirects=sum(r['edit'] is not None for r in info['records']),
                repairs=int(np.count_nonzero(info['arena'] == REPAIR)),
                gaps=sum(len(r['holes']) for r in info['records']),
                bead_colors=tuple(sorted(info['beads'])),
                bead_counts=tuple(sum(x in info['beads'] for x in r['samples']) for r in info['records']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            # Semantic checks of every index, independent of the renderer's
            # array flips: address cells by distances from the control corner.
            h = next(r for r in info['records'] if r['target'] in ('T', 'B'))
            v = next(r for r in info['records'] if r['target'] in ('L', 'R'))
            _require(out.shape == (len(v['samples'])+1, len(h['samples'])+1), "wrong loom extent")
            origin = (0 if h['target'] == 'T' else out.shape[0]-1,
                      0 if v['target'] == 'L' else out.shape[1]-1)
            dr = 1 if origin[0] == 0 else -1
            dc = 1 if origin[1] == 0 else -1
            _require(out[origin] == HEAD and np.count_nonzero(out == HEAD) == 1, "wrong loom origin")
            for i, a in enumerate(v['samples'], 1):
                _require(out[origin[0]+dr*i, origin[1]] == a, "wrong vertical tape sample")
            for j, b in enumerate(h['samples'], 1):
                _require(out[origin[0], origin[1]+dc*j] == b, "wrong horizontal tape sample")
            for i, a in enumerate(v['samples'], 1):
                for j, b in enumerate(h['samples'], 1):
                    expected = a if a == b and a in info['beads'] else BACKGROUND
                    _require(out[origin[0]+dr*i, origin[1]+dc*j] == expected, "wrong bead intersection")
            _require(not np.any(np.isin(out, [REPAIR, REDIRECT])), "annotation leaked into output")
            infos[split].append(info)
    # The first two teaching inputs have identical tapes and geometry but
    # opposite color correspondence. Non-square extents expose an axis swap.
    a, b = infos['train'][:2]
    _require(np.array_equal(a['arena'], b['arena']) and set(a['controls']) == set(b['controls']),
             "missing controlled identity contrast")
    _require(all(a['controls'][s] != b['controls'][s] for s in a['controls']), "controls were not swapped")
    _require(len({len(r['samples']) for r in a['records']}) == 2, "teaching loom must be rectangular")
    _require(len(a['beads']) == 2 and all(r['beads'] == a['beads'] for r in a['records']),
             "teaching tapes must expose both matches and mismatches")
    _require(all(r['edit'] is not None for r in a['records']), "redirection is not demonstrated")
    _require(any(_bends([r['edit']['pivot']]+r['edit']['old']) >= 1 for r in a['records']),
             "bent terminal deletion is not demonstrated")
    _require(all(r['holes'] for r in a['records']) and np.any(a['arena'] == REPAIR),
             "repair is not demonstrated")
    _require(any(len(r['samples']) != len(r['path'])-1 for r in a['records']),
             "redirection must change an extent in training")
    _require(all(any(a['arena'][p] in a['beads'] for p in r['edit']['old']) for r in a['records']),
             "deleted beads are not demonstrated")
    train_corners = {frozenset(i['controls']) for i in infos['train']}
    _require(all(frozenset(i['controls']) not in train_corners for i in infos['test']),
             "missing held-out-corner shift")
    return True


def self_test():
    """Bounded grid-based checks on all 42 supplied host seeds and legal counts."""
    measurements, signatures, orientation_codes = [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                measurements.append(_measure(info))
                # All 8 corner/correspondence states: use first head in scan
                # order to name the two tapes without looking at literal color.
                records = sorted(info['records'], key=lambda r: r['head'])
                orientation_codes.add(tuple(r['target'] for r in records))
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve mismatch")
        if seed in HOST_SEEDS[:3]:
            repeated = generate_puzzle(seed)
            _require(all(_key(p[k]) == _key(q[k]) for s in puzzle
                         for p, q in zip(puzzle[s], repeated[s]) for k in ('input', 'output')),
                     "nondeterministic generation")
        # Ablation A: keep the old terminals instead of rerouting. Both a
        # changed header and a lost bead relation must be observable.
        info = _decode(puzzle['train'][0]['input'])
        untouched = {}
        for r in info['records']:
            untouched[r['color']] = [r['color'] if info['arena'][p] in (BACKGROUND, REPAIR)
                                     else int(info['arena'][p]) for p in r['path'][1:]]
        ablated = _weave(untouched, {r['color']: r['target'] for r in info['records']})
        _require(not np.array_equal(ablated, puzzle['train'][0]['output']), "A ablation ineffective")
        # Repair ablation: a reader who skips blank holes shortens the retained
        # indexing domain, shifting every later bead and the whole output edge.
        skipped = {}
        for r in info['records']:
            prefix = r['path'][:r['edit']['index']+1]
            seq = [r['color'] if info['arena'][p] == REPAIR else int(info['arena'][p])
                   for p in prefix[1:] if info['arena'][p] != BACKGROUND]
            seq += [r['color']] * len(r['edit']['new'])
            skipped[r['color']] = seq
        without_gaps = _weave(skipped, {r['color']: r['target'] for r in info['records']})
        _require(not np.array_equal(without_gaps, puzzle['train'][0]['output']),
                 "gap-repair ablation ineffective")
        # Ablation B: swap only the identity assignment, as the paired input
        # does. The shape and cell arrangement must change.
        _require(not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output']),
                 "B ablation ineffective")
    _require(len(signatures) == 42, "cross-seed puzzle repetition")
    _require(len(orientation_codes) == 8, "missing orientation states")
    _require(len({m['canvas'] for m in measurements}) >= 45, "insufficient canvas variation")
    _require({m['redirects'] for m in measurements} == {0, 1, 2}, "missing redirect counts")
    _require({d for m in measurements for d in m['directions']} == set(DIRS), "missing redirect directions")
    _require({len(m['bead_colors']) for m in measurements} >= {1, 2, 3}, "missing bead palette sizes")
    _require(max(b for m in measurements for b in m['old_bends']) >= 2, "missing winding tail")
    _require(len({m['lengths'] for m in measurements}) >= 35, "insufficient extent variation")
    _require(len({m['turns'] for m in measurements}) >= 8, "insufficient route variation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(7919+nt*101+nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for kind in ('body', 'header', 'extent', 'frame'):
        bad = copy.deepcopy(original)
        pair = bad['test'][0]
        info = _decode(pair['input'])
        if kind == 'extent':
            pair['output'] = pair['output'][:-1, :]
        elif kind == 'frame':
            s = next(iter(info['controls']))
            _side(pair['input'], s)[2] = BACKGROUND
        else:
            out = pair['output']
            h = next(r for r in info['records'] if r['target'] in ('T', 'B'))
            v = next(r for r in info['records'] if r['target'] in ('L', 'R'))
            origin = (0 if h['target'] == 'T' else out.shape[0]-1,
                      0 if v['target'] == 'L' else out.shape[1]-1)
            dr, dc = (1 if origin[0] == 0 else -1), (1 if origin[1] == 0 else -1)
            p = (origin[0]+(dr if kind == 'body' else 0), origin[1]+dc)
            out[p] = (int(out[p])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt " + kind + " accepted")
    return dict(seeds=42, pairs=len(measurements), orientation_states=len(orientation_codes),
                canvas_shapes=len({m['canvas'] for m in measurements}),
                extent_pairs=len({m['lengths'] for m in measurements}),
                turn_pairs=len({m['turns'] for m in measurements}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
