"""Chromatic switchyard: equality-controlled rewiring and bead collection.

The reader uses only its argument. The sampler constructs colored lane cells,
then checks the actual graph; it never imports either parent generator.
"""
from __future__ import annotations

import json
import numpy as np


_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= g) & (g <= 9)), "colors must be 0..9")
    return g.astype(np.int8, copy=True)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _paths(mask):
    """Order every simple component; reject branches, loops and gravity traps."""
    cells = set(map(tuple, np.argwhere(mask)))
    neighbors = {p: [q for dr, dc in _DIRS
                     if (q := (p[0] + dr, p[1] + dc)) in cells] for p in cells}
    _require(all(len(v) in (1, 2) for v in neighbors.values()), "isolated cell or branch")
    unseen, result = set(cells), []
    while unseen:
        first = min(unseen)
        component, pending = {first}, [first]
        for p in pending:
            for q in neighbors[p]:
                if q not in component:
                    component.add(q)
                    pending.append(q)
        ends = sorted(p for p in component if len(neighbors[p]) == 1)
        _require(len(ends) == 2, "a component must have two terminals")
        ordered, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [q for q in neighbors[current] if q != previous]
            _require(len(choices) == 1, "ambiguous path")
            previous, current = current, choices[0]
            ordered.append(current)
        _require(len(ordered) == len(component), "incomplete path")
        vertical = {b[0] - a[0] for a, b in zip(ordered, ordered[1:]) if a[0] != b[0]}
        _require(len(vertical) <= 1, "path doubles back away from collector")
        result.append(ordered)
        unseen.difference_update(component)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _bars(core, bg, pivot):
    h, w = core.shape
    _require(h >= 3 and h % 2 == 1 and w >= 7, "lane core shape")
    _require(not np.any(core[::2] == pivot), "divider in a lane")
    _require(np.all(np.isin(core[1::2], [bg, pivot])), "separator contents")
    _require(np.all(core[::2, 0] != bg) and np.all(core[::2, -1] != bg), "outer terminals")
    centers, breaks = [], []
    for r in range(1, h, 2):
        cols = np.flatnonzero(core[r] == pivot).tolist()
        runs = []
        for c in cols:
            if runs and c == runs[-1][-1] + 1:
                runs[-1].append(c)
            else:
                runs.append([c])
        for run in runs:
            _require(len(run) == 3, "a divider must have length three")
            c = run[1]
            _require(2 <= c <= w - 3, "divider needs terminal clearance")
            centers.append((r, c))
    _require(bool(centers), "no dividers")
    for r in range(0, h, 2):
        cols = np.flatnonzero(core[r] == bg).tolist()
        _require(all(3 <= c <= w - 4 for c in cols), "break too near edge")
        _require(all(b - a >= 4 for a, b in zip(cols, cols[1:])), "short lane fragment")
        breaks.extend((r, c) for c in cols)
    return centers, breaks


def _topology(core, bg, pivot, bead, guides):
    """Read flanks, perform agreeing writes, and trace the resulting tracks."""
    centers, breaks = _bars(core, bg, pivot)
    structural = core.copy()
    dots = set(map(tuple, np.argwhere(core == bead)))
    original_ends = set()
    for r in range(0, core.shape[0], 2):
        for c in range(core.shape[1]):
            if core[r, c] == bg:
                continue
            neighbors = [(r, x) for x in (c - 1, c + 1)
                         if 0 <= x < core.shape[1] and core[r, x] != bg]
            _require(len(neighbors) >= 1, "isolated initial lane cell")
            if len(neighbors) == 1:
                original_ends.add((r, c))
    _require(dots <= original_ends, "beads must start at lane terminals")
    for p in dots:
        r, c = p
        values = [int(core[r, x]) for x in (c - 1, c + 1)
                  if 0 <= x < core.shape[1] and int(core[r, x]) in guides]
        _require(len(values) == 1, "bead substrate must be visible inward")
        structural[p] = values[0]
    # Every guide identity has a nonterminal occurrence, distinguishing it
    # from the bead identity without a frequency or literal-color assumption.
    for color in guides:
        positions = set(map(tuple, np.argwhere(core == color)))
        _require(bool(positions - original_ends), "guide identity appears only at terminals")
    active, inactive, writes, overlap = [], [], {}, 0
    for r, c in centers:
        upper, lower = core[r - 1, c - 1:c + 2], core[r + 1, c - 1:c + 2]
        _require(np.all(upper == upper[0]) and int(upper[0]) in guides, "upper flank")
        _require(np.all(lower == lower[0]) and int(lower[0]) in guides, "lower flank")
        if upper[0] != lower[0]:
            inactive.append((r, c))
            continue
        active.append((r, c))
        identity = int(upper[0])
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                value = pivot if cc == c else identity
                p = (rr, cc)
                if p in writes:
                    _require(writes[p] == value, "conflicting switch neighborhoods")
                    overlap += 1
                writes[p] = value
    for p, value in writes.items():
        _require(core[p] != bead and (core[p] != bg or p[0] % 2 == 1), "switch covers bead or break")
        structural[p] = value
    paths = _paths(np.isin(structural, list(guides)))
    ends = {p for path in paths for p in (path[0], path[-1])}
    _require(ends <= original_ends and dots <= ends, "rewiring creates unsupported terminal")
    return dict(core=core, bg=bg, pivot=pivot, bead=bead, guides=set(guides),
                centers=centers, breaks=breaks, active=active, inactive=inactive,
                writes=writes, overlap=overlap, structural=structural,
                dots=dots, paths=paths, original_ends=original_ends)


def _read(input_grid):
    inp = _grid(input_grid)
    frames = []
    for k in range(4):
        g = np.rot90(inp, k)
        wall = int(g[-1, 0])
        if not (np.all(g[-1] == wall) and np.count_nonzero(g == wall) == g.shape[1]):
            continue
        if g.shape[0] < 5:
            continue
        bg = int(g[-2, 0])
        if bg == wall or not np.all(g[-2] == bg):
            continue
        points = np.argwhere((g != bg) & (g != wall))
        if not len(points):
            continue
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0) + 1)
        core = g[r0:r1, c0:c1]
        if core.shape[0] < 3 or core.shape[0] % 2 != 1:
            continue
        markers = set(map(int, np.unique(core[1::2]))) - {bg}
        if len(markers) != 1:
            continue
        pivot = next(iter(markers))
        lane_colors = set(map(int, np.unique(core[::2]))) - {bg, pivot}
        for bead in sorted(lane_colors):
            guides = lane_colors - {bead}
            if len(guides) < 2:
                continue
            try:
                a = _topology(core, bg, pivot, bead, guides)
            except ValueError:
                continue
            if not a['dots'] or not a['active'] or not a['inactive']:
                continue
            a.update(grid=g, rotation=k, wall=wall, box=(r0, r1, c0, c1))
            frames.append(a)
    _require(len(frames) == 1, "input must have one identifiable collector and role interpretation")
    return frames[0]


def _targets(paths, dots):
    targets, single, collisions, levels, settled, empty = set(), 0, 0, 0, 0, 0
    for path in paths:
        here = dots & {path[0], path[-1]}
        if not here:
            empty += 1
            continue
        if path[0][0] == path[-1][0]:
            chosen = here
            levels += 1
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            chosen = set(inward[:len(here)])
            single += len(here) == 1 and chosen != here
            collisions += len(here) == 2
            settled += chosen == here
        targets.update(chosen)
    return targets, dict(single=single, collisions=collisions, levels=levels,
                         settled=settled, empty=empty)


def _answer(a):
    core = a['structural'].copy()
    targets, _ = _targets(a['paths'], a['dots'])
    for p in targets:
        core[p] = a['bead']
    out = a['grid'].copy()
    r0, r1, c0, c1 = a['box']
    out[r0:r1, c0:c1] = core
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible geometry and local equality."""
    return _answer(_read(input_grid))


def _all_open_paths(a):
    """Geometry-only ablation: open every bar, ignoring its flank identities."""
    mask = np.zeros(a['core'].shape, dtype=bool)
    mask[::2] = a['core'][::2] != a['bg']
    for r, c in a['centers']:
        mask[r - 1:r + 2, c - 1:c + 2] = True
        mask[r - 1:r + 2, c] = False
    return _paths(mask)


class _Union:
    def __init__(self, cells):
        self.parent = {p: p for p in cells}

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def _sample_centers(rng, lanes, width, rich):
    rows, centers = [], []
    for i in range(lanes - 1):
        previous = rows[-1] if rows else []
        desired = int(rng.choice([0, 1, 1, 2, 2, 3] if rich else [0, 1, 1, 1, 2]))
        chosen = []
        for cc in rng.permutation(np.arange(2, width - 2)):
            c = int(cc)
            if len(chosen) == desired:
                break
            if all(abs(c - old) >= 4 for old in chosen) and all(abs(c - old) >= 2 for old in previous):
                chosen.append(c)
        chosen.sort()
        rows.append(chosen)
        centers.extend((2 * i + 1, c) for c in chosen)
    return centers


def _make_breaks(rng, lanes, width, centers, extras):
    # Cutting same-side consecutive ports makes even the all-open color
    # ablation monotone. Locked bars need no special-case routing convention.
    breaks = []
    for r in range(0, 2 * lanes - 1, 2):
        ports = sorted((c, rr) for rr, c in centers if abs(rr - r) == 1)
        for (left, side_a), (right, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                _require(right - left >= 4, "ports too close")
                breaks.append((r, int(rng.integers(left + 2, right - 1))))
    for _ in range(extras):
        choices = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(3, width - 3)
                   if all(abs(c - cc) >= 3 for rr, cc in centers if abs(rr - r) == 1)
                   and all(abs(c - cc) >= 4 for rr, cc in breaks if rr == r)]
        if not choices:
            break
        breaks.append(choices[int(rng.integers(len(choices)))])
    return sorted(breaks)


def _colored_core(rng, lanes, width, centers, breaks, flags, palette_size):
    broken = set(breaks)
    cells = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(width) if (r, c) not in broken]
    union = _Union(cells)
    # The guide color under a terminal bead must be recoverable from its
    # inward neighbor, including terminals on either side of a break.
    for r in range(0, 2 * lanes - 1, 2):
        runs = []
        for c in range(width):
            if (r, c) in broken:
                continue
            if runs and c == runs[-1][-1] + 1:
                runs[-1].append(c)
            else:
                runs.append([c])
        for run in runs:
            _require(len(run) >= 3, "short raw track")
            union.join((r, run[0]), (r, run[1]))
            union.join((r, run[-1]), (r, run[-2]))
    unequal = []
    for (r, c), active in zip(centers, flags):
        upper = [(r - 1, x) for x in range(c - 1, c + 2)]
        lower = [(r + 1, x) for x in range(c - 1, c + 2)]
        for flank in (upper, lower):
            for p in flank[1:]:
                union.join(flank[0], p)
        if active:
            union.join(upper[0], lower[0])
        else:
            unequal.append((upper[0], lower[0]))
    adjacency = {}
    for p, q in unequal:
        a, b = union.find(p), union.find(q)
        _require(a != b, "inconsistent equality pattern")
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    palette = list(map(int, rng.permutation(np.arange(1, palette_size + 1))))
    ordered = list(adjacency)
    rng.shuffle(ordered)
    ordered.sort(key=lambda p: -len(adjacency[p]))
    assigned = {}
    for root in ordered:
        options = [v for v in palette if v not in {assigned[n] for n in adjacency[root] if n in assigned}]
        _require(bool(options), "guide palette cannot color inequalities")
        assigned[root] = int(rng.choice(options))
    core = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    for r in range(0, core.shape[0], 2):
        previous = int(rng.choice(palette))
        for c in range(width):
            p = (r, c)
            if p in broken:
                previous = int(rng.choice(palette))
                continue
            root = union.find(p)
            if root not in assigned:
                assigned[root] = previous if rng.random() < .82 else int(rng.choice(palette))
            core[p] = assigned[root]
            previous = int(core[p])
    for r, c in centers:
        core[r, c - 1:c + 2] = 7
    return core


def _place_beads(rng, base, a, density, multihop):
    forbidden = {(r + dr, c + dc) for r, c in a['centers'] for dr in (-1, 1) for dc in (-1, 0, 1)}
    eligible = a['original_ends'] - forbidden
    vertical = [p for p in a['paths'] if p[0][0] != p[-1][0]]
    singles = [p for p in vertical if min(p[0], p[-1]) in eligible]
    if multihop:
        singles = [p for p in singles if sum(int(r % 2) for r, c in p) >= 2
                   and len({int(a['structural'][q]) for q in p}) >= 2]
    doubles = [p for p in vertical if {p[0], p[-1]} <= eligible]
    level = [p for p in a['paths'] if p[0][0] == p[-1][0] and ({p[0], p[-1]} & eligible)]
    if not singles or not doubles or not level:
        return None
    all_paths = _all_open_paths(a)
    if multihop and not any(sum(r % 2 for r, c in path) >= 2 for path in vertical):
        return None
    for _ in range(18):
        dots = {p for p in sorted(eligible) if rng.random() < density}
        single = singles[int(rng.integers(len(singles)))]
        candidates = [p for p in doubles if p is not single]
        if not candidates:
            continue
        double = candidates[int(rng.integers(len(candidates)))]
        dots.difference_update((single[0], single[-1]))
        dots.add(min(single[0], single[-1]))
        dots.update((double[0], double[-1]))
        flat = level[int(rng.integers(len(level)))]
        flat_ends = sorted({flat[0], flat[-1]} & eligible)
        internal = [p for p in flat_ends if p[1] not in (0, base.shape[1] - 1)]
        dots.add((internal or flat_ends)[int(rng.integers(len(internal or flat_ends)))])
        targets, features = _targets(a['paths'], dots)
        other, _ = _targets(all_paths, dots)
        # Equality must change actual bead destinations, not just bar pixels.
        if targets == other or not all(features[k] for k in ('single', 'collisions', 'levels')):
            continue
        inp = base.copy()
        for p in dots:
            inp[p] = 8
        return inp
    return None


def _sample_pair(rng, lanes_range, width_range, rich, density_range, multihop, mapping, rotation):
    for attempt in range(240):
        lanes = int(rng.integers(*lanes_range))
        width = int(rng.integers(*width_range))
        centers = _sample_centers(rng, lanes, width, rich)
        if len(centers) < (3 if multihop else 2):
            continue
        if multihop and max(sum(rr == r for rr, c in centers) for r, c in centers) < 2:
            continue
        flags = list(map(bool, rng.random(len(centers)) < float(rng.uniform(.35, .8))))
        chosen = list(map(int, rng.permutation(len(flags))))
        flags[chosen[0]], flags[chosen[1]] = True, False
        if multihop:
            flags[chosen[2]] = True
        try:
            breaks = _make_breaks(rng, lanes, width, centers, int(rng.integers(1, lanes + 2)))
            base = _colored_core(rng, lanes, width, centers, breaks, flags, int(rng.integers(2, 7)))
            guides = set(map(int, np.unique(base[::2]))) - {0}
            a = _topology(base, 0, 7, 8, guides)
            inp = _place_beads(rng, base, a, float(rng.uniform(*density_range)), multihop)
            if inp is None:
                continue
            # Leave room for the collector and clear gap, optionally translate
            # the core. Padding belongs to the field, not to the track graph.
            top_room = min(2, 30 - inp.shape[0] - 2)
            top = int(rng.integers(top_room + 1))
            gap = int(rng.integers(1, min(2, 29 - inp.shape[0] - top) + 1))
            side_room = min(4, 30 - width)
            sides = int(rng.integers(side_room + 1))
            left = int(rng.integers(sides + 1))
            framed = np.pad(inp, ((top, gap + 1), (left, sides - left)), constant_values=0)
            framed[-1] = 9
            framed = np.asarray(mapping, dtype=np.int8)[framed]
            framed = np.rot90(framed, rotation).copy()
            return {'input': framed, 'output': solve(framed)}
        except ValueError:
            continue
    raise RuntimeError("bounded switchyard construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_mapping = list(map(int, rng.permutation(10)))
    mixed_palette = bool(rng.integers(2))
    mixed_orientation = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    grow = bool(rng.integers(2))
    rich = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    # Educational examples can occur at any training positions.
    educational = list(map(int, rng.permutation(num_train)))
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            is_test = split == 'test'
            large = is_test == grow
            lanes_range = (7, 14) if large else (3, 7)
            widths = (13, 31) if is_test else (11, 29)
            density = (.78, 1.0) if is_test and not grow else (.18, .68)
            multihop = is_test or index == educational[1]
            pair_rotation = rotation
            if mixed_orientation:
                pair_rotation = (rotation + (index if not is_test else index + 2)) % 4
            for _ in range(30):
                mapping = list(map(int, rng.permutation(10))) if mixed_palette else fixed_mapping
                pair = _sample_pair(rng, lanes_range, widths, rich or multihop,
                                    density, multihop, mapping, pair_rotation)
                key = _signature(pair['input'])
                if split == 'train' and index == count - 1:
                    readings = [_read(p['input']) for p in puzzle['train']] + [_read(pair['input'])]
                    matched = {int(a['core'][r - 1, c]) for a in readings for r, c in a['active']}
                    unmatched = {int(a['core'][r + dr, c]) for a in readings
                                 for r, c in a['inactive'] for dr in (-1, 1)}
                    if len(matched) < 2 or not (matched & unmatched):
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _read(inp)
            _require(np.array_equal(_answer(a), out), "incorrect output")
            targets, f = _targets(a['paths'], a['dots'])
            other, _ = _targets(_all_open_paths(a), a['dots'])
            _require(targets != a['dots'], "rewiring must affect bead locations")
            _require(targets != other, "flank equality must affect bead locations")
            _require(len(targets) == len(a['dots']), "bead conservation")
            _require(np.count_nonzero(inp == a['bead']) == np.count_nonzero(out == a['bead']), "global bead count")
            normal_out = np.rot90(out, a['rotation'])
            r0, r1, c0, c1 = a['box']
            out_core = normal_out[r0:r1, c0:c1]
            permitted = set(a['writes']) | a['dots'] | targets
            changes = set(map(tuple, np.argwhere(a['core'] != out_core)))
            _require(changes <= permitted, "unrelated cell changed")
            outside = np.ones(inp.shape if a['rotation'] % 2 == 0 else inp.shape[::-1], dtype=bool)
            outside[r0:r1, c0:c1] = False
            _require(np.array_equal(normal_out[outside], a['grid'][outside]), "field or collector changed")
            for p in a['breaks']:
                _require(out_core[p] == a['bg'], "break erased")
            for r, c in a['inactive']:
                _require(np.all(out_core[r, c - 1:c + 2] == a['pivot']), "locked bar rotated")
            if split == 'train':
                training.append((a, f))
    _require(all(any(f[k] for a, f in training) for k in ('single', 'collisions', 'levels')), "training coverage")
    _require(any(any(sum(int(r % 2) for r, c in path) >= 2
                        and len(a['dots'] & {path[0], path[-1]}) == 1
                        and min(path[0], path[-1]) in a['dots']
                        and len({int(a['structural'][q]) for q in path}) >= 2
                        for path in a['paths'])
                 for a, f in training), "moving multi-switch demonstration")
    _require(any(any(sum(rr == r for rr, c in a['centers']) >= 2 for r, c in a['centers']) for a, f in training), "busy-separator demonstration")
    _require(any(any(c not in (0, a['core'].shape[1] - 1) for r, c in a['dots']) for a, f in training), "break-adjacent demonstration")
    matched = {int(a['core'][r - 1, c]) for a, f in training for r, c in a['active']}
    unmatched = {int(a['core'][r + dr, c]) for a, f in training
                 for r, c in a['inactive'] for dr in (-1, 1)}
    _require(len(matched) >= 2 and bool(matched & unmatched), "equality must generalize across identities")
    return True


def _measure(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _read(pair['input'])
            _, f = _targets(a['paths'], a['dots'])
            normalized = np.zeros(a['core'].shape, dtype=np.int8)
            normalized[np.isin(a['core'], list(a['guides']))] = 1
            normalized[a['core'] == a['pivot']] = 2
            normalized[a['core'] == a['bead']] = 3
            records.append(dict(split=split, shape=pair['input'].shape,
                                lanes=(a['core'].shape[0] + 1) // 2,
                                width=a['core'].shape[1], active=len(a['active']),
                                inactive=len(a['inactive']), breaks=len(a['breaks']),
                                beads=len(a['dots']), guides=len(a['guides']),
                                overlap=a['overlap'], direction=a['rotation'],
                                roles=(a['bg'], a['pivot'], a['bead'], a['wall']),
                                hops=max(sum(int(r % 2) for r, c in path) for path in a['paths']),
                                layout=_signature(normalized), **f))
    return records


def self_test():
    """Bounded 42-seed variation and family invariants, plus metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in _measure(p)]
    signatures = [tuple(_signature(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles]
    assert len(set(signatures)) == 42
    assert len({r['shape'] for r in records}) >= 65
    assert len({r['layout'] for r in records}) >= 180
    assert len({r['active'] for r in records}) >= 10
    assert len({r['inactive'] for r in records}) >= 8
    assert len({r['breaks'] for r in records}) >= 10
    assert len({r['beads'] for r in records}) >= 15
    assert {r['direction'] for r in records} == set(range(4))
    assert max(r['hops'] for r in records) >= 4
    assert any(r['overlap'] for r in records)
    for role in range(4):
        assert {r['roles'][role] for r in records} == set(range(10))
    for p in puzzles:
        validate_generated_puzzle(p)
    # Whole-puzzle determinism, all legal explicit counts and strict typing.
    first = generate_puzzle(314159, 7, 3)
    again = generate_puzzle(314159, 7, 3)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = puzzles[0]['train'][0]
    original = pair['input'].copy()
    assert np.array_equal(solve(original), pair['output']) and np.array_equal(original, pair['input'])
    # No role or gravity literal: the visible collector and geometry travel.
    permutation = np.random.default_rng(2718).permutation(10)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(original, k)), np.rot90(pair['output'], k))
    assert np.array_equal(solve(permutation[original]), permutation[pair['output']])
    assert np.array_equal(solve(np.fliplr(original)), np.fliplr(pair['output']))
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt['test'][0]['output'] = corrupt['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    return dict(seeds=42, pairs=len(records), shapes=len({r['shape'] for r in records}),
                physical_layouts=len({r['layout'] for r in records}),
                active_range=[min(r['active'] for r in records), max(r['active'] for r in records)],
                bead_range=[min(r['beads'] for r in records), max(r['beads'] for r in records)],
                maximum_hops=max(r['hops'] for r in records),
                overlap_pairs=sum(bool(r['overlap']) for r in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
