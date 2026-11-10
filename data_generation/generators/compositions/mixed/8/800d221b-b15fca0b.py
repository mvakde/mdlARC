"""Shortest-route electorate: panel-labelled ports on a collector collar.

Standard library + NumPy only. No files, subprocesses, stored tasks, generation
metadata or random state are used by solve(). See task_spec.md for the full rule.
Generation/self_test keep at most one small puzzle and bounded hash sets in RAM;
they do not persist JSON, logs or execution traces.
"""
from collections import deque
import hashlib
import itertools

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 882509194, 113839124, 494734805, 791163165, 518387343,
              93835036, 757290144, 869915135, 325775757, 630784556,
              780658221, 926764754, 688332683, 289699062, 203267711,
              36891599, 446164190, 826064053, 985378722, 443409641,
              46697219, 137851085, 786074968, 955491727, 279751028,
              741630636, 140474429, 643520472, 308172151, 70743976,
              650850735, 242944542, 99612503, 355520108, 245283280,
              662375450, 694658162, 75662337, 820701529, 908493877,
              147525510)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "integer colors 0..9 required")
    return a


def _neighbors(p, shape):
    r, c = p
    for dr, dc in DIRS:
        q = r + dr, c + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    seen = set()
    result = []
    for p in zip(*np.nonzero(mask)):
        p = tuple(map(int, p))
        if p in seen:
            continue
        seen.add(p)
        cells, queue = [], [p]
        while queue:
            q = queue.pop()
            cells.append(q)
            for v in _neighbors(q, mask.shape):
                if mask[v] and v not in seen:
                    seen.add(v)
                    queue.append(v)
        result.append(cells)
    return result


def _route(passable, start, goal):
    """BFS with number of shortest paths capped at two, not DFS discovery order."""
    if not passable[start] or not passable[goal]:
        return None, 0
    distance, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, passable.shape):
            if not passable[q]:
                continue
            d = distance[p] + 1
            if q not in distance:
                distance[q], ways[q], parent[q] = d, ways[p], p
                queue.append(q)
            elif distance[q] == d:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in distance:
        return None, 0
    path, p = [goal], goal
    while p != start:
        p = parent[p]
        path.append(p)
    return path[::-1], ways[goal]


def _ring(box):
    """Clockwise, with each of the four corners occurring once."""
    r0, r1, c0, c1 = box
    return ([(r0, c) for c in range(c0, c1 + 1)] +
            [(r, c1) for r in range(r0 + 1, r1)] +
            [(r1, c) for c in range(c1, c0 - 1, -1)] +
            [(r, c0) for r in range(r1 - 1, r0, -1)])


def _majority(labels, signals):
    counts = [labels.count(s) for s in signals]
    if counts[0] == counts[1]:
        return None
    return signals[int(counts[1] > counts[0])]


def _interpret(a, scaffold, background, terminal, signals):
    free = (a == background) | (a == terminal)
    coords = np.argwhere(free)
    _require(len(coords) >= 8, "too few port sites")
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    _require(r1 - r0 >= 4 and c1 - c0 >= 4 and
             (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0,
             "collector must have odd dimensions >=3")
    _require(r0 >= 2 and c0 >= 2 and r1 < a.shape[0] - 2 and
             c1 < a.shape[1] - 2, "panels and outer wall must fit")
    box = r0, r1, c0, c1
    ring = _ring(box)
    ring_set = set(ring)
    _require(all(tuple(p) in ring_set for p in coords), "background outside collar")
    _require(all(free[p] for p in ((r0, c0), (r0, c1),
                                 (r1, c0), (r1, c1))), "corners cannot be blocked")
    _require(bool(np.all(a[r0 + 1:r1, c0 + 1:c1] == scaffold)),
             "collector interior must be solid scaffold")
    _require(all(int(a[p]) in (background, terminal, scaffold) for p in ring),
             "panels cannot lie in collar")
    panel_mask = np.isin(a, signals)
    panels = _components(panel_mask)
    _require(4 <= len(panels) <= 8, "expected four to eight distinct panels")
    owners, majorities, areas, boxes = {}, [], [], []
    for i, cells in enumerate(panels):
        rs, cs = zip(*cells)
        pr0, pr1, pc0, pc1 = min(rs), max(rs), min(cs), max(cs)
        _require(len(cells) == (pr1 - pr0 + 1) * (pc1 - pc0 + 1),
                 "each panel must be a filled rectangle")
        _require(not any(p in ring_set for p in cells), "panel overlaps collar")
        # A panel is directly outside exactly one complete or split side.
        side_ok = ((pr1 == r0 - 1 and c0 <= pc0 <= pc1 <= c1) or
                   (pr0 == r1 + 1 and c0 <= pc0 <= pc1 <= c1) or
                   (pc1 == c0 - 1 and r0 <= pr0 <= pr1 <= r1) or
                   (pc0 == c1 + 1 and r0 <= pr0 <= pr1 <= r1))
        _require(side_ok, "panel must face one collar side")
        labels = [int(a[p]) for p in cells]
        _require(set(labels) == set(signals), "both signals required in every panel")
        winner = _majority(labels, signals)
        _require(winner is not None, "local tie")
        majorities.append(winner)
        areas.append(len(cells))
        boxes.append((pr0, pr1, pc0, pc1))
        for p in cells:
            owners[p] = i
    port_owner = {}
    for p in ring:
        adjacent = {owners[q] for q in _neighbors(p, a.shape) if q in owners}
        if free[p]:
            _require(len(adjacent) == 1, "every available port needs one visible panel")
            port_owner[p] = adjacent.pop()
        else:
            _require(not adjacent, "wall gate must separate adjacent panels")
    _require(set(port_owner.values()) == set(range(len(panels))),
             "every panel must face at least one port")
    terminals = [tuple(map(int, p)) for p in np.argwhere(a == terminal)]
    _require(len(terminals) == 2, "exactly two terminals required")
    path, ways = _route(free, *terminals)
    _require(path is not None and ways == 1 and len(path) >= 7,
             "unique shortest route with at least five intermediate cells required")
    labels = [majorities[port_owner[p]] for p in path[1:-1]]
    winner = _majority(labels, signals)
    _require(winner is not None, "collector vote cannot tie")
    _require(np.count_nonzero(a == background) > len(path) - 2,
             "off-route background required")
    return dict(scaffold=scaffold, background=background, terminal=terminal,
                signals=signals, box=box, center=((r0 + r1) // 2, (c0 + c1) // 2),
                ring=ring, panels=panels, panel_boxes=boxes, panel_areas=areas,
                panel_majorities=majorities, port_owner=port_owner, path=path,
                labels=labels, winner=winner, gates=sum(not free[p] for p in ring))


def _inspect(grid):
    """Recover roles, panels, collector, route and votes entirely from input."""
    a = _array(grid)
    values = tuple(map(int, np.unique(a)))
    _require(len(values) == 5, "five semantic colors required")
    scaffold = int(a[0, 0])
    _require(bool(np.all(a[0] == scaffold) and np.all(a[-1] == scaffold) and
                  np.all(a[:, 0] == scaffold) and np.all(a[:, -1] == scaffold)),
             "outer frame must be scaffold")
    interpretations = []
    remaining = [v for v in values if v != scaffold]
    for terminal in remaining:
        if np.count_nonzero(a == terminal) != 2:
            continue
        for background in remaining:
            if background == terminal:
                continue
            signals = tuple(v for v in remaining if v not in (terminal, background))
            try:
                interpretations.append(_interpret(a, scaffold, background, terminal, signals))
            except ValueError:
                pass
    _require(len(interpretations) == 1, "input needs one geometric role interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _array(input_grid)
    info = _inspect(a)
    out = a.copy()
    for p, label in zip(info['path'][1:-1], info['labels']):
        out[p] = label
    out[info['center']] = info['winner']
    return out


def _pattern(rng, shape, majority, minority):
    area = int(np.prod(shape))
    strength = int(rng.integers(3))
    if strength == 0:
        count = (area - 1) // 2
    elif strength == 1:
        count = max(1, int(area * rng.uniform(.16, .35)))
    else:
        count = int(rng.integers(1, max(2, area // 10 + 1)))
    count = min(count, (area - 1) // 2)
    rr, cc = np.indices(shape)
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        order = np.argsort((abs(rr - r) + abs(cc - c) + rng.random(shape) * .2).ravel())
    elif style == 2:
        order = np.argsort(((rr + cc) % 2 * area + rr * shape[1] + cc).ravel())
    else:
        order = np.argsort((cc * shape[0] + rr).ravel())
    result = np.full(area, majority, dtype=np.int8)
    result[order[:count]] = minority
    return result.reshape(shape)


def _partition(rng, length, split):
    if not split:
        return [(0, length)], []
    gap = int(rng.integers(1, min(2, length - 6) + 1))
    first = int(rng.integers(3, length - gap - 2))
    return [(0, first), (first + gap, length)], list(range(first, first + gap))


def _layout(rng, inner_shape, gate_sides, depth_max, depth_extreme=False):
    ih, iw = inner_shape
    rh, rw = ih + 2, iw + 2
    lengths = (rw, ih, rw, ih)  # top, right, bottom, left
    eligible = [i for i, n in enumerate(lengths) if n >= 7]
    if gate_sides > len(eligible):
        return None
    split = set(map(int, rng.choice(eligible, gate_sides, replace=False)))
    parts = [_partition(rng, n, i in split) for i, n in enumerate(lengths)]
    depths = [[int(rng.integers(1, depth_max + 1)) for _ in p[0]] for p in parts]
    if depth_extreme:
        slots = [(side, i) for side, ds in enumerate(depths) for i in range(len(ds))]
        chosen = rng.choice(len(slots), 2, replace=False)
        for slot, value in zip(chosen, (6, 1)):
            side, i = slots[int(slot)]
            depths[side][i] = value
    # All segments have length >=3, hence contain both signals even at depth 1.
    top, right, bottom, left = [max(d) for d in depths]
    base_h, base_w = rh + top + bottom + 2, rw + left + right + 2
    if max(base_h, base_w) > 30:
        return None
    pad = [int(rng.integers(0, min(3, 30 - base_h) + 1)),
           int(rng.integers(0, min(3, 30 - base_w) + 1))]
    before = [int(rng.integers(n + 1)) for n in pad]
    shape = base_h + pad[0], base_w + pad[1]
    r0, c0 = 1 + top + before[0], 1 + left + before[1]
    r1, c1 = r0 + rh - 1, c0 + rw - 1
    box = r0, r1, c0, c1
    panels, owner, gates = [], {}, set()
    for side, ((segments, holes), side_depths) in enumerate(zip(parts, depths)):
        for (lo, hi), depth in zip(segments, side_depths):
            if side == 0:
                panel_box = r0 - depth, r0, c0 + lo, c0 + hi
                ports = [(r0, c) for c in range(c0 + lo, c0 + hi)]
            elif side == 1:
                panel_box = r0 + 1 + lo, r0 + 1 + hi, c1 + 1, c1 + 1 + depth
                ports = [(r, c1) for r in range(r0 + 1 + lo, r0 + 1 + hi)]
            elif side == 2:
                panel_box = r1 + 1, r1 + 1 + depth, c0 + lo, c0 + hi
                ports = [(r1, c) for c in range(c0 + lo, c0 + hi)]
            else:
                panel_box = r0 + 1 + lo, r0 + 1 + hi, c0 - depth, c0
                ports = [(r, c0) for r in range(r0 + 1 + lo, r0 + 1 + hi)]
            pid = len(panels)
            panels.append(panel_box)
            owner.update({p: pid for p in ports})
        for offset in holes:
            gates.add(((r0, c0 + offset), (r0 + 1 + offset, c1),
                       (r1, c0 + offset), (r0 + 1 + offset, c0))[side])
    return shape, box, panels, owner, gates


def _choose_electorate(rng, shape, box, owner, gates, witness, winner_index):
    ring = _ring(box)
    free = np.zeros(shape, dtype=bool)
    for p in ring:
        free[p] = p not in gates
    available = [p for p in ring if p not in gates]
    n = max(owner.values()) + 1
    # A shuffled finite set of endpoint pairs; no unbounded search or fallback stream.
    pairs = list(itertools.combinations(available, 2))
    rng.shuffle(pairs)
    for start, goal in pairs[:600]:
        path, count = _route(free, start, goal)
        if path is None or count != 1 or len(path) < 7:
            continue
        if len(path) - 2 >= len(available) - 2:
            continue
        selected = [owner[p] for p in path[1:-1]]
        if len(set(selected)) < 2:
            continue
        choices = []
        for bits in range(1, (1 << n) - 1):
            labels = [(bits >> i) & 1 for i in range(n)]
            votes = [labels[i] for i in selected]
            win = _majority(votes, (0, 1))
            if win != winner_index or set(votes) != {0, 1}:
                continue
            if witness:
                # Lose under panel counting and under colouring the entire field;
                # the unique *shorter* of two arcs must elect the opposite label.
                if _majority(labels, (0, 1)) != 1 - win:
                    continue
                if _majority([labels[owner[p]] for p in available if p not in (start, goal)],
                             (0, 1)) != 1 - win:
                    continue
                if _majority([labels[i] for i in set(selected)], (0, 1)) != 1 - win:
                    continue
            choices.append(labels)
        if choices:
            return start, goal, path, choices[int(rng.integers(len(choices)))]
    return None


def _ensure_minority_contact(rng, inp, panels, owner, path, signals):
    """Keep counts fixed, but guarantee a visible neighbor-copying counterexample."""
    contacts = []
    for p in path[1:-1]:
        i = owner[p]
        r0, r1, c0, c1 = panels[i]
        for q in _neighbors(p, inp.shape):
            if r0 <= q[0] < r1 and c0 <= q[1] < c1:
                contacts.append((i, q))
    rng.shuffle(contacts)
    for i, q in contacts:
        r0, r1, c0, c1 = panels[i]
        block = inp[r0:r1, c0:c1]
        majority = _majority(list(map(int, block.flat)), signals)
        minority = next(s for s in signals if s != majority)
        if inp[q] == minority:
            return
        other = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(block == minority)]
        p = other[int(rng.integers(len(other)))]
        inp[q], inp[p] = inp[p], inp[q]
        return


def _make_pair(rng, roles, winner_index, inner_choices, gate_choices,
               depth_max, witness=False, raw_witness=False, depth_extreme=False):
    bg, scaffold, terminal, s0, s1 = roles
    signals = (s0, s1)
    for _ in range(100):
        inner = tuple(map(int, rng.choice(inner_choices, 2)))
        if witness:
            inner = tuple(map(int, rng.choice((5, 7), 2)))
        gate_sides = 0 if witness else int(rng.choice(gate_choices))
        layout = _layout(rng, inner, gate_sides, depth_max, depth_extreme)
        if layout is None:
            continue
        shape, box, panels, owner, gates = layout
        elected = _choose_electorate(rng, shape, box, owner, gates, witness, winner_index)
        if elected is None:
            continue
        start, goal, path, labels = elected
        inp = np.full(shape, scaffold, dtype=np.int8)
        for p in _ring(box):
            if p not in gates:
                inp[p] = bg
        inp[start] = inp[goal] = terminal
        for i, (r0, r1, c0, c1) in enumerate(panels):
            inp[r0:r1, c0:c1] = _pattern(rng, (r1 - r0, c1 - c0),
                                         signals[labels[i]], signals[1 - labels[i]])
        _ensure_minority_contact(rng, inp, panels, owner, path, signals)
        winner = signals[winner_index]
        if raw_witness and _majority([int(x) for x in inp.flat if int(x) in signals],
                                     signals) != signals[1 - winner_index]:
            continue
        # Output is written from construction records, then rediscovered in validation.
        out = inp.copy()
        for p in path[1:-1]:
            out[p] = signals[labels[owner[p]]]
        r0, r1, c0, c1 = box
        out[(r0 + r1) // 2, (c0 + c1) // 2] = winner
        # Random quarter turns change which pair of sides owns the corner ports.
        # The parser recognizes the geometry, including its rotated version.
        if int(rng.integers(2)):
            inp, out = inp[::-1].copy(), out[::-1].copy()
        if int(rng.integers(2)):
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        # Transpose is supported through _inspect's orientation normalization.
        if int(rng.integers(2)):
            inp, out = inp.T.copy(), out.T.copy()
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded collar construction exhausted')


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, (int, np.integer)) and lo <= value <= hi,
                 'illegal example count')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    tr, te = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)), 'seed must be an integer')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    mode = int(rng.integers(3))
    vary_palette = bool(rng.integers(2))
    winners = list(map(int, rng.integers(2, size=tr)))
    if len(set(winners)) == 1:
        winners[int(rng.integers(tr))] ^= 1
    witness_slot = int(rng.integers(tr))
    gate_slot = (witness_slot + int(rng.integers(1, tr))) % tr
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', tr), ('test', te)):
        for i in range(count):
            witness = split == 'train' and i == witness_slot
            winner = winners[i] if split == 'train' else int(rng.integers(2))
            pair_roles = tuple(map(int, rng.choice(10, 5, replace=False))) if vary_palette else roles
            if split == 'train':
                inner = (3, 5, 7)
                gates = (0, 1) if mode == 1 else (0, 1, 2)
                if i == gate_slot:
                    gates = (1,)
                depths = 3 if mode == 2 else 5
            else:
                inner = (9, 11) if mode == 0 else ((5, 7, 9) if mode == 1 else (3, 5, 7))
                gates = (2, 3, 4) if mode == 1 else (0, 1, 2)
                depths = 6 if mode == 2 else 5
            for _ in range(12):
                pair = _make_pair(rng, pair_roles, winner, inner, gates, depths,
                                  witness=witness, raw_witness=witness,
                                  depth_extreme=split == 'test' and mode == 2)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(inp, info=None):
    a = np.asarray(inp)
    x = _inspect(a) if info is None else info
    s = x['signals']
    selected_owners = [x['port_owner'][p] for p in x['path'][1:-1]]
    occupied = set(selected_owners)
    all_ports = [p for p in x['port_owner'] if int(a[p]) == x['background']]
    full_vote = _majority([x['panel_majorities'][x['port_owner'][p]] for p in all_ports], s)
    panel_vote = _majority(x['panel_majorities'], s)
    active_vote = _majority([x['panel_majorities'][i] for i in occupied], s)
    raw_vote = _majority([int(v) for v in a.flat if int(v) in s], s)
    mismatch = False
    for p, label in zip(x['path'][1:-1], x['labels']):
        mismatch |= any(int(a[q]) in s and int(a[q]) != label for q in _neighbors(p, a.shape))
    minority_fractions = []
    for cells in x['panels']:
        counts = [sum(a[p] == v for p in cells) for v in s]
        minority_fractions.append(min(counts) / sum(counts))
    r0, r1, c0, c1 = x['box']
    terminals = x['path'][0], x['path'][-1]
    return dict(shape=tuple(a.shape), collector_shape=(r1-r0-1, c1-c0-1),
                panel_count=len(x['panels']), gates=x['gates'],
                occupied_ports=len(x['labels']),
                margin=abs(x['labels'].count(s[0]) - x['labels'].count(s[1])),
                weights=sorted(selected_owners.count(i) for i in occupied),
                panel_areas=x['panel_areas'], minority_fractions=minority_fractions,
                winner_index=s.index(x['winner']), roles=(x['background'], x['scaffold'], x['terminal'], *s),
                full_vote_opposes=full_vote is not None and full_vote != x['winner'],
                panel_vote_opposes=panel_vote is not None and panel_vote != x['winner'],
                active_vote_opposes=active_vote is not None and active_vote != x['winner'],
                raw_vote_opposes=raw_vote is not None and raw_vote != x['winner'],
                neighbor_copy_fails=bool(mismatch),
                detour=len(x['path']) - 1 - sum(abs(a-b) for a,b in zip(*terminals)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'pair lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _array(pair['input']), _array(pair['output'])
            _require(inp.shape == out.shape, 'output shape')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            x = _inspect(inp)
            predicted = solve(inp)
            _require(bool(np.array_equal(predicted, out)), 'output disagrees with input-only rule')
            changed = set(map(tuple, np.argwhere(inp != out)))
            expected = set(x['path'][1:-1]) | {x['center']}
            _require(changed == expected, 'only route intermediates and center may change')
            _require(all(int(inp[p]) == x['background'] for p in x['path'][1:-1]),
                     'non-background route intermediate')
            _require(set(x['labels']) == set(x['signals']),
                     'generated route must demonstrate both local labels')
            _require(all(int(out[p]) == int(inp[p]) for cells in x['panels'] for p in cells),
                     'panel minority preservation')
            m = _metrics(inp, x)
            _require(m['neighbor_copy_fails'], 'local majority must defeat neighbor copying')
            if split == 'train':
                train.append(m)
    # A geometrically exposed training witness defeats four tempting substitutes.
    _require(any(m['gates'] == 0 and all(m[k] for k in
                 ('full_vote_opposes', 'panel_vote_opposes', 'active_vote_opposes',
                  'raw_vote_opposes')) for m in train), 'missing causal training witness')
    _require(any(m['gates'] > 0 for m in train), 'missing blocked-collar demonstration')


def self_test():
    """42 supplied seeds, semantic mutations, equivariance and explicit counts.

    No artifact writes. At most 14 pairs of two <=30x30 int8 arrays are retained
    by generation (<26KB raw); 42 seed hashes and small diversity sets are kept.
    """
    shapes, cores, palettes, panel_counts, gates, weights, margins, hashes = (set() for _ in range(8))
    ports, fractions, areas = [], [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                m = _metrics(inp)
                shapes.add(m['shape'])
                cores.add(m['collector_shape'])
                palettes.add(m['roles'])
                panel_counts.add(m['panel_count'])
                gates.add(m['gates'])
                weights.add(tuple(m['weights']))
                margins.add(m['margin'])
                ports.append(m['occupied_ports'])
                fractions.extend(m['minority_fractions'])
                areas.extend(m['panel_areas'])
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(pair['output'].tobytes())
        hashes.add(digest.digest())
    _require(len(hashes) == 42 and len(shapes) >= 35 and len(cores) >= 10,
             'insufficient geometry variation')
    _require(len(palettes) >= 30 and len(panel_counts) >= 4 and len(gates) >= 4,
             'insufficient panel/gate/palette variation')
    _require(len(weights) >= 15 and len(margins) >= 5 and max(ports) >= 15,
             'insufficient electorate variation')
    _require(min(fractions) <= .05 and max(fractions) >= .45 and max(areas) >= 40,
             'insufficient local panel variation')
    first, repeat = generate_puzzle(0), generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for s in first
                 for a, b in zip(first[s], repeat[s]) for k in a), 'nondeterminism')
    pair = first['train'][0]
    before = pair['input'].copy()
    _require(np.array_equal(solve(before), pair['output']) and
             np.array_equal(before, pair['input']), 'solve mutates input')
    # Role recovery is geometrical: any bijection on colors must commute with solve.
    palette = np.array([9, 0, 7, 3, 8, 1, 6, 2, 5, 4], dtype=np.int8)
    _require(np.array_equal(solve(palette[before]), palette[pair['output']]),
             'palette permutation failed')
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(before, k)), np.rot90(pair['output'], k)),
                 'rotation failed')
    # Omit shortest selection: label every available background port instead.
    witness = next(p for p in first['train'] if all(_metrics(p['input'])[k] for k in
                   ('full_vote_opposes', 'panel_vote_opposes', 'active_vote_opposes', 'raw_vote_opposes')))
    x = _inspect(witness['input'])
    _require(x['gates'] == 0 and _metrics(witness['input'])['full_vote_opposes'],
             'B omission must change center')
    available = (witness['input'] == x['background']) | (witness['input'] == x['terminal'])
    available[x['path'][1]] = False
    longer, ways = _route(available, x['path'][0], x['path'][-1])
    _require(longer is not None and ways == 1 and len(longer) > len(x['path']),
             'witness must offer a longer alternate route')
    longer_vote = _majority([x['panel_majorities'][x['port_owner'][p]] for p in longer[1:-1]],
                             x['signals'])
    _require(longer_vote not in (None, x['winner']), 'longer route must change center')
    # Equal-length arcs are genuinely ambiguous and must be rejected.
    tied = witness['input'].copy()
    tied[tied == x['terminal']] = x['background']
    for p in (x['ring'][0], x['ring'][len(x['ring']) // 2]):
        tied[p] = x['terminal']
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('two shortest routes accepted')
    # Change one selected panel majority while preserving everything else.
    pivotal = None
    for i in set(x['port_owner'][p] for p in x['path'][1:-1]):
        labels = [next(s for s in x['signals'] if s != v) if x['port_owner'][p] == i else v
                  for p, v in zip(x['path'][1:-1], x['labels'])]
        if _majority(labels, x['signals']) not in (None, x['winner']):
            pivotal = i
            break
    _require(pivotal is not None, 'missing pivotal local majority')
    mutated = witness['input'].copy()
    for p in x['panels'][pivotal]:
        mutated[p] = next(s for s in x['signals'] if s != int(mutated[p]))
    _require(solve(mutated)[x['center']] != x['winner'], 'A dependency failed')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    center = _inspect(bad['train'][0]['input'])['center']
    bad['train'][0]['output'][center] = bad['train'][0]['input'][center]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong center accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(19, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds; no files written)')
