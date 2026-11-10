"""Contact ballots: assemble two-signal fragments around a fixed collector.

Only the standard library and NumPy are used. No files, parent generators,
generation metadata, or random state are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SEEDS = (0, 388949478, 459708468, 278202925, 927743992, 295802894,
         48059551, 312950467, 520845423, 207495578, 343034237, 434109724,
         766182552, 985007692, 318729676, 188813441, 957797062, 489319481,
         59053690, 587355433, 955426074, 634165822, 524708807, 29582285,
         308318528, 327067187, 968425676, 491799486, 466399137, 499350764,
         388156875, 384936525, 498872338, 575582027, 539696548, 165912538,
         854360480, 917445204, 760058739, 863098830, 459261105, 996375246)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _dilate(mask, diagonal=False):
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not diagonal and abs(dr) + abs(dc) != 1:
                continue
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            out[r0 + dr:r1 + dr, c0 + dc:c1 + dc] |= mask[r0:r1, c0:c1]
    return out


def _hubs(mask):
    h, w = mask.shape
    return [(r + 1, c + 1) for r in range(h - 2) for c in range(w - 2)
            if np.all(mask[r:r + 3, c:c + 3])]


def _shape(cells):
    return cells - cells.min(axis=0)


def _key(a):
    return a.shape, a.tobytes()


def _parse(grid):
    """Recover roles and pieces from visible connected components."""
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or np.sum(counts == counts.max()) != 1:
        raise ValueError("four roles and a uniquely most frequent background required")
    bg = int(colors[np.argmax(counts)])
    components = _components(a != bg)
    monos = [p for p in components if len(np.unique(a[p[:, 0], p[:, 1]])) == 1]
    if len(monos) != 1:
        raise ValueError("expected one monochrome fixed collector component")
    fixed = monos[0]
    scaffold = int(a[tuple(fixed[0])])
    signals = tuple(int(c) for c in colors if c not in (bg, scaffold))
    origin = fixed.min(axis=0)
    bh, bw = fixed.max(axis=0) - origin + 1
    core = np.zeros((bh, bw), dtype=bool)
    local = fixed - origin
    core[local[:, 0], local[:, 1]] = True
    hubs = _hubs(core)
    if len(hubs) != 1:
        raise ValueError("collector must contain exactly one solid 3x3 hub")
    pieces, shapes = [], set()
    for cells in components:
        values = a[cells[:, 0], cells[:, 1]]
        if np.all(values == scaffold):
            continue
        if tuple(map(int, np.unique(values))) != signals or len(cells) < 3:
            raise ValueError("each fragment must contain both signals")
        local = _shape(cells)
        signature = tuple(map(tuple, local.tolist()))
        if signature in shapes:
            raise ValueError("fragments must have distinct translation shapes")
        shapes.add(signature)
        # The target box is empty except for the collector in every input.
        if np.any(np.all((cells >= origin) & (cells < origin + (bh, bw)), axis=1)):
            raise ValueError("source fragments must lie outside the target rectangle")
        pieces.append(dict(source=cells, shape=local, values=values))
    if not 4 <= len(pieces) <= 8 or sum(len(p['shape']) for p in pieces) != int((~core).sum()):
        raise ValueError("four to eight fragments must exactly fill the collector box")
    box = a[origin[0]:origin[0] + bh, origin[1]:origin[1] + bw]
    if not np.all(box[~core] == bg):
        raise ValueError("target cavities must be empty")
    return dict(grid=a, background=bg, scaffold=scaffold, signals=signals,
                origin=origin, core=core, hub=hubs[0], pieces=pieces)


def _placements(core, pieces, node_limit=None):
    """Exact cover of ALL cavity cells, with each fragment used once.

    Every legal translation is considered, regardless of source position/color.
    Stop at the second geometric assembly. Construction uses a node budget to
    discard expensive proposals; public solving has no search cutoff.
    """
    h, w = core.shape
    target = sum(1 << int(r * w + c) for r, c in np.argwhere(~core))
    candidates, by_cell, by_piece = [], [[] for _ in range(h * w)], []
    for i, piece in enumerate(pieces):
        shape = piece['shape']
        ph, pw = shape.max(axis=0) + 1
        options = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                cells = shape + (r, c)
                if np.any(core[cells[:, 0], cells[:, 1]]):
                    continue
                indices = (cells[:, 0] * w + cells[:, 1]).tolist()
                bits = sum(1 << int(x) for x in indices)
                k = len(candidates)
                candidates.append((i, bits, cells))
                options.append(k)
                for x in indices:
                    by_cell[x].append(k)
        if not options:
            return []
        by_piece.append(options)
    full = (1 << len(pieces)) - 1
    answers, dead = [], set()
    nodes = 0

    def search(used, filled, chosen):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise ValueError("proposal exact-cover search exceeded budget")
        if len(answers) >= 2 or (used, filled) in dead:
            return
        if used == full:
            if filled == target:
                answers.append(chosen)
            return

        def feasible(k):
            i, bits, _ = candidates[k]
            return not (used >> i & 1) and not bits & filled

        options = None
        # Use both object and cell constraints, choosing the tightest one.
        for i, group in enumerate(by_piece):
            if not used >> i & 1:
                valid = [k for k in group if feasible(k)]
                if options is None or len(valid) < len(options):
                    options = valid
                if not options:
                    dead.add((used, filled))
                    return
        uncovered = target ^ filled
        while uncovered and len(options) > 1:
            bit = uncovered & -uncovered
            group = by_cell[bit.bit_length() - 1]
            valid = [k for k in group if feasible(k)]
            if len(valid) < len(options):
                options = valid
            if not options:
                dead.add((used, filled))
                return
            uncovered ^= bit
        before = len(answers)
        for k in options:
            i, bits, cells = candidates[k]
            search(used | (1 << i), filled | bits, chosen + [(i, cells)])
            if len(answers) >= 2:
                return
        if len(answers) == before:
            dead.add((used, filled))

    search(0, 0, [])
    return answers


def _analyze(input_grid, node_limit=None):
    info = _parse(input_grid)
    assemblies = _placements(info['core'], info['pieces'], node_limit)
    if len(assemblies) != 1:
        raise ValueError("input must have exactly one translation-only assembly")
    placement = dict(assemblies[0])
    contact = _dilate(info['core']) & ~info['core']
    labels, weights, edge_weights, raw_contacts = [], [], [], []
    for i, p in enumerate(info['pieces']):
        values = p['values']
        counts = [int(np.sum(values == s)) for s in info['signals']]
        if counts[0] == counts[1]:
            raise ValueError("local votes must be strict")
        labels.append(info['signals'][int(counts[1] > counts[0])])
        cells = placement[i]
        ports = contact[cells[:, 0], cells[:, 1]]
        weights.append(int(ports.sum()))
        raw_contacts.extend(map(int, values[ports]))
        edges = 0
        for r, c in cells:
            edges += sum(0 <= r + dr < contact.shape[0] and 0 <= c + dc < contact.shape[1]
                         and info['core'][r + dr, c + dc] for dr, dc in DIRS)
        edge_weights.append(int(edges))
    totals = [sum(w for w, label in zip(weights, labels) if label == s)
              for s in info['signals']]
    if min(totals) == 0 or totals[0] == totals[1]:
        raise ValueError("both signals must reach the collector with a strict result")
    winner = info['signals'][int(totals[1] > totals[0])]
    info.update(placement=placement, contact=contact, labels=labels, weights=weights,
                edge_weights=edge_weights, totals=totals, winner=winner,
                raw_contacts=raw_contacts)
    return info


def _render(info):
    out = info['grid'].copy()
    for p in info['pieces']:
        out[p['source'][:, 0], p['source'][:, 1]] = info['background']
    for i, cells in info['placement'].items():
        dest = cells + info['origin']
        out[dest[:, 0], dest[:, 1]] = info['pieces'][i]['values']
    center = info['origin'] + info['hub']
    out[tuple(center)] = info['winner']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble by geometry; then count fragment-majority labels at contact cells."""
    return _render(_analyze(input_grid))


def _core(rng, h, w, spurs):
    core = np.zeros((h, w), dtype=bool)
    r, c = int(rng.integers(1, h - 3)), int(rng.integers(1, w - 3))
    core[r:r + 3, c:c + 3] = True
    # Four independently wandering arms make the bounding box the target frame.
    for sign in (-1, 1):
        col = c + 1
        rows = range(r - 1, -1, -1) if sign == -1 else range(r + 3, h)
        for row in rows:
            new = int(np.clip(col + rng.integers(-1, 2), 0, w - 1))
            core[row, min(col, new):max(col, new) + 1] = True
            col = new
        row = r + 1
        cols = range(c - 1, -1, -1) if sign == -1 else range(c + 3, w)
        for col in cols:
            new = int(np.clip(row + rng.integers(-1, 2), 0, h - 1))
            core[min(row, new):max(row, new) + 1, col] = True
            row = new
    for _ in range(spurs):
        options = np.argwhere(_dilate(core) & ~core)
        if len(options):
            cell = options[int(rng.integers(len(options)))]
            trial = core.copy()
            trial[tuple(cell)] = True
            if len(_hubs(trial)) == 1:
                core = trial
    if _hubs(core) != [(r + 1, c + 1)]:
        raise ValueError("extra hub")
    return core, (r + 1, c + 1)


def _split(rng, cells):
    region = set(map(tuple, cells.tolist()))
    first = cells[int(rng.integers(len(cells)))]
    distance = np.abs(cells - first).sum(axis=1)
    second = cells[int(rng.choice(np.flatnonzero(distance == distance.max())))]
    owners = {tuple(first): 0, tuple(second): 1}
    frontier = []
    for cell, owner in owners.items():
        frontier.extend((owner, (cell[0] + dr, cell[1] + dc)) for dr, dc in DIRS)
    while frontier:
        index = int(rng.integers(len(frontier)))
        owner, cell = frontier[index]
        frontier[index] = frontier[-1]
        frontier.pop()
        if cell not in region or cell in owners:
            continue
        owners[cell] = owner
        frontier.extend((owner, (cell[0] + dr, cell[1] + dc)) for dr, dc in DIRS)
    return [np.array(sorted(p for p, owner in owners.items() if owner == i), dtype=int)
            for i in (0, 1)]


def _partition(rng, core, count):
    regions = _components(~core)
    if len(regions) > count or any(len(p) < 3 for p in regions):
        raise ValueError("tiny or too many chambers")
    for _ in range(80):
        if len(regions) == count:
            break
        choices = [i for i, p in enumerate(regions) if len(p) >= 8]
        if not choices:
            raise ValueError("cannot split further")
        i = int(rng.choice(choices))
        parts = _split(rng, regions[i])
        if min(map(len, parts)) >= 3:
            regions[i:i + 1] = parts
    if len(regions) != count:
        raise ValueError("partition retry bound")
    signatures = [tuple(map(tuple, _shape(p).tolist())) for p in regions]
    if len(set(signatures)) != len(signatures):
        raise ValueError("translation-congruent fragments")
    rng.shuffle(regions)
    return regions


def _ballots(rng, regions, core, signals, winner_index, evidence):
    contact = _dilate(core) & ~core
    weights = [int(contact[p[:, 0], p[:, 1]].sum()) for p in regions]
    assignments = []
    n = len(regions)
    for bits in range(1, (1 << n) - 1):
        win = sum(weights[i] for i in range(n) if bits >> i & 1)
        if not sum(weights) / 2 < win < sum(weights):
            continue
        if evidence == 0 and bits.bit_count() >= n / 2:
            continue
        if evidence == 1 and bits.bit_count() <= n / 2:
            continue
        assignments.append(bits)
    if not assignments:
        raise ValueError("no suitable contact-vote assignment")
    bits = int(rng.choice(assignments))
    winner, loser = signals[winner_index], signals[1 - winner_index]
    values, labels = [], []
    for i, cells in enumerate(regions):
        label = winner if bits >> i & 1 else loser
        other = loser if label == winner else winner
        area = len(cells)
        maximum = (area - 1) // 2
        minority = int(rng.integers(1, maximum + 1))
        ports = contact[cells[:, 0], cells[:, 1]]
        if evidence == 1:
            # Put losing-color minorities on the winning fragments' ports, and
            # winning-color minorities in losing fragments' interiors.
            rank = (~ports if label == winner else ports).astype(int) + rng.random(area) * .4
        else:
            mode = int(rng.integers(4))
            if mode == 0:
                rank = rng.random(area)
            elif mode == 1:
                pivot = cells[int(rng.integers(area))]
                rank = abs(cells - pivot).sum(axis=1) + rng.random(area) * .4
            elif mode == 2:
                rank = (cells.sum(axis=1) % 2) + rng.random(area) * .4
            else:
                axis = int(rng.integers(2))
                rank = cells[:, axis] * int(rng.choice([-1, 1])) + rng.random(area) * .4
        pattern = np.full(area, label, dtype=np.int8)
        pattern[np.argsort(rank)[:minority]] = other
        values.append(pattern)
        labels.append(label)
    if evidence == 0 and sum(int(np.sum(v == winner)) for v in values) * 2 >= sum(map(len, values)):
        raise ValueError("raw-cell shortcut not defeated")
    if evidence == 1:
        raw = np.concatenate([v[contact[p[:, 0], p[:, 1]]] for p, v in zip(regions, values)])
        if np.sum(raw == winner) * 2 >= len(raw):
            raise ValueError("raw-contact shortcut not defeated")
        # Do not teach that the color with weaker local majorities always wins.
        # This witness also shows winners at least as confident, on average,
        # as losing fragments. Vote strength never enters the semantic rule.
        fractions = [min(int(np.sum(v == signals[0])), int(np.sum(v == signals[1]))) / len(v)
                     for v in values]
        win_mean = np.mean([f for f, label in zip(fractions, labels) if label == winner])
        lose_mean = np.mean([f for f, label in zip(fractions, labels) if label != winner])
        if win_mean >= lose_mean:
            raise ValueError("local-strength shortcut not defeated")
    return values, labels, weights


def _scatter(rng, core, regions, values, roles):
    bg, scaffold, _, _ = roles
    bh, bw = core.shape
    # At most 900 cells, eight fragments and twelve in-memory scatter attempts.
    h, w = map(int, rng.integers(max(20, bh + 10), 31, size=2))
    r0, c0 = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    base = np.full((h, w), bg, dtype=np.int8)
    base[r0:r0 + bh, c0:c0 + bw][core] = scaffold
    reserved = np.zeros_like(base, dtype=bool)
    reserved[r0:r0 + bh, c0:c0 + bw] = True
    for _ in range(12):
        inp = base.copy()
        forbidden = _dilate(reserved, diagonal=True)
        order = sorted(range(len(regions)), key=lambda i: -len(regions[i]) + rng.random() * 5)
        for i in order:
            shape = _shape(regions[i])
            ph, pw = shape.max(axis=0) + 1
            valid = np.ones((h - ph + 1, w - pw + 1), dtype=bool)
            for r, c in shape:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            dest = shape + choices[int(rng.integers(len(choices)))]
            inp[dest[:, 0], dest[:, 1]] = values[i]
            occupied = np.zeros_like(forbidden)
            occupied[dest[:, 0], dest[:, 1]] = True
            forbidden |= _dilate(occupied, diagonal=True)
        else:
            return inp, base, np.array((r0, c0))
    raise ValueError("scatter clearance exhausted")


def _make_pair(rng, roles, count, winner_index, evidence, spurs, rotation, wide=False):
    h = int(rng.integers(7, 10 if wide else 11))
    w = int(rng.integers(h + 2 if wide else 8, 13))
    core, hub = _core(rng, h, w, spurs)
    regions = _partition(rng, core, count)
    signals = tuple(sorted(roles[2:]))
    values, labels, weights = _ballots(rng, regions, core, signals, winner_index, evidence)
    # Reject ambiguous geometry before spending effort on scattering or labels.
    fragments = [dict(shape=_shape(p)) for p in regions]
    assemblies = _placements(core, fragments, node_limit=15000)
    if len(assemblies) != 1:
        raise ValueError("ambiguous geometric assembly")
    for i, cells in assemblies[0]:
        if set(map(tuple, cells)) != set(map(tuple, regions[i])):
            raise ValueError("construction disagrees with assembly")
    inp, out, origin = _scatter(rng, core, regions, values, roles)
    for cells, pattern in zip(regions, values):
        dest = cells + origin
        out[dest[:, 0], dest[:, 1]] = pattern
    out[tuple(origin + hub)] = signals[winner_index]
    # Whole-scene transformations preserve translation between input and output.
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    mode = int(rng.integers(3))
    shared = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.random() < .3)
    first_winner = int(rng.integers(2))
    evidence_order = [0, 1] + [-1] * (ntrain - 2)
    rng.shuffle(evidence_order)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            evidence = evidence_order[i] if split == 'train' else -1
            winner_index = (first_winner if evidence == 0 else 1 - first_winner
                            if evidence == 1 else int(rng.integers(2)))
            roles = shared if stable_palette or mode == 2 else tuple(map(int, rng.choice(10, 4, replace=False)))
            if mode == 2 and split == 'test':
                fresh = tuple(map(int, rng.choice([c for c in range(10) if c not in shared[2:]], 2, replace=False)))
                fixed = tuple(map(int, rng.choice([c for c in range(10) if c not in fresh], 2, replace=False)))
                roles = fixed + fresh
            if mode == 0:
                fragments = int(rng.integers(5, 7) if split == 'train' else rng.integers(7, 9))
            elif evidence == 0:
                fragments = int(rng.integers(5, 7))
            else:
                fragments = int(rng.integers(4, 8 if split == 'train' else 9))
            spurs = int(rng.integers(5, 9) if mode == 2 and split == 'test'
                        else rng.integers(0, 3))
            rotation = (int(rng.choice([0, 2]) if split == 'train' else rng.choice([1, 3]))
                        if mode == 1 else int(rng.integers(4)))
            for _ in range(360):
                try:
                    pair = _make_pair(rng, roles, fragments, winner_index, evidence, spurs, rotation, mode == 1)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(info):
    win = info['winner']
    def sign(x):
        return int(x > 0) - int(x < 0)
    fractions = [min(int(np.sum(p['values'] == s)) for s in info['signals']) / len(p['values'])
                 for p in info['pieces']]
    strength = (float(np.mean([f for f, s in zip(fractions, info['labels']) if s != win]))
                - float(np.mean([f for f, s in zip(fractions, info['labels']) if s == win])))
    return dict(shape=info['grid'].shape, box=info['core'].shape,
                fragments=len(info['pieces']), core_area=int(info['core'].sum()),
                roles=(info['background'], info['scaffold'], *info['signals']),
                weights=tuple(sorted(info['weights'])), ports=sum(info['weights']),
                areas=tuple(sorted(len(p['shape']) for p in info['pieces'])),
                zero_ports=info['weights'].count(0),
                piece_vote=sign(2 * info['labels'].count(win) - len(info['pieces'])),
                raw_vote=sign(sum(int(np.sum(p['values'] == win)) * 2 - len(p['values'])
                                  for p in info['pieces'])),
                raw_contact_vote=sign(2 * info['raw_contacts'].count(win) - len(info['raw_contacts'])),
                edge_vote=sign(sum(w if s == win else -w
                                   for w, s in zip(info['edge_weights'], info['labels']))),
                margin=abs(info['totals'][0] - info['totals'][1]),
                winner_strength_advantage=strength,
                winner_index=info['signals'].index(win))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(_render(info), out), "incorrect assembled ballot"
            bg, scaffold = info['background'], info['scaffold']
            center = tuple(info['origin'] + info['hub'])
            # The only fixed cell that changes is the hub center.
            fixed = inp == scaffold
            fixed[center] = False
            assert np.all(out[fixed] == scaffold)
            assert out[center] == info['winner']
            assert np.sum(out == scaffold) == np.sum(inp == scaffold) - 1
            # Every source is erased; every translated fragment retains its
            # exact two-color pattern, including minorities and orientation.
            for i, piece in enumerate(info['pieces']):
                assert np.all(out[piece['source'][:, 0], piece['source'][:, 1]] == bg)
                dest = info['placement'][i] + info['origin']
                assert np.array_equal(out[dest[:, 0], dest[:, 1]], piece['values'])
                assert np.array_equal(_shape(dest), piece['shape'])
            origin = info['origin']
            h, w = info['core'].shape
            assert np.all(out[origin[0]:origin[0] + h, origin[1]:origin[1] + w] != bg)
            for signal in info['signals']:
                assert np.sum(out == signal) == np.sum(inp == signal) + (signal == info['winner'])
            if split == 'train':
                training.append(_metrics(info))
    assert any(m['piece_vote'] == -1 and m['raw_vote'] == -1 for m in training)
    assert any(m['piece_vote'] == 1 and m['raw_contact_vote'] == -1
               and m['winner_strength_advantage'] > 0 for m in training)
    assert {m['winner_index'] for m in training} == {0, 1}


def self_test():
    """Stream 42 seeds: no corpus, files, images, or execution traces retained."""
    shapes, boxes, structures, weights, areas, palettes, counts = set(), set(), set(), set(), set(), set(), set()
    totals = dict(pairs=0, minority_piece_wins=0, raw_contact_loses=0,
                  edge_disagrees=0, zero_port_fragments=0)
    fingerprints = set()
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                m = _metrics(info)
                a = pair['input']
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                shapes.add(a.shape)
                boxes.add(m['box'])
                weights.add(m['weights'])
                areas.update(m['areas'])
                palettes.add(m['roles'])
                counts.add(m['fragments'])
                # Hash structural data so diagnostics remain small in memory.
                remap = {}
                normalized = bytes(remap.setdefault(int(x), len(remap)) for x in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                totals['pairs'] += 1
                totals['minority_piece_wins'] += m['piece_vote'] == -1
                totals['raw_contact_loses'] += m['raw_contact_vote'] == -1
                totals['edge_disagrees'] += m['edge_vote'] == -1
                totals['zero_port_fragments'] += m['zero_ports']
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(structures) >= 205
    assert len(shapes) >= 35 and len(boxes) >= 20
    assert counts == {4, 5, 6, 7, 8} and len(weights) >= 35 and len(areas) >= 20
    assert len(palettes) >= 40
    assert totals['minority_piece_wins'] >= 42 and totals['raw_contact_loses'] >= 42
    first = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], again[s])
               for k in ('input', 'output'))
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(100 + 10 * tr + te, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Geometry and color roles commute with the transformation. These checks
    # also exercise family members beyond the sampled construction orientations.
    sample = first['train'][0]
    for turns in range(4):
        for reflect in (False, True):
            inp = np.rot90(sample['input'], turns)
            out = np.rot90(sample['output'], turns)
            if reflect:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    permutation = np.array([7, 9, 0, 5, 2, 8, 1, 4, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[sample['input']]), permutation[sample['output']])
    # Rearrange a fragment's internal pattern while preserving its two counts.
    # The geometric assembly and collector winner stay fixed; transported pixels
    # change in exactly the same order, including minority pixels on contact cells.
    info = _analyze(sample['input'])
    changed, expected = sample['input'].copy(), sample['output'].copy()
    for i, piece in enumerate(info['pieces']):
        pattern = np.roll(piece['values'], 1)
        source, target = piece['source'], info['placement'][i] + info['origin']
        changed[source[:, 0], source[:, 1]] = pattern
        expected[target[:, 0], target[:, 1]] = pattern
    assert np.array_equal(solve(changed), expected)
    # Wrong center and unchanged-input labels must both be rejected.
    for mutation in ('center', 'identity'):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = broken['test'][0]
        if mutation == 'center':
            info = _analyze(pair['input'])
            pair['output'][tuple(info['origin'] + info['hub'])] = info['scaffold']
        else:
            pair['output'] = pair['input'].copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            continue
        raise AssertionError("bad output accepted")
    return dict(seeds=42, **totals, grid_shapes=len(shapes), box_shapes=len(boxes),
                normalized_structures=len(structures), weight_profiles=len(weights),
                fragment_areas=len(areas), palettes=len(palettes), fragment_counts=sorted(counts))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
