"""Consensus camouflage: template differences, rather than ink, are walls.

Standard library + NumPy only. Nothing is written by this module, including its
self-test. Paths below are transient coordinate lists, not saved traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools

import numpy as np


_REQUIRED = frozenset({
    "cycling", "multiplicity", "bend_phase", "marker_phase", "restart",
    "template_needed", "removal_needed", "balanced_collision",
    "transparent_ink", "cross_separator", "edge_stop", "vary_columns",
})


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10
            and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4, with num_train > num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _walk(walls, start):
    """Right, up, right, up ...; stop immediately upon a canvas exit.

    A bend endpoint is counted only in its arriving leg. Collision cells never
    belong to the path, and an unavailable turn ends the route.
    """
    assert _inside(walls.shape, start) and not walls[start]
    p, direction = start, 0
    path, collisions, legs = [], [], []
    for _ in range(sum(walls.shape) + 1):
        dr, dc = ((0, 1), (-1, 0))[direction]
        leg = []
        while _inside(walls.shape, p) and not walls[p]:
            leg.append(p)
            p = (p[0] + dr, p[1] + dc)
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(walls.shape, p):
            end = "edge"
            break
        collisions.append(p)
        direction = 1 - direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        p = (leg[-1][0] + dr, leg[-1][1] + dc)
        if not _inside(walls.shape, p) or walls[p]:
            end = "blocked"
            break
    else:
        raise AssertionError("monotone route did not terminate")
    assert len(path) == len(set(path))
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end)


def _parse_canonical(g):
    """Read the three-cell header and the registered repetition lattice."""
    assert g.shape[0] >= 9 and g.shape[1] >= 13
    carrier = int(g[2, 0])
    assert np.all(g[2] == carrier)
    board = g[3:].copy()
    lattice = int(board[0, 0])
    assert lattice != carrier and np.all(board[0] == lattice)
    assert np.all(board[-1] == lattice)
    w = g.shape[1]
    marker = int(g[0, w - 3])
    assert marker not in (carrier, lattice)
    word = []
    for c in range(1, w - 4, 3):
        if int(g[0, c]) == carrier:
            break
        word.append(int(g[0, c]))
    assert 2 <= len(word) <= 6 and len(set(word)) >= 2
    assert w >= 3 * len(word) + 5
    assert not set(word) & {carrier, lattice, marker}
    header = np.full((3, w), carrier, dtype=np.int8)
    for j, color in enumerate(word):
        header[:2, 1 + 3*j:3 + 3*j] = color
    header[:2, w-3:w-1] = marker
    assert np.array_equal(g[:3], header)
    starts = tuple(sorted(tuple(map(int, p)) for p in np.argwhere(board == word[0])))
    assert 1 <= len(starts) <= 3
    assert all(c == 0 and 0 < r < board.shape[0]-1 for r, c in starts)
    assert all(b[0]-a[0] > 1 for a,b in zip(starts,starts[1:])), "launches must be isolated"
    for p in starts:
        board[p] = lattice
    assert not set(map(int, np.unique(board))) & (set(word) | {marker})
    rs = np.flatnonzero(np.all(board == lattice, axis=1))
    cs = np.flatnonzero(np.all(board == lattice, axis=0))
    assert len(rs) >= 2 and len(cs) >= 4
    assert rs[0] == cs[0] == 0
    assert rs[-1] == board.shape[0]-1 and cs[-1] == board.shape[1]-1
    assert np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])
    ph, pw = int(rs[1]-1), int(cs[1]-1)
    assert ph >= 3 and pw >= 3
    panels = [[board[r+1:s, c+1:d] for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    flat = [p for row in panels for p in row]
    frequencies = Counter(p.tobytes() for p in flat)
    best = max(frequencies.values())
    winners = [k for k, v in frequencies.items() if v == best]
    assert len(winners) == 1
    master = next(p for p in flat if p.tobytes() == winners[0]).copy()
    colors = set(map(int, np.unique(master)))
    assert carrier in colors and len(colors) == 2 and lattice not in colors
    motif = (colors - {carrier}).pop()
    assert set(map(int, np.unique(board))) == {carrier, motif, lattice}
    walls = np.zeros(board.shape, dtype=bool)
    columns, edits = [], []
    for b, row in enumerate(panels):
        odd = [j for j, p in enumerate(row) if not np.array_equal(p, master)]
        assert len(odd) == 1
        columns.append(odd[0])
        for p in row:
            assert set(map(int, np.unique(p))) <= {carrier, motif}
            assert np.all(p[[0, -1]] == carrier) and np.all(p[:, [0, -1]] == carrier)
        variant = row[odd[0]]
        diff = variant != master
        r, c = 1 + b*(ph+1), 1 + odd[0]*(pw+1)
        walls[r:r+ph, c:c+pw] = diff
        additions = int(np.count_nonzero(diff & (master == carrier)))
        removals = int(np.count_nonzero(diff & (master == motif)))
        edits.append((additions, removals))
    assert all(r not in rs for r, _ in starts)
    records = [_walk(walls, p) for p in starts]
    occupied = set()
    for record in records:
        footprint = set(record['path']) | set(record['collisions'])
        assert not footprint & occupied, "routes may not merge or share collision cells"
        occupied |= footprint
    return dict(board=board, carrier=carrier, motif=motif, lattice=lattice,
                marker=marker, word=tuple(word), starts=starts, records=records,
                walls=walls, master=master, ph=ph, pw=pw, bands=len(panels),
                copies=len(panels[0]), columns=tuple(columns), edits=tuple(edits))


def _decode(g):
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(g, -rotation))
        except (AssertionError, IndexError):
            continue
        info['rotation'] = rotation
        candidates.append(info)
    assert len(candidates) == 1, "input must have one header/entry interpretation"
    return candidates[0]


def _paint(info, mode="correct"):
    """The other modes are explicit wrong-rule witnesses, not family modes."""
    word, walls = info['word'], info['walls']
    if mode == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    if mode == 'raw_ink':
        walls = info['board'] == info['motif']
    elif mode == 'additions_only':
        walls = walls & (info['board'] == info['motif'])
    board = info['board'].copy()
    offset = 0
    for start in info['starts']:
        record = _walk(walls, start)
        cursor = 0
        for j, length in enumerate(record['legs']):
            for i in range(length):
                phase = cursor + i
                if mode == 'reset_bend':
                    phase = i
                elif mode == 'consume_marker':
                    phase += j
                elif mode == 'global_phase':
                    phase += offset
                board[record['path'][cursor+i]] = word[phase % len(word)]
            cursor += length
            if mode == 'first_leg':
                break
        marks = record['collisions'][:1] if mode == 'first_leg' else record['collisions']
        for p in marks:
            board[p] = info['marker']
        offset += len(record['path'])
    out = np.full((3 + board.shape[0], board.shape[1]), info['carrier'], dtype=np.int8)
    out[3:] = board
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use input-visible consensus and symbols only; never consult a seed."""
    info = _decode(_grid(input_grid))
    return np.rot90(_paint(info), info['rotation']).copy()


def _evidence(info):
    evidence = set()
    expected = _paint(info)
    for mode, name in (
        ('deduplicate', 'multiplicity'), ('reset_bend', 'bend_phase'),
        ('consume_marker', 'marker_phase'), ('global_phase', 'restart'),
        ('raw_ink', 'template_needed'), ('additions_only', 'removal_needed'),
    ):
        if not np.array_equal(expected, _paint(info, mode)):
            evidence.add(name)
    if len(set(info['columns'])) > 1:
        evidence.add('vary_columns')
    for record in info['records']:
        if len(record['path']) >= 2*len(info['word']):
            evidence.add('cycling')
        if any(info['board'][p] == info['motif'] for p in record['path']):
            evidence.add('transparent_ink')
        if any(info['board'][p] == info['lattice'] and 0 < p[0] < info['board'].shape[0]-1
               and 0 < p[1] < info['board'].shape[1]-1 for p in record['path']):
            evidence.add('cross_separator')
        for r, c in record['collisions']:
            b = (r-1)//(info['ph']+1)
            a, d = info['edits'][b]
            if a == d > 0:
                evidence.add('balanced_collision')
        if record['end'] == 'edge':
            r, c = record['path'][-1]
            corner = (r-1, c) if len(record['legs']) % 2 else (r, c+1)
            if _inside(info['walls'].shape, corner) and not info['walls'][corner]:
                evidence.add('edge_stop')
    return evidence


def _sample_word(rng, length, pool, repeat=False):
    distinct = int(rng.integers(2, min(len(pool), length - int(repeat)) + 1))
    symbols = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length-distinct)))
    rng.shuffle(symbols)
    return tuple(int(pool[i]) for i in symbols)


def _master(rng, ph, pw, carrier, motif):
    h, w = ph-2, pw-2
    rr, cc = np.indices((h, w))
    style = int(rng.integers(6))
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(.22, .78)
    elif style == 1:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    elif style == 2:
        mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
    elif style == 3:
        mask = (rr + int(rng.integers(3))*cc) % int(rng.integers(2, 4)) == 0
    elif style == 4:
        mask = rng.random((h, w)) < .4
        mask |= mask[:, ::-1]
    else:
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        target = int(rng.integers(2, max(3, h*w-1)))
        for _ in range(target-1):
            border = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                      and any(0 <= y < h and 0 <= x < w and mask[y, x]
                              for y, x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))]
            if border:
                mask[border[int(rng.integers(len(border)))]] = True
    if not 2 <= mask.sum() <= mask.size-2:
        mask[:] = False
        mask.flat[rng.permutation(mask.size)[:mask.size//2]] = True
    master = np.full((ph, pw), carrier, dtype=np.int8)
    master[1:-1, 1:-1][mask] = motif
    return master


def _changes(rng, master, motif):
    core = master[1:-1, 1:-1] == motif
    h, w = core.shape
    diff = np.zeros(core.shape, dtype=bool)
    mode = int(rng.integers(5))
    cap = max(1, core.size//2)
    if mode == 0:  # Count-preserving exchanges still produce real walls.
        ink, blank = np.flatnonzero(core), np.flatnonzero(~core)
        n = int(rng.integers(1, min(len(ink), len(blank), max(1, cap//2))+1))
        diff.flat[rng.choice(ink, n, replace=False)] = True
        diff.flat[rng.choice(blank, n, replace=False)] = True
    elif mode == 1:  # A short horizontal or vertical barrier.
        axis = int(rng.integers(2))
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if axis == 0:
            diff[r, c:min(w, c+int(rng.integers(1, w+1)))] = True
        else:
            diff[r:min(h, r+int(rng.integers(1, h+1))), c] = True
    elif mode == 2:  # A local rectangular change.
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dh, dw = int(rng.integers(1, h-r+1)), int(rng.integers(1, w-c+1))
        diff[r:r+dh, c:c+dw] = True
    else:
        n = int(rng.integers(1, cap+1))
        diff.flat[rng.choice(diff.size, n, replace=False)] = True
    if int(diff.sum()) > cap:
        selected = np.flatnonzero(diff)
        diff[:] = False
        diff.flat[rng.choice(selected, cap, replace=False)] = True
    result = np.zeros(master.shape, dtype=bool)
    result[1:-1, 1:-1] = diff
    return result


def _board(master, differences, columns, copies, lattice, carrier, motif):
    ph, pw = master.shape
    bands = len(differences)
    board = np.full((bands*(ph+1)+1, copies*(pw+1)+1), lattice, dtype=np.int8)
    walls = np.zeros(board.shape, dtype=bool)
    for b, diff in enumerate(differences):
        for j in range(copies):
            panel = master.copy()
            if j == columns[b]:
                panel[diff] = np.where(master[diff] == motif, carrier, motif)
            r, c = 1+b*(ph+1), 1+j*(pw+1)
            board[r:r+ph, c:c+pw] = panel
            if j == columns[b]:
                walls[r:r+ph, c:c+pw] = diff
    return board, walls


def _finish(board, walls, starts, word, marker, carrier, rotation):
    """Construct labels directly from the sampled board and transient routes."""
    inp = np.full((3+board.shape[0], board.shape[1]), carrier, dtype=np.int8)
    inp[3:] = board
    for j, color in enumerate(word):
        inp[:2, 1+3*j:3+3*j] = color
    inp[:2, -3:-1] = marker
    out = np.full_like(inp, carrier)
    out[3:] = board
    for start in starts:
        inp[start[0]+3, start[1]] = word[0]
        record = _walk(walls, start)
        for i, (r, c) in enumerate(record['path']):
            out[r+3, c] = word[i % len(word)]
        for r, c in record['collisions']:
            out[r+3, c] = marker
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _witness(rng, length=None):
    """Construct two disjoint, several-leg routes with balanced hidden walls.

    Coordinates vary with tile geometry and exception columns. This is training
    evidence, not an extra solving convention or a finite set of task templates.
    """
    for _ in range(64):
        ph, pw = int(rng.integers(6, 8)), int(rng.integers(6, 9))
        copies = int(rng.integers(3, 5)) if pw == 6 else 3
        bands = int(rng.integers(2, 4))
        w = copies*(pw+1)+1
        n = int(rng.integers(3, 5)) if length is None else length
        if w < 3*n+5:
            continue
        p0, p1 = sorted(map(int, rng.choice(copies, 2, replace=False)))
        y0 = 1 + (bands-2)*(ph+1) + ph-2
        if (1+p0*(pw+1)+y0+4) % n == 0:
            continue
        carrier, lattice, motif, marker, *pool = map(int, rng.permutation(10))
        master = _master(rng, ph, pw, carrier, motif)
        lower = ((ph-2,2), (ph-5,1), (ph-4,4), (ph-3,4))
        upper = ((ph-2,3), (ph-4,2), (ph-3,4), (ph-5,1))
        for p, ink in list(zip(lower, (True,False,True,False))) + list(zip(upper, (True,True,False,False))):
            master[p] = motif if ink else carrier
        differences = [_changes(rng, master, motif) for _ in range(bands)]
        columns = list(map(int, rng.integers(copies, size=bands)))
        for b, cells, column in ((bands-2, upper, p0), (bands-1, lower, p1)):
            differences[b][:] = False
            for p in cells:
                differences[b][p] = True
            columns[b] = column
        # Earlier bands may have arbitrary edits, except at these two exit rays.
        exit_columns = {1+p0*(pw+1)+3, 1+p1*(pw+1)+3}
        for b in range(bands-2):
            for c in range(1, pw-1):
                if 1+columns[b]*(pw+1)+c in exit_columns:
                    differences[b][:, c] = False
            if not differences[b].any():
                differences[b][1, 1] = True  # local column 1 is never an exit ray
        board, walls = _board(master, differences, columns, copies, lattice, carrier, motif)
        starts = ((y0, 0), (1+(bands-1)*(ph+1)+ph-2, 0))
        for _ in range(32):
            word = _sample_word(rng, n, pool, repeat=n > 2)
            pair = _finish(board, walls, starts, word, marker, carrier, int(rng.integers(4)))
            info = _decode(pair['input'])
            needed = _REQUIRED if n > 2 else _REQUIRED - {'multiplicity'}
            if needed <= _evidence(info):
                return pair
    # A guaranteed parameter choice, still sampled in palette and orientation.
    # It obeys the same rule if geometry draws were exceptionally unlucky.
    return _fixed_witness(rng, 3 if length is None else length)


def _fixed_witness(rng, length):
    ph, pw, bands, copies = 6, 7, 3, 3
    carrier, lattice, motif, marker, *pool = map(int, rng.permutation(10))
    master = _master(rng, ph, pw, carrier, motif)
    lo = ((4,2),(1,1),(2,4),(3,4))
    up = ((4,3),(2,2),(3,4),(1,1))
    for p, ink in list(zip(lo, (1,0,1,0))) + list(zip(up, (1,1,0,0))):
        master[p] = motif if ink else carrier
    ds = [np.zeros(master.shape, dtype=bool) for _ in range(3)]
    ds[0][1,1] = True
    for b, cells in ((1, up), (2, lo)):
        for p in cells:
            ds[b][p] = True
    board, walls = _board(master, ds, (2,0,1), copies, lattice, carrier, motif)
    # The first route has 17 cells: not divisible by any allowed word length.
    patterns = itertools.product(range(2), repeat=length)
    for pattern in patterns:
        if len(set(pattern)) < 2 or (length > 2 and len(set(pattern)) == length):
            continue
        word = tuple(pool[i] for i in pattern)
        pair = _finish(board, walls, ((12,0),(19,0)), word, marker, carrier, int(rng.integers(4)))
        needed = _REQUIRED if length > 2 else _REQUIRED - {'multiplicity'}
        if needed <= _evidence(_decode(pair['input'])):
            return pair
    raise AssertionError("fixed informative construction failed")


def _random_pair(rng, test=False, excluded_geometry=None):
    length = int(rng.integers(5, 7) if test else rng.integers(2, 5))
    choices = [(ph,pw,bands,copies) for ph in range(4,10) for pw in range(4,9)
               for bands in range(1,6) for copies in range(3,6)
               if 3+bands*(ph+1)+1 <= 30 and 3*length+5 <= copies*(pw+1)+1 <= 30
               and (ph,pw,bands,copies) != excluded_geometry]
    for _ in range(128):
        ph,pw,bands,copies = choices[int(rng.integers(len(choices)))]
        carrier, lattice, motif, marker, *pool = map(int, rng.permutation(10))
        word = _sample_word(rng, length, pool)
        master = _master(rng, ph, pw, carrier, motif)
        differences = [_changes(rng, master, motif) for _ in range(bands)]
        columns = list(map(int, rng.integers(copies, size=bands)))
        board, walls = _board(master, differences, columns, copies, lattice, carrier, motif)
        candidates = []
        for r in range(1, board.shape[0]-1):
            if r % (ph+1):
                record = _walk(walls, (r,0))
                candidates.append(((r,0), record))
        good = [x for x in candidates if x[1]['collisions'] and len(x[1]['path']) >= 2*length]
        if not good:
            continue
        first = good[int(rng.integers(len(good)))]
        selected = [first[0]]
        occupied = set(first[1]['path']) | set(first[1]['collisions'])
        target = int(rng.choice([1,2,3], p=[.50,.37,.13]))
        rng.shuffle(candidates)
        for start, record in candidates:
            if len(selected) == target:
                break
            footprint = set(record['path']) | set(record['collisions'])
            if not footprint & occupied and all(abs(start[0]-p[0]) > 1 for p in selected):
                selected.append(start)
                occupied |= footprint
        return _finish(board, walls, sorted(selected), word, marker, carrier, int(rng.integers(4)))
    return _witness(rng, length)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    witness = _witness(rng)
    wi = _decode(witness['input'])
    witness_geometry = tuple(wi[k] for k in ('ph','pw','bands','copies'))
    witness_slot = int(rng.integers(nt))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train",nt),("test",ne)):
        for j in range(count):
            if split == 'train' and j == witness_slot:
                pair = witness
            else:
                for _ in range(16):
                    pair = _random_pair(rng, split == 'test', witness_geometry if split == 'train' else None)
                    if _key(pair['input']) not in seen and _key(pair['input']) != _key(witness['input']):
                        break
                else:
                    raise AssertionError("distinct-input retry budget exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), set()
    train_lengths, test_lengths, train_geometries = set(), set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, np.rot90(_paint(info), info['rotation']))
            assert not np.array_equal(inp, out)
            # Input-defined masks and exact output preservation, not edit labels.
            assert info['walls'].any()
            assert any(r['collisions'] for r in info['records'])
            assert any(len(r['path']) >= 2*len(info['word']) for r in info['records'])
            canonical_out = np.rot90(out, -info['rotation'])
            footprint = {p for record in info['records'] for p in record['path'] + record['collisions']}
            untouched = np.ones(info['board'].shape, dtype=bool)
            for p in footprint:
                untouched[p] = False
            assert np.array_equal(canonical_out[3:][untouched], info['board'][untouched])
            for record in info['records']:
                assert all(info['walls'][p] for p in record['collisions'])
                assert all(not info['walls'][p] for p in record['path'])
            if split == 'train':
                evidence |= _evidence(info)
                train_lengths.add(len(info['word']))
                train_geometries.add(tuple(info[k] for k in ('ph','pw','bands','copies')))
            else:
                test_lengths.add(len(info['word']))
    assert _REQUIRED <= evidence, _REQUIRED - evidence
    assert max(train_lengths) < min(test_lengths), "generated shift is a longer visible program"
    assert len(train_geometries) >= 2


def _digest(puzzle):
    h = hashlib.sha256()
    for s in ('train','test'):
        for p in puzzle[s]:
            for role in ('input','output'):
                a = np.asarray(p[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _regressions():
    # Hand-computed right/up route; no generative code constructs its answer.
    walls = np.zeros((6,9), dtype=bool)
    walls[4,5] = walls[1,4] = walls[2,7] = True
    record = _walk(walls, (4,0))
    expected = tuple([(4,c) for c in range(5)] + [(3,4),(2,4)]
                     + [(2,5),(2,6)] + [(1,6),(0,6)])
    assert record['path'] == expected and record['legs'] == (5,2,2,2)
    assert record['collisions'] == ((4,5),(1,4),(2,7)) and record['end'] == 'edge'
    # Available terrain after a forward exit must remain unused.
    assert (0,7) not in record['path'] and not walls[0,7]
    horizontal = _walk(np.zeros((4,7), dtype=bool), (2,0))
    assert horizontal['path'] == tuple((2,c) for c in range(7))
    assert not horizontal['collisions'] and horizontal['end'] == 'edge'
    blocked = np.zeros((4,7), dtype=bool)
    blocked[2,3] = blocked[1,2] = True
    stopped = _walk(blocked, (2,0))
    assert stopped['path'] == ((2,0),(2,1),(2,2))
    assert stopped['collisions'] == ((2,3),) and stopped['end'] == 'blocked'


def self_test():
    """Bounded: 42 puzzles streamed in memory; no files or unbounded retries."""
    _regressions()
    signatures, shapes, geometries, masters, rotations = set(), set(), set(), set(), set()
    word_patterns, route_counts, leg_counts, edits, palettes = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add(pair['input'].shape)
                geometries.add(tuple(info[k] for k in ('ph','pw','bands','copies')))
                masters.add(_key(info['master'] == info['motif']))
                rotations.add(info['rotation'])
                labels = {}
                word_patterns.add(tuple(labels.setdefault(c, len(labels)) for c in info['word']))
                route_counts.add(len(info['starts']))
                leg_counts.update(len(r['legs']) for r in info['records'])
                edits.update(info['edits'])
                palettes.add((info['carrier'],info['lattice'],info['motif'],info['marker']))
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(geometries) >= 50 and len(masters) >= 80
    assert rotations == set(range(4)) and route_counts == {1,2,3}
    assert {1,2,3,4} <= leg_counts and len(word_patterns) >= 20
    assert any(a == r > 0 for a,r in edits) and any(a and not r for a,r in edits)
    assert any(r and not a for a,r in edits) and len(palettes) >= 100
    assert _digest(generate_puzzle(81723, 7, 3)) == _digest(generate_puzzle(81723, 7, 3))
    for a in range(2,11):
        for b in range(1, min(4,a-1)+1):
            p = generate_puzzle(1000+10*a+b, a, b)
            assert (len(p['train']),len(p['test'])) == (a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Directly exercise the constructive backstop, including long test words.
    for length in range(2,7):
        p = _fixed_witness(np.random.default_rng(90+length), length)
        needed = _REQUIRED if length > 2 else _REQUIRED - {'multiplicity'}
        assert needed <= _evidence(_decode(p['input']))
        assert np.array_equal(solve(p['input']),p['output'])
    # Mutation checks: exact preservation and the two inherited mechanisms.
    base = generate_puzzle(0,2,1)
    for mode in ('raw_ink','additions_only','reset_bend','consume_marker','deduplicate','global_phase'):
        bad = {s:[{k:v.copy() for k,v in p.items()} for p in base[s]] for s in base}
        target = next(p for p in bad['train'] if not np.array_equal(
            _paint(_decode(p['input'])),_paint(_decode(p['input']),mode)))
        info = _decode(target['input'])
        target['output'] = np.rot90(_paint(info,mode),info['rotation']).copy()
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("validator accepted wrong rule: " + mode)


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, semantic regressions and mutations')
