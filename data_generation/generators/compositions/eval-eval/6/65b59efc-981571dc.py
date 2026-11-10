"""Fragments across the folds: identity-linked reconstruction of mirrored stamps.

Only standard-library modules and NumPy are used. The parser and solver consume
the current input alone; the generator does not supply them latent parameters.
"""
from functools import lru_cache
import numpy as np


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orbit(r, c, h, w, fy, fx):
    rr, cc = 2 * fy - 1 - r, 2 * fx - 1 - c
    rows = (r, rr) if 0 <= rr < h else (r,)
    cols = (c, cc) if 0 <= cc < w else (c,)
    return tuple(sorted(set((y, x) for y in rows for x in cols)))


@lru_cache(maxsize=256)
def _orbits(h, w, fy, fx):
    seen, result = set(), []
    for r in range(h):
        for c in range(w):
            if (r, c) not in seen:
                group = _orbit(r, c, h, w, fy, fx)
                seen.update(group)
                result.append(group)
    return tuple(result)


def _flips(r, c, h, w, fy, fx):
    """Higher member of an in-bounds pair is reflected; fringes are canonical."""
    rr, cc = 2 * fy - 1 - r, 2 * fx - 1 - c
    return 0 <= rr < h and r > rr, 0 <= cc < w and c > cc


def _view(a, vertical, horizontal):
    if vertical:
        a = a[::-1, :]
    if horizontal:
        a = a[:, ::-1]
    return a.copy()


def _parse(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for n in range(2, 6):
        try:
            H, W = grid.shape
            start = n + 4
            assert H > start and (H - start - 1) % (n + 2) == 0
            h = (H - start - 1) // (n + 2)
            assert 2 <= h <= 5
            bg, frame = int(grid[n + 3, 0]), int(grid[0, 0])
            assert bg != frame and np.all(grid[n + 3] == bg)
            top = grid[0]
            libw = 0
            while libw < W and top[libw] == frame:
                libw += 1
            assert np.all(top[libw:] == bg)
            assert (libw - 1) % (n + 1) == 0
            k = (libw - 1) // (n + 1) - 1
            assert 2 <= k <= 4
            assert np.all(grid[:n + 3, libw:] == bg)
            assert np.all(grid[n + 2, :libw] == frame)
            for c in range(0, libw, n + 1):
                assert np.all(grid[:n + 3, c] == frame)
            assert np.all(grid[1, 1:n + 1] == bg)
            erase = int(grid[2, 1])
            assert erase not in (bg, frame)
            assert np.all(grid[2:n + 2, 1:n + 1] == erase)
            labels, targets, cards = [], {}, {}
            for j in range(k):
                c = 1 + (j + 1) * (n + 1)
                header = grid[1, c:c + n]
                identity, target = int(header[0]), int(header[-1])
                assert identity not in (bg, frame, erase) and identity not in labels
                assert target not in (bg, erase) and np.all(header[1:-1] == bg)
                card = grid[2:n + 2, c:c + n].copy()
                assert set(map(int, np.unique(card))) <= {bg, erase, identity}
                labels.append(identity)
                targets[identity], cards[identity] = target, card
            board = grid[start:]
            # Its bottom border is nonbackground all the way to the right edge.
            nonblank = np.flatnonzero(board[-1] != bg)
            assert len(nonblank)
            bw = int(nonblank[-1]) + 1
            assert (bw - 1) % (n + 1) == 0
            w = (bw - 1) // (n + 1)
            assert 2 <= w <= 7 and max(bw, libw) == W
            assert np.all(board[:, bw:] == bg)
            board = board[:, :bw]
            cx = np.flatnonzero(board[0] != frame)
            ry = np.flatnonzero(board[:, 0] != frame)
            assert len(cx) == len(ry) == 1
            assert cx[0] % (n + 1) == ry[0] % (n + 2) == 0
            fx, fy = int(cx[0] // (n + 1)), int(ry[0] // (n + 2))
            assert 1 <= fx < w and 1 <= fy < h
            axis = int(board[0, cx[0]])
            assert axis not in (bg, frame, erase, *labels)
            stencil = np.full(board.shape, bg, dtype=np.int8)
            metadata = np.zeros(board.shape, dtype=bool)
            metadata[::n + 2, :] = True
            metadata[:, ::n + 1] = True
            stencil[metadata] = frame
            stencil[fy * (n + 2), :] = axis
            stencil[:, fx * (n + 1)] = axis
            assert np.array_equal(board[metadata], stencil[metadata])
            assignment = np.full((h, w), bg, dtype=np.int8)
            tiles = {}
            for r in range(h):
                for c in range(w):
                    y, x = 1 + r * (n + 2), 1 + c * (n + 1)
                    header = board[y, x:x + n]
                    identity = int(header[0])
                    assert identity in (bg, *labels) and np.all(header[1:] == bg)
                    tile = board[y + 1:y + 1 + n, x:x + n].copy()
                    if identity == bg:
                        assert np.all(tile == bg), "blank command contains a fragment"
                    else:
                        assert set(map(int, np.unique(tile))) <= {bg, erase, identity}
                        tiles[r, c] = tile
                    assignment[r, c] = identity
            assert set(map(int, np.unique(assignment))) - {bg} == set(labels)
            for group in _orbits(h, w, fy, fx):
                assert len({int(assignment[p]) for p in group}) == 1, "identity fold mismatch"
            parsed = dict(n=n, h=h, w=w, k=k, bg=bg, frame=frame, erase=erase,
                          axis=axis, fy=fy, fx=fx, labels=tuple(labels), targets=targets,
                          cards=cards, tiles=tiles, assignment=assignment)
            _recover(parsed)  # Require unique, consistent values, including backgrounds.
            candidates.append(parsed)
        except (AssertionError, IndexError):
            continue
    assert len(candidates) == 1, f"expected one interpretation, got {len(candidates)}"
    return candidates[0]


def _recover(p, normalize=True):
    n, erase = p['n'], p['erase']
    values = {label: np.full((n, n), -1, dtype=np.int8) for label in p['labels']}

    def observe(label, fragment):
        known = fragment != erase
        bits = (fragment == label).astype(np.int8)
        previous = values[label]
        assert not np.any(known & (previous != -1) & (previous != bits)), "inconsistent stamp evidence"
        previous[known] = bits[known]

    for label in p['labels']:
        observe(label, p['cards'][label])
    for (r, c), tile in p['tiles'].items():
        label = int(p['assignment'][r, c])
        v, u = _flips(r, c, p['h'], p['w'], p['fy'], p['fx'])
        observe(label, _view(tile, v, u) if normalize else tile)
    for mask in values.values():
        assert np.all(mask != -1), "stamp coordinate has no witness"
        assert mask.any(), "empty stamp"
    return {label: mask.astype(bool) for label, mask in values.items()}


def _render(p, masks=None, flip_vertical=True, flip_horizontal=True, keyed=True):
    masks = _recover(p) if masks is None else masks
    n, bg = p['n'], p['bg']
    result = np.full((p['h'] * n, p['w'] * n), bg, dtype=np.int8)
    for r in range(p['h']):
        for c in range(p['w']):
            label = int(p['assignment'][r, c])
            if label == bg:
                continue
            v, u = _flips(r, c, p['h'], p['w'], p['fy'], p['fx'])
            mask = _view(masks[label], v and flip_vertical, u and flip_horizontal)
            color = p['targets'][label] if keyed else label
            result[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.where(mask, color, bg)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover canonical motifs from all fragments, then render their folded views."""
    return _render(_parse(input_grid))


@lru_cache(maxsize=16)
def _geometries(n, k):
    result = []
    for h in range(2, min(5, (30 - n - 5) // (n + 2)) + 1):
        for w in range(2, min(7, 29 // (n + 1)) + 1):
            for fy in range(1, h):
                for fx in range(1, w):
                    groups = _orbits(h, w, fy, fx)
                    if len(groups) >= k + 1 and any(len(g) == 4 for g in groups):
                        result.append((h, w, fy, fx))
    return tuple(result)


def _choose_geometry(rng, choices):
    centered = [g for g in choices if 2 * g[2] == g[0] and 2 * g[3] == g[1]]
    pool = centered if centered and rng.random() < .3 else choices
    return pool[int(rng.integers(len(pool)))]


def _motif(rng, n):
    rr, cc = np.indices((n, n))
    mode = int(rng.integers(9))
    if mode == 0:
        mask = rng.random((n, n)) < rng.uniform(.15, .85)
    elif mode == 1:
        mask = np.zeros((n, n), dtype=bool)
        r, c = map(int, rng.integers(n, size=2))
        mask[r, :] = mask[:, c] = True
    elif mode == 2:
        mask = (rr == 0) | (cc == 0) | (rr == n - 1) | (cc == n - 1)
    elif mode == 3:
        mask = (rr + cc) % 2 == int(rng.integers(2))
    elif mode == 4:
        mask = rr >= cc if rng.random() < .5 else rr == cc
    elif mode == 5:
        mask = np.zeros((n, n), dtype=bool)
        r0, r1 = sorted(map(int, rng.integers(n, size=2)))
        c0, c1 = sorted(map(int, rng.integers(n, size=2)))
        mask[r0:r1 + 1, c0:c1 + 1] = True
    elif mode == 6:
        mask = (rr % 2 == int(rng.integers(2)))
    elif mode == 7:
        mask = np.ones((n, n), dtype=bool)
    else:
        mask = np.zeros((n, n), dtype=bool)
        r, c = map(int, rng.integers(n, size=2))
        for _ in range(int(rng.integers(n, n * n + 1))):
            mask[r, c] = True
            dr, dc = ((0, 1), (1, 0), (-1, 0), (0, -1))[int(rng.integers(4))]
            r, c = int(np.clip(r + dr, 0, n - 1)), int(np.clip(c + dc, 0, n - 1))
    if not mask.any():
        mask.flat[int(rng.integers(n * n))] = True
    return _view(mask, bool(rng.integers(2)), bool(rng.integers(2)))


def _primary_motif(rng, n):
    for _ in range(64):
        mask = _motif(rng, n)
        special = np.argwhere(mask & ~mask[::-1, ::-1])
        if (mask.sum() >= 2 and len(special) and
                not np.array_equal(mask, mask[::-1]) and
                not np.array_equal(mask, mask[:, ::-1])):
            return mask, tuple(map(int, special[int(rng.integers(len(special)))]))
    # Constructive termination guard, still an asymmetric nonempty stamp.
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :2] = True
    mask[1, 0] = True
    return mask, (0, 0)


def _assignment(rng, h, w, fy, fx, labels, bg):
    groups = list(_orbits(h, w, fy, fx))
    four = [i for i, g in enumerate(groups) if len(g) == 4]
    primary_group = int(rng.choice(four))
    remaining = [i for i in range(len(groups)) if i != primary_group]
    other_primary = int(rng.choice(remaining))
    remaining.remove(other_primary)
    values = [bg] * len(groups)
    values[primary_group] = values[other_primary] = labels[0]
    # Different orbit orders give scattered, banded, diagonal and border layouts.
    mode = int(rng.integers(4))
    if mode == 0:
        rng.shuffle(remaining)
    else:
        remaining.sort(key=lambda i: (sum(groups[i][0]) if mode == 1 else
                                     groups[i][0][0] if mode == 2 else
                                     min(groups[i][0]), groups[i][0]))
        if rng.random() < .5:
            remaining.reverse()
    for label, i in zip(labels[1:], remaining):
        values[i] = label
    extra = remaining[len(labels) - 1:]
    density = float(rng.choice([.15, .45, .8, 1.]))
    for i in extra:
        if rng.random() < density:
            values[i] = int(rng.choice(labels))
    assignment = np.full((h, w), bg, dtype=np.int8)
    for group, label in zip(groups, values):
        for point in group:
            assignment[point] = label
    return assignment, groups[primary_group]


def _encode(p):
    n, h, w, bg = p['n'], p['h'], p['w'], p['bg']
    libw = (p['k'] + 1) * (n + 1) + 1
    bw, start = w * (n + 1) + 1, n + 4
    grid = np.full((start + h * (n + 2) + 1, max(libw, bw)), bg, dtype=np.int8)
    grid[[0, n + 2], :libw] = p['frame']
    grid[:n + 3, :libw:n + 1] = p['frame']
    grid[2:n + 2, 1:n + 1] = p['erase']
    for j, label in enumerate(p['labels']):
        c = 1 + (j + 1) * (n + 1)
        grid[1, c] = label
        grid[1, c + n - 1] = p['targets'][label]
        grid[2:n + 2, c:c + n] = p['cards'][label]
    board = grid[start:, :bw]
    board[::n + 2, :] = p['frame']
    board[:, ::n + 1] = p['frame']
    board[p['fy'] * (n + 2), :] = p['axis']
    board[:, p['fx'] * (n + 1)] = p['axis']
    for r in range(h):
        for c in range(w):
            y, x = 1 + r * (n + 2), 1 + c * (n + 1)
            label = int(p['assignment'][r, c])
            board[y, x] = label
            if label != bg:
                board[y + 1:y + n + 1, x:x + n] = p['tiles'][r, c]
    return grid


def _make_pair(rng, n, k, geometry, mode, roles=None):
    h, w, fy, fx = geometry
    bg, frame, erase, axis = map(int, rng.choice(10, 4, replace=False)) if roles is None else roles
    labels = list(map(int, rng.choice([c for c in range(10) if c not in (bg, frame, erase, axis)], k, replace=False)))
    masks = {label: _motif(rng, n) for label in labels}
    primary, q2 = _primary_motif(rng, n)
    masks[labels[0]] = primary
    if k >= 3 and rng.random() < .2:
        masks[labels[-1]] = masks[labels[1]].copy()
    pool = [c for c in range(10) if c not in (bg, erase)]
    targets = {label: int(rng.choice(pool)) for label in labels}
    targets[labels[0]] = int(rng.choice([c for c in pool if c != labels[0]]))
    assignment, lost_group = _assignment(rng, h, w, fy, fx, labels, bg)
    cards = {label: np.where(mask, label, bg).astype(np.int8) for label, mask in masks.items()}
    tiles = {}
    for r in range(h):
        for c in range(w):
            label = int(assignment[r, c])
            if label != bg:
                v, u = _flips(r, c, h, w, fy, fx)
                tiles[r, c] = np.where(_view(masks[label], v, u), label, bg).astype(np.int8)
    p = dict(n=n, h=h, w=w, k=k, bg=bg, frame=frame, erase=erase, axis=axis,
             fy=fy, fx=fx, assignment=assignment, cards=cards, tiles=tiles,
             targets=targets, labels=tuple(labels))
    # Ground truth is drawn before any evidence is erased.
    output = _render(p, masks)
    card_density = .12 if mode == 'basic' else .45
    tile_density = .28 if mode == 'basic' else (.47 if mode == 'complementary' else .62)
    cm = {label: rng.random((n, n)) < card_density for label in labels}
    tm = {point: rng.random((n, n)) < tile_density for point in tiles}
    forced_cards = {label: np.zeros((n, n), dtype=bool) for label in labels}
    forced_tiles = {point: np.zeros((n, n), dtype=bool) for point in tiles}
    primary_label = labels[0]
    q1s = [tuple(map(int, q)) for q in np.argwhere(primary) if tuple(q) != q2]
    q1 = q1s[int(rng.integers(len(q1s)))]
    survivor_tile = max(lost_group)  # This member uses both flips.

    def actual(point, q):
        v, u = _flips(*point, h, w, fy, fx)
        return (n - 1 - q[0] if v else q[0], n - 1 - q[1] if u else q[1])

    if mode == 'test':
        # No stamp card and no direct reflection witness for this entire orbit.
        forced_cards[primary_label][:] = True
        for point in lost_group:
            forced_tiles[point][:] = True
    else:
        # A foreground pixel vanishes from every member of a reflection orbit.
        for point in lost_group:
            forced_tiles[point][actual(point, q1)] = True
        # Another pixel has a sole witness in a doubly reflected occurrence.
        forced_cards[primary_label][q2] = True
        for point in tiles:
            if assignment[point] == primary_label and point != survivor_tile:
                forced_tiles[point][actual(point, q2)] = True
        if mode == 'complementary':
            missing_tile = lost_group[int(rng.integers(len(lost_group) - 1))]
            forced_tiles[missing_tile][:] = True
    for label in labels:
        cm[label] |= forced_cards[label]
    for point in tiles:
        tm[point] |= forced_tiles[point]
    if mode != 'test':
        tm[survivor_tile][actual(survivor_tile, q2)] = False
    # Every canonical coordinate, foreground OR background, retains a witness.
    # Choose among occurrences not forbidden by the deliberate held-out gaps.
    for label in labels:
        positions = [point for point in tiles if assignment[point] == label]
        for r in range(n):
            for c in range(n):
                q = (r, c)
                sites = [(cm[label], q, forced_cards[label][q])]
                sites += [(tm[point], actual(point, q), forced_tiles[point][actual(point, q)]) for point in positions]
                if all(mask[at] for mask, at, _ in sites):
                    possible = [(mask, at) for mask, at, forbidden in sites if not forbidden]
                    assert possible, "constructive witness allocation failed"
                    mask, at = possible[int(rng.integers(len(possible)))]
                    mask[at] = False
    for label in labels:
        cards[label][cm[label]] = erase
    for point in tiles:
        tiles[point][tm[point]] = erase
    # The key order is unrelated to spatial placement order or frequency.
    p['labels'] = tuple(map(int, rng.permutation(labels)))
    return {'input': _encode(p), 'output': output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer('num_train', num_train, 2, 10)
    ns = _integer('num_test', num_test, 1, 4)
    if nt <= ns:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(2, 6))
    max_k = max(k for k in range(2, 5) if _geometries(n, k))
    k = int(rng.integers(2, max_k + 1))
    regime = int(rng.integers(3))
    geometries = _geometries(n, k)
    base = _choose_geometry(rng, geometries)
    fixed_roles = tuple(map(int, rng.choice(10, 4, replace=False))) if rng.random() < .35 else None
    seen, puzzle = set(), {'train': [], 'test': []}
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            # The first two demonstrations have complementary evidence regimes.
            mode = ('basic' if i == 0 else 'complementary') if split == 'train' else 'test'
            pair_k = k
            if regime == 2 and n <= 4 and i >= 2:
                pair_k = int(rng.integers(2, max_k + 1))
            choices = _geometries(n, pair_k)
            geometry = base if regime == 0 else _choose_geometry(rng, choices)
            for _ in range(32):
                pair = _make_pair(rng, n, pair_k, geometry, mode, fixed_roles)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    # Shuffle demonstration order; there is no index-dependent solving convention.
    order = rng.permutation(nt)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _mirror_only(p):
    """Counterfactual: retain the key but ignore identity links across orbits."""
    n, bg, erase = p['n'], p['bg'], p['erase']
    raw = np.full((p['h'] * n, p['w'] * n), bg, dtype=np.int8)
    for (r, c), tile in p['tiles'].items():
        raw[r * n:(r + 1) * n, c * n:(c + 1) * n] = tile
    result = raw.copy()
    H, W = raw.shape
    a, b = 2 * p['fy'] * n - 1, 2 * p['fx'] * n - 1
    for r in range(H):
        for c in range(W):
            if raw[r, c] == erase:
                rows = [r] + ([a - r] if 0 <= a - r < H else [])
                cols = [c] + ([b - c] if 0 <= b - c < W else [])
                visible = {int(raw[y, x]) for y in rows for x in cols if raw[y, x] != erase}
                assert len(visible) <= 1
                result[r, c] = next(iter(visible)) if visible else bg
    source = result.copy()
    for label, target in p['targets'].items():
        result[source == label] = target
    return result


def _features(p):
    masks = _recover(p)
    counts = [int(np.count_nonzero(p['assignment'] == label)) for label in p['labels']]
    empty_orbits = linked_losses = 0
    for group in _orbits(p['h'], p['w'], p['fy'], p['fx']):
        if p['assignment'][group[0]] != p['bg']:
            missing = all(np.all(p['tiles'][point] == p['erase']) for point in group)
            empty_orbits += missing
            label = int(p['assignment'][group[0]])
            linked_losses += (missing and len(group) == 4 and
                              np.all(p['cards'][label] == p['erase']))
    double_only = 0
    for label, mask in masks.items():
        for i, j in zip(*np.nonzero(mask)):
            if p['cards'][label][i, j] != p['erase']:
                continue
            witnesses = []
            for point, tile in p['tiles'].items():
                if p['assignment'][point] != label:
                    continue
                v, u = _flips(*point, p['h'], p['w'], p['fy'], p['fx'])
                at = (p['n'] - 1 - i if v else i, p['n'] - 1 - j if u else j)
                if tile[at] != p['erase']:
                    witnesses.append((v, u))
            double_only += witnesses == [(True, True)]
    return dict(scale=p['n'], count=p['k'], geometry=(p['h'], p['w'], p['fy'], p['fx']),
                offsets=(2 * p['fy'] - p['h'], 2 * p['fx'] - p['w']),
                roles=(p['bg'], p['frame'], p['erase'], p['axis']),
                density=[float(m.mean()) for m in masks.values()],
                blank_tiles=int(np.count_nonzero(p['assignment'] == p['bg'])),
                occurrences=counts, erased_cards=sum(np.all(card == p['erase']) for card in p['cards'].values()),
                erased_tiles=sum(np.all(tile == p['erase']) for tile in p['tiles'].values()),
                erased_orbits=int(empty_orbits),
                linked_losses=int(linked_losses), double_only_pixels=int(double_only),
                fringes=sum(len(g) < 4 for g in _orbits(p['h'], p['w'], p['fy'], p['fx'])))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, parsed = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            p = _parse(inp)
            masks = _recover(p)
            expected = _render(p, masks)
            assert np.array_equal(out, expected), "incorrect output"
            assert out.shape == (p['h'] * p['n'], p['w'] * p['n'])
            assert set(map(int, np.unique(out))) <= {p['bg'], *p['targets'].values()}
            assert p['erase'] not in np.unique(out)
            # Every known body cell constrains a precise output pixel after keying.
            for (r, c), tile in p['tiles'].items():
                label = int(p['assignment'][r, c])
                known = tile != p['erase']
                block = out[r * p['n']:(r + 1) * p['n'], c * p['n']:(c + 1) * p['n']]
                assert np.array_equal(block[known], np.where(tile == label, p['targets'][label], p['bg'])[known])
            parsed[split].append((p, out))
    all_p = [p for split in parsed.values() for p, _ in split]
    assert len({p['n'] for p in all_p}) == 1, "one scale convention per puzzle"
    train = parsed['train']
    assert any(not np.array_equal(out, _mirror_only(p)) for p, out in train), "no identity-link witness"
    for vertical, horizontal in ((False, True), (True, False), (False, False)):
        assert any(not np.array_equal(out, _render(p, flip_vertical=vertical, flip_horizontal=horizontal))
                   for p, out in train), "missing reflection-orientation witness"
    assert any(not np.array_equal(out, _render(p, keyed=False)) for p, out in train), "key is unused"
    assert any(not np.array_equal(out, _render(p, {label: p['cards'][label] == label for label in p['labels']}))
               for p, out in train), "cards alone already suffice"
    assert any(_features(p)['erased_tiles'] for p, _ in train), "no complete-tile training gap"
    assert any(_features(p)['double_only_pixels'] for p, _ in train), "no sole doubly-reflected witness"
    for p, out in parsed['test']:
        f = _features(p)
        assert f['linked_losses'] >= 1, "missing same-identity whole-card/whole-orbit shift"
        assert not np.array_equal(out, _mirror_only(p)), "test admits mirror-only shortcut"


def self_test() -> dict:
    """Bounded 42-seed study, meaningful ablations and malformed-evidence checks."""
    shapes, geometries, roles, motifs, maps, signatures = set(), set(), set(), set(), set(), set()
    records = []
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_signature(np.asarray(pair['input'])) for split in ('train', 'test') for pair in puzzle[split])
        assert signature not in signatures
        signatures.add(signature)
        repeated = generate_puzzle(seed)
        assert all(np.array_equal(puzzle[s][i][key], repeated[s][i][key])
                   for s in puzzle for i in range(len(puzzle[s])) for key in ('input', 'output'))
        if seed == 0:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = np.asarray(pair['input'])
                snapshot = inp.copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, snapshot), "solve mutates the input"
                p = _parse(inp)
                f = _features(p)
                f['split'] = split
                records.append(f)
                shapes.add((inp.shape, np.asarray(pair['output']).shape))
                geometries.add(f['geometry'])
                roles.add(f['roles'])
                motifs.update((p['n'], m.tobytes()) for m in _recover(p).values())
                normalized = np.zeros(p['assignment'].shape, dtype=np.int8)
                for i, label in enumerate(p['labels']):
                    normalized[p['assignment'] == label] = i + 1
                maps.add(_signature(normalized))
    assert {r['scale'] for r in records} == {2, 3, 4, 5}
    assert {r['count'] for r in records} >= {2, 3, 4}
    assert len(shapes) >= 25 and len(geometries) >= 25
    assert len(roles) >= 20 and len(motifs) >= 70 and len(maps) >= 50
    assert any(r['blank_tiles'] for r in records)
    assert any(r['fringes'] for r in records) and any(not r['fringes'] for r in records)
    assert any(r['offsets'][0] < 0 for r in records) and any(r['offsets'][0] > 0 for r in records)
    assert any(r['offsets'][1] < 0 for r in records) and any(r['offsets'][1] > 0 for r in records)
    # Counts are checked exhaustively here; verification also exercises them.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corrupt an answer, contradict a known canonical bit, and remove all
    # evidence for a bit. Each must be rejected, not guessed from the seed.
    bad = {s: [{key: np.asarray(a).copy() for key, a in pair.items()} for pair in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong answer accepted')
    for mutation in ('contradiction', 'no_witness'):
        p = _parse(first['train'][0]['input'])
        label = p['labels'][0]
        q = (0, 0)
        for point in p['tiles']:
            if p['assignment'][point] == label:
                v, u = _flips(*point, p['h'], p['w'], p['fy'], p['fx'])
                at = (p['n'] - 1 if v else 0, p['n'] - 1 if u else 0)
                p['tiles'][point][at] = p['erase']
        if mutation == 'no_witness':
            p['cards'][label][q] = p['erase']
        else:
            p['cards'][label][q] = label
            point = next(point for point in p['tiles'] if p['assignment'][point] == label)
            v, u = _flips(*point, p['h'], p['w'], p['fy'], p['fx'])
            at = (p['n'] - 1 if v else 0, p['n'] - 1 if u else 0)
            p['tiles'][point][at] = p['bg']
        try:
            solve(_encode(p))
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{mutation} accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                shapes=len(shapes), geometries=len(geometries), role_assignments=len(roles),
                canonical_motifs=len(motifs), normalized_maps=len(maps),
                scales=sorted({r['scale'] for r in records}), counts=sorted({r['count'] for r in records}),
                tests_with_whole_erased_orbits=sum(r['split'] == 'test' and r['erased_orbits'] > 0 for r in records))


if __name__ == '__main__':
    print(self_test())
