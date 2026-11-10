import numpy as np

# Every pair expresses its roles locally. The invariant is directional branching
# at a maximal transverse run, independent of literal palette and orientation.


def _ensure(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validated_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validated_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validated_count(num_train, "num_train")
    num_test = _validated_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    count_rng = np.random.default_rng(np.random.SeedSequence([seed, 918273]))
    if num_train is None and num_test is None:
        num_test = int(count_rng.integers(1, 5))
        num_train = int(count_rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(count_rng.integers(num_test + 1, 11))
    else:
        num_test = int(count_rng.integers(1, min(4, num_train - 1) + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("impossible train/test count combination")

    return num_train, num_test


def _to_canonical(grid: np.ndarray, direction: int) -> np.ndarray:
    """
    Convert an actual grid to a representation whose entry boundary is the
    top edge and whose forward direction is downward.

    direction:
        0: down
        1: up
        2: right
        3: left
    """
    if direction == 0:
        return grid
    if direction == 1:
        return np.flipud(grid)
    if direction == 2:
        return grid.T
    if direction == 3:
        return np.fliplr(grid).T
    raise ValueError("invalid direction")


def _from_canonical(grid: np.ndarray, direction: int) -> np.ndarray:
    if direction == 0:
        result = grid
    elif direction == 1:
        result = np.flipud(grid)
    elif direction == 2:
        result = grid.T
    elif direction == 3:
        result = np.fliplr(grid.T)
    else:
        raise ValueError("invalid direction")

    return np.ascontiguousarray(result, dtype=np.int8)


def _maximal_run(
    row: np.ndarray,
    column: int,
    barrier: int,
) -> tuple[int, int]:
    left = column
    right = column

    while left > 0 and int(row[left - 1]) == barrier:
        left -= 1
    while right + 1 < len(row) and int(row[right + 1]) == barrier:
        right += 1

    return left, right


def _barrier_runs(
    canonical: np.ndarray,
    barrier: int,
) -> set[tuple[int, int, int]]:
    runs: set[tuple[int, int, int]] = set()
    height, width = canonical.shape

    for row in range(height):
        column = 0
        while column < width:
            if int(canonical[row, column]) != barrier:
                column += 1
                continue

            left = column
            while (
                column + 1 < width
                and int(canonical[row, column + 1]) == barrier
            ):
                column += 1
            runs.add((row, left, column))
            column += 1

    return runs


def _candidate_parameters(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Recognize separated, boundary-attached stubs, including edge entries.

    Background is dominant; all guide pixels must belong to straight prefixes
    of length at least two. Obstacles may occupy the entry layers elsewhere.
    Reject competing interpretations instead of consulting generation metadata.
    """
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or sum(counts == max(counts)) != 1:
        return []
    background = int(colors[np.argmax(counts)])
    candidates = []
    for direction in range(4):
        a = _to_canonical(grid, direction)
        h, w = a.shape
        for guide in map(int, colors):
            if guide == background:
                continue
            entries = np.flatnonzero(a[0] == guide)
            if not len(entries) or np.any(np.diff(entries) < 2):
                continue
            mask = np.zeros(a.shape, dtype=bool)
            for c in entries:
                rows = np.flatnonzero(a[:, c] == guide)
                if not 2 <= len(rows) < h or not np.array_equal(rows, np.arange(len(rows))):
                    break
                mask[:len(rows), c] = True
            else:
                if np.array_equal(mask, a == guide):
                    barrier = next(int(x) for x in colors if x not in (background, guide))
                    candidates.append((direction, background, guide, barrier))
    return sorted(set(candidates))


def _advance(row, active, barrier):
    hits = {}
    flowing = set()
    for c in sorted(active):
        if row[c] == barrier:
            hits.setdefault(_maximal_run(row, c, barrier), []).append(c)
        else:
            flowing.add(c)
    return hits, flowing


def _geometry_features(canonical, barrier, runs):
    singletons = {(r, a) for r, a, b in runs if a == b}
    thin_links = sum((r+1, c) in singletons for r, c in singletons)
    remaining = set(zip(*np.nonzero(canonical == barrier)))
    bent = stepped = 0
    while remaining:
        start = remaining.pop()
        component, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    pending.append(p)
        thin = singletons & component
        rows, cols = {r for r, _ in component}, {c for _, c in component}
        bent += bool(thin) and len(rows) > 1 and len(cols) > 1
        stepped += len({c for _, c in thin}) > 1
    return {'singleton_forward_links': thin_links, 'bent_formations': bent,
            'stepped_formations': stepped}


def _simulate_canonical(canonical, background, guide, barrier):
    """Layer simulator. Each incident run is handled once, on immutable input."""
    out = canonical.copy()
    h, w = canonical.shape
    entries = set(map(int, np.flatnonzero(canonical[0] == guide)))
    active = entries.copy()
    reached = set()
    events = []
    incoming = []
    simultaneous = initial_simultaneous = descendant_simultaneous = 0
    multi_hit = two_flank = clipped = interrupted = mergers = shared = gaps = threads = 0
    stub_lengths = [int(np.count_nonzero(canonical[:, c] == guide)) for c in sorted(entries)]
    for r in range(1, h):
        incoming.append((r, sorted(active)))
        hits, following = _advance(canonical[r], active, barrier)
        if len(hits) >= 2:
            simultaneous += 1
            if reached:
                descendant_simultaneous += 1
            else:
                initial_simultaneous += 1
        gaps += bool(hits and following)
        threads += sum(0 < c < w-1 and canonical[r, c-1] == barrier and canonical[r, c+1] == barrier
                       for c in following)
        for c in following:
            if canonical[r, c] == background:
                out[r, c] = guide
        new_flanks = set()
        for (left, right), columns in hits.items():
            reached.add((r, left, right))
            events.append((r, left, right, columns))
            multi_hit += len(columns) > 1
            clipped += left == 0 or right == w - 1
            two_flank += left > 0 and right < w - 1
            span = slice(max(0, left - 1), min(w, right + 2))
            interrupted += bool(np.any(canonical[r-1, span] == barrier))
            line = out[r-1, span]
            line[canonical[r-1, span] == background] = guide
            for c in (left - 1, right + 1):
                if 0 <= c < w:
                    shared += c in new_flanks
                    mergers += c in following
                    new_flanks.add(c)
                    following.add(c)
                    if canonical[r, c] == background:
                        out[r, c] = guide
        active = following
    all_runs = _barrier_runs(canonical, barrier)
    inert = all_runs - reached
    changed = int(np.count_nonzero(out != canonical))
    stats = {
        'forward_length': h, 'transverse_width': w, 'entry_count': len(entries),
        'stub_lengths': stub_lengths, 'edge_entry': bool(entries & {0, w-1}),
        'collision_runs': len(reached), 'collision_rows': len({r for r, _, _ in reached}),
        'total_barrier_runs': len(all_runs), 'distractor_runs': len(inert),
        'reached_lengths': sorted(b-a+1 for _, a, b in reached),
        'unreached_lengths': sorted(b-a+1 for _, a, b in inert),
        'two_flank_events': two_flank, 'clipped_events': clipped,
        'multi_hit_events': multi_hit, 'simultaneous_layers': simultaneous,
        'initial_simultaneous': initial_simultaneous,
        'descendant_simultaneous': descendant_simultaneous,
        'interrupted_crosspieces': interrupted, 'mergers': mergers,
        'shared_flanks': shared, 'gap_passes': gaps, 'narrow_gap_threads': int(threads),
        'prefix_barriers': int(np.count_nonzero(canonical[:2] == barrier)),
        'final_collision': any(r == h-1 for r, _, _ in reached),
        'immediate_collision': any(canonical[n, c] == barrier for c, n in zip(sorted(entries), stub_lengths)),
        'vertical_links': int(np.count_nonzero((canonical[:-1] == barrier) & (canonical[1:] == barrier))),
        'barrier_cells': int(np.count_nonzero(canonical == barrier)),
        'new_guide_cells': changed, 'changed_cells': changed,
        'events': events, 'incoming': incoming,
    }
    stats.update(_geometry_features(canonical, barrier, all_runs))
    return out, stats


def _worklist_prediction(grid, parameters):
    """Separate ray work-list algorithm in actual coordinates, without layers.

    It neither calls the generating simulator nor shares its run enumerator.
    Visited ray positions make merging and repeated encounters order-independent.
    """
    direction, background, guide, barrier = parameters
    dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[direction]
    tr, tc = -dc, dr
    h, w = grid.shape
    out = grid.copy()
    def inside(r, c):
        return 0 <= r < h and 0 <= c < w
    queue = []
    for r, c in zip(*np.nonzero(grid == guide)):
        if not inside(r-dr, c-dc):
            queue.append((int(r), int(c)))
    visited = set()
    while queue:
        r, c = queue.pop()
        if not inside(r, c) or (r, c) in visited:
            continue
        visited.add((r, c))
        if grid[r, c] != barrier:
            if grid[r, c] == background:
                out[r, c] = guide
            queue.append((r+dr, c+dc))
            continue
        a = b = 0
        while inside(r+(a-1)*tr, c+(a-1)*tc) and grid[r+(a-1)*tr, c+(a-1)*tc] == barrier:
            a -= 1
        while inside(r+(b+1)*tr, c+(b+1)*tc) and grid[r+(b+1)*tr, c+(b+1)*tc] == barrier:
            b += 1
        for t in range(a-1, b+2):
            rr, cc = r-dr+t*tr, c-dc+t*tc
            if inside(rr, cc) and grid[rr, cc] == background:
                out[rr, cc] = guide
        queue.extend([(r+(a-1)*tr, c+(a-1)*tc), (r+(b+1)*tr, c+(b+1)*tc)])
    return out


def predict_from_training(train_pairs, test_input):
    """Prediction interface receives only training labels and the query input."""
    _ensure(len(train_pairs) >= 1, 'training examples required')
    for pair in train_pairs:
        choices = _candidate_parameters(pair['input'])
        _ensure(len(choices) == 1, 'ambiguous training roles')
        _ensure(np.array_equal(_worklist_prediction(pair['input'], choices[0]), pair['output']),
                'training does not establish the common ray rule')
    choices = _candidate_parameters(test_input)
    _ensure(len(choices) == 1, 'ambiguous query roles')
    return _worklist_prediction(test_input, choices[0])


def _counterfactual(canonical, background, guide, barrier, shortcut):
    """Fault injection checks whether a demonstration actually teaches a case.

    Counting an event is insufficient when overlapping paths hide its effect.
    These deliberately wrong rules are never used to construct ground truth.
    """
    out = canonical.copy()
    h, w = canonical.shape
    active = set(map(int, np.flatnonzero(canonical[0] == guide)))
    for r in range(1, h):
        hits, following = _advance(canonical[r], active, barrier)
        runs = sorted(hits)
        if shortcut == 'one_per_layer':
            runs = runs[:1]
        elif shortcut == 'ignore_singletons':
            runs = [(a, b) for a, b in runs if a != b]
        elif shortcut == 'all_bars':
            runs = sorted((a, b) for _, a, b in _barrier_runs(canonical[r:r+1], barrier))
        for c in following:
            if canonical[r, c] == background:
                out[r, c] = guide
        for a, b in runs:
            for c in range(max(0, a-1), min(w, b+2)):
                if canonical[r-1, c] == background or shortcut == 'overwrite_crosspiece':
                    out[r-1, c] = guide
            for c in (a-1, b+1):
                if 0 <= c < w:
                    following.add(c)
                    if canonical[r, c] == background:
                        out[r, c] = guide
        active = following
    return out


def _length(rng, available):
    """The same length distribution is used for causal and inert geometry."""
    p = rng.random()
    if p < .18:
        return 1
    if p < .62:
        return int(rng.integers(1, min(6, available)+1))
    if p < .82:
        return int(rng.integers(max(1, available-3), available+1))
    return int(rng.integers(1, available+1))


def _dimensions(rng, scale, regime, witness):
    if scale == 'matched':
        h, w = int(rng.integers(12, 22)), int(rng.integers(10, 23))
    elif scale == 'scale' and regime == 'target_test':
        h, w = int(rng.integers(24, 31)), int(rng.integers(22, 31))
    elif scale == 'scale':
        h, w = int(rng.integers(8, 19)), int(rng.integers(7, 22))
    else:
        shape = int(rng.integers(5))
        if shape == 0:
            h, w = int(rng.integers(8, 31)), int(rng.integers(4, 9))
        elif shape == 1:
            h, w = int(rng.integers(6, 11)), int(rng.integers(15, 31))
        elif shape == 2:
            h, w = 30, 30
        else:
            h, w = int(rng.integers(7, 31)), int(rng.integers(7, 31))
    if witness:
        h, w = max(h, 9), max(w, 9)
    return h, w


def _add_scenery(rng, grid, guide, barrier, budget, protected_rows):
    """Compose thin stems, elbows, stairs, and bars before causal scheduling."""
    h, w = grid.shape
    for _ in range(budget):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        kind = int(rng.integers(4))
        cells = set()
        if kind == 0:
            n = _length(rng, w-1)
            c = int(rng.integers(w-n+1))
            cells.update((r, x) for x in range(c, c+n))
        else:
            n = int(rng.integers(2, min(9, h-r)+1)) if h-r >= 2 else 1
            x = c
            for y in range(r, r+n):
                cells.add((y, x))
                if kind == 2 and rng.random() < .45:
                    nx = max(0, min(w-1, x + int(rng.choice([-1, 1]))))
                    cells.add((y, nx))
                    x = nx
            if kind == 3:
                length = _length(rng, w-1)
                left = int(rng.integers(max(0, x-length+1), min(x, w-length)+1))
                cells.update((r+n-1, y) for y in range(left, left+length))
        if any(y in protected_rows or grid[y, x] == guide for y, x in cells):
            continue
        if np.count_nonzero(grid == barrier) + len(cells) > h*w*.24:
            continue
        for y, x in cells:
            grid[y, x] = barrier


def _segment_at(rng, row, pivot, guide, barrier, singleton=False):
    if row[pivot] != 0:
        return
    lo = hi = pivot
    # Reserve one background cell between distinct maximal runs.
    while lo > 0 and row[lo-1] == 0 and (lo < 2 or row[lo-2] != barrier):
        lo -= 1
    while hi+1 < len(row) and row[hi+1] == 0 and (hi+2 == len(row) or row[hi+2] != barrier):
        hi += 1
    n = 1 if singleton else _length(rng, min(hi-lo+1, len(row)-1))
    left = int(rng.integers(max(lo, pivot-n+1), min(pivot, hi-n+1)+1))
    row[left:left+n] = barrier


def _add_unreached(rng, grid):
    """Add runs in ray-free intervals, using the very same length sampler.

    An entire interval is eligible only when it is separated from existing runs;
    therefore these additions cannot change any already determined collision.
    They may interrupt an upstream crosspiece, whose barrier precedence is real.
    """
    _, stats = _simulate_canonical(grid, 0, 1, 2)
    active = {r: set(cols) for r, cols in stats['incoming']}
    active[0] = set(map(int, np.flatnonzero(grid[0] == 1)))
    h, w = grid.shape
    for _ in range(int(rng.integers(1, 5))):
        intervals = []
        for r in range(h):
            free = [grid[r, c] == 0 and c not in active[r]
                    and (c == 0 or grid[r, c-1] != 2)
                    and (c == w-1 or grid[r, c+1] != 2) for c in range(w)]
            c = 0
            while c < w:
                if not free[c]:
                    c += 1
                    continue
                lo = c
                while c+1 < w and free[c+1]:
                    c += 1
                intervals.append((r, lo, c))
                c += 1
        if not intervals:
            return
        # Bias toward wide *available space*, not toward a causal-status length.
        weights = np.array([(hi-lo+1)**2 for _, lo, hi in intervals], dtype=float)
        r, lo, hi = intervals[int(rng.choice(len(intervals), p=weights/weights.sum()))]
        n = _length(rng, hi-lo+1)
        left = int(rng.integers(lo, hi-n+2))
        if np.count_nonzero(grid == 2) + n < h*w*.35:
            grid[r, left:left+n] = 2


def _construct_canonical_input(rng, regime, settings, witness=None, quiet=False):
    """Compose freely sized scenes; witnesses constrain events, not whole layouts.

    Canonical colors 0/1/2 are mapped to sampled visible roles afterwards.
    No layout bank or split-specific bar-length limits are used.
    """
    h, w = _dimensions(rng, settings['scale'], regime, witness)
    grid = np.zeros((h, w), dtype=np.int8)
    rich = settings['richness']
    max_entries = 2 if rich == 0 else min(5, (w+1)//2)
    count = int(rng.integers(1, max_entries+1))
    if witness == 'split':
        count = max(2, count) if settings['simultaneous'] else 1
    if witness == 'merge':
        count = 2
    # Random separated entries; include both transverse boundaries naturally.
    slots = list(map(int, rng.permutation(w)))
    entries = []
    if witness is None and rng.random() < .25:
        count = 1
        entries = [int(rng.integers(max(2, w//5)))]
    elif witness == 'merge':
        left = int(rng.integers(1, w-5))
        entries = [left, int(rng.integers(left+2, w-2))]
    else:
        for c in slots:
            if all(abs(c-d) >= 3 if witness == 'split' else abs(c-d) >= 2 for d in entries):
                entries.append(c)
                if len(entries) == count:
                    break
    entries.sort()
    if witness == 'split' and settings['simultaneous'] and len(entries) < 2:
        entries = [2, w-3]
    lengths = [2 if rich == 0 or rng.random() < .6 else int(rng.integers(3, min(6, h-2))) for _ in entries]
    for c, n in zip(entries, lengths):
        grid[:n, c] = 1
    first = max(lengths) + int(rng.integers(0, 3))
    first = min(first, h-4) if witness else min(first, h-1)
    protected = set(range(0 if witness == 'merge' else 2, first+3)) if witness else set()
    budget = 0 if quiet or rich == 0 else int(rng.integers(0, 3+3*rich))
    _add_scenery(rng, grid, 1, 2, budget, protected)
    if witness == 'merge':
        # Same-bar multi-hit, then an edge detour. Match the inert bar's length
        # to a reached one in the prefix, where it cannot be encountered.
        a, b = entries
        grid[first, a:b+1] = 2
        grid[first+2, :a] = 2
        grid[1, :a] = 2
    elif witness == 'split':
        # Disjoint bars hit together by initial rays. Singleton collisions are
        # optional puzzle-level extensions and get a clear training witness.
        for j, c in enumerate(entries):
            if j == 0 and settings['singleton']:
                grid[first, c] = 2
            else:
                left = c if c == 0 or rng.random() < .5 else c-1
                right = min(w-1, c+int(rng.integers(0, 2)))
                grid[first, left:right+1] = 2
    if not witness:
        style = str(rng.choice(['general', 'straight', 'single', 'clipped'], p=[.72, .09, .09, .10]))
    else:
        style = 'general'
    if style == 'straight':
        # Restrict geometry to columns that no ray visits. Still extend stubs.
        grid[grid == 2] = 0
        available = [c for c in range(w) if c not in entries]
        c = int(rng.choice(available))
        grid[int(rng.integers(h)), c] = 2
        if rich and rng.random() < .7:
            r = int(rng.integers(2, h))
            _segment_at(rng, grid[r], c, 1, 2)
            grid[r, entries] = 0
        if rng.random() < .75:
            _add_unreached(rng, grid)
        return grid
    if style in ('single', 'clipped'):
        grid[grid == 2] = 0
        # One entry makes a sequence of entirely clipped detours possible.
        if style == 'clipped':
            grid[grid == 1] = 0
            entries, lengths = [int(rng.integers(w))], [2]
            grid[:2, entries[0]] = 1
    event_budget = 1 if style == 'single' else int(rng.integers(1, min(12, h-2)+1))
    rows = set(map(int, rng.choice(np.arange(2, h), size=min(event_budget, h-2), replace=False)))
    if witness:
        rows -= protected
    active = set(entries)
    for r in range(1, h):
        if r in rows:
            available = [c for c in sorted(active) if grid[r, c] == 0 and (c not in entries or r >= lengths[entries.index(c)])]
            if available and style == 'clipped':
                c = available[0]
                if c == w-1 or (c > 0 and rng.random() < .5):
                    left, right = int(rng.integers(1, c+1)), w-1
                else:
                    left, right = 0, int(rng.integers(c, w-1))
                grid[r, left:right+1] = 2
            elif available:
                rng.shuffle(available)
                quota = 1 if not settings['simultaneous'] else int(rng.integers(1, min(3, len(available))+1))
                for c in available[:quota]:
                    _segment_at(rng, grid[r], c, 1, 2,
                                singleton=settings['singleton'] and rng.random() < .15)
        hits, following = _advance(grid[r], active, 2)
        for left, right in hits:
            following.update(c for c in (left-1, right+1) if 0 <= c < w)
        active = following
    if not np.any(grid == 2):
        # Bounded fallback for a scene whose selected events missed long stubs.
        c = min(entries)
        grid[h-1, c] = 2
    if not quiet and rng.random() < .55:
        _add_unreached(rng, grid)
    return grid


def _features(stats):
    """Case extensions whose test use needs a training demonstration."""
    flags = set()
    for name in ('simultaneous_layers', 'multi_hit_events', 'clipped_events',
                 'interrupted_crosspieces', 'shared_flanks'):
        if stats[name]:
            flags.add(name)
    if 1 in stats['reached_lengths']:
        flags.add('struck_singleton')
    if stats['entry_count'] >= 3:
        flags.add('many_entries')
    if max(stats['stub_lengths']) > 2:
        flags.add('long_stub')
    return flags


def _demonstrated_features(pair, stats=None):
    stats = _pair_stats(pair) if stats is None else stats
    flags = _features(stats)
    d, bg, guide, barrier = _candidate_parameters(pair['input'])[0]
    canonical = _to_canonical(pair['input'], d)
    expected = _to_canonical(pair['output'], d)
    for feature, shortcut in (('simultaneous_layers', 'one_per_layer'),
                              ('struck_singleton', 'ignore_singletons')):
        if feature in flags and np.array_equal(_counterfactual(canonical, bg, guide, barrier, shortcut), expected):
            flags.remove(feature)
    return flags


def _pair_stats(pair):
    candidates = _candidate_parameters(pair['input'])
    _ensure(len(candidates) == 1, 'input roles/direction must be uniquely visible')
    d, bg, g, b = candidates[0]
    canonical = _to_canonical(pair['input'], d)
    _, stats = _simulate_canonical(canonical, bg, g, b)
    stats.update(direction=d, palette=[bg, g, b])
    return stats


def _grid_signature(grid):
    return (grid.shape, np.asarray(grid, dtype=np.int8).tobytes())


def _pair_signature(pair):
    return (_grid_signature(pair['input']), _grid_signature(pair['output']))


def _make_pair(rng, regime, settings, palette, direction, witness=None, allowed=None, seen=None):
    for attempt in range(120):
        # After the exploratory budget, a clean composition is a bounded fallback.
        a = _construct_canonical_input(rng, regime, settings, witness, quiet=attempt >= 80)
        if rng.random() < .5:
            a = np.fliplr(a).copy()
        o, stats = _simulate_canonical(a, 0, 1, 2)
        if stats['changed_cells'] == 0 or np.count_nonzero(a == 0) <= max(np.count_nonzero(a == 1), np.count_nonzero(a == 2)):
            continue
        if witness == 'split':
            if not stats['two_flank_events']:
                continue
            if settings['simultaneous'] and not stats['simultaneous_layers']:
                continue
            if settings['singleton'] and 1 not in stats['reached_lengths']:
                continue
            if settings['singleton'] and np.array_equal(_counterfactual(a, 0, 1, 2, 'ignore_singletons'), o):
                continue
            if settings['simultaneous'] and np.array_equal(_counterfactual(a, 0, 1, 2, 'one_per_layer'), o):
                continue
        if witness == 'merge' and not (stats['multi_hit_events'] and stats['clipped_events'] and stats['distractor_runs']):
            continue
        if allowed is not None and not _features(stats) <= allowed:
            continue
        mapping = np.array(palette, dtype=np.int8)
        pair = {'input': _from_canonical(mapping[a], direction), 'output': _from_canonical(mapping[o], direction)}
        if _candidate_parameters(pair['input']) != [(direction, *palette)]:
            continue
        if _pair_signature(pair) in _SOURCE_SIGNATURES or (seen is not None and _grid_signature(pair['input']) in seen):
            continue
        return pair
    # A finite clean family guarantees recovery even after extreme rejection.
    # More than fourteen distinct candidates are available for at most fourteen
    # puzzle pairs. This path has no random retry or source-like numeric limit.
    mapping = np.array(palette, dtype=np.int8)
    for h in range(9, 31):
        w = 11
        a = np.zeros((h, w), dtype=np.int8)
        if witness == 'merge':
            a[:2, [2, 7]] = 1
            a[3, 2:8] = 2
            a[5, :2] = a[1, :2] = 2
        elif witness == 'split':
            a[:2, [3, 7]] = 1
            a[3, 3:4 if settings['singleton'] else 5] = 2
            a[3, 7:9] = 2
            a[6, :3] = 2
        else:
            a[:2, 4] = 1
            a[h-2, 4:6] = 2
        o, stats = _simulate_canonical(a, 0, 1, 2)
        pair = {'input': _from_canonical(mapping[a], direction), 'output': _from_canonical(mapping[o], direction)}
        if seen is not None and _grid_signature(pair['input']) in seen:
            continue
        _ensure(_candidate_parameters(pair['input']) == [(direction, *palette)], 'fallback ambiguity')
        _ensure(_pair_signature(pair) not in _SOURCE_SIGNATURES, 'fallback source copy')
        _ensure(allowed is None or _features(stats) <= allowed, 'fallback training coverage')
        return pair
    raise AssertionError('more unique inputs excluded than legal puzzle size')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validated_seed(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    richness = int(rng.choice(3, p=[.20, .55, .25]))
    settings = {
        'richness': richness,
        'simultaneous': richness == 2 or rng.random() < .7,
        'singleton': richness == 2 or rng.random() < .65,
        'scale': str(rng.choice(['overlap', 'matched', 'scale'], p=[.55, .25, .20])),
    }
    role_mode = str(rng.choice(['fixed', 'mixed'], p=[.45, .55]))
    direction_mode = str(rng.choice(['fixed', 'mixed', 'transfer'], p=[.40, .40, .20]))
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    # Explicitly permit the original role assignment, without banning any colors.
    if rng.random() < .08:
        palette = (7, 6, 2)
    direction = int(rng.integers(4))
    transfer = int(rng.integers(4))
    train_directions = [d for d in range(4) if d != transfer]
    rng.shuffle(train_directions)
    puzzle = {'train': [], 'test': []}
    seen = set()
    covered = set()
    for split, count, regime in (('train', nt, 'source_train'), ('test', nq, 'target_test')):
        for i in range(count):
            local_palette = palette
            if role_mode == 'mixed' and i+int(split == 'test') > 0:
                local_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
            local_direction = direction
            if direction_mode == 'mixed':
                local_direction = (direction + i + int(split == 'test')) % 4
            elif direction_mode == 'transfer':
                local_direction = transfer if split == 'test' else train_directions[i % 3]
            witness = ('split' if i == 0 else 'merge') if split == 'train' and i < 2 else None
            pair = _make_pair(rng, regime, settings, local_palette, local_direction,
                              witness, covered if split == 'test' else None, seen)
            puzzle[split].append(pair)
            seen.add(_grid_signature(pair['input']))
            if split == 'train':
                covered |= _demonstrated_features(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _ensure(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid pair counts')
    seen, covered = set(), set()
    train_stats = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            for grid in pair.values():
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a matrix')
                _ensure(all(1 <= d <= 30 for d in grid.shape), 'invalid grid dimensions')
                _ensure(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)), 'invalid colors')
            a, o = pair['input'], pair['output']
            _ensure(a.shape == o.shape and not np.array_equal(a, o), 'invalid output shape or unchanged output')
            key = _grid_signature(a)
            _ensure(key not in seen and _pair_signature(pair) not in _SOURCE_SIGNATURES, 'duplicate/source pair')
            seen.add(key)
            parameters = _candidate_parameters(a)
            _ensure(len(parameters) == 1, 'ambiguous role or direction')
            d, bg, g, b = parameters[0]
            expected = _worklist_prediction(a, parameters[0])
            _ensure(np.array_equal(o, expected), 'incorrect ray propagation')
            _ensure(np.all(o[a != bg] == a[a != bg]), 'original pixels overwritten')
            _ensure(np.all(o[o != a] == g), 'only guide cells may be added')
            stats = _pair_stats(pair)
            if split == 'train':
                covered |= _demonstrated_features(pair, stats)
                train_stats.append(stats)
            else:
                _ensure(_features(stats) <= covered, 'test uses an undemonstrated collision case')
    _ensure(any(s['two_flank_events'] for s in train_stats), 'training lacks branching')
    _ensure(any(s['collision_runs'] >= 2 for s in train_stats), 'training lacks repeated collisions')
    _ensure(any(s['distractor_runs'] for s in train_stats), 'training lacks causal reachability evidence')
    _ensure(any(set(s['reached_lengths']) & set(s['unreached_lengths']) for s in train_stats),
            'training must distinguish equally long reached and unreached runs')
    _ensure(any(not np.array_equal(
        _counterfactual(_to_canonical(p['input'], s['direction']), *s['palette'], 'all_bars'),
        _to_canonical(p['output'], s['direction'])) for p, s in zip(puzzle['train'], train_stats)),
        'training must visibly distinguish reached from unreached barriers')
    # Mixed roles are locally explicit, not hidden puzzle-wide equalities.
    # The only held-out geometric regime is scale/orientation/composition.


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(_pair_signature(x) == _pair_signature(y)
               for x, y in zip(a[s], b[s])) for s in ('train', 'test'))


def coverage_measurements(puzzles):
    """Measurements reconstructed from returned grids, available to scratch QA."""
    records, groups = [], []
    for puzzle_index, puzzle in enumerate(puzzles):
        group = []
        for split in ('train', 'test'):
            for i, pair in enumerate(puzzle[split]):
                stats = _pair_stats(pair)
                stats.update(puzzle=puzzle_index, split=split, pair=i)
                records.append(stats)
                group.append(stats)
        groups.append(group)
    flags = {
        'simultaneous': lambda s: s['simultaneous_layers'] > 0,
        'initial_simultaneous': lambda s: s['initial_simultaneous'] > 0,
        'descendant_simultaneous': lambda s: s['descendant_simultaneous'] > 0,
        'same_bar_multi_hit': lambda s: s['multi_hit_events'] > 0,
        'shared_flanks': lambda s: s['shared_flanks'] > 0,
        'merging': lambda s: s['mergers'] > 0,
        'gap_passes': lambda s: s['gap_passes'] > 0,
        'struck_singleton': lambda s: 1 in s['reached_lengths'],
        'inert_singleton': lambda s: 1 in s['unreached_lengths'],
        'long_reached': lambda s: max(s['reached_lengths'], default=0) >= .75*s['transverse_width'],
        'long_unreached': lambda s: max(s['unreached_lengths'], default=0) >= .75*s['transverse_width'],
        'connected': lambda s: s['vertical_links'] > 0,
        'thin_stems': lambda s: s['singleton_forward_links'] > 0,
        'bent_formations': lambda s: s['bent_formations'] > 0,
        'stepped_formations': lambda s: s['stepped_formations'] > 0,
        'narrow_gap_threads': lambda s: s['narrow_gap_threads'] > 0,
        'prefix_obstacles': lambda s: s['prefix_barriers'] > 0,
        'edge_entry': lambda s: s['edge_entry'],
        'immediate_collision': lambda s: s['immediate_collision'],
        'final_collision': lambda s: s['final_collision'],
        'many_entries': lambda s: s['entry_count'] >= 3,
        'long_stub': lambda s: max(s['stub_lengths']) > 2,
        'unequal_stubs': lambda s: len(set(s['stub_lengths'])) > 1,
        'zero_distractors': lambda s: s['distractor_runs'] == 0,
        'one_sided_only': lambda s: s['clipped_events'] > 0 and not s['two_flank_events'],
        'single_collision': lambda s: s['collision_runs'] == 1,
        'straight_through': lambda s: s['collision_runs'] == 0,
        'interrupted_crosspiece': lambda s: s['interrupted_crosspieces'] > 0,
        'small_narrow': lambda s: s['transverse_width'] <= 6,
        'short_wide': lambda s: s['forward_length'] <= 10 and s['transverse_width'] >= 18,
        'full_size': lambda s: s['forward_length'] == s['transverse_width'] == 30,
        'source_palette': lambda s: set(s['palette']) == {2, 6, 7},
        'combined_rich': lambda s: s['simultaneous_layers'] > 0 and s['vertical_links'] > 0 and s['clipped_events'] > 0 and max(s['stub_lengths']) > 2,
    }
    totals = {split: {name: sum(bool(test(s)) for s in records if s['split'] == split)
                      for name, test in flags.items()} for split in ('train', 'test')}
    within = []
    for group in groups:
        training = [s for s in group if s['split'] == 'train']
        testing = [s for s in group if s['split'] == 'test']
        within.append({
            'palettes': len({tuple(s['palette']) for s in group}),
            'directions': len({s['direction'] for s in group}),
            'shapes': len({(s['forward_length'], s['transverse_width']) for s in group}),
            'entry_counts': len({s['entry_count'] for s in group}),
            'stub_lengths': len({tuple(s['stub_lengths']) for s in group}),
            'orientation_transfer': bool({s['direction'] for s in testing} - {s['direction'] for s in training}),
            'test_not_deeper': min(s['forward_length'] for s in testing) <= max(s['forward_length'] for s in training),
            'test_not_more_collisions': min(s['collision_runs'] for s in testing) <= max(s['collision_runs'] for s in training),
            'test_not_more_distractors': min(s['distractor_runs'] for s in testing) <= max(s['distractor_runs'] for s in training),
        })
    return {'records': records, 'feature_counts': totals, 'within_puzzle': within}


def self_test() -> None:
    # Regression labels are data only; prediction never receives held-out labels.
    source = [{'input': _decode(a), 'output': _decode(b)} for a, b in _SOURCE_REGRESSIONS]
    for pair in source:
        choices = _candidate_parameters(pair['input'])
        _ensure(choices == [(0, 7, 6, 2)], 'source recognition regression')
        actual, _ = _simulate_canonical(pair['input'], 7, 6, 2)
        _ensure(np.array_equal(actual, pair['output']), 'source layer regression')
        _ensure(np.array_equal(_worklist_prediction(pair['input'], choices[0]), pair['output']), 'source work-list regression')
    for query in source[6:]:
        prediction = predict_from_training(source[:6], query['input'].copy())
        _ensure(np.array_equal(prediction, query['output']), 'source prediction regression')

    # Explicit counterexample to at most one collision per depth: two initial
    # rays, disjoint bars, a shared flank, and a later single-run multi-hit.
    a = np.zeros((9, 11), dtype=np.int8)
    a[:2, [3, 7]] = 1
    a[2, 2:5] = a[2, 6:9] = 2
    a[5, 1:6] = 2
    expected = _worklist_prediction(a, (0, 0, 1, 2))
    actual, s = _simulate_canonical(a, 0, 1, 2)
    _ensure(np.array_equal(actual, expected) and s['simultaneous_layers'] == 1 and s['shared_flanks'] > 0, 'simultaneous collision regression')
    _ensure(np.all(actual[1, 1:10] == 1) and actual[3, 9] == 1, 'second distinct bar was skipped')

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        train = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            prediction = predict_from_training(train, pair['input'].copy())
            _ensure(np.array_equal(prediction, pair['output']), 'held-out prediction mismatch')
        for shortcut in ('one_per_layer', 'all_bars', 'ignore_singletons', 'overwrite_crosspiece'):
            def matches(pair):
                d, bg, g, b = _candidate_parameters(pair['input'])[0]
                wrong = _counterfactual(_to_canonical(pair['input'], d), bg, g, b, shortcut)
                return np.array_equal(_from_canonical(wrong, d), pair['output'])
            if all(matches(pair) for pair in puzzle['train']):
                _ensure(all(matches(pair) for pair in puzzle['test']), 'training admits a test-changing shortcut')
        for pair in puzzle['train'] + puzzle['test']:
            d, bg, g, b = _candidate_parameters(pair['input'])[0]
            canonical = _to_canonical(pair['input'], d)
            expected, _ = _simulate_canonical(canonical, bg, g, b)
            _ensure(np.array_equal(_from_canonical(expected, d), pair['output']), 'two solvers disagree')
    report = coverage_measurements(puzzles)
    train_counts = {len(p['train']) for p in puzzles}
    test_counts = {len(p['test']) for p in puzzles}
    _ensure(len(train_counts) >= 4 and len(test_counts) >= 3, 'count diversity')
    for split, metrics in report['feature_counts'].items():
        for name, count in metrics.items():
            _ensure(count > 0, f'42-seed coverage missing {split} {name}')
    for s in report['records']:
        _ensure(s['barrier_cells'] > 0 and s['changed_cells'] > 0, 'trivial scene')
    within = report['within_puzzle']
    for name in ('palettes', 'directions', 'entry_counts', 'stub_lengths'):
        _ensure(sum(s[name] > 1 for s in within) >= 8, f'within-puzzle {name} collapsed')
    for name in ('orientation_transfer', 'test_not_deeper', 'test_not_more_collisions', 'test_not_more_distractors'):
        _ensure(sum(s[name] for s in within) >= 3, f'puzzle regime {name} collapsed')
    records = report['records']
    _ensure({s['direction'] for s in records} == set(range(4)), 'direction coverage')
    for role in range(3):
        _ensure({s['palette'][role] for s in records} == set(range(10)), 'color-role coverage')
    _ensure(len({(s['forward_length'], s['transverse_width']) for s in records}) >= 60, 'shape diversity')
    for split in ('train', 'test'):
        hit = {n for s in records if s['split'] == split for n in s['reached_lengths']}
        inert = {n for s in records if s['split'] == split for n in s['unreached_lengths']}
        _ensure(len(hit & inert) >= 10 and max(hit & inert) >= 18, 'bar size predicts reachability')
    joint = {(len(p['test']), _pair_stats(p['train'][0])['direction']) for p in puzzles}
    _ensure(len(joint) >= 10, 'direction and test count are coupled')
    offsets = {c-a for s in records for _, a, _, columns in s['events'] for c in columns}
    gaps = {b-a for s in records for a, b in zip(
        sorted({r for r, _, _, _ in s['events']}), sorted({r for r, _, _, _ in s['events']})[1:])}
    _ensure(len(offsets) >= 10 and 0 in offsets, 'collision offsets lack variation')
    _ensure(1 in gaps and max(gaps) >= 5, 'collision spacing lacks variation')
    _ensure(any(s['palettes'] == s['directions'] == 1 for s in within), 'homogeneous puzzles disappeared')

    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(7109, nt, nq)
        b = generate_puzzle(7109, nt, nq)
        _ensure(_puzzles_equal(a, b), 'nondeterminism')
        _ensure(nt is None or len(a['train']) == nt, 'train count not honored')
        _ensure(nq is None or len(a['test']) == nq, 'test count not honored')
    _ensure(not _puzzles_equal(generate_puzzle(7109), generate_puzzle(7110)), 'identical seeds')
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')


def _decode(rows):
    return np.array([[int(c) for c in row] for row in rows.split()], dtype=np.int8)


# Exact source data for regressions and content-based exclusion only.
_SOURCE_REGRESSIONS = (
    ('77776777777767 77776777777767 77777777777777 77722277777222 77777777777777 22222777777777 77777777777777 77777777777777 77777772222222 77777777777777 77222227777777 77777777777777 77777222222777 77777777777777',
     '77776777777767 77776777777767 77666667776666 77622267776222 66666667776777 22222667776777 77777667776777 77777666666666 77777662222222 76666666777777 76222226777777 76776666666677 76776222222677 76776777777677'),
    ('7777677777767 7777677777767 7777777777777 7777777777777 7777777772222 7777777777777 7777777777777 7777777777777 7777777777777 7222277777777 7777777777777 7777777777777 7777777777777 7777222277777 7777777777777 7777777777777',
     '7777677777767 7777677777767 7777677777767 7777677766666 7777677762222 7777677767777 7777677767777 7777677767777 6666667767777 6222267767777 6777767767777 6777767767777 6776666667777 6776222267777 6776777767777 6776777767777'),
    ('776777722 776777727 722277777 777777777 777777777 777222277 777777777 222222777',
     '776777722 666667727 622267777 677767777 676666667 676222267 666666667 222222667'),
    ('77767 22262 77777 77272 77777 77222 77777 77777 77777 77777 22272 77777 77222 77777',
     '77767 22262 77767 77262 76666 76222 76777 76777 76777 66667 22262 76666 76222 76777'),
    ('67777777777777777777 67777777777777777777 22222222222222222277 77777777777777777777 77777722227777772222 77222277777777777777 77777777777772222277 77777777777777777777',
     '67777777777777777777 66666666666666666667 22222222222222222267 77777777777777766666 77777722227777762222 77222277777766666667 77777777777762222267 77777777777767777767'),
    ('7777776777777777 2222226772222222 7777777777777777 7777777777777777 7777222222222777 7777777777777777 7777777777777777 2222222222777777 7777777777777777 7777777777777777 7777777222222227 7777777777777777 7777777777777777 2277222222222277 7777777777777777',
     '7777776777777777 2222226772222222 7777776777777777 7776666666666677 7776222222222677 7776777777777677 6666666666677677 2222222222677677 7777777777677677 7777776666666666 7777776222222226 7777776777777776 7776666666666666 2276222222222266 7776777777777766'),
    ('7767777777776777 7767777777776777 7777777777777777 7777777777777777 2272277777722222 7777777777777777 7777777722222777 7777777777777777 7777772222777777 7777777777777777 7777777777777777 7777222277777777 7777777777777777 7777777777777777 7777772222777777 7777777777777777',
     '7767777777776777 7767777777776777 7767777777776777 7767777777666666 2262277777622222 7767777666666677 7767777622222677 7767766666677677 7767762222677677 7767767777677677 7766666667677677 7766222267677677 7766777767677677 7766766666677677 7766762222677677 7766767777677677'),
    ('777767777777777777777767777777 777767777777777777777767777777 772222777777777772222222227777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777222277777777777777777 777777777777777777777777777777 777777777777777777777772222777 777777777772777777777777777777 777777777772777777777777777777 772777777772777777777777777777 772777777777777777727777777777 772777777777777777727777777777 772777777777777777727777777777 772777777777777777727777777777 772222227777777777727777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777722222222227777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777777777777777777777 777777777777772222777777722227 777777777777777777777777777777 777777777777777777777777777777',
     '777767777777777777777767777777 766666677777777766666666666777 762222677777777762222222226777 767777677777777767777777776777 767777677777777767777777776777 767777677777777767777777776777 767777677777777767777777776777 767777677222277767777777776777 767777677777777767777766666677 767777677777777767777762222677 767777677772777767777767777677 767777677772777767777767777677 762777677772777767777767777677 762777677777777767727767777677 762777677777777767727767777677 762777677777777767727767777677 762666666777777767727767777677 762222226777777767727767777677 767777776777777767777767777677 767777776777777767777767777677 767777776777777767777767777677 767777776666666666666767777677 767777776622222222226767777677 767777776677777777776767777677 767777776677777777776767777677 767777776677777777776767777677 767777776677777777776767666666 767777776677772222776767622226 767777776677777777776767677776 767777776677777777776767677776'),
)
_SOURCE_SIGNATURES = {(_grid_signature(_decode(a)), _grid_signature(_decode(b)))
                      for a, b in _SOURCE_REGRESSIONS}
