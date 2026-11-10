"""Occurrence-addressed, simultaneous lookup between hollow and solid queues.

Only standard-library modules and NumPy are used. No parent code or files are
loaded. Public solving depends exclusively on the visible input.
"""
from collections import Counter
import numpy as np


# Both dimensions are odd; a 3x3 rim plus center would have no visible hole.
_SHAPES = ((3, 5), (5, 3), (5, 5), (3, 7), (7, 3),
           (5, 7), (7, 5), (7, 7), (3, 9), (9, 3))
_SHORTCUTS = ("no_A", "no_B", "whole_queue", "center_rank", "global_donors", "nearest", "reverse",
              "horizontal", "horizontal_reverse", "updated", "pruned", "greedy")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(int, copy=True)


def _tile(height, width, kind, body, value, background):
    tile = np.full((height, width), body, dtype=int)
    if kind == 0:
        tile[1:-1, 1:-1] = background
    tile[height // 2, width // 2] = value
    return tile


def _parse(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
        raise ValueError("input needs a complete background border")
    remaining = grid != background
    tokens = []
    size = None
    while np.any(remaining):
        top, left = map(int, np.argwhere(remaining)[0])
        body = int(grid[top, left])
        width = height = 0
        while left + width < grid.shape[1] and grid[top, left + width] == body:
            width += 1
        while top + height < grid.shape[0] and grid[top + height, left] == body:
            height += 1
        if (height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0
                or height * width == 9 or 2 * width > 30):
            raise ValueError("tokens need a visible hollow interior and a unique center")
        if size is not None and size != (height, width):
            raise ValueError("all tokens in an input must have the same dimensions")
        size = height, width
        patch = grid[top:top + height, left:left + width]
        if patch.shape != size:
            raise ValueError("clipped token")
        value = int(patch[height // 2, width // 2])
        if value == background:
            raise ValueError("every token needs a nonbackground center")
        kinds = [k for k in (0, 1) if np.array_equal(
            patch, _tile(height, width, k, body, value, background))]
        if len(kinds) != 1:
            raise ValueError("malformed or ambiguous token")
        # Consume the entire footprint, including a disconnected center dot.
        neighborhood = grid[top - 1:top + height + 1, left - 1:left + width + 1].copy()
        neighborhood[1:height + 1, 1:width + 1] = background
        if not np.all(neighborhood == background):
            raise ValueError("token footprints require a complete one-cell gap")
        remaining[top:top + height, left:left + width] = False
        tokens.append(dict(top=top, left=left, kind=kinds[0], body=body, value=value))
    if not tokens:
        raise ValueError("no tokens")
    for kind in (0, 1):
        queue = [t for t in tokens if t['kind'] == kind]
        if not queue or len(queue) * size[0] > 30:
            raise ValueError("both input classes are required and queues must fit")
        if len({t['top'] for t in queue}) != len(queue):
            raise ValueError("same-class top-row ties are not in this family")
    return dict(background=background, size=size, tokens=tokens)


def _queues(a, order="vertical", subset=None):
    key = (lambda i: (a['tokens'][i]['left'], a['tokens'][i]['top'])) if order.startswith(
        "horizontal") else (lambda i: (a['tokens'][i]['top'], a['tokens'][i]['left']))
    reverse = order in ("reverse", "horizontal_reverse")
    return [sorted((i for i, t in enumerate(a['tokens'])
                    if t['kind'] == k and (subset is None or i in subset)),
                   key=key, reverse=reverse) for k in (0, 1)]


def _links(a, mode="rule", order="vertical", subset=None):
    """Map each successful source instance to its original donor instance."""
    tokens = a['tokens']
    queues = _queues(a, order, subset)
    links = {}
    for kind, queue in enumerate(queues):
        occurrence = Counter()
        for slot, i in enumerate(queue, 1):
            t = tokens[i]
            occurrence_key = t['value'] if mode == "center_rank" else t['body']
            occurrence[occurrence_key] += 1
            rank = occurrence[occurrence_key]
            if mode == "no_A":
                rank = 1
            elif mode == "whole_queue":
                rank = slot
            if mode == "global_donors":
                candidates = [j for q in queues for j in q if j != i]
                candidates.sort(key=lambda j: (tokens[j]['top'], tokens[j]['left']))
            else:
                candidates = queues[1 - kind]
            donors = [j for j in candidates if tokens[j]['body'] == t['value']]
            if rank <= len(donors):
                links[i] = min(donors, key=lambda j: (
                    abs(t['top'] - tokens[j]['top']) + abs(t['left'] - tokens[j]['left']),
                    tokens[j]['top'], tokens[j]['left'])) if mode == "nearest" else donors[rank - 1]
    return links


def _render(a, links, order="vertical", style="compact", updated=False):
    tokens, (height, width) = a['tokens'], a['size']
    queues = _queues(a, order, set(links))
    if style == "greedy":
        rows = [[None, None]]
        for i in sorted(links, key=lambda j: (tokens[j]['top'], tokens[j]['left'])):
            kind = tokens[i]['kind']
            if rows[-1][kind] is not None:
                rows.append([None, None])
            rows[-1][kind] = i
    else:
        rows = [[q[r] if r < len(q) else None for q in queues]
                for r in range(max(1, *(len(q) for q in queues)))]
    out = np.full((height * len(rows), 2 * width), a['background'], dtype=int)
    for row, slots in enumerate(rows):
        for kind, i in enumerate(slots):
            if i is None:
                continue
            donor = links[i]
            if updated and donor in links:
                donor = links[donor]
            out[row * height:(row + 1) * height, kind * width:(kind + 1) * width] = _tile(
                height, width, kind, tokens[i]['body'], tokens[donor]['value'], a['background'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _parse(input_grid)
    links = _links(a)
    if not links:
        raise ValueError("family inputs must contain at least one successful address")
    return _render(a, links)


def _alternative(a, mode):
    """Generator-owned diagnostics, not an independent solver or oracle."""
    if mode == "no_B":
        return _render(a, {i: i for i in range(len(a['tokens']))})
    order = mode if mode in ("reverse", "horizontal", "horizontal_reverse") else "vertical"
    links = _links(a, mode, order)
    if mode == "pruned":
        links = _links(a, subset=set(links))
    return _render(a, links, order, style="greedy" if mode == "greedy" else "compact",
                   updated=mode == "updated")


def _features(a):
    tokens, queues, links = a['tokens'], _queues(a), _links(a)
    ranks, slots = {}, {}
    for queue in queues:
        count = Counter()
        for slot, i in enumerate(queue, 1):
            count[tokens[i]['body']] += 1
            ranks[i], slots[i] = count[tokens[i]['body']], slot
    survivors = [sum(i in links for i in q) for q in queues]
    colors = [{tokens[i]['body'] for i in q} for q in queues]
    conflicts = any(len({tokens[i]['value'] for i in q if tokens[i]['body'] == body}) > 1
                    for q in queues for body in {tokens[i]['body'] for i in q})
    early_failure = any(i not in links and j in links and tokens[i]['body'] == tokens[j]['body']
                        for q in queues for n, i in enumerate(q) for j in q[n + 1:])
    return dict(counts=tuple(map(len, queues)), survivors=tuple(survivors),
                max_success_rank=max((ranks[i] for i in links), default=0),
                conflicting_values=conflicts, shared_body=bool(colors[0] & colors[1]),
                early_failure=early_failure,
                per_body=any(ranks[i] != slots[i] for i in links),
                deleted_donor=any(j not in links for j in links.values()),
                changing_donor=any(j in links and tokens[links[j]]['value'] != tokens[j]['value']
                                   for j in links.values()),
                monochrome=sum(t['body'] == t['value'] for t in tokens),
                unchanged=sum(tokens[i]['value'] == tokens[j]['value'] for i, j in links.items()),
                holes=sum(i not in links for i in range(len(tokens))),
                cross_tie=len({t['top'] for t in tokens}) < len(tokens))


def _packing_options(size, count):
    height, width = size
    capacity = 29 // (width + 1)  # 28 interior cells, one-cell inter-tile gaps.
    return [bands for bands in range(1, count + 1)
            if bands * capacity >= count and count + bands * height - 1 <= 28]


def _layout(rng, size, counts):
    height, width = size
    count = sum(counts)
    bands = int(rng.choice(_packing_options(size, count)))
    capacity = 29 // (width + 1)
    amounts = [1] * bands
    for _ in range(count - bands):
        choices = [i for i, amount in enumerate(amounts) if amount < capacity]
        amounts[int(rng.choice(choices))] += 1
    min_h = count + bands * height - 1
    min_w = max(amounts) * (width + 1) - 1
    mode = int(rng.integers(6))
    if mode == 0:
        h = int(rng.integers(min_h, min(28, min_h + 3) + 1))
        w = int(rng.integers(min_w, min(28, min_w + 3) + 1))
    elif mode == 1:
        h = w = int(rng.integers(max(min_h, min_w), 29))
    elif mode == 2:
        h, w = 28, int(rng.integers(min_w, 29))
    elif mode == 3:
        h, w = int(rng.integers(min_h, 29)), 28
    elif mode == 4:
        h = w = 28
    else:
        h, w = int(rng.integers(min_h, 29)), int(rng.integers(min_w, 29))
    kinds = [0] * counts[0] + [1] * counts[1]
    sequence_mode = int(rng.integers(3))
    if sequence_mode == 0:
        rng.shuffle(kinds)
    elif sequence_mode == 1:
        kinds = sorted(kinds, reverse=bool(rng.integers(2)))
    else:
        kinds = [k for r in range(max(counts)) for k in (0, 1) if r < counts[k]]
    vertical = rng.multinomial(h - min_h, [1 / (bands + 1)] * (bands + 1))
    positions = []
    top = 1 + int(vertical[0])
    for band, amount in enumerate(amounts):
        gaps = rng.multinomial(w - (amount * (width + 1) - 1), [1 / (amount + 1)] * (amount + 1))
        left = 1 + int(gaps[0])
        offsets = rng.permutation(amount)
        for i in range(amount):
            positions.append((top + int(offsets[i]), left))
            left += width + 1 + int(gaps[i + 1])
        top += height + amount + int(vertical[band + 1])
    rr, cc = np.meshgrid(np.arange(1, h + 2 - height), np.arange(1, w + 2 - width), indexing="ij")
    # These are free relocation sweeps, not a retained shelf template.
    for _ in range(3):
        for i in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if i != j:
                    legal &= (np.abs(rr - r) > height) | (np.abs(cc - c) > width)
                    if kinds[i] == kinds[j]:
                        legal &= rr != r
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(i)] = int(rr[r, c]), int(cc[r, c])
    queues = [sorted(p for p, k in zip(positions, kinds) if k == kind) for kind in (0, 1)]
    return (h + 2, w + 2), queues


def _abstract(relations):
    return dict(size=(3, 5), background=9, tokens=[
        dict(top=2 * rank + kind, left=rank, kind=kind, body=b, value=v)
        for kind, queue in enumerate(relations) for rank, (b, v) in enumerate(queue)])


def _random_relations(rng, size):
    limit = min(6, 30 // size[0])
    for _ in range(100):
        counts = tuple(map(int, rng.integers(1, limit + 1, size=2)))
        if not _packing_options(size, sum(counts)):
            continue
        pool = int(rng.integers(2, 6))
        queues = [[(int(rng.integers(pool)), int(rng.integers(pool + 1)))
                   for _ in range(n)] for n in counts]
        # Make at least one visible address succeed, retaining arbitrary others.
        queues[0][0] = queues[0][0][0], queues[1][0][0]
        if rng.random() < .35:
            queues[1][0] = queues[1][0][0], queues[0][0][0]
        if any(max(Counter(b for b, _ in q).values()) > 3 for q in queues):
            continue
        a = _abstract(queues)
        if _links(a):
            return queues
    # A small reciprocal pair is always feasible for the sampled vocabulary.
    return [[(0, 1)], [(1, 0)]]


def _witness_relations(rng, size, primary, mode):
    if mode == "address":
        k = int(rng.integers(2, min(3, 30 // size[0] - 1) + 1))
        p = [(0, 1)] * (k + 1)
        q = [(1, 1)] + [(1, j + 2) for j in range(k - 2)] + [(1, 0)]
    else:
        # X->U fails before X->Y succeeds at occurrence TWO. Y and X also
        # occur in the opposite queues with misleading values.
        p = [(0, 2), (2, 1), (0, 1), (1, 2)]
        q = [(1, 3), (0, 3), (1, 0)]
    return [p, q] if primary == 0 else [q, p]


def _depth_relations(rng, size, primary):
    limit = 30 // size[0]
    depths = [k for k in range(4, min(6, limit - 1) + 1)
              if _packing_options(size, 2 * k)]
    k = int(rng.choice(depths))
    tail = int(rng.integers(0, min(2, limit - k) + 1))
    p = [(0, 1)] * (k + tail)
    q = [(1, 1)] + [(1, j + 2) for j in range(k - 2)] + [(1, 0)]
    if len(p) < limit and rng.random() < .65:
        p.insert(int(rng.integers(len(p) + 1)), (1, 2))
    if len(q) < limit and rng.random() < .65:
        q.insert(int(rng.integers(len(q) + 1)), (0, 2))
    # Some wide tokens cannot pack every draw; drop optional distractors/tails.
    if not _packing_options(size, len(p) + len(q)):
        p, q = [(0, 1)] * (k + 1), [(1, 1)] + [(1, j + 2) for j in range(k - 2)] + [(1, 0)]
    if not _packing_options(size, len(p) + len(q)):
        p = [(0, 1)] * k
    return [p, q] if primary == 0 else [q, p]


def _make_pair(rng, size, relations, background):
    shape, positions = _layout(rng, size, tuple(map(len, relations)))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    grid = np.full(shape, background, dtype=int)
    height, width = size
    for kind, (queue, sites) in enumerate(zip(relations, positions)):
        for (body, value), (top, left) in zip(queue, sites):
            grid[top:top + height, left:left + width] = _tile(
                height, width, kind, palette[body], palette[value], background)
    # All output information is rediscovered from returned pixels.
    return dict(input=grid, output=solve(grid))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _training_suffices(pairs):
    analyses = [_parse(p['input']) for p in pairs]
    return all(any(not np.array_equal(p['output'], _alternative(a, mode))
                   for p, a in zip(pairs, analyses)) for mode in _SHORTCUTS)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    background = int(rng.integers(10))
    varied_background = bool(rng.random() < .7)
    constant_size = bool(rng.random() < .25)
    # The seven-token contrast needs four source-class slots; nine-high tiles
    # remain useful for ordinary, smaller examples but cannot hold this witness.
    witness_shapes = [s for s in _SHAPES if 4 * s[0] <= 30 and _packing_options(s, 7)]
    common_size = witness_shapes[int(rng.integers(len(witness_shapes)))]
    primary = int(rng.integers(2))
    modes = ["address", "holes"]
    rng.shuffle(modes)
    for _ in range(96):
        train = []
        for i, mode in enumerate(modes):
            size = common_size if constant_size else witness_shapes[int(rng.integers(len(witness_shapes)))]
            bg = background if not varied_background or i == 0 else int(rng.choice(
                [c for c in range(10) if c != background]))
            role = primary if mode == "address" else 1 - primary
            train.append(_make_pair(rng, size, _witness_relations(rng, size, role, mode), bg))
        if _training_suffices(train) and _key(train[0]['input']) != _key(train[1]['input']):
            break
    else:
        raise RuntimeError("bounded sampling could not expose all ordering contrasts")
    puzzle = dict(train=train, test=[])
    seen = {_key(p['input']) for p in train}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for i in range(count):
            for _ in range(64):
                depth = split == "test" and (i == 0 or rng.random() < .4)
                if depth:
                    choices = [s for s in _SHAPES if s[0] <= 5 and _packing_options(s, 9)]
                    size = common_size if constant_size and common_size in choices else choices[int(rng.integers(len(choices)))]
                    relations = _depth_relations(rng, size, int(rng.integers(2)))
                else:
                    size = common_size if constant_size else _SHAPES[int(rng.integers(len(_SHAPES)))]
                    relations = _random_relations(rng, size)
                bg = int(rng.integers(10)) if varied_background else background
                pair = _make_pair(rng, size, relations, bg)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded sampling could not make a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _parse(inp)
            links = _links(a)
            assert links, "no successful address"
            expected = _render(a, links)
            assert np.array_equal(out, expected), "lookup, shape copying, or padding mismatch"
            assert not np.array_equal(inp, out)
            f = _features(a)
            height, width = a['size']
            assert out.shape == (max(f['survivors']) * height, 2 * width)
            # Explicitly inspect padding, survivor bodies, and unchanged geometry.
            for kind, queue in enumerate(_queues(a, subset=set(links))):
                assert np.all(out[len(queue) * height:, kind * width:(kind + 1) * width] == a['background'])
                for rank, index in enumerate(queue):
                    t = a['tokens'][index]
                    patch = out[rank * height:(rank + 1) * height, kind * width:(kind + 1) * width]
                    assert np.array_equal(patch, _tile(height, width, kind, t['body'],
                                                      a['tokens'][links[index]]['value'], a['background']))
            features[split].append(f)
    train = features['train']
    for feature in ('conflicting_values', 'shared_body', 'early_failure', 'per_body',
                    'deleted_donor', 'changing_donor', 'monochrome', 'unchanged'):
        assert any(f[feature] for f in train), f"missing training witness: {feature}"
    signs = {int(np.sign(f['survivors'][1] - f['survivors'][0])) for f in train}
    assert {-1, 1} <= signs, "both padding directions need demonstrations"
    assert _training_suffices(puzzle['train']), "a documented shortcut still fits training"
    assert max(f['max_success_rank'] for f in features['test']) > max(
        f['max_success_rank'] for f in train), "missing successful-rank extrapolation"


def self_test():
    """Bounded 42-seed checks; return small, descriptive structural measurements."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, shapes, layouts, palettes, sizes, counts, survivors, depths = [], set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _parse(pair['input'])
                f = _features(a)
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
                shapes.add(pair['input'].shape)
                sizes.add(a['size'])
                palettes.add(a['background'])
                counts.add(f['counts'])
                survivors.add(f['survivors'])
                depths.add(f['max_success_rank'])
                # Translation-normalized geometry, without literal palette data.
                mr = min(t['top'] for t in a['tokens'])
                mc = min(t['left'] for t in a['tokens'])
                layouts.add((a['size'], tuple((t['top'] - mr, t['left'] - mc, t['kind']) for t in a['tokens'])))
                records.append(f)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 50 and len(layouts) >= 180
    assert set(_SHAPES) <= sizes and palettes == set(range(10))
    assert len(counts) >= 20 and len(survivors) >= 15 and {1, 2, 3, 4, 5} <= depths
    assert any(f['monochrome'] for f in records) and any(f['unchanged'] for f in records)
    assert any(f['cross_tie'] for f in records)
    assert any(0 in f['survivors'] for f in records)
    assert any(p['input'].shape == (30, 30) for puzzle in puzzles for s in ('train', 'test') for p in puzzle[s])
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(101, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject actual corruption of output contents and input centers.
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    inp = puzzles[0]['train'][0]['input'].copy()
    a = _parse(inp)
    t = a['tokens'][0]
    inp[t['top'] + a['size'][0] // 2, t['left'] + a['size'][1] // 2] = a['background']
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("missing center accepted")
    # The solver also handles valid dimensions outside the sampling vocabulary,
    # nonblack backgrounds, monochrome results, and opposite-class top-row ties.
    for height, width in ((11, 13), (27, 3), (3, 13)):
        grid = np.full((height + 2, 2 * width + 3), 8, dtype=int)
        grid[1:height + 1, 1:width + 1] = _tile(height, width, 0, 0, 2, 8)
        grid[1:height + 1, width + 2:2 * width + 2] = _tile(height, width, 1, 2, 0, 8)
        expected = np.concatenate((_tile(height, width, 0, 0, 0, 8),
                                   _tile(height, width, 1, 2, 2, 8)), axis=1)
        assert np.array_equal(solve(grid), expected)
    tied = np.full((9, 21), 0, dtype=int)
    tied[1:4, 1:6] = _tile(3, 5, 0, 1, 2, 0)
    tied[1:4, 8:13] = _tile(3, 5, 0, 1, 2, 0)
    tied[5:8, 15:20] = _tile(3, 5, 1, 2, 1, 0)
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("same-class tie accepted")
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes), layouts=len(layouts),
                tile_shapes=sorted(sizes), backgrounds=sorted(palettes), count_patterns=len(counts),
                survivor_patterns=len(survivors), successful_ranks=sorted(depths),
                semantic_verification="generator-owned invariants and finite shortcut comparisons only")
