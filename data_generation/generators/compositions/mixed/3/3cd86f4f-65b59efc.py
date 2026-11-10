"""Keyed template tiling, followed by bottom-anchored whole-image row shear.

Only the standard library and NumPy are required. No parent code, files, seed,
or demonstration state is consulted by solve(). See task_spec.md for the grammar.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 rectangular integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()).hexdigest()


def _turn(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key_offsets(n):
    return (0, 1) if n == 2 else (1,)


def _parse(value):
    """Read the fixed upright grammar, including arbitrary blank outer margins."""
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("a complete background border is required")
    rr, cc = np.nonzero(a != bg)
    if not len(rr):
        raise ValueError("empty encoding")
    top, left = int(rr.min()), int(cc.min())
    bottom, right = int(rr.max()) + 1, int(cc.max()) + 1
    core = a[top:bottom, left:right]
    height, width = core.shape
    n = (height - 3) // 2
    if n not in (2, 3) or height != 2*n + 3 or (width + 1) % (n + 1):
        raise ValueError("invalid panel scale or registration")
    k = (width + 1) // (n + 1)
    if not 2 <= k <= min(n*n, 7):
        raise ValueError("invalid library count")
    sep = int(core[n, 0])
    if sep == bg or not np.all(core[[n, 2*n+1]] == sep):
        raise ValueError("horizontal separators must be solid and share one color")
    for c in range(n, width, n+1):
        if not np.all(core[:2*n+2, c] == sep):
            raise ValueError("broken panel separator")

    upper, lower, templates, targets, layers = [], [], {}, {}, {}
    expected_key = np.full(width, bg, dtype=np.int8)
    occupied = np.zeros((n, n), dtype=bool)
    assignment = np.full((n, n), -1, dtype=np.int8)
    for j in range(k):
        c = j*(n+1)
        panel = core[:n, c:c+n]
        colors = np.unique(panel[panel != bg])
        if len(colors) != 1 or int(colors[0]) == sep or int(colors[0]) in templates:
            raise ValueError("each upper panel needs a unique nonstructural label")
        label = int(colors[0])
        upper.append(label)
        templates[label] = panel == label
        swatch = core[-1, [c+d for d in _key_offsets(n)]]
        if not np.all(swatch == swatch[0]) or int(swatch[0]) == bg:
            raise ValueError("a target swatch must be visible and uniform")
        targets[label] = int(swatch[0])
        expected_key[[c+d for d in _key_offsets(n)]] = swatch[0]

        panel = core[n+1:2*n+1, c:c+n]
        colors = np.unique(panel[panel != bg])
        if len(colors) != 1 or int(colors[0]) == sep or int(colors[0]) in layers:
            raise ValueError("each lower panel needs one unique visible identity")
        identity = int(colors[0])
        mask = panel == identity
        if np.any(occupied & mask):
            raise ValueError("placement layers must not overlap")
        occupied |= mask
        assignment[mask] = identity
        lower.append(identity)
        layers[identity] = mask
    if set(upper) != set(lower) or not np.array_equal(core[-1], expected_key):
        raise ValueError("unmatched labels or malformed key row")
    return dict(n=n, k=k, background=bg, separator=sep, upper=tuple(upper),
                lower=tuple(lower), templates=templates, targets=targets,
                layers=layers, assignment=assignment,
                margins=(top, left, a.shape[0]-bottom, a.shape[1]-right))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute final pixel coordinates directly from the visible input."""
    p = _parse(input_grid)
    n, bg = p['n'], p['background']
    side = n*n
    answer = np.full((side, 2*side-1), bg, dtype=np.int8)
    for label in p['upper']:
        tr, tc = np.nonzero(p['templates'][label])
        for mr, mc in zip(*np.nonzero(p['layers'][label])):
            rows = int(mr)*n + tr
            cols = int(mc)*n + tc + side-1-rows
            answer[rows, cols] = p['targets'][label]
    return answer


def _blocks(p, correspondence='identity', recolor=True, stamp_turn=0,
            layer_turn=0, filled=False, first_only=False, exchange=False):
    """Block reconstruction for validation and explicit wrong-rule witnesses."""
    n, bg = p['n'], p['background']
    a = np.full((n*n, n*n), bg, dtype=np.int8)
    upper, lower = p['upper'], p['lower']
    start = upper.index(lower[0])
    for j, label in enumerate(lower):
        if first_only and j:
            continue
        chosen = (label if correspondence == 'identity' else upper[j]
                  if correspondence == 'position' else upper[(start+j) % p['k']])
        mask, layer = p['templates'][chosen], p['layers'][label]
        if exchange:
            mask, layer = p['layers'][chosen], p['templates'][label]
        mask, layer = _turn(mask, stamp_turn), _turn(layer, layer_turn)
        if filled:
            mask = np.ones_like(mask)
        tile = np.where(mask, p['targets'][chosen] if recolor else chosen, bg)
        for r, c in zip(*np.nonzero(layer)):
            a[r*n:(r+1)*n, c*n:(c+1)*n] = tile
    return a


def _shear(a, bg, kind='correct', n=None):
    h, w = a.shape
    result = np.full((h, h+w-1), bg, dtype=np.int8)
    for r in range(h):
        shift = (h-1-r if kind == 'correct' else 0 if kind == 'none'
                 else r if kind == 'top' else n*(n-1-r//n) if kind == 'macro'
                 else n-1-r//n if kind == 'macro_one' else n-1-r % n)
        result[r, shift:shift+w] = a[r]
    return result


def _teaches(p):
    """Evidence checks concern actual answers, not hidden sampler flags."""
    n, bg = p['n'], p['background']
    a = _blocks(p)
    if not (np.any(a[0] != bg) and np.any(a[-1] != bg)
            and any(np.sum(row != bg) >= 2 for row in a)
            and np.sum(p['assignment'] != -1) > p['k']):
        return False
    order = [p['upper'].index(x) for x in p['lower']]
    if all(x == (order[0]+i) % p['k'] for i, x in enumerate(order)):
        return False
    wrong_blocks = [dict(correspondence='position'), dict(correspondence='cyclic'),
                    dict(recolor=False), dict(filled=True), dict(first_only=True),
                    dict(exchange=True)]
    wrong_blocks += [dict(stamp_turn=t) for t in range(1, 8)]
    wrong_blocks += [dict(layer_turn=t) for t in range(1, 8)]
    if any(np.array_equal(a, _blocks(p, **kw)) for kw in wrong_blocks):
        return False
    correct = _shear(a, bg)
    return all(not np.array_equal(correct, _shear(a, bg, kind, n))
               for kind in ('none', 'top', 'macro', 'macro_one', 'local'))


def _motif(rng, n):
    r, c = np.indices((n, n))
    style = int(rng.integers(10))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (r == 0) | (r == n-1) | (c == 0) | (c == n-1)
    elif style == 2:
        mask = r == c
    elif style == 3:
        mask = (r == n//2) | (c == n//2)
    elif style == 4:
        mask = (r+c) % 2 == int(rng.integers(2))
    elif style == 5:
        mask = r == int(rng.integers(n))
    elif style == 6:
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[int(rng.integers(n*n))] = True
    elif style == 7:
        r0, c0 = map(int, rng.integers(n, size=2))
        mask = (r >= r0) & (c >= c0)
    else:
        mask = rng.random((n, n)) < rng.uniform(.15, .90)
    if not np.any(mask):
        mask.flat[int(rng.integers(n*n))] = True
    return _turn(mask, int(rng.integers(8)))


def _assignment(rng, n, labels, count):
    coords = [(r, c) for r in range(n) for c in range(n)]
    style = int(rng.integers(5))
    if style == 0:
        positions = rng.permutation(n*n)
    else:
        center = tuple(map(int, rng.integers(n, size=2)))
        def rank(q):
            r, c = q
            if style == 1:
                return r*n + (c if r % 2 == 0 else n-1-c)
            if style == 2:
                return (r+c, r)
            if style == 3:
                return (abs(r-center[0])+abs(c-center[1]), r, c)
            return (min(r, c, n-1-r, n-1-c), r, c)
        positions = np.array([r*n+c for r, c in sorted(coords, key=rank)])
    weights = rng.dirichlet(np.full(len(labels), .6))
    counts = 1 + rng.multinomial(count-len(labels), weights)
    identities = np.repeat(rng.permutation(labels), counts)
    if style == 0:
        rng.shuffle(identities)
    a = np.full((n, n), -1, dtype=np.int8)
    a.flat[positions[:count]] = identities
    return _turn(a, int(rng.integers(8)))


def _encode(rng, p):
    n, k, bg, sep = p['n'], p['k'], p['background'], p['separator']
    width = k*(n+1)-1
    core = np.full((2*n+3, width), bg, dtype=np.int8)
    for j, label in enumerate(p['upper']):
        c = j*(n+1)
        core[:n, c:c+n] = np.where(p['templates'][label], label, bg)
        core[-1, [c+d for d in _key_offsets(n)]] = p['targets'][label]
    for j, label in enumerate(p['lower']):
        c = j*(n+1)
        core[n+1:2*n+1, c:c+n] = np.where(p['assignment'] == label, label, bg)
    core[[n, 2*n+1], :] = sep
    for c in range(n, width, n+1):
        core[:2*n+2, c] = sep
    top, bottom = map(int, rng.integers(1, 4, size=2))
    left = int(rng.integers(1, min(3, 29-width)+1))
    right = int(rng.integers(1, min(3, 30-width-left)+1))
    return np.pad(core, ((top, bottom), (left, right)), constant_values=bg)


def _make_pair(rng, n, k, density, anchor=False, fallback=False):
    colors = list(map(int, rng.permutation(10)))
    bg, sep, labels = colors[0], colors[1], colors[2:2+k]
    masks = [_motif(rng, n) for _ in labels]
    if not anchor and rng.random() < .23:
        a, b = map(int, rng.choice(k, 2, replace=False))
        masks[b] = masks[a].copy()
    pool = [x for x in range(10) if x != bg]
    if anchor:
        targets = list(map(int, rng.choice(pool, k, replace=False)))
        # Repair only the informative anchor's key; ordinary keys are unrestricted.
        if any(a == b for a, b in zip(labels, targets)):
            targets = labels[1:] + labels[:1]
    else:
        targets = list(map(int, rng.choice(pool, k, replace=True)))
    if anchor:
        lower = [labels[0], labels[2], labels[1]]
        count = n*n if n == 2 else int(rng.integers(k+1, n*n))
    else:
        mode = int(rng.integers(4))
        lower = (labels.copy() if mode == 0 else np.roll(labels, 1).tolist()
                 if mode == 1 else list(map(int, rng.permutation(labels))))
        count = (k if density == 'sparse' else n*n if density == 'full'
                 else int(rng.integers(k, n*n if density == 'moderate' else n*n+1)))
    assignment = _assignment(rng, n, labels, count)
    if fallback:
        # Rare bounded construction fallback, only for the three-label anchor.
        if n == 2:
            codes = [1, 3, 14]
            assignment = np.array([[labels[0], labels[1]], [labels[2], labels[1]]], dtype=np.int8)
        else:
            codes = [19, 356, 325]
            assignment = np.array([[labels[0], labels[0], labels[1]],
                                   [-1, labels[2], -1], [labels[1], -1, labels[2]]], dtype=np.int8)
        masks = [np.array([(code >> i) & 1 for i in range(n*n)], dtype=bool).reshape(n, n)
                 for code in codes]
    p = dict(n=n, k=k, background=bg, separator=sep, upper=tuple(labels), lower=tuple(lower),
             templates=dict(zip(labels, masks)), targets=dict(zip(labels, targets)), assignment=assignment)
    # Ground truth constructs the latent payload independently of solve's pixel formula.
    payload = np.full((n*n, n*n), bg, dtype=np.int8)
    for label, mask, target in zip(labels, masks, targets):
        payload[np.kron(assignment == label, mask)] = target
    return {'input': _encode(rng, p), 'output': _shear(payload, bg)}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    a, b = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0 or not 2 <= a <= 10 or not 1 <= b <= 4 or a <= b:
        raise ValueError("require seed >= 0, 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    if regime == 0:  # Familiar compact scale -> larger scale and more possible uses.
        training = [(2, 3, 'full'), (2, 2, 'sparse')]
        training += [(2, int(rng.integers(2, 4)), 'any') for _ in range(a-2)]
        tests = [(3, int(rng.integers(3, 8)), 'any') for _ in range(b)]
    elif regime == 1:  # Familiar scale, larger keyed library.
        training = [(3, 3, 'moderate'), (3, 2, 'sparse')]
        training += [(3, int(rng.integers(2, 4)), 'any') for _ in range(a-2)]
        tests = [(3, int(rng.integers(4, 8)), 'any') for _ in range(b)]
    else:  # Both scales demonstrated; extrapolate placement density.
        training = [(3, 3, 'moderate'), (2, 2, 'sparse')]
        for _ in range(a-2):
            n = int(rng.integers(2, 4))
            training.append((n, int(rng.integers(2, min(4, n*n))), 'moderate'))
        tests = []
        for _ in range(b):
            n = int(rng.integers(2, 4))
            tests.append((n, int(rng.integers(2, min(7, n*n)+1)), 'full'))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', training), ('test', tests)):
        for i, (n, k, density) in enumerate(schedule):
            anchor = split == 'train' and i == 0
            for attempt in range(96):
                pair = _make_pair(rng, n, k, density, anchor, fallback=anchor and attempt >= 64)
                if anchor and not _teaches(_parse(pair['input'])):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            p = _parse(inp)
            n, bg = p['n'], p['background']
            side = n*n
            assert out.shape == (side, 2*side-1)
            payload = _blocks(p)
            for r in range(side):
                start = side-1-r
                assert np.all(out[r, :start] == bg)
                assert np.all(out[r, start+side:] == bg)
                assert np.array_equal(out[r, start:start+side], payload[r])
            assert np.array_equal(solve(inp), out)
            # Counts must survive both substitution and shear, even with shared targets.
            expected_counts = {}
            for label in p['upper']:
                target = p['targets'][label]
                count = int(np.sum(p['layers'][label])) * int(np.sum(p['templates'][label]))
                expected_counts[target] = expected_counts.get(target, 0) + count
            assert set(map(int, np.unique(out))) <= {bg, *p['targets'].values()}
            for target, count in expected_counts.items():
                assert int(np.sum(out == target)) == count
            if split == 'train':
                training.append(p)
    assert any(_teaches(p) for p in training), "insufficient counterfactual training evidence"
    assert any(np.any(p['assignment'] == -1) for p in training), "demonstrate empty macro-cells"


def self_test():
    """Stream 42 seeds, all count combinations, and bounded semantic mutations."""
    shapes, motifs, layouts, roles, signatures, ns, ks = set(), set(), set(), set(), set(), set(), set()
    pairs = sparse = full = duplicate_motifs = shared_targets = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                n, k = p['n'], p['k']
                ns.add(n)
                ks.add(k)
                shapes.add(pair['input'].shape)
                roles.add((p['background'], p['separator']))
                motifs.update((n, mask.tobytes()) for mask in p['templates'].values())
                normalized = np.full((n, n), -1, dtype=np.int8)
                for j, label in enumerate(p['upper']):
                    normalized[p['assignment'] == label] = j
                layouts.add(_key(normalized))
                count = int(np.sum(normalized != -1))
                sparse += count == k
                full += count == n*n
                duplicate_motifs += len({m.tobytes() for m in p['templates'].values()}) < k
                shared_targets += len(set(p['targets'].values())) < k
                pairs += 1
                signature.append((_digest(pair['input']), _digest(pair['output'])))
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        signatures.add(tuple(signature))
    assert len(signatures) == 42 and ns == {2, 3}
    assert {2, 3, 4, 5} <= ks and len(shapes) >= 15
    assert len(motifs) >= 40 and len(layouts) >= 60 and len(roles) >= 25
    assert min(sparse, full, duplicate_motifs, shared_targets) >= 5
    # solve also covers legal symmetric/degenerate scenes without teaching quotas.
    for n in (2, 3):
        rng = np.random.default_rng(400+n)
        for k in range(2, min(n*n, 7)+1):
            for density in ('sparse', 'full'):
                pair = _make_pair(rng, n, k, density)
                assert np.array_equal(solve(pair['input']), pair['output'])
        fallback = _make_pair(rng, n, 3, 'any', anchor=True, fallback=True)
        assert _teaches(_parse(fallback['input']))
        # All silhouettes and targets can coincide; identity remains syntactically visible.
        p = _parse(fallback['input'])
        mask = np.zeros((n, n), dtype=bool)
        mask[-1, -1] = True
        p['templates'] = {label: mask.copy() for label in p['upper']}
        p['targets'] = dict.fromkeys(p['upper'], p['separator'])
        scene = _encode(rng, p)
        assert np.array_equal(solve(scene), _shear(_blocks(p), p['background']))
    first, again = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s])
               for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(42, a, b)
            assert (len(q['train']), len(q['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Bad answers, duplicate inputs, overlaps, missing identities, invisible swatches.
    bad = {s: [dict(input=p['input'].copy(), output=p['output'].copy()) for p in first[s]]
           for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("wrong answer accepted")
    bad['test'][0] = {k: v.copy() for k, v in first['train'][0].items()}
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("duplicate input accepted")
    anchor = first['train'][0]['input']
    p = _parse(anchor)
    top, left = p['margins'][:2]
    n = p['n']
    for mutation in ('overlap', 'empty', 'key', 'border'):
        a = anchor.copy()
        if mutation == 'border':
            a[0, 0] = p['separator']
        elif mutation == 'key':
            a[top+2*n+2, left+np.array(_key_offsets(n))] = p['background']
        elif mutation == 'empty':
            a[top+n+1:top+2*n+1, left:left+n] = p['background']
        else:
            other = p['lower'][1]
            r, c = map(int, np.argwhere(p['layers'][other])[0])
            a[top+n+1+r, left+c] = p['lower'][0]
        try:
            solve(a)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed input accepted: " + mutation)
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), template_bitmaps=len(motifs),
                normalized_macro_layouts=len(layouts), structural_palettes=len(roles),
                panel_sides=sorted(ns), library_counts=sorted(ks), single_use_pairs=sparse,
                full_pairs=full, duplicate_silhouette_pairs=duplicate_motifs,
                shared_target_pairs=shared_targets)
