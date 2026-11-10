"""Periodic, position-addressed deflector walls with selective error repair.

Only Python's standard library and NumPy are used. No generation metadata is
needed by solve(), and no files are written by this module.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


_D = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 64004041, 518655911, 673028137, 531533865, 724008658,
              184370641, 77682818, 549111335, 730397570, 973378289,
              267779501, 832647938, 363925043, 417720569, 228842192,
              261451769, 824753657, 436382529, 217524378, 497854660,
              567495472, 309137715, 863810307, 838347615, 988040771,
              503228068, 828172980, 616600832, 407926014, 832648890,
              800366163, 588353768, 89712024, 750913233, 952687448,
              47953647, 348408211, 256818627, 400365728, 933726134, 698744231)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9)),
             "grid must contain integer colors 0..9")
    return g.astype(np.int8, copy=False)


def _repair(bits):
    """All supported strict-majority periodic explanations must agree."""
    bits = np.asarray(bits, dtype=np.int8)
    n = len(bits)
    candidates = []
    for p in range(1, n // 3 + 1):
        word = []
        errors = 0
        for phase in range(p):
            values = bits[phase::p]
            ones = int(values.sum())
            zeros = len(values) - ones
            if ones == zeros:
                break
            word.append(int(ones > zeros))
            errors += min(ones, zeros)
        else:
            if errors <= max(2, n // 8) and len(set(word)) == 2:
                clean = np.asarray(word, dtype=np.int8)[np.arange(n) % p]
                candidates.append((p, clean))
    _require(bool(candidates), "no supported nonconstant periodic code")
    p, answer = candidates[0]
    _require(all(np.array_equal(answer, x[1]) for x in candidates),
             "ambiguous periodic repair")
    return answer, p


def _components(g, background):
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    result = []
    for x, y in np.argwhere(g != background):
        r, c = int(x), int(y)
        if seen[r, c]:
            continue
        todo = [(r, c)]
        seen[r, c] = True
        cells = []
        while todo:
            a, b = todo.pop()
            cells.append((a, b))
            for dr, dc in _D:
                rr, cc = a + dr, b + dc
                if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                        and g[rr, cc] != background):
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(cells)
    return result


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    background = int(g[0, 0])
    _require(all(int(g[r, c]) == background for r, c in
                 ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "corners must be background")
    sources = []
    rails = []
    for cells in _components(g, background):
        if len(cells) == 1:
            sources.append(cells[0])
            continue
        a, b = np.min(cells, axis=0)
        e, f = np.max(cells, axis=0) + 1
        a, b, e, f = map(int, (a, b, e, f))
        rh, rw = e - a, f - b
        _require(len(cells) == rh * rw and min(rh, rw) == 3 and max(rh, rw) >= 14,
                 "deflectors must be solid three-cell-wide rectangular components")
        _require(a >= 1 and b >= 1 and e < h and f < w, "wall needs an exterior margin")
        block = g[a:e, b:f]
        frame = int(block[0, 0])
        _require(np.all(block[0] == frame) and np.all(block[-1] == frame)
                 and np.all(block[:, 0] == frame) and np.all(block[:, -1] == frame),
                 "wall has an inconsistent outline")
        axis = 0 if rh == 3 else 1  # horizontal / vertical long axis
        stripe = block[1, 1:-1] if axis == 0 else block[1:-1, 1]
        accents = set(map(int, np.unique(stripe))) - {frame}
        _require(len(accents) == 1, "wall stripe must use outline and one accent")
        accent = next(iter(accents))
        raw = (stripe == accent).astype(np.int8)
        clean, period = _repair(raw)
        rails.append(dict(r=a, c=b, axis=axis, frame=frame, accent=accent,
                          raw=raw, code=clean, period=period))
    _require(len(sources) == 1 and 1 <= len(rails) <= 5, "need one point source and 1..5 walls")
    source = sources[0]
    source_color = int(g[source])
    frames = {z['frame'] for z in rails}
    accents = {z['accent'] for z in rails}
    _require(len(frames) == 1 and len(accents) == len(rails), "outlines common; wall accents distinct")
    frame = next(iter(frames))
    _require(len({background, source_color, frame}) == 3
             and not accents & {background, source_color, frame}, "semantic colors overlap")
    # Separation also makes point-source identification and side collisions clear.
    labels = np.full(g.shape, -1, dtype=np.int8)
    for j, z in enumerate(rails):
        a, b, e, f = _box(z)
        labels[a:e, b:f] = j
    labels[source] = len(rails)
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
        aa = labels[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
        bb = labels[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
        _require(not np.any((aa >= 0) & (bb >= 0) & (aa != bb)), "objects touch")
    return dict(grid=g, background=background, source=source, source_color=source_color,
                frame=frame, rails=rails, owner=labels)


def _box(z):
    r, c, n = z['r'], z['c'], len(z['code'])
    return (r, c, r + 3, c + n + 2) if z['axis'] == 0 else (r, c, r + n + 2, c + 3)


def _code_cell(z, index):
    return (z['r'] + 1, z['c'] + 1 + index) if z['axis'] == 0 else (z['r'] + 1 + index, z['c'] + 1)


def _trace(info, codes=None, mapping=(0, 1), first_only=False):
    """Route four independent rays; read the *input* walls, never painted paths."""
    g, rails, owner = info['grid'], info['rails'], info['owner']
    h, w = g.shape
    if codes is None:
        codes = [z['code'] for z in rails]
    mask = np.zeros_like(g, dtype=bool)
    queries = [set() for _ in rails]
    events, lengths = [], []
    for branch in range(4):
        r, c = info['source']
        d, turns, seen = branch, 0, set()
        while True:
            state = (r, c, d)
            _require(state not in seen, "cyclic ray")
            seen.add(state)
            dr, dc = _D[d]
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            j = int(owner[rr, cc])
            if 0 <= j < len(rails):
                z = rails[j]
                _require(z['axis'] == d % 2, "ray hits a wall end cap")
                index = c - z['c'] - 1 if z['axis'] == 0 else r - z['r'] - 1
                _require(0 <= index < len(z['code']), "ray hits a wall corner")
                if first_only and turns:
                    break
                bit = int(codes[j][index])
                queries[j].add(index)
                events.append((branch, j, index, bit, d))
                d = (d + (-1 if mapping[bit] == 0 else 1)) % 4
                turns += 1
            else:
                if g[rr, cc] == info['background']:
                    mask[rr, cc] = True
                r, c = rr, cc
        lengths.append(turns)
    return dict(mask=mask, queries=queries, events=events, lengths=lengths)


def _apply(info, route):
    out = info['grid'].copy()
    out[route['mask']] = info['source_color']
    for z, indices in zip(info['rails'], route['queries']):
        for index in indices:
            out[_code_cell(z, index)] = z['accent'] if z['code'][index] else z['frame']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, periodic code and encounter directly from the input."""
    info = _decode(input_grid)
    return _apply(info, _trace(info))


def _has_mixed_wall(route):
    return any(len({e[3] for e in route['events'] if e[1] == j}) == 2
               for j in {e[1] for e in route['events']})


def _constant_wall_fits(info, expected):
    """Ablation: even a separately chosen fixed turn for each wall must fail."""
    for bits in itertools.product((0, 1), repeat=len(info['rails'])):
        codes = [np.full(len(z['code']), bit, dtype=np.int8) for z, bit in zip(info['rails'], bits)]
        try:
            route = _trace(info, codes)
        except ValueError:
            continue
        if np.array_equal(_apply(info, route), expected):
            return True
    return False


def _primitive_word(rng, p):
    for _ in range(32):
        word = rng.integers(0, 2, p, dtype=np.int8)
        if len(set(map(int, word))) == 2 and not any(
                p % q == 0 and np.array_equal(word, np.tile(word[:q], p // q))
                for q in range(1, p)):
            return word
    word = np.zeros(p, dtype=np.int8)
    word[-1] = 1
    return word


def _render(shape, rails, source):
    g = np.zeros(shape, dtype=np.int8)
    g[source] = 1
    for j, z in enumerate(rails):
        a, b, e, f = _box(z)
        g[a:e, b:f] = 2
        values = np.where(z['code'], j + 3, 2)
        if z['axis'] == 0:
            g[a + 1, b + 1:f - 1] = values
        else:
            g[a + 1:e - 1, b + 1] = values
    return g


def _info_for_construction(shape, rails, source):
    g = _render(shape, rails, source)
    owner = np.full(shape, -1, dtype=np.int8)
    local = []
    for j, z in enumerate(rails):
        a, b, e, f = _box(z)
        owner[a:e, b:f] = j
        local.append(dict(z, raw=z['code'].copy(), frame=2, accent=j + 3))
    return dict(grid=g, background=0, source=source, source_color=1,
                frame=2, rails=local, owner=owner)


def _random_layout(rng, test, strong, profile):
    low = 24 if test else profile
    h, w = map(int, rng.integers(low, 31, size=2))
    target = int(rng.integers(2 if strong or test else 1, 6))
    occupied = np.zeros((h, w), dtype=bool)
    rails = []
    for _ in range(target * 24):
        if len(rails) >= target:
            break
        axis = int(rng.integers(2))
        cap = (w if axis == 0 else h) - 4
        choices = [p for p in (range(4, 6) if test and not rails else range(2, 6 if test else 4))
                   if 4 * p <= cap]
        if not choices:
            continue
        p = int(rng.choice(choices))
        n = int(rng.integers(max(12, 4 * p), cap + 1))
        rh, rw = (3, n + 2) if axis == 0 else (n + 2, 3)
        r, c = int(rng.integers(1, h - rh)), int(rng.integers(1, w - rw))
        if np.any(occupied[r - 1:r + rh + 1, c - 1:c + rw + 1]):
            continue
        code = _primitive_word(rng, p)[np.arange(n) % p]
        rails.append(dict(r=r, c=c, axis=axis, code=code, period=p))
        occupied[r:r + rh, c:c + rw] = True
    if len(rails) < (2 if strong or test else 1):
        return
    padded = np.pad(occupied, 1, constant_values=True)
    free = ~np.logical_or.reduce([padded[dr:dr + h, dc:dc + w]
                                 for dr in range(3) for dc in range(3)])
    sources = np.argwhere(free)
    if not len(sources):
        return
    for index in rng.permutation(len(sources))[:12]:
        source = tuple(map(int, sources[index]))
        info = _info_for_construction((h, w), rails, source)
        try:
            route = _trace(info)
        except ValueError:
            continue
        if not route['events']:
            continue
        if strong and (not _has_mixed_wall(route) or max(route['lengths']) < 2):
            continue
        if test and not any(rails[e[1]]['period'] >= 4 for e in route['events']):
            continue
        yield info, route


def _add_noise(rng, info, route, strong):
    rails = info['rails']
    options = []
    for j, (z, indices) in enumerate(zip(rails, route['queries'])):
        p = z['period']
        for hit in sorted(indices):
            misses = [i for i in range(len(z['code'])) if i not in indices and i % p != hit % p]
            if strong:
                queried_phases = {i % p for i in indices}
                misses = [i for i in misses if i % p in queried_phases]
            if misses:
                options.append((j, hit, misses))
    if not options:
        return None
    clean_answer = _apply(info, route)
    if strong:
        if np.array_equal(_trace(info, first_only=True)['mask'], route['mask']):
            return None
        if _constant_wall_fits(info, clean_answer):
            return None
        try:
            reversed_route = _trace(info, mapping=(1, 0))
        except ValueError:
            pass
        else:
            if np.array_equal(reversed_route['mask'], route['mask']):
                return None
    for _ in range(28):
        primary, hit, misses = options[int(rng.integers(len(options)))]
        noisy = info['grid'].copy()
        for j, z in enumerate(rails):
            if j == primary:
                flips = [hit, int(rng.choice(misses))]
            else:
                flips = [int(rng.integers(len(z['code'])))] if rng.random() < .65 else []
            raw = z['code'].copy()
            raw[flips] ^= 1
            try:
                repaired, p = _repair(raw)
            except ValueError:
                break
            if p != z['period'] or not np.array_equal(repaired, z['code']):
                break
            for i, bit in enumerate(raw):
                noisy[_code_cell(z, i)] = z['accent'] if bit else z['frame']
        else:
            local = dict(info, grid=noisy)
            raw_codes = [np.array([int(noisy[_code_cell(z, i)] == z['accent'])
                                  for i in range(len(z['code']))], dtype=np.int8) for z in rails]
            try:
                raw_route = _trace(local, raw_codes)
            except ValueError:
                continue
            if np.array_equal(raw_route['mask'], route['mask']):
                continue
            return dict(input=noisy, output=_apply(local, route))
    return None


def _fallback_layouts(rng, test):
    """Parametric two-wall construction; used only if random packing is exhausted.

    The long upper wall is read at two different phases, after one branch first
    meets a perpendicular wall. Dimensions, code words and addresses are sampled.
    """
    for _ in range(160):
        h, w = map(int, rng.integers(26, 31, size=2))
        r, c = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        p = int(rng.choice([4, 5] if test else [2, 3]))
        n = int(rng.integers(max(16, 4 * p), w - c - 2))
        code = _primitive_word(rng, p)[np.arange(n) % p]
        vr = r + 4
        vp = int(rng.choice([2, 3]))
        vn = int(rng.integers(12, h - vr - 2))
        vcode = _primitive_word(rng, vp)[np.arange(vn) % vp]
        addresses = [(a, b) for a in range(1, n - 4) for b in range(a + 3, n)
                     if code[a] == 0 and code[b] == 1]
        vzeros = np.flatnonzero(vcode == 0)
        if not addresses or not len(vzeros):
            continue
        a, b = addresses[int(rng.integers(len(addresses)))]
        vc = c + b + 2
        if vc + 3 >= w:
            continue
        source = (vr + 1 + int(rng.choice(vzeros)), c + 1 + a)
        rails = [dict(r=r, c=c, axis=0, code=code, period=p),
                 dict(r=vr, c=vc, axis=1, code=vcode, period=vp)]
        info = _info_for_construction((h, w), rails, source)
        try:
            route = _trace(info)
        except ValueError:
            continue
        yield info, route


def _make_pair(rng, test, strong, profile):
    for _ in range(240 if strong else 80):
        for info, route in _random_layout(rng, test, strong, profile):
            pair = _add_noise(rng, info, route, strong)
            if pair is not None:
                return pair
    for info, route in _fallback_layouts(rng, test):
        pair = _add_noise(rng, info, route, strong)
        if pair is not None:
            return pair
    raise RuntimeError("bounded construction exhausted")


def _headings(info, route):
    return [{e[4] for e in route['events'] if e[3] == bit} for bit in (0, 1)]


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_palette = rng.permutation(10)
    profile = int(rng.choice([19, 22, 24]))
    puzzle = {'train': [], 'test': []}
    seen, heading_sets, train_periods = set(), [set(), set()], set()
    for serial in range(nt + nq):
        split = 'train' if serial < nt else 'test'
        test, strong = split == 'test', serial < 2
        for _ in range(16):
            pair = _make_pair(rng, test, strong, profile)
            info = _decode(pair['input'])
            route = _trace(info)
            periods = {info['rails'][e[1]]['period'] for e in route['events']}
            if serial == 1 and not {2, 3} <= train_periods | periods:
                continue
            rotations = list(map(int, rng.permutation(4)))
            if serial == 1:
                current = _headings(info, route)
                rotations.sort(key=lambda k: -sum(min(2, len(heading_sets[b] |
                                 {(d - k) % 4 for d in current[b]})) for b in (0, 1)))
            k = rotations[0]
            palette = np.roll(base_palette, serial % 10).copy()
            # Extra permutations of non-background roles keep literal codes local.
            palette[1:] = palette[1:][rng.permutation(9)]
            pair = {name: palette[np.rot90(g, k)].astype(np.int8) for name, g in pair.items()}
            if _key(pair['input']) in seen:
                continue
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                info = _decode(pair['input'])
                route = _trace(info)
                for bit, heads in enumerate(_headings(info, route)):
                    heading_sets[bit].update(heads)
                train_periods.update(periods)
            break
        else:
            raise RuntimeError("bounded teaching/uniqueness selection exhausted")
    rng.shuffle(puzzle['train'])
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    g, out = _grid(pair['input']), _grid(pair['output'])
    info = _decode(g)
    route = _trace(info)
    _require(out.shape == g.shape and np.array_equal(_apply(info, route), out), "incorrect answer")
    active, retained = 0, 0
    for z, indices in zip(info['rails'], route['queries']):
        n, p = len(z['code']), z['period']
        _require(12 <= n <= 26 and 2 <= p <= 6 and n >= 4 * p, "unsupported code dimensions")
        changed = np.flatnonzero(z['raw'] != z['code'])
        _require(len(changed) <= max(2, n // 8), "too much noise")
        _require(len({int(i) % p for i in changed}) == len(changed), "multiple flips in a phase")
        active += sum(int(i) in indices for i in changed)
        retained += sum(int(i) not in indices for i in changed)
    _require(active >= 1 and retained >= 1, "need queried and unqueried errors")
    raw = _trace(info, [z['raw'] for z in info['rails']])
    _require(not np.array_equal(raw['mask'], route['mask']), "raw code would give the same route")
    info.update(route=route, active_errors=active, retained_errors=retained)
    return info


def _phase_repair_witness(info):
    """A queried phase also has an erroneous *unqueried copy* that survives."""
    for z, indices in zip(info['rails'], info['route']['queries']):
        phases = {i % z['period'] for i in indices}
        if any(i not in indices and i % z['period'] in phases
               and z['raw'][i] != z['code'][i] for i in range(len(z['code']))):
            return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos['train']
    heads = [set(), set()]
    periods = set()
    for q in train:
        for bit, values in enumerate(_headings(q, q['route'])):
            heads[bit].update(values)
        periods.update(q['rails'][e[1]]['period'] for e in q['route']['events'])
    _require(all(len(x) >= 2 for x in heads), "both relative turns need multiple incoming headings")
    _require({2, 3} <= periods, "training must query two different periods")
    _require(any(_has_mixed_wall(q['route']) and not _constant_wall_fits(q, _apply(q, q['route']))
                 for q in train), "wall-position dependence is undemonstrated")
    _require(any(not np.array_equal(_trace(q, first_only=True)['mask'], q['route']['mask'])
                 for q in train), "recursive continuation is undemonstrated")
    _require(any(_phase_repair_witness(q) for q in train),
             "address-only repair is indistinguishable from repairing queried phases")
    for mapping in ((1, 0), (0, 0), (1, 1)):
        fits = True
        for q in train:
            try:
                candidate = _apply(q, _trace(q, mapping=mapping))
            except ValueError:
                fits = False
                break
            if not np.array_equal(candidate, _apply(q, q['route'])):
                fits = False
                break
        _require(not fits, "training leaves an alternative turn convention")
    for q in infos['test']:
        _require(any(q['rails'][e[1]]['period'] >= 4 for e in q['route']['events']),
                 "test must use a longer code period")


def self_test():
    """Bounded 42-seed checks, all explicit count combinations, no disk writes."""
    signatures, shapes, layouts, periods, counts, axes, palettes = set(), set(), set(), set(), set(), set(), set()
    hit_counts, turn_counts, mixed = set(), set(), 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                q = _analyze_pair(pair)
                digest.update(pair['input'].tobytes())
                shapes.add(q['grid'].shape)
                counts.add(len(q['rails']))
                layouts.add((q['grid'].shape, q['source'], tuple(_box(z) for z in q['rails'])))
                periods.update(z['period'] for z in q['rails'])
                axes.update(z['axis'] for z in q['rails'])
                palettes.add((q['background'], q['source_color'], q['frame']))
                hit_counts.add(len(q['route']['events']))
                turn_counts.add(max(q['route']['lengths']))
                mixed += _has_mixed_wall(q['route'])
                assert np.array_equal(solve(pair['input']), pair['output'])
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(layouts) >= 180 and len(shapes) >= 35
    assert {2, 3, 4, 5} <= periods and axes == {0, 1}
    assert len(counts) >= 3 and len(palettes) >= 60 and len(hit_counts) >= 4
    assert max(turn_counts) >= 3 and mixed >= 84
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(41, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, again = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[n], b[n]) for s in p for a, b in zip(p[s], again[s])
               for n in ('input', 'output'))
    # Tied evidence, a constant program, and two conflicting supported repairs.
    for bits, message in (([0] * 6 + [1] * 6, 'no supported'),
                          ([0] * 12, 'no supported'),
                          ([int(c) for c in '101010001000'], 'ambiguous')):
        try:
            _repair(np.asarray(bits))
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("unsupported code accepted")
    # Unsampled family boundary: a period-six wall, two distinct noisy phases,
    # a queried error, and an error that must survive.
    word = np.tile(np.array([0, 0, 0, 0, 0, 1], dtype=np.int8), 4)
    rail = dict(r=2, c=1, axis=0, code=word, period=6)
    extension = _info_for_construction((9, 28), [rail], (7, 14))
    route = _trace(extension)
    noisy = extension['grid'].copy()
    for i in (12, 5):
        cell = _code_cell(rail, i)
        noisy[cell] = 3 if noisy[cell] == 2 else 2
    expected = _apply(dict(extension, grid=noisy), route)
    extension_pair = dict(input=noisy, output=expected)
    assert _analyze_pair(extension_pair)['rails'][0]['period'] == 6
    assert np.array_equal(solve(noisy), expected)
    permutation = np.random.default_rng(782).permutation(10)
    for k in range(4):
        rotated = np.rot90(noisy, k)
        assert np.array_equal(solve(rotated), np.rot90(expected, k))
        assert np.array_equal(solve(permutation[rotated]), permutation[np.rot90(expected, k)])
    # End-cap impact has no invented meaning.
    end_cap = noisy.copy()
    end_cap[7, 14], end_cap[7, 1] = 0, 1
    try:
        solve(end_cap)
    except ValueError:
        pass
    else:
        raise AssertionError("end-cap impact accepted")
    # Real separated long walls forming a right-turn cycle.
    cycle = _info_for_construction((30, 30), [
        dict(r=r, c=c, axis=axis, code=np.tile([1, 0], 6).astype(np.int8), period=2)
        for r, c, axis in ((1, 17, 1), (17, 15, 0), (15, 7, 1), (7, 1, 0))], (12, 12))
    try:
        solve(cycle['grid'])
    except ValueError as error:
        assert 'cyclic' in str(error)
    else:
        raise AssertionError("cyclic route accepted")
    # Exercise the fallback directly rather than waiting for packing to fail.
    for test in (False, True):
        rng = np.random.default_rng(137 + int(test))
        for info, route in _fallback_layouts(rng, test):
            pair = _add_noise(rng, info, route, True)
            if pair is not None:
                _analyze_pair(pair)
                assert np.array_equal(solve(pair['input']), pair['output'])
                break
        else:
            raise AssertionError("fallback construction failed")
    return {'seeds': 42, 'puzzles': len(signatures), 'shapes': len(shapes),
            'layouts': len(layouts), 'periods': sorted(periods), 'wall_counts': sorted(counts),
            'max_turns': max(turn_counts), 'mixed_wall_pairs': mixed}
