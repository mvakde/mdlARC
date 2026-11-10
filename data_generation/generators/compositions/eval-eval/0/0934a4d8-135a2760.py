"""Accordion ribbons: jointly decode reflected repetition, then extract a shutter.

Only NumPy and the standard library are used. The prediction path receives pixels
only; construction templates never enter solve or validation.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 773380375, 148058214, 863137057, 220068967, 445899654,
              687554206, 611238619, 687282282, 405678780, 411396955,
              961452612, 375684482, 108490997, 59436529, 961205929,
              470530856, 789089039, 588862208, 45401963, 90895853,
              955189379, 674136137, 770201060, 593933808, 434312004,
              49751654, 473837182, 406126331, 803560555, 386065077,
              577171950, 600673388, 469920987, 290866812, 213681104,
              20884955, 69704698, 308441610, 240163852, 684080054,
              219809353)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _mask(a):
    candidates = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        box = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        r, s, c, d = box
        if len(rr) >= 2 and len(rr) == (s-r)*(d-c):
            candidates.append((int(color), box))
    _require(len(candidates) == 1, "The shutter must be the unique rectangular color")
    return candidates[0]


def _parse_horizontal(a, mask):
    h, w = a.shape
    structure = int(a[0, 0])
    _require(mask != structure and np.all(a[0] == structure)
             and np.all(a[-1] == structure) and np.all(a[:, 0] == structure)
             and np.all(a[:, -1] == structure), "Frame must be intact")
    dividers = np.flatnonzero(np.all((a == structure) | (a == mask), axis=1))
    _require(2 <= len(dividers) <= 5 and dividers[0] == 0
             and dividers[-1] == h-1 and np.all(np.diff(dividers) >= 5),
             "Invalid ribbon dividers")
    length = w-4
    _require(12 <= length <= 26, "Invalid ribbon length")
    rectangles = [(int(lo)+2, int(hi)-1, 2, w-2)
                  for lo, hi in zip(dividers[:-1], dividers[1:])]
    _require(all(2 <= s-r <= 10 for r, s, _, _ in rectangles),
             "Invalid ribbon thickness")
    frame = np.zeros(a.shape, bool)
    frame[:, [0, w-1]] = True
    frame[dividers] = True
    motif = np.zeros(a.shape, bool)
    for r, s, c, d in rectangles:
        motif[r:s, c:d] = True
    padding_mask = ~(frame | motif)
    values = np.unique(a[padding_mask & (a != mask)])
    _require(len(values) == 1 and values[0] not in (mask, structure),
             "Padding must have a distinct visible color")
    padding = int(values[0])
    _require(np.all((a[frame] == structure) | (a[frame] == mask)),
             "Damaged structural cells")
    accents = []
    for r, s, c, d in rectangles:
        values = set(int(x) for x in np.unique(a[r:s, c:d])) - {mask, structure}
        _require(len(values) == 1, "Each ribbon must have one visible accent")
        accent = next(iter(values))
        _require(accent != padding, "Accent cannot be padding")
        accents.append(accent)
    _require(len(set(accents)) == len(accents), "Ribbon accents must differ")
    return dict(structure=structure, padding=padding, dividers=dividers,
                rectangles=rectangles, frame=frame, motif=motif,
                length=length, accents=accents)


def _orbit_ids(thickness, length, period):
    q, x = np.indices((thickness, length))
    folded_q = np.where((x // period) % 2 == 0, q, thickness-1-q)
    return folded_q*period + x % period


def _recover_lane(lane, structure, accent, mask):
    t, length = lane.shape
    visible = lane != mask
    candidates = []
    for period in range(1, length//3 + 1):
        ids = _orbit_ids(t, length, period)
        count = np.bincount(ids[visible], minlength=t*period)
        ones = np.bincount(ids[visible], weights=(lane[visible] == accent),
                           minlength=t*period).astype(int)
        if np.any(count < 3) or np.any(2*ones == count):
            continue
        errors = int(np.minimum(ones, count-ones).sum())
        if errors > max(1, int(visible.sum())//10):
            continue
        template = np.where(2*ones > count, accent, structure).astype(np.int8)
        if len(np.unique(template)) < 2:
            continue
        repaired = template[ids]
        candidates.append((period, repaired, errors, ids, count))
    _require(bool(candidates), "No supported reflected period")
    repaired = candidates[0][1]
    _require(all(np.array_equal(c[1], repaired) for c in candidates),
             "Supported reflected periods give competing completions")
    p, _, errors, ids, counts = candidates[0]
    return repaired, dict(period=p, errors=errors, ids=ids, counts=counts,
                          candidates=tuple(c[0] for c in candidates))


def _infer(input_grid):
    original = _grid(input_grid)
    mask, original_box = _mask(original)
    options = []
    for transpose in (False, True):
        a = original.T.copy() if transpose else original.copy()
        try:
            parsed = _parse_horizontal(a, mask)
            clean = np.full(a.shape, parsed['padding'], dtype=np.int8)
            clean[parsed['frame']] = parsed['structure']
            lanes = []
            for rect, accent in zip(parsed['rectangles'], parsed['accents']):
                r, s, c, d = rect
                repaired, record = _recover_lane(a[r:s, c:d], parsed['structure'], accent, mask)
                clean[r:s, c:d] = repaired
                lanes.append(record)
            options.append(dict(a=a, clean=clean, mask=mask, transpose=transpose,
                                parsed=parsed, lanes=lanes))
        except ValueError:
            continue
    _require(len(options) == 1, "Ribbon axis or completion is ambiguous")
    record = options[0]
    full = record['clean'].T if record['transpose'] else record['clean']
    r, s, c, d = original_box
    record['original_box'] = original_box
    record['full'] = full.copy()
    return full[r:s, c:d].copy(), record


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover from this input alone, and return the target-oriented shutter crop."""
    return _infer(input_grid)[0]


def _nearest_donor(lane, ids, q, x, mask):
    witnesses = np.argwhere((ids == ids[q, x]) & (lane != mask))
    _require(len(witnesses) > 0, "Orbit has no donor")
    # np.argwhere is row-major; argmin retains that order on distance ties.
    i = int(np.argmin(np.abs(witnesses[:, 0]-q) + np.abs(witnesses[:, 1]-x)))
    return tuple(int(v) for v in witnesses[i])


def _nearest_crop(record):
    """Ablation: copy one closest orbit donor, without periodic consensus."""
    a, mask, parsed = record['a'], record['mask'], record['parsed']
    naive = record['clean'].copy()
    for rect, info in zip(parsed['rectangles'], record['lanes']):
        r, s, c, d = rect
        lane = a[r:s, c:d]
        for q, x in np.argwhere(lane == mask):
            dq, dx = _nearest_donor(lane, info['ids'], int(q), int(x), mask)
            naive[r+q, c+x] = lane[dq, dx]
    if record['transpose']:
        naive = naive.T
    r, s, c, d = record['original_box']
    return naive[r:s, c:d].copy()


def _reflection_sensitive(record):
    """Cells where repeating the first tile without reflection gives wrong data."""
    total = 0
    for rect, info in zip(record['parsed']['rectangles'], record['lanes']):
        r, s, c, d = rect
        repaired = record['clean'][r:s, c:d]
        ordinary = repaired[:, np.arange(d-c) % info['period']]
        hidden = record['a'][r:s, c:d] == record['mask']
        total += int(np.count_nonzero(hidden & (ordinary != repaired)))
    return total


def _crosses(rng, count, length):
    cap = min(10, length-1)
    budget = min(29-3*count, cap*count)
    if rng.random() < .25:
        return [int(rng.integers(2, budget//count+1))]*count
    total = int(rng.integers(2*count, budget+1))
    values = np.full(count, 2, dtype=int)
    for _ in range(total-2*count):
        eligible = np.flatnonzero(values < cap)
        values[int(rng.choice(eligible))] += 1
    rng.shuffle(values)
    return [int(x) for x in values]


def _bits(rng, thickness, period, style):
    if style == 0:
        return rng.integers(0, 2, (thickness, period), dtype=np.int8)
    if style == 1:
        q, x = np.indices((thickness, period))
        offset = int(rng.integers(period))
        a = ((x+q+offset) % period == 0).astype(np.int8)
    else:
        word = rng.integers(0, 2, period, dtype=np.int8)
        a = np.stack([np.roll(word, int(rng.integers(period)))
                      for _ in range(thickness)])
    return 1-a if rng.random() < .5 else a


def _make_pair(rng, *, kind, count, axis, test, palette, style):
    # Every attempt constructs its own geometry and patterns. No template bank.
    for _ in range(120):
        minimum_length = 22 if test else (18 if kind == 'cross' else 12)
        length = int(rng.integers(minimum_length, 27))
        crosses = _crosses(rng, count, length)
        periods = [int(rng.integers(2, length//6+1)) for _ in crosses]
        if count > 1 and length >= 18:
            periods[1] = int(rng.choice([p for p in range(2, length//6+1)
                                         if p != periods[0]]))
        structure, padding, mask = (int(x) for x in palette[:3])
        accents = [int(x) for x in palette[3:3+count]]
        dividers = np.cumsum([0] + [t+3 for t in crosses])
        shape = int(dividers[-1])+1, length+4
        clean = np.full(shape, padding, dtype=np.int8)
        clean[:, [0, shape[1]-1]] = structure
        clean[dividers] = structure
        rects = []
        usable = True
        for index, (t, p, accent) in enumerate(zip(crosses, periods, accents)):
            r, s = int(dividers[index])+2, int(dividers[index+1])-1
            rects.append((r, s, 2, length+2))
            bits = _bits(rng, t, p, style if rng.random() < .7 else 0)
            if len(np.unique(bits)) != 2 or np.array_equal(bits, bits[::-1]):
                usable = False
                break
            template = np.where(bits, accent, structure).astype(np.int8).ravel()
            clean[r:s, 2:length+2] = template[_orbit_ids(t, length, p)]
        if not usable:
            continue
        if kind == 'long':
            index = int(rng.integers(count))
            r, s, _, _ = rects[index]
            depth = int(rng.integers(1, (s-r)//2+1))
            lo, hi = (r, r+depth) if rng.random() < .5 else (s-depth, s)
            box = lo, hi, 2, length+2
        else:
            if kind == 'cross':
                span = int(rng.integers(2, count+1)) if test else 2
                if not test:
                    first = int(rng.choice([i for i in range(count-1)
                                            if periods[i] != periods[i+1]]))
                else:
                    first = int(rng.integers(0, count-span+1))
                last = first+span-1
                lo = int(rng.integers(rects[first][0], rects[first][1]))
                hi = int(rng.integers(rects[last][0], rects[last][1]))+1
                involved = list(range(first, last+1))
            else:
                index = int(rng.integers(count))
                r, s, _, _ = rects[index]
                lo = int(rng.integers(r, s))
                hi = int(rng.integers(lo+1, s+1))
                involved = [index]
            minimum_p = min(periods[i] for i in involved)
            low_width = 2*minimum_p if test else minimum_p
            high_width = 3*minimum_p-1 if test else 2*minimum_p
            width = int(rng.integers(low_width, high_width+1))
            start = int(rng.integers(0, length-width+1))
            box = lo, hi, start+2, start+width+2
        a = clean.copy()
        r0, r1, c0, c1 = box
        a[r0:r1, c0:c1] = mask
        for index, (rect, p, accent) in enumerate(zip(rects, periods, accents)):
            r, s, c, d = rect
            lane = a[r:s, c:d]
            ids = _orbit_ids(s-r, length, p)
            visible = lane != mask
            counts = np.bincount(ids[visible], minlength=(s-r)*p)
            if np.any(counts < 3):
                usable = False
                break
            cap = max(1, int(visible.sum())//10)
            error_count = int(rng.integers(1, min(4, cap, len(counts))+1))
            hidden = np.argwhere(~visible)
            selected = set()
            if len(hidden):
                q, x = (int(v) for v in hidden[int(rng.integers(len(hidden)))])
                dq, dx = _nearest_donor(lane, ids, q, x, mask)
                selected.add(int(ids[dq, dx]))
                lane[dq, dx] = accent if lane[dq, dx] == structure else structure
            while len(selected) < error_count:
                eligible = [i for i in range(len(counts)) if i not in selected]
                group = int(rng.choice(eligible))
                donors = np.argwhere((ids == group) & visible)
                dq, dx = donors[int(rng.integers(len(donors)))]
                selected.add(group)
                lane[dq, dx] = accent if lane[dq, dx] == structure else structure
        if not usable:
            continue
        output = clean[r0:r1, c0:c1].copy()
        if axis:
            a, output = a.T.copy(), output.T.copy()
        try:
            predicted, record = _infer(a)
            if not np.array_equal(predicted, output):
                continue
            expected_full = clean.T if axis else clean
            if not np.array_equal(record['full'], expected_full):
                continue
            if tuple(x['period'] for x in record['lanes']) != tuple(periods):
                continue
            if np.array_equal(_nearest_crop(record), output):
                continue
            if kind == 'long' and _reflection_sensitive(record) == 0:
                continue
        except ValueError:
            continue
        return {'input': a, 'output': output}
    raise RuntimeError("Bounded accordion construction exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # Nonnegative seeds use the requested seed directly; signed seeds normalize.
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    regime = int(rng.integers(3))  # shared axis, mixed axes, or a held-out axis
    preferred_axis = int(rng.integers(2))
    style = int(rng.integers(3))
    preferred_palette = rng.permutation(10)
    vary_palette = rng.random() < .75
    long_count = int(rng.integers(1, 4))
    cross_count = int(rng.choice([n for n in range(2, 5) if n != long_count]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for serial in range(nt+nq):
        test = serial >= nt
        if serial == 0:
            kind, count = 'long', long_count
        elif serial == 1:
            kind, count = 'cross', cross_count
        elif test:
            kind, count = 'cross', int(rng.integers(2, 5))
        else:
            kind = str(rng.choice(['long', 'window', 'cross']))
            count = int(rng.integers(2 if kind == 'cross' else 1, 5))
        if regime == 1:
            axis = (preferred_axis if serial == 0 else 1-preferred_axis) if serial < 2 else int(rng.integers(2))
        else:
            axis = (1-preferred_axis) if test and regime == 2 else preferred_axis
        palette = rng.permutation(10) if vary_palette else preferred_palette
        for _ in range(12):
            pair = _make_pair(rng, kind=kind, count=count, axis=axis,
                              test=test, palette=palette, style=style)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['test' if test else 'train'].append(pair)
                break
        else:
            raise RuntimeError("Unique-input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    predicted, record = _infer(pair['input'])
    a, mask, parsed = record['a'], record['mask'], record['parsed']
    full_row_loss = False
    erased_groups_without_reflection = 0
    masked_periods = []
    for rect, info in zip(parsed['rectangles'], record['lanes']):
        r, s, c, d = rect
        hidden = a[r:s, c:d] == mask
        if np.any(hidden):
            masked_periods.append(info['period'])
        full_row_loss |= bool(np.any(np.all(hidden, axis=1)))
        # Give a repetition-only ablation even the true ordinary period 2p.
        # Empty coordinate-phase groups have two possible binary completions.
        p = 2*info['period']
        for row in hidden:
            erased_groups_without_reflection += sum(np.all(row[phase::p])
                                                    for phase in range(p))
    return dict(axis=int(record['transpose']), shape=np.asarray(pair['input']).shape,
                output_shape=predicted.shape, count=len(parsed['rectangles']),
                crosses=tuple(s-r for r, s, _, _ in parsed['rectangles']),
                length=parsed['length'], periods=tuple(x['period'] for x in record['lanes']),
                masked_periods=tuple(masked_periods),
                errors=tuple(x['errors'] for x in record['lanes']),
                mask=mask, full_row_loss=full_row_loss,
                crossed_divider=any(np.any(a[int(r)] == mask)
                                    for r in parsed['dividers'][1:-1]),
                repetition_only_unknown=int(erased_groups_without_reflection),
                reflection_sensitive=_reflection_sensitive(record),
                nearest_wrong=int(np.count_nonzero(_nearest_crop(record) != predicted)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             "Puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "Splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 2, 10, 'num_train')
    _count(nq, 1, 4, 'num_test')
    _require(nt > nq, "Invalid split counts")
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     "Each pair must contain exactly input and output")
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            _require(key not in seen, "Inputs must be distinct")
            seen.add(key)
            predicted, record = _infer(a)
            _require(np.array_equal(predicted, b), "Incorrect shutter crop")
            _require(not np.any(b == record['mask']), "Shutter color remains in answer")
            canon, clean, parsed = record['a'], record['clean'], record['parsed']
            outside = canon != record['mask']
            _require(np.array_equal(canon[outside & ~parsed['motif']], clean[outside & ~parsed['motif']]),
                     "Frames or padding changed")
            for rect, accent, info in zip(parsed['rectangles'], parsed['accents'], record['lanes']):
                r, s, c, d = rect
                lane, repaired = canon[r:s, c:d], clean[r:s, c:d]
                visible = lane != record['mask']
                changed = visible & (lane != repaired)
                _require(1 <= int(changed.sum()) <= max(1, int(visible.sum())//10),
                         "Each ribbon needs sparse visible binary errors")
                _require(np.all(np.isin(lane[visible], [parsed['structure'], accent])),
                         "Noise must be a binary color flip")
                p = info['period']
                _require(np.array_equal(repaired[:, p:], repaired[::-1, :-p]),
                         "Clean ribbon does not repeat with reflection")
                _require(not np.array_equal(repaired, repaired[::-1]),
                         "Cross-sectional reflection has no effect")
            f = _features(pair)
            _require(f['nearest_wrong'] > 0, "Single-donor copying must fail")
            if split == 'train':
                train_features.append(f)
    _require(any(f['full_row_loss'] and f['repetition_only_unknown'] > 0
                 and f['reflection_sensitive'] > 0 for f in train_features),
             "Training must require reflected witnesses")
    _require(len({f['count'] for f in train_features}) >= 2,
             "Training must expose different ribbon counts")
    # The crossing example demonstrates structural cells in the extracted crop.
    _require(any(f['crossed_divider'] and len(set(f['masked_periods'])) >= 2
                 for f in train_features),
             "Training must demonstrate a shutter spanning a divider")


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed diversity run, ablations, mutations, and legal counts."""
    signatures, shapes, outputs, structures, periods, crosses, masks = (set() for _ in range(7))
    counts, axes = Counter(), Counter()
    full_losses, nearest_errors, partials = 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed in (0, HOST_SEEDS[1], HOST_SEEDS[-1]):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                before = a.copy()
                assert np.array_equal(solve(a), pair['output'])
                assert np.array_equal(a, before), "solve mutated its input"
                f = _features(pair)
                shapes.add(f['shape'])
                outputs.add(f['output_shape'])
                periods.update(f['periods'])
                crosses.update(f['crosses'])
                masks.add(f['mask'])
                counts[f['count']] += 1
                axes[f['axis']] += 1
                full_losses += f['repetition_only_unknown'] > 0
                nearest_errors += f['nearest_wrong']
                partials += any(f['length'] % p for p in f['periods'])
                roles = {}
                norm = tuple(roles.setdefault(int(x), len(roles)) for x in a.ravel())
                structures.add((a.shape, norm))
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 50 and len(outputs) >= 40
    assert set(counts) == {1, 2, 3, 4} and set(axes) == {0, 1}
    assert periods == {2, 3, 4} and len(crosses) >= 8 and len(masks) >= 8
    assert full_losses >= 42 and nearest_errors >= 210 and 0 < partials < 210
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1700+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(77, 2, 1)
    # Rotation, reflection and local palette changes preserve this fixed rule.
    transformed = copy.deepcopy(p)
    for i, pair in enumerate(transformed['train']+transformed['test']):
        palette = np.roll(np.arange(10, dtype=np.int8), i+1)
        for role in pair:
            pair[role] = palette[np.rot90(pair[role])].copy()
    validate_generated_puzzle(transformed)
    # Hand-derived, asymmetric two-column unit, six copies. The whole top
    # motif row is erased. Its closest bottom donor is wrong; other cycles
    # establish the answer. These literal rows do not use the constructor.
    rows = ('3333333333333333', '3222222222222223',
            '3283388338833823', '3233333333333323',
            '3238833883388323', '3222222222222223',
            '3333333333333333')
    clean = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    masked = clean.copy()
    masked[2, 2:14] = 9
    masked[4, 4] = 3
    answer, record = _infer(masked)
    assert np.array_equal(answer, clean[2:3, 2:14])
    assert not np.array_equal(answer, _nearest_crop(record))
    assert _reflection_sensitive(record) > 0
    erased = clean.copy()
    erased[2:5, 2:14] = 9
    try:
        solve(erased)
    except ValueError:
        pass
    else:
        raise AssertionError("Completely erased motif accepted")
    broken = copy.deepcopy(p)
    b = broken['test'][0]['output']
    b[0, 0] = (int(b[0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), output_shapes=len(outputs),
                normalized_structures=len(structures), ribbon_counts=dict(counts),
                axes=dict(axes), periods=sorted(periods), thicknesses=sorted(crosses),
                mask_colors=sorted(masks), repetition_only_incomplete=full_losses,
                nearest_copy_wrong_cells=nearest_errors, partial_cycle_pairs=partials)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
