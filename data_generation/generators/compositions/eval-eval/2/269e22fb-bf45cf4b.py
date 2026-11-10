"""Registered remainder printing: a colored crop is both key and printing tile.

The input contains a framed binary stencil and a framed colored fragment.
Fragment support identifies its unique location and D4 orientation in the
stencil. Undo that orientation, remove the sampled footprint, and print the
remaining active stencil cells with copies of the registered fragment.
"""
from __future__ import annotations

import json
import numpy as np


def _transform(a, index):
    """0..3: CCW turns; 4..7: those turns followed by a left/right flip."""
    b = np.rot90(a, index % 4)
    return (np.fliplr(b) if index >= 4 else b).copy()


def _inverse(index):
    probe = np.arange(6).reshape(2, 3)
    return next(j for j in range(8)
                if np.array_equal(_transform(_transform(probe, index), j), probe))


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        stack = [remaining.pop()]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nxt in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        result.append(cells)
    return result


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("the canvas must have a uniform background rim")
    readings = []
    for frame in map(int, np.unique(a)):
        if frame == bg:
            continue
        components = _components(a == frame)
        if len(components) != 2:
            continue
        panels, boxes = [], []
        for cells in components:
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            h, w = r1-r0+1, c1-c0+1
            if min(h, w) < 4:
                break
            expected = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                        if r in (r0, r1) or c in (c0, c1)}
            if set(cells) != expected:
                break
            panels.append(a[r0+1:r1, c0+1:c1].copy())
            boxes.append((r0, r1, c0, c1))
        if len(panels) != 2:
            continue
        outside = np.ones(a.shape, dtype=bool)
        for r0, r1, c0, c1 in boxes:
            outside[r0:r1+1, c0:c1+1] = False
        if np.any(a[outside] != bg):
            continue
        payloads = [set(map(int, np.unique(p))) - {bg} for p in panels]
        mono = [i for i, colors in enumerate(payloads) if len(colors) == 1]
        multi = [i for i, colors in enumerate(payloads) if len(colors) >= 2]
        if len(mono) != 1 or len(multi) != 1:
            continue
        stencil, stamp = panels[mono[0]], panels[multi[0]]
        signal = next(iter(payloads[mono[0]]))
        if frame in stencil or frame in stamp or signal in stamp:
            continue
        reference = stencil == signal
        if np.all(reference) or not np.any(stamp == bg):
            continue
        readings.append(dict(reference=reference, stamp=stamp, background=bg,
                             signal=signal, frame=frame, boxes=boxes))
    if len(readings) != 1:
        raise ValueError("expected two separated frames with unique stencil/stamp roles")
    return readings[0]


def _matches(reference, patch):
    h, w = patch.shape
    result = []
    for orientation in range(8):
        full = _transform(reference, orientation)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        positions = np.argwhere(np.all(windows == patch, axis=(-2, -1)))
        result.extend((orientation, int(r), int(c)) for r, c in positions)
    return result


def _registration(info):
    reference, stamp, bg = info['reference'], info['stamp'], info['background']
    matches = _matches(reference, stamp != bg)
    if len(matches) != 1:
        raise ValueError("fragment support must uniquely identify orientation and offset")
    orientation, r, c = matches[0]
    inv = _inverse(orientation)
    registered = _transform(stamp, inv)
    footprint = np.zeros(_transform(reference, orientation).shape, dtype=bool)
    h, w = stamp.shape
    footprint[r:r+h, c:c+w] = True
    footprint = _transform(footprint, inv)
    residual = reference & ~footprint
    if (np.count_nonzero(residual) < 2 or np.count_nonzero(reference & footprint) < 2
            or reference.size <= footprint.sum()):
        raise ValueError("require a proper crop with at least two sampled and remaining active cells")
    if any(n * s > 30 for n, s in zip(reference.shape, registered.shape)):
        raise ValueError("expanded result exceeds the ARC size limit")
    return dict(orientation=orientation, offset=(r, c), registered=registered,
                footprint=footprint, residual=residual)


def _expand(mask, stamp, bg):
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0]*sh, mask.shape[1]*sw))
    return np.where(mask[rr//sh, cc//sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction; no reference bank, seed, or examples are consulted."""
    info = _read(input_grid)
    registration = _registration(info)
    return _expand(registration['residual'], registration['registered'], info['background'])


def _crop_bank(reference):
    """Canonical-axis crops admissible under every possible D4 registration."""
    H, W = reference.shape
    bank = []
    for h in range(2, min(6, H, 30//H)+1):
        for w in range(2, min(6, W, 30//W)+1):
            if h*w == H*W:
                continue
            for r in range(H-h+1):
                for c in range(W-w+1):
                    patch = reference[r:r+h, c:c+w]
                    active = int(patch.sum())
                    if active < 2 or active == patch.size or reference.sum()-active < 2:
                        continue
                    if _matches(reference, patch) == [(0, r, c)]:
                        bank.append((r, c, h, w))
    return bank


def _offset_matters(reference, crop):
    r, c, h, w = crop
    actual = np.zeros(reference.shape, dtype=bool)
    guessed = np.zeros(reference.shape, dtype=bool)
    actual[r:r+h, c:c+w] = True
    guessed[:h, :w] = True
    return not np.array_equal(reference & actual, reference & guessed)


def _training_choices(reference, bank):
    """Reserve two shapes/addresses, including a causal non-top-left address."""
    meaningful = [b for b in bank if _offset_matters(reference, b)]
    first = [b for b in bank if any(q[2:] != b[2:] and q[:2] != b[:2]
                                  for q in meaningful)]
    return first, meaningful


def _draw_reference(rng, H, W):
    """Mix bars, bends, loops, staircases, combs and sparse satellites."""
    a = np.zeros((H, W), dtype=bool)
    for _ in range(int(rng.integers(3, 7))):
        kind = int(rng.integers(6))
        r0, r1 = sorted(map(int, rng.choice(H, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(W, 2, replace=False)))
        if kind == 0:
            a[r0, c0:c1+1] = True
        elif kind == 1:
            a[r0:r1+1, c0] = True
            a[r1, c0:c1+1] = True
        elif kind == 2:
            a[r0:r1+1, c0] = a[r0:r1+1, c1] = True
            a[r0, c0:c1+1] = a[r1, c0:c1+1] = True
        elif kind == 3:
            n = max(r1-r0, c1-c0)+1
            rr = np.rint(np.linspace(r0, r1, n)).astype(int)
            cc = np.rint(np.linspace(c0, c1, n)).astype(int)
            a[rr, cc] = True
        elif kind == 4:
            a[r0:r1+1, c0] = True
            a[r0:r1+1:2, c0:c1+1] = True
        else:
            a[rng.integers(H, size=3), rng.integers(W, size=3)] = True
    # Erased short runs add concavities without replacing the motif mixture by noise.
    if rng.random() < .5:
        r, c = int(rng.integers(H)), int(rng.integers(W-1))
        a[r, c:c+int(rng.integers(1, 3))] = False
    return a


def _reference(rng):
    H = int(rng.integers(5, 10))
    W = H if rng.random() < .35 else int(rng.integers(5, 10))
    for _ in range(160):
        reference = _draw_reference(rng, H, W)
        if not .22 <= reference.mean() <= .77:
            continue
        bank = _crop_bank(reference)
        shapes = {(h, w) for _, _, h, w in bank}
        if len(bank) >= 14 and len(shapes) >= 2 and _training_choices(reference, bank)[0]:
            return reference, bank
    # Retry with smaller sampled dimensions, where larger recognizable windows
    # are possible. This changes no semantic convention or individual example.
    for _ in range(160):
        H, W = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        reference = _draw_reference(rng, H, W)
        bank = _crop_bank(reference)
        if (.22 <= reference.mean() <= .77 and len(bank) >= 14
                and len({b[2:] for b in bank}) >= 2 and _training_choices(reference, bank)[0]):
            return reference, bank
    raise RuntimeError("bounded reference construction exhausted")


def _colored_stamp(rng, patch, bg, payload):
    h, w = patch.shape
    r, c = np.indices(patch.shape)
    k = len(payload)
    family = int(rng.integers(5))
    if family == 0:
        labels = r + int(rng.integers(k))
    elif family == 1:
        labels = c + int(rng.integers(k))
    elif family == 2:
        labels = r + c
    elif family == 3:
        labels = np.minimum.reduce([r, c, h-1-r, w-1-c])
    else:
        labels = rng.integers(k, size=patch.shape)
    stamp = np.where(patch, np.asarray(payload)[labels % k], bg).astype(np.int8)
    cells = np.argwhere(patch)
    # Two actual payload colors are necessary to visibly separate panel roles.
    if len(set(map(int, stamp[patch]))) < 2:
        r0, c0 = cells[int(rng.integers(len(cells)))]
        stamp[r0, c0] = payload[1]
        if len(set(map(int, stamp[patch]))) < 2:
            r1, c1 = next(cell for cell in cells if tuple(cell) != (r0, c0))
            stamp[r1, c1] = payload[0]
    return stamp


def _framed(content, frame):
    panel = np.full((content.shape[0]+2, content.shape[1]+2), frame, dtype=np.int8)
    panel[1:-1, 1:-1] = content
    return panel


def _canvas(rng, reference, stamp, bg, frame, signal):
    panels = [_framed(np.where(reference, signal, bg), frame), _framed(stamp, frame)]
    if rng.integers(2):
        panels.reverse()
    horizontal = bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    p, q = panels
    if horizontal:
        H = max(p.shape[0], q.shape[0])+int(rng.integers(2, 7))
        W = p.shape[1]+q.shape[1]+gap+int(rng.integers(2, 6))
        c0 = int(rng.integers(1, W-p.shape[1]-q.shape[1]-gap))
        positions = [(int(rng.integers(1, H-p.shape[0])), c0),
                     (int(rng.integers(1, H-q.shape[0])), c0+p.shape[1]+gap)]
    else:
        H = p.shape[0]+q.shape[0]+gap+int(rng.integers(2, 6))
        W = max(p.shape[1], q.shape[1])+int(rng.integers(2, 7))
        r0 = int(rng.integers(1, H-p.shape[0]-q.shape[0]-gap))
        positions = [(r0, int(rng.integers(1, W-p.shape[1]))),
                     (r0+p.shape[0]+gap, int(rng.integers(1, W-q.shape[1])))]
    a = np.full((H, W), bg, dtype=np.int8)
    for panel, (r, c) in zip(panels, positions):
        a[r:r+panel.shape[0], c:c+panel.shape[1]] = panel
    return a


def _counts(num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= value <= upper):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    reference, bank = _reference(rng)
    palette = list(map(int, rng.permutation(10)))
    varied = bool(rng.integers(2))
    quarter = int(rng.choice([1, 3]))
    anchors = [0, quarter]
    full_d4 = nt >= 3 and bool(rng.integers(2))
    if full_d4:
        anchors.append(int(rng.integers(4, 8)))
    train_orientations = anchors + [int(rng.choice(anchors)) for _ in range(nt-len(anchors))]
    held_out = sorted(set(range(8) if full_d4 else range(4))-set(anchors))
    test_orientations = [int(rng.choice(held_out)) for _ in range(nq)]
    # Two large, differently shaped demonstrations; tests favor smaller windows.
    first_choices, meaningful = _training_choices(reference, bank)
    ordered = sorted(first_choices, key=lambda b: b[2]*b[3], reverse=True)
    first = ordered[int(rng.integers(min(5, len(ordered))))]
    different = sorted([b for b in meaningful if b[2:] != first[2:] and b[:2] != first[:2]],
                       key=lambda b: b[2]*b[3], reverse=True)
    second = different[int(rng.integers(min(5, len(different))))]
    crops = [first, second]
    crops += [bank[int(rng.integers(len(bank)))] for _ in range(nt-2)]
    small = sorted(bank, key=lambda b: b[2]*b[3])[:max(1, len(bank)//2)]
    tests = [small[int(rng.integers(len(small)))] for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, orientations, selected in (('train', train_orientations, crops),
                                           ('test', test_orientations, tests)):
        for orientation, (r, c, h, w) in zip(orientations, selected):
            for _ in range(32):
                colors = list(map(int, rng.permutation(10))) if varied else palette
                bg, frame, signal = colors[:3]
                payload = colors[3:3+int(rng.integers(2, 6))]
                canonical_stamp = _colored_stamp(rng, reference[r:r+h, c:c+w], bg, payload)
                displayed_stamp = _transform(canonical_stamp, orientation)
                inp = _canvas(rng, reference, displayed_stamp, bg, frame, signal)
                if _key(inp) in seen:
                    continue
                # Ground truth is written block by block from construction
                # coordinates; solve recovers them exclusively from the input.
                residual = reference.copy()
                residual[r:r+h, c:c+w] = False
                out = np.full((reference.shape[0]*h, reference.shape[1]*w), bg, dtype=np.int8)
                for rr, cc in np.argwhere(residual):
                    out[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = canonical_stamp
                puzzle[split].append({'input': inp, 'output': out})
                seen.add(_key(inp))
                break
            else:
                raise RuntimeError("distinct input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    shared = None
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read(inp)
            reg = _registration(info)
            refkey = _key(info['reference'])
            if shared is None:
                shared = refkey
            assert refkey == shared, "reference must remain fixed in displayed coordinates"
            assert np.array_equal(solve(inp), out), "incorrect registered remainder print"
            residual, stamp, bg = reg['residual'], reg['registered'], info['background']
            H, W = info['reference'].shape
            sh, sw = stamp.shape
            assert out.shape == (H*sh, W*sw)
            assert info['signal'] not in out and info['frame'] not in out
            # Check every block, including holes and completely blank footprint.
            for r in range(H):
                for c in range(W):
                    block = out[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                    assert np.array_equal(block, stamp if residual[r, c] else np.full(stamp.shape, bg))
            # Essential-contribution checks; all pairs require footprint removal
            # and scale substitution, rather than only geometric reconstruction.
            assert not np.array_equal(out, _expand(info['reference'], stamp, bg))
            assert out.shape != info['reference'].shape
            assert np.any(reg['footprint'] & info['reference']) and np.any(residual)
            records[split].append((info, reg))
    training = records['train']
    assert len({reg['registered'].shape for _, reg in training}) >= 2
    assert any(reg['orientation'] == 0 for _, reg in training)
    assert any(reg['orientation'] in (1, 3) for _, reg in training)
    origins = []
    causal_address = False
    for info, reg in training:
        rows, cols = np.nonzero(reg['footprint'])
        r, c = int(rows.min()), int(cols.min())
        h, w = reg['registered'].shape
        origins.append((r, c))
        causal_address |= _offset_matters(info['reference'], (r, c, h, w))
    assert len(set(origins)) >= 2 and causal_address, "demonstrate the recovered crop address"
    # Quarter-turn evidence supports C4; a reflected training crop supports D4.
    if not any(reg['orientation'] >= 4 for _, reg in training):
        assert all(reg['orientation'] < 4 for _, reg in records['test'])
    assert any(not np.array_equal(
        _expand(reg['residual'], info['stamp'], info['background']),
        _expand(reg['residual'], reg['registered'], info['background']))
        for info, reg in training), "demonstrate inverse registration of the tile"


def _canonical(reference):
    return min(_key(_transform(reference, i)) for i in range(8))


def self_test():
    """Bounded semantic and diversity audit of 42 default puzzles."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    def signature(p):
        return tuple((s, _key(pair['input']), _key(pair['output']))
                     for s in ('train', 'test') for pair in p[s])
    assert signature(corpus[0]) == signature(generate_puzzle(0))
    assert len({signature(p) for p in corpus}) == 42
    references, crop_shapes, offsets, orientations = set(), set(), set(), set()
    output_shapes, backgrounds, payload_counts, layouts = set(), set(), set(), set()
    for puzzle in corpus:
        references.add(_canonical(_read(puzzle['train'][0]['input'])['reference']))
        example = puzzle['train'][0]
        info = _read(example['input'])
        # Metamorphic input-only checks: physical layout and literal color
        # identities carry no latent instruction. Keep the exact panel contents.
        moved = _canvas(np.random.default_rng(2026), info['reference'], info['stamp'],
                        info['background'], info['frame'], info['signal'])
        assert np.array_equal(solve(moved), example['output'])
        permutation = np.random.default_rng(7).permutation(10)
        assert np.array_equal(solve(permutation[example['input']]), permutation[example['output']])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read(pair['input'])
                reg = _registration(info)
                crop_shapes.add(reg['registered'].shape)
                offsets.add(reg['offset'])
                orientations.add(reg['orientation'])
                output_shapes.add(np.asarray(pair['output']).shape)
                backgrounds.add(info['background'])
                payload_counts.add(len(set(map(int, np.unique(info['stamp'])))-{info['background']}))
                a, b = sorted(info['boxes'])
                layouts.add('vertical' if a[1] < b[0] else 'horizontal')
                assert _matches(info['reference'], info['stamp'] != info['background']) == [
                    (reg['orientation'], *reg['offset'])]
    assert len(references) >= 40, "measure reference geometry after D4 normalization"
    assert len(crop_shapes) >= 10 and len(offsets) >= 15
    assert orientations == set(range(8))
    assert len(output_shapes) >= 20 and len(backgrounds) == 10
    assert {2, 3, 4} <= payload_counts and layouts == {'vertical', 'horizontal'}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(123, nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in corpus[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed output accepted")
    return {'seeds': 42, 'reference_classes': len(references), 'crop_shapes': len(crop_shapes),
            'offsets': len(offsets), 'output_shapes': len(output_shapes),
            'orientations': sorted(orientations), 'backgrounds': sorted(backgrounds)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
