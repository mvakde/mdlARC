"""Rectangular occlusion in cropped, reflected pattern fields.

All reconstruction checks use the returned pixels.  Construction labels are used
only to form a solved example, never to select a completion during validation.
"""
import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset((
    'a1ff5569b48bc084c829ba909c7aeae40b0b0e1c4283bba48bd279feaff45969',
    'ca227920425ef8333a3df43aca9d317afcebb4017c479419ed2e46b92b79cbd3',
    '5aa45b64c0f672e84c67654313f4d45709f3c499581cdf6ae2393c766418f674',
    '681dfeb3562dd9f8f5e589485240588faad58c7f5167f642f98184b1b8e44ef5',
    'c8efc897a6a0784f0e4a6b91fdee7712c79a4b10e2237b846b6c15d61dc76cfd',
))


def _transform(a, code):
    if code < 4:
        return (a, a[::-1], a[:, ::-1], a[::-1, ::-1])[code].copy()
    return _transform(a.T, code - 4)


def _array_key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _source_copy(a):
    return a.shape == (30, 30) and hashlib.sha256(
        a.astype(np.int8).tobytes()).hexdigest() in _SOURCE_INPUT_DIGESTS


def _mask(a):
    candidates = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if (s-r)*(d-c) == len(rr) and len(rr) >= 2:
            candidates.append((int(color), (r, s, c, d)))
    assert len(candidates) == 1, 'The overwrite must be the unique rectangular color'
    return candidates[0]


def _axes(a, color):
    """Exact reflections on their in-bounds overlap, including offset axes."""
    h, w = a.shape
    rr, cc = np.indices(a.shape)
    result = []
    for axis, length in ((0, h), (1, w)):
        coord = rr if axis == 0 else cc
        for total in range(length // 2, (3 * length) // 2):
            mate = total - coord
            inside = (mate >= 0) & (mate < length) & (coord < mate)
            r, c = rr[inside], cc[inside]
            if len(r) < max(12, a.size * .20):
                continue
            x = a[r, c]
            y = a[total-r, c] if axis == 0 else a[r, total-c]
            visible = (x != color) & (y != color)
            if visible.sum() < max(10, len(r) * .40):
                continue
            if np.array_equal(x[visible], y[visible]) and len(np.unique(x[visible])) >= 2:
                result.append((axis, total))
    return result


def _assessment(a, color, ids, regions=None):
    """A candidate relation needs substantial visible witnesses, not just a fit
    to the missing patch. Singletons make no prediction; conflicts invalidate it.
    """
    _, labels = np.unique(ids, return_inverse=True)
    labels = labels.reshape(a.shape)
    flat = labels.ravel()
    visible = a.ravel() != color
    size = int(flat.max()) + 1
    lo = np.full(size, 10, dtype=np.int16)
    hi = np.full(size, -1, dtype=np.int16)
    count = np.bincount(flat[visible], minlength=size)
    np.minimum.at(lo, flat[visible], a.ravel()[visible])
    np.maximum.at(hi, flat[visible], a.ravel()[visible])
    if np.any((count > 0) & (lo != hi)):
        return None
    witnessed = visible & (count[flat] >= 2)
    if witnessed.sum() < max(20, a.size * .28):
        return None
    if regions is not None:
        for region in regions:
            selected = region.ravel() & visible
            if (witnessed & selected).sum() < max(6, selected.sum() * .25):
                return None
    prediction = hi[labels[a == color]]
    if not np.any(prediction >= 0):
        return None
    return prediction


def _orthogonal_ids(shape, axes):
    rr, cc = np.indices(shape)
    for axis, total in axes:
        if axis == 0:
            rr = np.minimum(rr, total-rr)
        else:
            cc = np.minimum(cc, total-cc)
    return (rr + 60) * 128 + cc + 60


def _models(a, color):
    """Infer region-bounded models from pixels; accept extra agreeing symmetry.

    Axial models have arbitrary half-integer axes. The two regional models
    enumerate size and crop, testing every visible cell in all related regions.
    A transpose inside one block is never applied to the rest of the canvas.
    """
    axes = _axes(a, color)
    models = []
    h_axes = [s for axis, s in axes if axis == 0]
    v_axes = [s for axis, s in axes if axis == 1]
    for relation in [[x] for x in axes] + [[(0, h), (1, v)] for h in h_axes for v in v_axes]:
        prediction = _assessment(a, color, _orthogonal_ids(a.shape, relation))
        if prediction is not None:
            models.append(('axial', tuple(relation), prediction))

    # The retained four-block family, with independent cropping and rotations.
    for transposed in (False, True):
        work = a.T if transposed else a
        sums = v_axes if transposed else h_axes
        h, w = work.shape
        rr, cc = np.indices(work.shape)
        for n in range(max(4, (max(h, w)+1)//2), min(17, (min(h, w)+8)//2)+1):
            for total in sums:
                delta = 2*n - 1 - total
                if delta % 2:
                    continue
                top = delta // 2
                if not 0 <= top <= 2*n-h:
                    continue
                for left in range(2*n-w+1):
                    r, c = rr+top, cc+left
                    folded = np.minimum(r, 2*n-1-r)
                    right = c >= n
                    for code in range(4, 8):
                        x, y = (c % n).copy(), folded.copy()
                        # A transformed array's coordinate lookup (not a rotation label).
                        if code in (5, 7):
                            y = n-1-y
                        if code in (6, 7):
                            x = n-1-x
                        ids = np.where(right, x*n+y, folded*n+c)
                        regions = [right, ~right]
                        if transposed:
                            ids = ids.T
                            regions = [v.T for v in regions]
                        prediction = _assessment(a, color, ids, regions)
                        if prediction is not None:
                            models.append(('transposed_blocks', (transposed, n, top, left, code), prediction))

    # Orthogonal reflections of [A B; B.T C]. A and C are independent:
    # only the off-diagonal blocks have a transpose relation.
    h, w = a.shape
    rr, cc = np.indices(a.shape)
    for n in range(6, 17, 2):
        if 2*n < max(h, w) or 2*n-min(h, w) > 8:
            continue
        for hs in h_axes:
            for vs in v_axes:
                if (2*n-1-hs) % 2 or (2*n-1-vs) % 2:
                    continue
                top, left = (2*n-1-hs)//2, (2*n-1-vs)//2
                if not (0 <= top <= 2*n-h and 0 <= left <= 2*n-w):
                    continue
                r = np.minimum(rr+top, 2*n-1-rr-top)
                c = np.minimum(cc+left, 2*n-1-cc-left)
                k = n//2
                region = (r >= k) & (c < k)
                other = (r < k) & (c >= k)
                ids = np.where(region, c*n+r, r*n+c)
                prediction = _assessment(a, color, ids, [region, other])
                if prediction is not None:
                    models.append(('local_transpose', (n, top, left), prediction))
    return axes, models


def _infer(a):
    color, bbox = _mask(a)
    axes, models = _models(a, color)
    assert models, 'No reflection relation has adequate visible support'
    predictions = np.stack([m[2] for m in models])
    lo = np.where(predictions >= 0, predictions, 10).min(axis=0)
    hi = predictions.max(axis=0)
    assert np.all(hi >= 0), 'An orbit was erased completely'
    assert np.array_equal(lo, hi), 'Supported correspondences disagree'
    r, s, c, d = bbox
    answer = hi.reshape(s-r, d-c).astype(np.int8)
    direct = [m[2] for m in models if m[0] == 'axial']
    axial_complete = bool(direct) and np.all(np.max(np.stack(direct), axis=0) >= 0)
    return answer, {'color': color, 'bbox': bbox, 'axes': axes,
                    'models': models, 'needs_transpose': not axial_complete,
                    'single_source': any(np.all(m[2] >= 0) for m in models if m[0] == 'axial' and len(m[1]) == 1)}


def solve(train_pairs, test_input):
    """Prediction interface: receives no test label, seed, or construction data."""
    assert train_pairs
    demonstrated_transpose = False
    for pair in train_pairs:
        predicted, record = _infer(pair['input'])
        assert np.array_equal(predicted, pair['output']), 'Inconsistent demonstration'
        demonstrated_transpose |= any(m[0] != 'axial' for m in record['models'])
    predicted, record = _infer(test_input)
    assert not record['needs_transpose'] or demonstrated_transpose, 'Undemonstrated reflection mechanism'
    return predicted


def _pattern(rng, shape, palette, style):
    h, w = shape
    palette = np.asarray(palette, dtype=np.int8)
    if style == 0:
        a = rng.choice(palette, size=shape)
    elif style == 1:
        weights = rng.dirichlet(np.full(len(palette), .65)) + .025
        a = rng.choice(palette, size=shape, p=weights/weights.sum())
    elif style == 2:
        scale = int(rng.integers(2, 4))
        small = rng.choice(palette, size=((h+scale-1)//scale, (w+scale-1)//scale))
        a = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)[:h, :w].copy()
    else:
        side = int(rng.integers(2, 5))
        motif = rng.choice(palette, size=(side, side))
        a = np.empty(shape, dtype=np.int8)
        for r in range(0, h, side):
            for c in range(0, w, side):
                if style == 3 or rng.random() < .3:
                    motif = rng.choice(palette, size=(side, side))
                block = motif.copy()
                rr, cc = np.indices(block.shape)
                block = block[np.minimum(rr, cc), np.maximum(rr, cc)]
                block = _transform(block, int(rng.integers(0, 8)))
                a[r:r+side, c:c+side] = block[:min(side, h-r), :min(side, w-c)]
    # Rare palette members have visible redundancy after reflection, too.
    positions = rng.choice(a.size, size=min(a.size, 2*len(palette)), replace=False)
    a.flat[positions] = np.resize(palette, len(positions))
    if rng.random() < .13:
        a = a[np.minimum(np.arange(h), h-1-np.arange(h))[:, None], np.arange(w)[None, :]]
    return a.astype(np.int8)


def _reflect(a, single_row=False, single_col=False):
    upper = np.concatenate((a, a[:, -2::-1] if single_col else a[:, ::-1]), axis=1)
    return np.concatenate((upper, upper[-2::-1] if single_row else upper[::-1]), axis=0)


def _field(rng, family, palette, style, cropped):
    if family == 'orthogonal':
        h, w = (int(rng.integers(4, 16)), int(rng.integers(4, 16)))
        if rng.random() < .25:
            w = h
        base = _pattern(rng, (h, w), palette, style)
        full = _reflect(base, rng.random() < .3, rng.random() < .3)
    elif family == 'blocks':
        n = int(rng.integers(5, 16))
        base = _pattern(rng, (n, n), palette, style)
        upper = np.concatenate((base, _transform(base, int(rng.integers(4, 8)))), axis=1)
        full = np.concatenate((upper, upper[::-1]), axis=0)
    else:
        k = int(rng.integers(3, 9))
        a, b, c = [_pattern(rng, (k, k), palette, style) for _ in range(3)]
        base = np.block([[a, b], [b.T, c]])
        full = _reflect(base)
    cuts = []
    for length in full.shape:
        maximum = max(0, min(8, length//3, length-8))
        amount = int(rng.integers(max(0, length-30), maximum+1)) if cropped else max(0, length-30)
        start = int(rng.integers(0, amount+1))
        cuts.append((start, length-amount+start))
    full = full[cuts[0][0]:cuts[0][1], cuts[1][0]:cuts[1][1]].copy()
    return _transform(full, int(rng.integers(0, 8)))


def _colors(rng):
    mask = int(rng.integers(0, 10))
    n = int(rng.integers(2, 10))
    palette = tuple(int(v) for v in rng.choice([c for c in range(10) if c != mask], n, replace=False))
    return mask, palette


def _orientation(shape):
    h, w = shape
    return 'square' if h == w else ('tall' if h > w else 'wide')


def _edges(shape, bbox):
    r, s, c, d = bbox
    return tuple(k for k, yes in enumerate((r == 0, d == shape[1], s == shape[0], c == 0)) if yes)


def _window(rng, full, orientation, edge, forbidden_edge, challenge):
    h, w = full.shape
    if orientation is None:
        orientation = str(rng.choice(['thin', 'square', 'tall', 'wide'], p=[.14, .18, .34, .34]))
    if orientation == 'thin':
        if rng.random() < .5:
            rh, rw = 1, int(rng.integers(2, min(w, 14)+1))
        else:
            rh, rw = int(rng.integers(2, min(h, 14)+1)), 1
    elif orientation == 'square':
        rh = rw = int(rng.integers(2, max(3, min(h, w)//2)+1))
    elif orientation == 'tall':
        rh = int(rng.integers(3, max(4, min(h-1, 18))+1))
        rw = int(rng.integers(1, min(w-1, rh-1, max(2, w//2))+1))
    else:
        rw = int(rng.integers(3, max(4, min(w-1, 18))+1))
        rh = int(rng.integers(1, min(h-1, rw-1, max(2, h//2))+1))
    r, c = int(rng.integers(0, h-rh+1)), int(rng.integers(0, w-rw+1))
    if challenge:
        axes = _axes(full, -1)
        if axes:
            axis, total = axes[int(rng.integers(len(axes)))]
            length = rh if axis == 0 else rw
            center = (total+1)//2
            depth = length//2 if rng.random() < .35 else int(rng.integers(1, max(2, length)))
            start = int(np.clip(center-depth, 0, (h-rh) if axis == 0 else (w-rw)))
            if axis == 0:
                r = start
            else:
                c = start
    if edge is not None:
        if edge == 0:
            r = 0
        elif edge == 1:
            c = w-rw
        elif edge == 2:
            r = h-rh
        elif edge == 3:
            c = 0
        elif h-rh >= 2 and w-rw >= 2:
            r, c = max(1, min(r, h-rh-1)), max(1, min(c, w-rw-1))
    box = r, r+rh, c, c+rw
    if forbidden_edge in _edges(full.shape, box) or rh*rw > full.size*.26:
        return None
    return box


def _make_pair(rng, params, family, regime, orientation, edge, forbidden_edge, need_transpose, seen):
    color, palette = _colors(rng) if params['vary_colors'] else params['colors']
    style = int(rng.integers(0, 5)) if rng.random() < params['intensity'] else params['style']
    for attempt in range(90):
        # A bounded fallback retains the rule while dropping optional crop/style preferences.
        fallback = attempt >= 60
        chosen_family = 'blocks' if fallback and need_transpose else ('orthogonal' if fallback else family)
        full = _field(rng, chosen_family, palette, 0 if fallback else style,
                      False if fallback else rng.random() < params['crop_probability'])
        for _ in range(18):
            challenge = need_transpose or rng.random() < (.65 if regime == 'target_test' else .42)
            box = _window(rng, full, orientation, edge, forbidden_edge, challenge)
            if box is None:
                continue
            r, s, c, d = box
            output = full[r:s, c:d].copy()
            inp = full.copy()
            inp[r:s, c:d] = color
            if _array_key(inp) in seen or _source_copy(inp):
                continue
            try:
                inferred, record = _infer(inp)
            except AssertionError:
                continue
            if need_transpose and not record['needs_transpose']:
                continue
            if not np.array_equal(inferred, output):
                continue
            return {'input': inp, 'output': output}
    # Finite deterministic fallback: a well-witnessed legacy field with a seam window.
    # Color values are generated arithmetically to avoid an unbounded random retry.
    for phase in range(32):
        rr, cc = np.indices((10, 10))
        indices = ((rr*17 + cc*31 + rr*cc*7 + (rr+phase)**2 + cc**3) % 101) % len(palette)
        base = np.asarray(palette, dtype=np.int8)[indices]
        upper = np.concatenate((base, base.T), axis=1)
        full = np.concatenate((upper, upper[::-1]), axis=0)
        for turn in range(4):
            transformed = np.rot90(full, turn).copy()
            for box in ((7, 13, 0, 3), (7, 13, 17, 20), (0, 3, 7, 13), (17, 20, 7, 13), (7, 13, 2, 5)):
                r, s, c, d = box
                if orientation in ('tall', 'wide') and _orientation((s-r, d-c)) != orientation:
                    continue
                edges = _edges(transformed.shape, box)
                if forbidden_edge in edges or (edge in range(4) if edge is not None else False) and edge not in edges:
                    continue
                inp = transformed.copy()
                inp[r:s, c:d] = color
                try:
                    inferred, record = _infer(inp)
                except AssertionError:
                    continue
                if need_transpose and not record['needs_transpose']:
                    continue
                if _array_key(inp) not in seen and np.array_equal(inferred, transformed[r:s, c:d]):
                    return {'input': inp, 'output': transformed[r:s, c:d].copy()}
    raise AssertionError('No supported mask after bounded construction and fallback')
def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer count")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer count")
    if not lower <= int(value) <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        seed_key = abs(int(seed))
        resolved_test = 1 + seed_key % 4
        possible_train_counts = 10 - resolved_test
        resolved_train = (
            resolved_test
            + 1
            + (seed_key // 4) % possible_train_counts
        )
    elif num_train is None:
        resolved_test = int(num_test)
        minimum_train = max(2, resolved_test + 1)
        resolved_train = int(
            rng.integers(minimum_train, 11)
        )
    elif num_test is None:
        resolved_train = int(num_train)
        maximum_test = min(4, resolved_train - 1)
        resolved_test = int(
            rng.integers(1, maximum_test + 1)
        )
    else:
        resolved_train = int(num_train)
        resolved_test = int(num_test)

    if resolved_train <= resolved_test:
        raise ValueError("num_train must be greater than num_test")

    return resolved_train, resolved_test




def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    nt, nq = _resolve_counts(seed, rng, num_train, num_test)
    intensity = float(rng.choice([.15, .5, .9]))
    preferred = str(rng.choice(['orthogonal', 'blocks', 'local']))
    params = {'intensity': intensity, 'style': int(rng.integers(5)),
              'vary_colors': rng.random() < .75, 'colors': _colors(rng),
              'crop_probability': float(rng.uniform(.25, .95))}
    regime = str(rng.choice(['consistent', 'mixed', 'held_edge', 'held_orientation'], p=[.22, .36, .24, .18]))
    train_orientation = str(rng.choice(['tall', 'wide'])) if regime in ('consistent', 'held_orientation') else None
    test_orientation = ('wide' if train_orientation == 'tall' else 'tall') if regime == 'held_orientation' else train_orientation
    held_edge = int(rng.integers(4)) if regime == 'held_edge' else None
    families = [preferred if rng.random() >= intensity else str(rng.choice(['orthogonal', 'blocks', 'local'])) for _ in range(nt+nq)]
    needs_demonstration = any(f != 'orthogonal' for f in families)
    if needs_demonstration:
        families[0] = 'blocks'
    puzzle = {'train': [], 'test': []}
    seen = set()
    for index, family in enumerate(families):
        train = index < nt
        split = 'train' if train else 'test'
        # All four edges and interior windows occur in either split. A held-out
        # edge/orientation is a placement shift, never a new transformation.
        edge = held_edge if not train and held_edge is not None else int(rng.integers(-1, 4))
        if train and edge == held_edge:
            edge = -1
        pair = _make_pair(rng, params, family, 'source_train' if train else 'target_test',
                          train_orientation if train else test_orientation, edge,
                          held_edge if train else None,
                          needs_demonstration and index == 0, seen)
        puzzle[split].append(pair)
        seen.add(_array_key(pair['input']))
    # Demonstrations establish target ordering; symmetric answers are still legal
    # elsewhere, including monochrome and one-dimensional answers.
    if not any(p['output'].size > 2 and not np.array_equal(p['output'], p['output'][::-1, ::-1]) for p in puzzle['train']):
        puzzle['train'][0] = _make_pair(rng, params, 'blocks', 'source_train', train_orientation,
                                        -1, held_edge, True, seen)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen_inputs, seen_pairs = set(), set()
    records = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = pair['input'], pair['output']
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _array_key(a)
            pair_key = key, _array_key(b)
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert not _source_copy(a)
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            # Locate the mask and infer without using output dimensions or colors.
            predicted, record = _infer(a)
            assert np.array_equal(predicted, b), 'Incorrect target-oriented crop'
            assert record['color'] not in b
            assert 2 <= len(np.unique(a[a != record['color']])) <= 9
            records[split].append(record)
    demonstrated = any(any(m[0] != 'axial' for m in r['models']) for r in records['train'])
    assert demonstrated or not any(r['needs_transpose'] for r in records['test'])
    assert any(p['output'].size > 2 and not np.array_equal(p['output'], p['output'][::-1, ::-1]) for p in puzzle['train'])
    # No color/side/parity/order convention is hidden: each is observed locally.


def measurements(puzzle):
    """Review features reconstructed from actual arrays, not sampler metadata."""
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a, b = pair['input'], pair['output']
            _, record = _infer(a)
            r, s, c, d = record['bbox']
            full = a.copy()
            full[r:s, c:d] = b
            palette, counts = np.unique(full, return_counts=True)
            equality = (np.mean(full[1:] == full[:-1]) + np.mean(full[:, 1:] == full[:, :-1]))/2
            overlaps = []
            for axis, total in record['axes']:
                start, end = (r, s) if axis == 0 else (c, d)
                if 2*start < total < 2*(end-1):
                    delta = start+end-1-total
                    overlaps.append('balanced' if delta == 0 else ('negative' if delta < 0 else 'positive'))
            # Internal diagonal motifs are measured on completed 3x3 neighborhoods.
            patches = np.lib.stride_tricks.sliding_window_view(full, (3, 3))
            symmetry = np.all(patches == patches.swapaxes(-1, -2), axis=(-1, -2))
            nonconstant = patches.max(axis=(-1, -2)) != patches.min(axis=(-1, -2))
            # Count repetitions inside a corner, without counting the field's
            # compulsory opposite-half reflection as a repeated motif.
            core = full[:max(3, full.shape[0]//2), :max(3, full.shape[1]//2)]
            small = np.lib.stride_tricks.sliding_window_view(core, (2, 2)).reshape(-1, 4)
            small = small[small.max(axis=1) != small.min(axis=1)]
            motif_repeats = len(small) - len(np.unique(small, axis=0))
            result.append({'split': split, 'input_shape': a.shape, 'output_shape': b.shape,
                'mask_color': record['color'], 'palette': tuple(int(v) for v in palette),
                'palette_size': len(palette), 'output_colors': len(np.unique(b)),
                'orientation': _orientation(b.shape), 'thin': min(b.shape) == 1,
                'edges': _edges(a.shape, record['bbox']), 'bbox': record['bbox'],
                'axes': tuple(record['axes']), 'offset': any(t != a.shape[k]-1 for k, t in record['axes']),
                'overlaps': tuple(overlaps), 'models': tuple(sorted(set(m[0] for m in record['models']))),
                'needs_transpose': record['needs_transpose'], 'single_source': record['single_source'],
                'dominant_fraction': float(counts.max()/counts.sum()), 'adjacent_equal': float(equality),
                'symmetric_motifs': int(np.count_nonzero(symmetry & nonconstant)),
                'motif_repeats': motif_repeats,
                'whole_symmetries': tuple(k for k in range(1, 8) if _transform(full, k).shape == full.shape and np.array_equal(full, _transform(full, k)))})
    return result


def self_test() -> None:
    all_features, signatures, train_counts, test_counts = [], set(), set(), set()
    changing_colors, changing_shapes, changing_orientation, changing_layout = 0, 0, 0, 0
    held_orientation, consistent_orientation, held_edges = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        key = tuple(tuple((_array_key(p['input']), _array_key(p['output'])) for p in puzzle[sp]) for sp in ('train', 'test'))
        signatures.add(key)
        if seed in (0, 7, 41):
            again = generate_puzzle(seed)
            assert key == tuple(tuple((_array_key(p['input']), _array_key(p['output'])) for p in again[sp]) for sp in ('train', 'test'))
        train = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            prediction = solve(train, pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
        features = measurements(puzzle)
        all_features.extend(features)
        changing_colors += len({f['mask_color'] for f in features}) > 1
        changing_shapes += len({f['input_shape'] for f in features}) > 1
        changing_orientation += len({f['orientation'] for f in features}) > 1
        changing_layout += len({f['models'] for f in features}) > 1
        tr = {f['orientation'] for f in features if f['split'] == 'train'}
        te = {f['orientation'] for f in features if f['split'] == 'test'}
        held_orientation += bool(te-tr)
        consistent_orientation += len(tr | te) == 1
        train_edges = {e for f in features if f['split'] == 'train' for e in f['edges']}
        test_edges = {e for f in features if f['split'] == 'test' for e in f['edges']}
        held_edges += bool(test_edges-train_edges)
    assert len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert changing_colors >= 20 and changing_shapes >= 38 and changing_orientation >= 22 and changing_layout >= 10
    assert held_orientation >= 3 and consistent_orientation >= 3 and held_edges >= 5
    assert len({f['input_shape'] for f in all_features}) >= 60
    assert len({f['output_shape'] for f in all_features}) >= 45
    assert {f['palette_size'] for f in all_features} == set(range(2, 10))
    assert {f['mask_color'] for f in all_features} == set(range(10))
    assert sum(0 not in f['palette'] and f['mask_color'] != 0 for f in all_features) >= 10
    assert sum(f['thin'] for f in all_features) >= 10
    assert sum(f['output_colors'] < 3 for f in all_features) >= 10
    assert sum(f['offset'] for f in all_features) >= 60
    assert sum(f['input_shape'][0] != f['input_shape'][1] for f in all_features) >= 80
    assert sum(any(n % 2 for n in f['input_shape']) for f in all_features) >= 80
    assert min(min(f['input_shape']) for f in all_features) <= 9
    assert max(max(f['input_shape']) for f in all_features) == 30
    assert sum(f['needs_transpose'] for f in all_features) >= 20
    assert sum(f['single_source'] for f in all_features) >= 40
    assert {'balanced', 'negative', 'positive'} <= {v for f in all_features for v in f['overlaps']}
    assert sum(f['dominant_fraction'] > .5 for f in all_features) >= 10
    assert sum(f['adjacent_equal'] > .5 for f in all_features) >= 10
    assert sum(f['adjacent_equal'] < .22 for f in all_features) >= 10
    assert sum(f['symmetric_motifs'] >= 8 for f in all_features) >= 10
    assert sum(f['motif_repeats'] >= 10 for f in all_features) >= 10
    assert sum(len(f['whole_symmetries']) >= 2 for f in all_features) >= 10
    for split in ('train', 'test'):
        subset = [f for f in all_features if f['split'] == split]
        assert {0, 1, 2, 3} == {e for f in subset for e in f['edges']}
        assert any(not f['edges'] for f in subset)
        assert {'tall', 'wide', 'square'} <= {f['orientation'] for f in subset}
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000+index, *counts)
        validate_generated_puzzle(p)
        assert all(x is None or len(p[sp]) == x for x, sp in zip(counts, ('train', 'test')))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    # Mutations must fail without consulting construction state.
    p = generate_puzzle(6, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong label accepted')
    # Rotation and independent color bijections preserve semantics.
    p = generate_puzzle(19, 3, 1)
    rotated = {sp: [{k: np.rot90(v).copy() for k, v in pair.items()} for pair in p[sp]] for sp in p}
    validate_generated_puzzle(rotated)
    # Source-style 32x32 field cropped to 30x30: offset axes sum to 31,
    # independent diagonal blocks, and a left-edge query needing local transpose.
    rng = np.random.default_rng(81723)
    aa, bb, cc = [rng.integers(1, 8, (8, 8), dtype=np.int8) for _ in range(3)]
    full = _reflect(np.block([[aa, bb], [bb.T, cc]]))[:30, :30]
    masked = full.copy()
    masked[14:23, :3] = 8
    answer, record = _infer(masked)
    assert np.array_equal(answer, full[14:23, :3])
    assert record['needs_transpose'] and set(record['axes']) == {(0, 31), (1, 31)}
    assert not np.array_equal(full, full.T)
    # A rectangle crossing both axes can erase a whole orbit and must fail.
    base = rng.integers(1, 8, (6, 7), dtype=np.int8)
    full = _reflect(base)
    erased = full.copy()
    erased[5:7, 6:8] = 8
    corrupted = full.copy()
    corrupted[1:3, 1:4] = 8
    corrupted[-1, -1] = 1 + int(corrupted[-1, -1]) % 7
    for invalid in (erased, corrupted):
        try:
            _infer(invalid)
        except AssertionError:
            continue
        raise AssertionError('Erased orbit or conflicting visible evidence accepted')
    # Internal symmetries and cropped odd canvases are legitimate when answers agree.
    base = rng.integers(1, 8, (7, 7), dtype=np.int8)
    rr, cc = np.indices(base.shape)
    base = base[np.minimum(rr, cc), np.maximum(rr, cc)]
    upper = np.concatenate((base, base.T), axis=1)
    full = np.concatenate((upper, upper[::-1]), axis=0)[:-1, :-1]
    masked = full.copy()
    masked[1:4, 7:10] = 8
    answer, record = _infer(masked)
    assert np.array_equal(answer, full[1:4, 7:10]) and len(record['models']) > 1
    for index, pair in enumerate(rotated['train']+rotated['test']):
        perm = np.roll(np.arange(10, dtype=np.int8), index+1)
        for k in pair:
            pair[k] = perm[pair[k]]
    validate_generated_puzzle(rotated)
