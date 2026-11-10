import numpy as np


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _check_count("num_train", num_train, 2, 10)
    ns = _check_count("num_test", num_test, 1, 4)

    if nt is not None and ns is not None:
        if nt <= ns:
            raise ValueError("num_train must be greater than num_test")
        return nt, ns

    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    else:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))

    if not (2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns):
        raise ValueError("impossible train/test count combination")
    return nt, ns


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = []

    for start_r, start_c in zip(*np.nonzero(mask)):
        start_r = int(start_r)
        start_c = int(start_c)

        if seen[start_r, start_c]:
            continue

        seen[start_r, start_c] = True
        stack = [(start_r, start_c)]
        cells = []

        while stack:
            r, c = stack.pop()
            cells.append((r, c))

            for nr, nc in (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ):
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and mask[nr, nc]
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(cells)

    return components


def _rectangles_overlap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    return not (
        ar1 <= br0
        or br1 <= ar0
        or ac1 <= bc0
        or bc1 <= ac0
    )


# The source rule has no hidden color dictionary. Difficulty and allowed clue
# axes are puzzle-level sampling choices; all geometric/color evidence is local.

def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _draw(plan, background, shape, mini, base, side, stride, axis, match, length):
    inp = np.full(shape, background, dtype=np.int8)
    mr, mc = mini
    inp[mr:mr + 2 * plan.shape[0]:2, mc:mc + 2 * plan.shape[1]:2] = plan
    for k in range(length):
        r, c = match[0] + (axis == 0) * k, match[1] + (axis == 1) * k
        y, x = base[0] + r * stride, base[1] + c * stride
        inp[y:y + side, x:x + side] = plan[r, c]
    out = inp.copy()
    for (r, c), color in np.ndenumerate(plan):
        if color != background:
            y, x = base[0] + r * stride, base[1] + c * stride
            out[y:y + side, x:x + side] = color
    return {"input": inp, "output": out}


def _source_pairs():
    # Compact exact-content fixtures, transcribed from the staged source grids.
    specs = (
        (8, 23, [[4, 4], [1, 1], [1, 1]], (1, 8), 3, 5, (0, 0)),
        (8, 23, [[2, 2, 8], [1, 1, 8], [1, 1, 3], [1, 1, 3]],
         (4, 10), 2, 3, (0, 0)),
        (1, 23, [[2, 3], [3, 3], [3, 4]], (10, 11), 2, 5, (2, 0)),
        (3, 27, [[1, 1, 1], [4, 4, 2], [1, 4, 1], [8, 3, 3]],
         (5, 8), 4, 5, (1, 1)),
    )
    return [_draw(np.array(p, dtype=np.int8), bg, (n, n), (1, 1), b, s, t, 1, m, 2)
            for bg, n, p, b, s, t, m in specs]


def _asymmetric(plan):
    variants = [np.flip(plan, 0), np.flip(plan, 1), np.flip(plan, (0, 1))]
    if plan.shape[0] == plan.shape[1]:
        variants += [np.rot90(plan), np.rot90(plan, 3), plan.T,
                     np.flip(plan.T, (0, 1))]
    return all(not np.array_equal(plan, other) for other in variants)


def _segments(plan, bg, axis):
    """All uniquely located occupied sequences, including repeated-color runs."""
    h, w = plan.shape
    found = []
    for length in range(2, plan.shape[axis] + 1):
        sequences = {}
        for r in range(h - (length - 1 if axis == 0 else 0)):
            for c in range(w - (length - 1 if axis == 1 else 0)):
                seq = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                            for k in range(length))
                if bg not in seq:
                    sequences.setdefault(seq, []).append((r, c))
        for seq, positions in sequences.items():
            if len(positions) == 1:
                found.append((positions[0], length, seq))
    return found


def _sample_plan(rng, shape, bg, palette, axis, difficulty, regime, demo):
    h, w = shape
    for attempt in range(80):
        plan = rng.choice(palette, size=shape).astype(np.int8)
        # Copying arbitrary rows/columns encourages legitimate repetition. Only
        # uniqueness of the selected seed matters, not uniqueness of every row.
        if rng.random() < .28:
            dim = int(rng.integers(2))
            if shape[dim] >= 3:
                a, b = rng.choice(shape[dim], 2, replace=False)
                if dim == 0:
                    plan[a] = plan[b]
                else:
                    plan[:, a] = plan[:, b]
        mode = 0 if difficulty == "simple" else int(rng.choice(5, p=[.23, .18, .22, .22, .15]))
        if difficulty == "rich":
            mode = int(rng.choice([0, 1, 2, 3, 4], p=[.08, .08, .30, .30, .24]))
        mask = np.zeros(shape, dtype=bool)
        if mode == 1:
            mask[int(rng.integers(h)), int(rng.integers(w))] = True
        elif mode in (2, 4):
            count = int(rng.integers(2, max(3, (h * w) // 3 + 1)))
            mask.flat[rng.choice(h * w, count, replace=False)] = True
        if mode in (3, 4):
            dim = int(rng.integers(2))
            extent = shape[dim]
            length = int(rng.integers(2, extent + 1))
            start = int(rng.integers(extent - length + 1))
            cross = int(rng.integers(shape[1 - dim]))
            if dim == 0:
                mask[start:start + length, cross] = True
            else:
                mask[cross, start:start + length] = True
        plan[mask] = bg
        # Extent is witnessed by at least one swatch on each extreme. Boundary
        # blanks and completely empty interior lines remain legal.
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        occupied = np.argwhere(plan != bg)
        if any(not np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
            continue
        if demo and not _asymmetric(plan):
            continue
        choices = []
        weights = []
        for match, length, seq in _segments(plan, bg, axis):
            rest = plan.copy()
            for k in range(length):
                rest[match[0] + (axis == 0) * k, match[1] + (axis == 1) * k] = bg
            added = set(int(v) for v in rest.flat) - {bg}
            if not added or (demo and len(added) < 2):
                continue
            start, cross = match[axis], match[1 - axis]
            extent = shape[axis]
            interior_cross = 0 < cross < shape[1 - axis] - 1
            if difficulty == "simple" and (length != extent or interior_cross):
                continue
            if difficulty == "source_shift" and regime == "target_test":
                if not (interior_cross and length < extent and start + length == extent):
                    continue
            weight = 1.0
            if difficulty == "rich":
                weight *= (3 if length >= 3 else 1) * (3 if interior_cross else 1)
            elif regime == "target_test":
                weight *= 2 if interior_cross else 1
            choices.append((match, length))
            weights.append(weight)
        if choices:
            weights = np.array(weights) / sum(weights)
            match, length = choices[int(rng.choice(len(choices), p=weights))]
            return plan, match, length
    if difficulty == 'source_shift' and regime == 'target_test':
        plan = np.full(shape, palette[0], dtype=np.int8)
        match = (1, w - 2) if axis == 1 else (h - 2, 1)
        plan[match] = palette[1]
        return plan, match, 2
    # Bounded, constructive fallback: a unique full boundary sequence, a second
    # payload color among the additions, and visibly established plan extent.
    plan = np.full(shape, palette[0], dtype=np.int8)
    if axis == 1:
        plan[0, 0] = palette[1]
        plan[-1, -1] = palette[1]
        if h > 2:
            plan[1, -1] = palette[1]
        else:
            plan[-1, 0] = palette[1]
    else:
        plan[0, 0] = palette[1]
        plan[-1, -1] = palette[1]
        if w > 2:
            plan[-1, 1] = palette[1]
        else:
            plan[0, -1] = palette[1]
    # A third distinct corner avoids accidental symmetries in the 2x2 case.
    if not _asymmetric(plan):
        extra = next(v for v in range(10) if v != bg and v not in palette[:2])
        plan[-1, -1] = extra
    return plan, (0, 0), shape[axis]


def _geometry_options(difficulty, regime):
    options = []
    for h in range(2, 7):
        for w in range(2, 7):
            if difficulty == "simple" and max(h, w) > 4:
                continue
            if difficulty == "rich" and max(h, w) < 4:
                continue
            if difficulty == "source_shift" and regime == "target_test":
                if not (3 <= h <= 4 and 3 <= w <= 4):
                    continue
            for side in range(2, 6):
                if difficulty == "simple" and side > 4:
                    continue
                if difficulty == "source_shift" and regime == "target_test" and side != 4:
                    continue
                for gutter in range(1, 5):
                    box = ((h - 1) * (side + gutter) + side,
                           (w - 1) * (side + gutter) + side)
                    mini = (2 * h - 1, 2 * w - 1)
                    for separation_axis in (0, 1):
                        required = [max(box[d], mini[d]) for d in (0, 1)]
                        required[separation_axis] = box[separation_axis] + mini[separation_axis] + 1
                        if max(required) <= 30:
                            options.append((h, w, side, gutter, separation_axis))
    return options


def _place(rng, plan_shape, side, gutter, separation_axis):
    box = tuple((n - 1) * (side + gutter) + side for n in plan_shape)
    mini = tuple(2 * n - 1 for n in plan_shape)
    d, other = separation_axis, 1 - separation_axis
    gap = int(rng.integers(1, min(4, 30 - box[d] - mini[d]) + 1))
    required = [max(box[k], mini[k]) for k in (0, 1)]
    required[d] = box[d] + mini[d] + gap
    canvas_mode = int(rng.choice(4, p=[.35, .30, .25, .10]))
    if canvas_mode == 3:
        shape = (30, 30)
    elif canvas_mode == 2:
        n = int(rng.integers(max(required), 31))
        shape = (n, n)
    else:
        shape = tuple(int(rng.integers(n, min(30, n + (3 if canvas_mode == 0 else 10)) + 1))
                      for n in required)
    base, anchor = [0, 0], [0, 0]
    extra = shape[d] - required[d]
    margins = sorted(int(x) for x in rng.integers(0, extra + 1, size=2))
    if rng.integers(2):
        base[d] = margins[0]
        anchor[d] = base[d] + box[d] + gap + margins[1] - margins[0]
    else:
        anchor[d] = margins[0]
        base[d] = anchor[d] + mini[d] + gap + margins[1] - margins[0]
    base[other] = int(rng.integers(shape[other] - box[other] + 1))
    anchor[other] = int(rng.integers(shape[other] - mini[other] + 1))
    return shape, tuple(anchor), tuple(base)


def _make_pair(rng, difficulty, regime, axis, options, avoid=None):
    choices = options
    if avoid is not None:
        choices = [x for x in choices if (x[0], x[1]) != avoid['plan_shape'] and x[2] != avoid['side']]
    h, w, side, gutter, separation = choices[int(rng.integers(len(choices)))]
    colors = rng.permutation(10)
    bg = int(colors[0])
    if avoid is not None and bg == avoid['background']:
        colors[0], colors[1] = colors[1], colors[0]
        bg = int(colors[0])
    size = int(rng.integers(2, 4 if difficulty == 'simple' else 7))
    palette = colors[1:size + 1]
    plan, match, length = _sample_plan(rng, (h, w), bg, palette, axis, difficulty,
                                      regime, regime == 'source_train')
    shape, mini, base = _place(rng, plan.shape, side, gutter, separation)
    return _draw(plan, bg, shape, mini, base, side, side + gutter, axis, match, length)


def _recover(inp):
    """Recover all task geometry from input pixels, without a split or ordinal."""
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > np.partition(counts, -2)[-2], 'ambiguous background'
    components = _components(inp != bg)
    swatches = [c[0] for c in components if len(c) == 1]
    assert len(swatches) >= 3
    squares = []
    for cells in components:
        if len(cells) == 1:
            continue
        a = np.array(cells)
        r, c = a.min(axis=0)
        bottom, right = a.max(axis=0) + 1
        side = int(bottom - r)
        assert side >= 2 and right - c == side and len(cells) == side * side
        color = int(inp[r, c])
        assert np.all(inp[r:bottom, c:right] == color), 'seed must be solid'
        squares.append((int(r), int(c), side, color))
    assert len(squares) >= 2
    points = np.array(swatches)
    mr, mc = (int(x) for x in points.min(axis=0))
    er, ec = (int(x) for x in points.max(axis=0) + 1)
    assert np.all((points - (mr, mc)) % 2 == 0)
    assert any(np.diff(np.unique(points[:, 0])) == 2)
    assert any(np.diff(np.unique(points[:, 1])) == 2)
    plan = inp[mr:er:2, mc:ec:2].copy()
    h, w = plan.shape
    assert min(h, w) >= 2
    mini_mask = np.zeros((er - mr, ec - mc), dtype=bool)
    mini_mask[::2, ::2] = True
    assert np.all(inp[mr:er, mc:ec][~mini_mask] == bg)
    assert np.count_nonzero(plan != bg) == len(swatches)
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        assert len({s[1] for s in squares}) == 1, 'seed must lie on one axis'
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    assert all(s[2] == side for s in squares)
    differences = np.diff([s[axis] for s in squares])
    stride = int(differences[0])
    assert np.all(differences == stride) and stride > side
    sequence = tuple(s[3] for s in squares)
    length = len(sequence)
    matches = []
    for r in range(h - (length - 1 if axis == 0 else 0)):
        for c in range(w - (length - 1 if axis == 1 else 0)):
            seq = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                        for k in range(length))
            if seq == sequence:
                matches.append((r, c))
    # Distinct translations of a finite nonempty plan yield different grids.
    assert len(matches) == 1, 'seed colors must determine one lattice origin'
    match = matches[0]
    base = (squares[0][0] - match[0] * stride, squares[0][1] - match[1] * stride)
    br, bc = base
    bottom, right = br + (h - 1) * stride + side, bc + (w - 1) * stride + side
    assert 0 <= br < bottom <= inp.shape[0] and 0 <= bc < right <= inp.shape[1]
    assert not _rectangles_overlap((br, bc, bottom, right), (mr, mc, er, ec))
    seed_sites = {(match[0] + (axis == 0) * k, match[1] + (axis == 1) * k)
                  for k in range(length)}
    expected = inp.copy()
    additions = []
    for (r, c), value in np.ndenumerate(plan):
        y, x = br + r * stride, bc + c * stride
        region = inp[y:y + side, x:x + side]
        assert np.all(region == (value if (r, c) in seed_sites else bg))
        if value != bg:
            expected[y:y + side, x:x + side] = value
            if (r, c) not in seed_sites:
                additions.append((r, c, int(value)))
    assert additions, 'at least one square must be added'
    holes = plan == bg
    edge_mask = np.zeros(plan.shape, dtype=bool)
    edge_mask[0] = edge_mask[-1] = True
    edge_mask[:, 0] = edge_mask[:, -1] = True
    start = match[axis]
    extent = plan.shape[axis]
    segment = ('whole' if length == extent else 'prefix' if start == 0 else
               'suffix' if start + length == extent else 'interior')
    cross = match[1 - axis]
    cross_kind = 'first' if cross == 0 else 'last' if cross == plan.shape[1 - axis] - 1 else 'interior'
    direction = tuple(name for name, present in (
        ('up', any(r < match[0] for r, c, v in additions)),
        ('down', any(r > match[0] + (length - 1 if axis == 0 else 0) for r, c, v in additions)),
        ('left', any(c < match[1] for r, c, v in additions)),
        ('right', any(c > match[1] + (length - 1 if axis == 1 else 0) for r, c, v in additions))) if present)
    relations = tuple(name for name, yes in (('above', bottom <= mr), ('below', er <= br),
                                             ('left', right <= mc), ('right', ec <= bc)) if yes)
    short_matches = sum(
        tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k]) for k in range(2)) == sequence[:2]
        for r in range(h - (axis == 0)) for c in range(w - (axis == 1)))
    meta = dict(background=bg, palette=tuple(sorted(set(int(v) for v in plan.flat) - {bg})),
                plan_shape=plan.shape, holes=int(holes.sum()), edge_holes=int((holes & edge_mask).sum()),
                blank_run=bool(np.any(holes[:-1] & holes[1:]) or np.any(holes[:, :-1] & holes[:, 1:])),
                separated_blanks=len(_components(holes)) >= 2,
                repeated_rows=len({tuple(row) for row in plan.tolist()}) < h,
                repeated_columns=len({tuple(col) for col in plan.T.tolist()}) < w,
                equal_seed=len(set(sequence)) == 1, side=side, gutter=stride - side, stride=stride,
                longer_resolves_repetition=length > 2 and short_matches > 1,
                interior_holes=int((holes & ~edge_mask).sum()),
                seed_line_holes=int(np.count_nonzero(plan[match[0], :] == bg) if axis == 1
                                    else np.count_nonzero(plan[:, match[1]] == bg)),
                axis='horizontal' if axis == 1 else 'vertical', length=length, match=match,
                segment=segment, cross=cross_kind, directions=direction, base=base, miniature=(mr, mc),
                placement=relations, canvas=inp.shape, asymmetric=_asymmetric(plan),
                added_colors=tuple(sorted({v for r, c, v in additions})))
    return expected, meta


def solve(train_pairs, test_input):
    """Prediction interface: labels of tests and generation state are inaccessible."""
    demonstrated = set()
    for pair in train_pairs:
        expected, meta = _recover(pair['input'])
        assert np.array_equal(expected, pair['output'])
        if meta['asymmetric'] and len(meta['added_colors']) >= 2:
            demonstrated.add(meta['axis'])
    prediction, meta = _recover(test_input)
    assert meta['axis'] in demonstrated, 'training must demonstrate order-preserving transfer for this axis'
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, source_keys = set(), {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
    metas = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            assert (key, _grid_key(out)) not in source_keys, 'exact source copy'
            expected, meta = _recover(inp)
            assert np.array_equal(expected, out), 'incorrect completion or changed unrelated cell'
            metas[split].append(meta)
    teaching = {m['axis'] for m in metas['train'] if m['asymmetric'] and len(m['added_colors']) >= 2}
    assert {m['axis'] for m in metas['train'] + metas['test']} <= teaching
    # Two distinct demonstrations expose scale inference and local color roles.
    for feature in ('plan_shape', 'side', 'background'):
        assert len({m[feature] for m in metas['train']}) >= 2


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    difficulty = str(rng.choice(['simple', 'mixed', 'rich', 'source_shift'], p=[.18, .40, .25, .17]))
    axis_mode = str(rng.choice(['horizontal', 'vertical', 'mixed'], p=[.48, .17, .35]))
    axes = [1] if axis_mode == 'horizontal' else [0] if axis_mode == 'vertical' else [0, 1]
    option_sets = {regime: _geometry_options(difficulty, regime)
                   for regime in ('source_train', 'target_test')}
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    rng.shuffle(train_axes)
    result, seen = {'train': [], 'test': []}, set()
    previous = None
    for split, regime, count in (('train', 'source_train', nt), ('test', 'target_test', ns)):
        for index in range(count):
            axis = train_axes[index] if split == 'train' else int(rng.choice(axes))
            # Continuous RNG consumption, with bounded retries only for exact
            # duplicate exclusion. No alignment depends on the returned ordinal.
            for attempt in range(32):
                pair = _make_pair(rng, difficulty, regime, axis, option_sets[regime],
                                  previous if split == 'train' and index == 1 else None)
                key = _grid_key(pair['input'])
                source_copy = any(np.array_equal(pair['input'], p['input']) and
                                  np.array_equal(pair['output'], p['output']) for p in _source_pairs())
                if key not in seen and not source_copy:
                    break
            else:
                # Ninety injective recolorings exceed the fourteen-pair bound
                # plus source exclusions. They also cover two-color inputs.
                original = pair
                active = sorted(set(int(v) for v in pair['input'].flat))
                candidates = []
                for first in range(10):
                    for second in range(10):
                        if first == second:
                            continue
                        targets = [first, second] + [v for v in range(10) if v not in (first, second)]
                        domain = active + [v for v in range(10) if v not in active]
                        mapping = np.empty(10, dtype=np.int8)
                        mapping[domain] = targets
                        candidate = {k: mapping[v] for k, v in original.items()}
                        candidate_key = _grid_key(candidate['input'])
                        if candidate_key in seen:
                            continue
                        if any(all(np.array_equal(candidate[k], p[k]) for k in ('input', 'output'))
                               for p in _source_pairs()):
                            continue
                        if split == 'train' and index == 1:
                            if _recover(candidate['input'])[1]['background'] == previous['background']:
                                continue
                        candidates.append(candidate)
                assert candidates
                pair = candidates[0]
                key = _grid_key(pair['input'])
            seen.add(key)
            result[split].append(pair)
            if split == 'train' and index == 0:
                previous = _recover(pair['input'])[1]
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def _signature(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def diversity_measurements(puzzles):
    """Public analysis helper: every feature is reconstructed from returned pixels."""
    records = []
    for i, puzzle in enumerate(puzzles):
        for split in ('train', 'test'):
            for index, pair in enumerate(puzzle[split]):
                _, meta = _recover(pair['input'])
                records.append(dict(puzzle=i, split=split, index=index, **meta))
    return records


def self_test() -> None:
    puzzles = [generate_puzzle(4200 + i) for i in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert _signature(generate_puzzle(9137, 7, 3)) == _signature(generate_puzzle(9137, 7, 3))
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + i, *counts)
        for split, count in zip(('train', 'test'), counts):
            assert count is None or len(puzzle[split]) == count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.5, 1), (True, 1)):
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for p in puzzles:
        validate_generated_puzzle(p)
        training = [{k: v.copy() for k, v in pair.items()} for pair in p['train']]
        for pair in p['test']:
            predicted = solve(training, pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
    for pair in _source_pairs():
        assert np.array_equal(_recover(pair['input'])[0], pair['output'])
    records = diversity_measurements(puzzles)
    for split in ('train', 'test'):
        group = [r for r in records if r['split'] == split]
        for key, minimum in (('plan_shape', 12), ('side', 4), ('gutter', 4), ('length', 4),
                             ('match', 12), ('miniature', 30), ('base', 30), ('background', 9),
                             ('palette', 30), ('canvas', 30), ('directions', 6)):
            assert len({r[key] for r in group}) >= minimum, (split, key)
        assert {r['segment'] for r in group} == {'whole', 'prefix', 'suffix', 'interior'}
        assert {r['cross'] for r in group} == {'first', 'last', 'interior'}
        assert {r['axis'] for r in group} == {'horizontal', 'vertical'}
        assert {v for r in group for v in r['placement']} == {'above', 'below', 'left', 'right'}
        for key in ('edge_holes', 'blank_run', 'separated_blanks', 'equal_seed', 'repeated_rows', 'repeated_columns'):
            assert sum(bool(r[key]) for r in group) >= 3, (split, key)
        assert any(r['longer_resolves_repetition'] for r in group)
        assert any(r['interior_holes'] for r in group)
        assert any(r['seed_line_holes'] for r in group)
        assert any(r['holes'] == 0 for r in group) and any(r['holes'] == 1 for r in group)
        assert any(r['holes'] >= 4 for r in group)
        assert sum(len(r['palette']) == 2 for r in group) >= 3
        assert any(r['canvas'] == (30, 30) for r in group)
        assert any(r['side'] >= 4 and r['cross'] == 'interior' and r['segment'] == 'suffix' for r in group)
    assert min(max(r['canvas']) for r in records) <= 14
    assert any(r['plan_shape'] == (5, 5) for r in records)
    assert sum(r['holes'] >= 2 and max(r['plan_shape']) >= 4 and r['length'] >= 3 and
               r['cross'] == 'interior' and r['miniature'] != (1, 1) for r in records) >= 8
    for key in ('plan_shape', 'side', 'background', 'miniature', 'base', 'match', 'palette', 'canvas'):
        varied = sum(len({r[key] for r in records if r['puzzle'] == i}) > 1 for i in range(42))
        assert varied >= 35, (key, varied)
    for key in ('holes', 'gutter', 'length', 'directions'):
        assert sum(len({r[key] for r in records if r['puzzle'] == i}) > 1 for i in range(42)) >= 22, key
    axis_sets = {tuple(sorted({r['axis'] for r in records if r['puzzle'] == i})) for i in range(42)}
    assert len(axis_sets) == 3
    assert sum(all(r['holes'] == 0 and r['cross'] != 'interior' and r['segment'] == 'whole'
                   for r in records if r['puzzle'] == i) for i in range(42)) >= 3
    test_records = [r for r in records if r['split'] == 'test']
    assert sum(r['side'] in {t['side'] for t in records if t['split'] == 'train' and t['puzzle'] == r['puzzle']}
               for r in test_records) > len(test_records) // 2
    first = [r for r in records if r['split'] == 'test' and r['index'] == 0]
    assert len({r['match'] for r in first}) >= 10
    # The former ordinal-only attack must fail on most first tests.
    assert sum(r['match'] == (1, 1) for r in first) < 21
    ambiguous = _draw(np.array([[4, 4], [4, 4], [1, 1]], dtype=np.int8),
                      8, (20, 20), (1, 1), (1, 8), 2, 4, 1, (0, 0), 2)
    try:
        _recover(ambiguous['input'])
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted two seed matches')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('validator accepted a changed unrelated cell')
