import numpy as np


_BACKGROUND = 0
_MARKER = 2
_REVEAL = 4
_ALLOWED_CONTEXTS = (1, 3, 5, 6, 7, 8, 9)

# Rejecting a source input is stronger than merely rejecting an exact
# source input/output pair, because no generated pair can then be a copy.
_SOURCE_INPUT_ROWS = (
    (
        "8000080088888000",
        "0800000008080800",
        "0088808888880808",
        "0080800008088280",
        "0028222800028280",
        "8028288800080088",
        "8008808888088000",
        "8080808088088808",
        "8000800000000000",
        "8088288808000888",
        "8028828808008808",
        "0800088002880888",
        "8008888002820008",
        "0880888000808888",
        "8880808000888888",
    ),
    (
        "100001100000010001",
        "112111111001011100",
        "111212222011100110",
        "102122222010001111",
        "011100111000101100",
        "101001100011110000",
        "101000101000001101",
        "000100100010001210",
        "010110000100002211",
        "000001000000001100",
        "011011212121010000",
        "000000111110011001",
        "000001121220010111",
        "010000001101011100",
        "001000100100000110",
        "000000111010100111",
        "100100111011000001",
    ),
    (
        "33000003033000",
        "00300330300030",
        "00330003330000",
        "30000003330033",
        "00022223000303",
        "03322332000330",
        "03022232003000",
        "00000330300003",
        "00330330330033",
        "33320330003030",
        "03230003300030",
        "03303300330303",
        "00303300303303",
        "03303030300000",
        "30030000033033",
    ),
    (
        "000999009900000090",
        "929229000900900000",
        "022992009990090090",
        "000000000999999090",
        "099000909909009999",
        "999909292290090000",
        "000009222290990000",
        "909909009099099909",
        "000900099999090000",
        "990909099009900009",
        "099090929009009999",
        "099009299900099909",
        "909909990090009990",
        "999999000092292290",
        "099999909002929929",
        "090900990902292290",
        "900000000000909990",
    ),
)


def _is_plain_int(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _validate_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if not _is_plain_int(value):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
    rng: np.random.Generator,
) -> tuple[int, int]:
    n_train = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    n_test = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if n_train is not None and n_test is not None:
        if n_train <= n_test:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return n_train, n_test

    if n_train is None and n_test is None:
        # This schedule guarantees broad count diversity for sequential
        # fixed seeds while remaining deterministic.
        n_test = 1 + (seed % 4)
        lower = max(2, n_test + 1)
        span = 11 - lower
        n_train = lower + ((seed // 4) % span)
        return n_train, n_test

    if n_train is None:
        assert n_test is not None
        lower = max(2, n_test + 1)
        n_train = int(rng.integers(lower, 11))
        return n_train, n_test

    upper_test = min(4, n_train - 1)
    if upper_test < 1:
        raise ValueError(
            "no legal num_test exists for num_train"
        )
    n_test = int(rng.integers(1, upper_test + 1))
    return n_train, n_test


def _prefix(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.int32)
    return np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _sum(prefix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1]
               - prefix[r1 + 1, c0] + prefix[r0, c0])


def _overlap(a: tuple, b: tuple) -> bool:
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _blank_gap(a: tuple, b: tuple) -> bool:
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _contains(outer: tuple, inner: tuple) -> bool:
    return (outer[0] <= inner[0] and inner[1] <= outer[1]
            and outer[2] <= inner[2] and inner[3] <= outer[3])


def _recover_rectangles(grid: np.ndarray, marker: int,
                        context: int) -> list[tuple[int, int, int, int]] | None:
    """Recover inclusion-maximal, solid, tight marker rectangles.

    Smaller touching marker subsets can form valid boxes in the originals.
    The convention is to take complete maximal groups. A crossing or
    overlapping maximal choice is ambiguous and is rejected.
    """
    points = np.argwhere(grid == marker)
    if len(points) < 2:
        return None
    rows = sorted(set(int(x) for x in points[:, 0]))
    cols = sorted(set(int(x) for x in points[:, 1]))
    allowed = _prefix((grid == marker) | (grid == context))
    marks = _prefix(grid == marker)
    candidates = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1:]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1:]:
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _sum(allowed, r0, r1, c0, c1) != area:
                        continue
                    nmarks = _sum(marks, r0, r1, c0, c1)
                    if nmarks < 2 or nmarks == area:
                        continue
                    if (_sum(marks, r0, r0, c0, c1) == 0
                            or _sum(marks, r1, r1, c0, c1) == 0
                            or _sum(marks, r0, r1, c0, c0) == 0
                            or _sum(marks, r0, r1, c1, c1) == 0):
                        continue
                    candidates.append((r0, r1, c0, c1))
    if not candidates:
        return None
    candidates.sort(key=lambda b: -((b[1] - b[0] + 1) * (b[3] - b[2] + 1)))
    maximal = []
    for box in candidates:
        if not any(_contains(outer, box) for outer in maximal):
            maximal.append(box)
    if any(_overlap(a, b) for i, a in enumerate(maximal)
           for b in maximal[i + 1:]):
        return None
    covered = np.zeros(grid.shape, bool)
    for r0, r1, c0, c1 in maximal:
        covered[r0:r1 + 1, c0:c1 + 1] = True
    if np.any((grid == marker) & ~covered):
        return None
    return sorted(maximal)


def _sample_shape(rng: np.random.Generator, height: int,
                  width: int) -> tuple[int, int]:
    # A compositional mix of small, thin, wide, tall, and larger objects.
    mode = int(rng.integers(0, 5))
    if mode == 0:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    elif mode == 1:
        h, w = 2, int(rng.integers(3, 9))
    elif mode == 2:
        h, w = int(rng.integers(3, 9)), 2
    elif mode == 3:
        h, w = int(rng.integers(3, 7)), int(rng.integers(4, 10))
    else:
        h, w = int(rng.integers(3, 9)), int(rng.integers(3, 7))
    return min(h, height - 2), min(w, width - 2)


def _place(rng: np.random.Generator, height: int, width: int,
           shapes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0] * shapes[i][1])
    for _ in range(24):
        boxes = [None] * len(shapes)
        placed = []
        for index in order:
            h, w = shapes[index]
            options = [(r, r + h - 1, c, c + w - 1)
                       for r in range(height - h + 1)
                       for c in range(width - w + 1)]
            rng.shuffle(options)
            box = next((b for b in options
                        if all(_blank_gap(b, other) for other in placed)), None)
            if box is None:
                break
            boxes[index] = box
            placed.append(box)
        if all(box is not None for box in boxes):
            return boxes
    return None


def _marker_pattern(rng: np.random.Generator, h: int, w: int,
                    mode: int) -> np.ndarray:
    area = h * w
    if mode == 0:
        pattern = np.zeros((h, w), bool)
        # Sparse extrema; two diagonal markers are legal for 2x2.
        for r, c in ((0, int(rng.integers(w))),
                     (h - 1, int(rng.integers(w))),
                     (int(rng.integers(h)), 0),
                     (int(rng.integers(h)), w - 1)):
            pattern[r, c] = True
        if area > 6:
            for index in rng.choice(area, size=int(rng.integers(0, 3)), replace=False):
                pattern.flat[int(index)] = True
    elif mode == 1:
        pattern = rng.random((h, w)) < float(rng.uniform(0.22, 0.45))
    elif mode == 2:
        pattern = rng.random((h, w)) < float(rng.uniform(0.55, 0.82))
    elif mode == 3:
        pattern = np.zeros((h, w), bool)
        if rng.random() < 0.5:
            pattern[0, :] = True
            pattern[:, 0] = True
        else:
            pattern[h - 1, :] = True
            pattern[:, w - 1] = True
    else:
        parity = int(rng.integers(2))
        pattern = (np.indices((h, w)).sum(0) % 2) == parity
    # Ensure tight bounds without imposing four marked corners.
    for side in (0, 1, 2, 3):
        if side == 0 and not pattern[0].any():
            pattern[0, int(rng.integers(w))] = True
        elif side == 1 and not pattern[-1].any():
            pattern[-1, int(rng.integers(w))] = True
        elif side == 2 and not pattern[:, 0].any():
            pattern[int(rng.integers(h)), 0] = True
        elif side == 3 and not pattern[:, -1].any():
            pattern[int(rng.integers(h)), -1] = True
    if pattern.all():
        pattern[int(rng.integers(h)), int(rng.integers(w))] = False
    if pattern.sum() < 2:
        pattern[0, 0] = True
        pattern[-1, -1] = True
    return pattern


def _outside(shape: tuple[int, int], boxes: list[tuple]) -> np.ndarray:
    outside = np.ones(shape, bool)
    for r0, r1, c0, c1 in boxes:
        outside[r0:r1 + 1, c0:c1 + 1] = False
    return outside


def _add_clutter(rng: np.random.Generator, grid: np.ndarray,
                 outside: np.ndarray, context: int, density: float) -> None:
    h, w = grid.shape
    score = rng.random((h, w))
    # Local patches and short runs make the distractors more than iid noise.
    for _ in range(int(rng.integers(3, 10))):
        rh = int(rng.integers(1, 4))
        rw = int(rng.integers(2, 6))
        if rng.random() < 0.4:
            rh, rw = rw, rh
        r = int(rng.integers(0, max(1, h - rh + 1)))
        c = int(rng.integers(0, max(1, w - rw + 1)))
        score[r:min(h, r + rh), c:min(w, c + rw)] += float(rng.uniform(0.4, 1.2))
    cells = np.argwhere(outside)
    desired = max(1, min(len(cells) - 1, int(round(density * len(cells)))))
    scores = score[outside]
    chosen = np.argpartition(scores, -desired)[-desired:]
    locations = cells[chosen]
    grid[locations[:, 0], locations[:, 1]] = context


def _touches(grid: np.ndarray, boxes: list[tuple], context: int) -> bool:
    h, w = grid.shape
    outside = _outside(grid.shape, boxes)
    for r0, r1, c0, c1 in boxes:
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                if outside[r, c] and grid[r, c] == context:
                    return True
    return False


def _make_pair(rng: np.random.Generator, marker: int, reveal: int,
               background: int, context: int, count: int,
               regime: str, force_thin: bool = False) -> dict:
    for attempt in range(90):
        lower = 9 + 2 * (count - 1)
        height = int(rng.integers(lower, 31))
        width = int(rng.integers(lower, 31))
        shapes = [_sample_shape(rng, height, width) for _ in range(count)]
        if force_thin:
            shapes[int(rng.integers(count))] = (2, 2)
        boxes = _place(rng, height, width, shapes)
        if boxes is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        mask = np.zeros(grid.shape, bool)
        for index, (r0, r1, c0, c1) in enumerate(boxes):
            h, w = shapes[index]
            mode = int(rng.integers(5))
            if force_thin and (h, w) == (2, 2):
                # The original includes a two-marker diagonal 2x2 object.
                pattern = np.eye(2, dtype=bool)
                if rng.random() < 0.5:
                    pattern = pattern[:, ::-1]
            else:
                pattern = _marker_pattern(rng, h, w, mode)
            view = grid[r0:r1 + 1, c0:c1 + 1]
            view[:] = context
            view[pattern] = marker
            mask[r0:r1 + 1, c0:c1 + 1] = True
        outside = ~mask
        # Density is sampled per pair, including both sparse and source-like fields.
        density = float(rng.uniform(0.14, 0.51))
        _add_clutter(rng, grid, outside, context, density)
        if not _touches(grid, boxes, context):
            continue
        recovered = _recover_rectangles(grid, marker, context)
        if recovered != sorted(boxes):
            continue
        # The two non-marker colors must be identifiable by the rule.
        wrong = _recover_rectangles(grid, marker, background)
        if wrong is not None:
            continue
        result = grid.copy()
        result[mask & (grid == context)] = reveal
        if np.array_equal(result, grid):
            continue
        if any(grid.shape == (len(rows), len(rows[0])) and
               all(''.join(str(int(v)) for v in row) == source_row
                   for row, source_row in zip(grid, rows))
               for rows in _SOURCE_INPUT_ROWS):
            continue
        return {'input': grid, 'output': result}
    raise ValueError(f'could not construct {regime} pair')


def _infer_marker_reveal(train: list[dict]) -> tuple[int, int]:
    reveals = set()
    markers = None
    for pair in train:
        x, y = pair['input'], pair['output']
        changed = x != y
        assert changed.any()
        reveals.update(int(v) for v in np.unique(y[changed]))
        values = set(int(v) for v in np.unique(x))
        markers = values if markers is None else markers & values
    assert len(reveals) == 1
    reveal = reveals.pop()
    assert markers is not None
    # The shared marker is tested first by its visible frequency. In these
    # scenes it is generally far rarer than the background; the geometric
    # check, rather than the frequency, decides whether it is correct.
    ranked = sorted(markers, key=lambda color: sum(
        int(np.count_nonzero(pair['input'] == color)) for pair in train))
    for marker in ranked:
        if marker == reveal:
            continue
        works = True
        for pair in train:
            x, y = pair['input'], pair['output']
            contexts = np.unique(x[x != y])
            if len(contexts) != 1:
                works = False
                break
            context = int(contexts[0])
            boxes = _recover_rectangles(x, marker, context)
            if boxes is None:
                works = False
                break
            predicted = x.copy()
            for r0, r1, c0, c1 in boxes:
                view = predicted[r0:r1 + 1, c0:c1 + 1]
                view[view == context] = reveal
            if not np.array_equal(predicted, y):
                works = False
                break
        if works:
            return marker, reveal
    raise AssertionError('training does not establish marker and reveal')


def _predict_pair(grid: np.ndarray, marker: int, reveal: int) -> np.ndarray | None:
    others = [int(v) for v in np.unique(grid) if v != marker]
    if len(others) != 2 or reveal in others:
        return None
    answers = []
    for context in others:
        boxes = _recover_rectangles(grid, marker, context)
        if boxes is None:
            continue
        output = grid.copy()
        for r0, r1, c0, c1 in boxes:
            view = output[r0:r1 + 1, c0:c1 + 1]
            view[view == context] = reveal
        if not np.array_equal(output, grid):
            answers.append(output)
    if len(answers) != 1:
        return None
    return answers[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_plain_int(seed):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    n_train, n_test = _resolve_counts(int(seed), num_train, num_test, rng)
    palette = [int(v) for v in rng.permutation(10)]
    marker, reveal = palette[:2]
    # Both roles are learned from training, then kept fixed for this puzzle.
    stable_background = bool(rng.random() < 0.48)
    fixed_background = palette[2] if stable_background else None
    contexts = palette[3:]
    train_context_pool = contexts[:int(rng.integers(2, min(5, len(contexts)) + 1))]
    pair_contexts = [int(train_context_pool[i % len(train_context_pool)])
                     for i in rng.permutation(n_train)]
    # A mix of familiar and novel test camouflage colors.
    test_contexts = []
    for _ in range(n_test):
        if rng.random() < 0.5:
            test_contexts.append(int(rng.choice(train_context_pool)))
        else:
            unused = [v for v in contexts if v not in train_context_pool]
            test_contexts.append(int(rng.choice(unused or train_context_pool)))
    all_contexts = pair_contexts + test_contexts
    backgrounds = []
    for context in all_contexts:
        if stable_background:
            backgrounds.append(fixed_background)
        else:
            choices = [v for v in palette if v not in (marker, reveal, context)]
            backgrounds.append(int(rng.choice(choices)))
    train_counts = [int(rng.integers(1, 6)) for _ in range(n_train)]
    if all(c == 1 for c in train_counts):
        train_counts[int(rng.integers(n_train))] = 2
    test_counts = [int(rng.integers(1, 7)) for _ in range(n_test)]
    thin_index = int(rng.integers(n_train))
    pairs = []
    for i in range(n_train + n_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        count = train_counts[i] if i < n_train else test_counts[i - n_train]
        pair = _make_pair(pair_rng, marker, reveal, backgrounds[i],
                          all_contexts[i], count,
                          'source_train' if i < n_train else 'target_test',
                          force_thin=(i == thin_index))
        pairs.append(pair)
    puzzle = {'train': pairs[:n_train], 'test': pairs[n_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        x, y = pair['input'], pair['output']
        for grid in (x, y):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        assert x.shape == y.shape
        sx, sy = _signature(x), _signature(y)
        assert sx not in seen_inputs and (sx, sy) not in seen_pairs
        seen_inputs.add(sx)
        seen_pairs.add((sx, sy))
        assert not any(x.shape == (len(rows), len(rows[0])) and
                       all(''.join(str(int(v)) for v in row) == source_row
                           for row, source_row in zip(x, rows))
                       for rows in _SOURCE_INPUT_ROWS)
    marker, reveal = _infer_marker_reveal(train)
    all_boxes = []
    for pair in train + test:
        x, y = pair['input'], pair['output']
        predicted = _predict_pair(x, marker, reveal)
        assert predicted is not None and np.array_equal(predicted, y)
        changed = x != y
        assert changed.any() and np.all(y[changed] == reveal)
        assert np.all(x[x == marker] == y[x == marker])
        contexts = set(int(v) for v in np.unique(x[changed]))
        assert len(contexts) == 1
        context = contexts.pop()
        other = [int(v) for v in np.unique(x) if v not in (marker, context)]
        assert len(other) == 1
        background = other[0]
        assert len({marker, reveal, context, background}) == 4
        boxes = _recover_rectangles(x, marker, context)
        assert boxes is not None and 1 <= len(boxes) <= 6
        mask = ~_outside(x.shape, boxes)
        assert np.all(x[mask] != background)
        assert np.all(x[~mask] == y[~mask])
        assert np.all(y[(x == context) & mask] == reveal)
        assert np.any((x == context) & ~mask)
        assert _touches(x, boxes, context)
        all_boxes.append(boxes)
    assert any(any(b[1] - b[0] + 1 == 2 or b[3] - b[2] + 1 == 2
                   for b in boxes) for boxes in all_boxes[:len(train)])
    assert any(any(np.count_nonzero(pair['input'][r0:r1 + 1, c0:c1 + 1]
                                    == marker) == 2
                   for r0, r1, c0, c1 in boxes)
               for pair, boxes in zip(train, all_boxes[:len(train)]))


def _equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p['input'], q['input']) and
        np.array_equal(p['output'], q['output'])
        for p, q in zip(a[s], b[s])) for s in ('train', 'test'))


def self_test() -> None:
    a = generate_puzzle(314159)
    assert _equal(a, generate_puzzle(314159))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(9000 + i, nt, ns)
        assert len(p['train']) == nt and len(p['test']) == ns
    for i, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(9100 + i, nt, ns)
        assert nt is None or len(p['train']) == nt
        assert ns is None or len(p['test']) == ns
    for kw in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
               {'num_test': 5}, {'num_train': 2, 'num_test': 2},
               {'num_train': 3, 'num_test': 3}):
        try:
            generate_puzzle(123, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal pair counts')
    train_sizes, test_sizes, grid_shapes, object_counts = set(), set(), set(), set()
    box_shapes, colors, layouts = set(), set(), set()
    densities, corner_counts, total_boxes = [], 0, 0
    total_pairs = 0
    edge_boxes = two_marker_boxes = 0
    stable_backgrounds = variable_backgrounds = 0
    familiar_tests = novel_tests = 0
    larger_tests = smaller_tests = 0
    varied_densities = 0
    puzzles = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        train_sizes.add(len(p['train']))
        test_sizes.add(len(p['test']))
        puzzles.add(tuple(_signature(pair['input']) for pair in p['train'] + p['test']))
        marker, reveal = _infer_marker_reveal(p['train'])
        colors.add((marker, reveal))
        backgrounds = set()
        train_contexts, test_contexts = set(), set()
        train_dimensions, test_dimensions = [], []
        local_densities = []
        for pair_index, pair in enumerate(p['train'] + p['test']):
            x = pair['input']
            total_pairs += 1
            grid_shapes.add(x.shape)
            context = int(np.unique(x[x != pair['output']])[0])
            background = next(int(v) for v in np.unique(x)
                              if v not in (marker, context))
            backgrounds.add(background)
            if pair_index < len(p['train']):
                train_contexts.add(context)
                train_dimensions.append(x.shape)
            else:
                test_contexts.add(context)
                test_dimensions.append(x.shape)
            boxes = _recover_rectangles(x, marker, context)
            assert boxes is not None
            object_counts.add(len(boxes))
            layouts.add((x.shape, tuple(boxes)))
            outside = _outside(x.shape, boxes)
            density = float(np.mean(x[outside] == context))
            densities.append(density)
            local_densities.append(density)
            for r0, r1, c0, c1 in boxes:
                box_shapes.add((r1-r0+1, c1-c0+1))
                block = x[r0:r1+1, c0:c1+1]
                corners = (block[0,0], block[0,-1], block[-1,0], block[-1,-1])
                corner_counts += all(v == marker for v in corners)
                edge_boxes += (r0 == 0 or c0 == 0 or r1 == x.shape[0] - 1
                               or c1 == x.shape[1] - 1)
                two_marker_boxes += np.count_nonzero(block == marker) == 2
                total_boxes += 1
        stable_backgrounds += len(backgrounds) == 1
        variable_backgrounds += len(backgrounds) > 1
        familiar_tests += bool(train_contexts & test_contexts)
        novel_tests += bool(test_contexts - train_contexts)
        larger_tests += any(h > max(v[0] for v in train_dimensions)
                            or w > max(v[1] for v in train_dimensions)
                            for h, w in test_dimensions)
        smaller_tests += any(h < min(v[0] for v in train_dimensions)
                             or w < min(v[1] for v in train_dimensions)
                             for h, w in test_dimensions)
        varied_densities += max(local_densities) - min(local_densities) >= 0.1
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert len(puzzles) >= 40 and len(grid_shapes) >= 20
    assert len(layouts) >= 0.8 * total_pairs
    assert len(box_shapes) >= 20 and len(object_counts) >= 5
    assert len(colors) >= 20
    assert min(densities) < 0.24 and max(densities) > 0.42
    assert corner_counts < total_boxes * 0.5
    assert two_marker_boxes >= 42 and edge_boxes >= 42
    assert stable_backgrounds >= 8 and variable_backgrounds >= 8
    assert familiar_tests >= 8 and novel_tests >= 8
    assert larger_tests >= 8 and smaller_tests >= 8
    assert varied_densities >= 30
