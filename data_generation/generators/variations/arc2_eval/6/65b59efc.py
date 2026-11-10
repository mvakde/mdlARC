"""Identity-matched, keyed stamp expansion with recoverable registration.

Only presentation (a common whole-scene dihedral transform), variation intensity,
and the sampling schedule are shared. All command data are visible per input.
"""

import hashlib
import numpy as np

def _check_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
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


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _check_optional_count("num_train", num_train, 2, 10)
    test_count = _check_optional_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        lower = max(2, test_count + 1)
        train_count = lower + (seed // 4) % (11 - lower)
    elif train_count is None:
        assert test_count is not None
        lower = max(2, test_count + 1)
        train_count = lower + (seed // 7) % (11 - lower)
    elif test_count is None:
        upper = min(4, train_count - 1)
        test_count = 1 + (seed // 11) % upper

    assert train_count is not None and test_count is not None
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _separator_marker(
    n: int,
    background: int,
    separator: int,
) -> np.ndarray:
    marker = np.full(n, background, dtype=np.int8)
    marker[0] = separator
    marker[-1] = separator

    # This reproduces the registered 5-0-5 and 5-0-5-0-5 patterns,
    # while the even-sized test scale uses 5-0-0-5.
    if n >= 5 and n % 2 == 1:
        marker[n // 2] = separator

    return marker


def _legend_offsets(n: int) -> tuple[int, ...]:
    if n % 2:
        return (n // 2,)
    return (n // 2 - 1, n // 2)


def _motif_candidate(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    style = int(rng.integers(0, 9))
    mask = np.zeros((n, n), dtype=bool)

    if style == 0:
        mask[:, :] = True

    elif style == 1:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

    elif style == 2:
        np.fill_diagonal(mask, True)
        if int(rng.integers(0, 2)):
            mask |= np.fliplr(np.eye(n, dtype=bool))

    elif style == 3:
        centers = [n // 2] if n % 2 else [n // 2 - 1, n // 2]
        row = centers[int(rng.integers(0, len(centers)))]
        col = centers[int(rng.integers(0, len(centers)))]
        mask[row, :] = True
        mask[:, col] = True

    elif style == 4:
        parity = int(rng.integers(0, 2))
        rows, cols = np.indices((n, n))
        mask = (rows + cols) % 2 == parity

    elif style == 5:
        mask[0, :] = True
        mask[:, 0] = True
        mask = np.rot90(mask, int(rng.integers(0, 4)))
        if int(rng.integers(0, 2)):
            mask = np.fliplr(mask)

    elif style == 6:
        if int(rng.integers(0, 2)):
            count = int(rng.integers(1, n))
            rows = rng.choice(n, size=count, replace=False)
            mask[rows, :] = True
        else:
            count = int(rng.integers(1, n))
            cols = rng.choice(n, size=count, replace=False)
            mask[:, cols] = True

    elif style == 7:
        row0 = int(rng.integers(0, n - 1))
        row1 = int(rng.integers(row0 + 1, n))
        col0 = int(rng.integers(0, n - 1))
        col1 = int(rng.integers(col0 + 1, n))

        mask[row0 : row1 + 1, col0 : col1 + 1] = True

        if (
            int(rng.integers(0, 2))
            and row1 - row0 >= 2
            and col1 - col0 >= 2
        ):
            mask[row0 + 1 : row1, col0 + 1 : col1] = False

    else:
        density = float(rng.uniform(0.22, 0.78))
        mask = rng.random((n, n)) < density

    return mask


def _sample_targets(
    rng: np.random.Generator,
    labels: list[int],
    background: int,
) -> list[int]:
    pool = [color for color in range(10) if color != background]
    k = len(labels)

    for _ in range(128):
        targets = [
            int(color)
            for color in rng.choice(pool, size=k, replace=False)
        ]
        if all(
            target != label
            for target, label in zip(targets, labels)
        ):
            return targets

    # Guaranteed backtracking fallback.
    order = [int(color) for color in rng.permutation(pool)]
    chosen: list[int] = []
    used: set[int] = set()

    def search(index: int) -> bool:
        if index == k:
            return True

        for color in order:
            if color not in used and color != labels[index]:
                used.add(color)
                chosen.append(color)

                if search(index + 1):
                    return True

                chosen.pop()
                used.remove(color)

        return False

    assert search(0)
    return chosen


def _build_input(
    n: int,
    upper_order: list[int],
    lower_order: list[int],
    masks_by_label: dict[int, np.ndarray],
    targets_by_label: dict[int, int],
    assignment: np.ndarray,
    background: int,
    separator: int,
    right_padding: int,
    bottom_padding: int,
) -> np.ndarray:
    k = len(upper_order)
    core_width = k * n + (k - 1)
    core_height = 2 * n + 4

    grid = np.full(
        (
            core_height + bottom_padding,
            core_width + right_padding,
        ),
        background,
        dtype=np.int8,
    )
    marker = _separator_marker(n, background, separator)

    # Upper template library.
    for panel_index, label in enumerate(upper_order):
        col0 = panel_index * (n + 1)
        grid[:n, col0 : col0 + n] = np.where(
            masks_by_label[label],
            label,
            background,
        ).astype(np.int8)

    # Lower placement layers.
    for panel_index, label in enumerate(lower_order):
        col0 = panel_index * (n + 1)
        layer = assignment == label
        grid[n + 1 : 2 * n + 1, col0 : col0 + n] = np.where(
            layer,
            label,
            background,
        ).astype(np.int8)

    # Registered separator columns.
    for panel_index in range(k - 1):
        gap = panel_index * (n + 1) + n
        grid[:n, gap] = marker
        grid[n + 1 : 2 * n + 1, gap] = marker

    # Registered separator rows.
    for row in (n, 2 * n + 1):
        for panel_index in range(k):
            col0 = panel_index * (n + 1)
            grid[row, col0 : col0 + n] = marker

        for panel_index in range(k - 1):
            gap = panel_index * (n + 1) + n
            grid[row, gap] = separator

    # Spacer row preserving panel registration.
    spacer_row = 2 * n + 2
    for panel_index in range(k - 1):
        gap = panel_index * (n + 1) + n
        grid[spacer_row, gap] = separator

    # Target-color legend aligned with the upper panels.
    legend_row = 2 * n + 3
    for panel_index, label in enumerate(upper_order):
        col0 = panel_index * (n + 1)
        for offset in _legend_offsets(n):
            grid[legend_row, col0 + offset] = targets_by_label[label]

    return grid


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


# Untouched source regression inputs and exact canonical grid fingerprints.
_SOURCE = ((('22251115444', '22201010040', '22251115444', '50555055505', '00050445100', '00000040010', '20050005001', '50555055505', '00050005000', '06000700010'), '670f4d8f2fb8906c288d68bd9bfece3a3baa9ea51ee12a627182246c153d5b67', '11179e50bf2bf09803485fdc34b13a9da525757ab0ecc637ead279d340853ee8'), (('01052225404', '11102020444', '01052225040', '50555055505', '00054005001', '00004000001', '22050005000', '50555055505', '00050005000', '07000900030'), 'f71f7cabd0c1ab0389d5a595b00c43ff9b58ef305aaf3af0e4db8862ead03bfa', 'e2e6d2f4aa85f62f4a226d07d6da9b1edbccfc70598393a3b50360060e9750e0'), (('11101522222504004', '10111002020044444', '11101520202504004', '10001020202004444', '11111522222544044', '50505550505550505', '40000500000502222', '40000000000002200', '44000500000000000', '04400010000000000', '00400511000500000', '50505550505550505', '00000500000500000', '00300000800000600'), '92576a00eb5dd529e96eb0bd7f48b8f5c1695d848a49dce3a56c11fde7eaebc7', 'ac433ea5ba8694e54a1e619775bd3a00fe095a4ad47073d36aebb8f0e5fc85d2'), (('888885202025770075444440000000', '880080222220777770444440000000', '800085202025077705004000000000', '800880222220077770044400000000', '888885202025770775444440000000', '505055505055505055505050000000', '000775002005800005000000000000', '000070002200800000000000000000', '000005000225800005044000000000', '000000000000800000044000000000', '000005000005888885000000000000', '505055505055505055505050000000', '000005000005000005000000000000', '003000004000001000006000000000', '000000000000000000000000000000', '000000000000000000000000000000'), '0747f44cb567a2319952fb82425a3c773630c14d7b2eab48145cb15bdb6cc26a', '1a6056f0e2bec25dd7674afbcc1dc2d557c0bf92249122afc05dfa7b1d85f0cd'), (('66665044053300510015', '60660444403330011110', '60060444400333011110', '66665044050033510010', '50055500555005550005', '06605300350000500000', '00000033000000010010', '66660000000000000000', '00005000054444500005', '50055500555005550055', '00005000050000500005', '01100099000880007700'), 'adb06cb1a6d03c47fe53a1ec611c71a47e5bda9fc128c52cb99c246b9e7ee960', 'cdf3502d7b659f459e4df59a7a40eb7c1b8420ea699c0f4b8492cf88e9953478'))

def _digest(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _transform(grid, presentation):
    if presentation >= 4:
        grid = np.fliplr(grid)
    return np.rot90(grid, presentation % 4).copy()


def _registration(n, k, background, separator):
    """A reference stencil, used for measuring omissions, not for parsing."""
    masks = {i: np.ones((n, n), dtype=bool) for i in range(k)}
    return _build_input(n, list(range(k)), list(range(k)), masks,
                        dict.fromkeys(range(k), background),
                        np.full((n, n), -1), background, separator, 0, 0)


def _read_core(core, n, k, background):
    width = k * (n + 1) - 1
    assert core.shape[1] in (width, width + 1)
    # Both row dividers, all column dividers, and the spacer may use
    # redundant registration. Their remaining marks must agree in color.
    metadata = np.zeros(core.shape, dtype=bool)
    metadata[n, :width] = True
    metadata[2*n+1, :width] = True
    for c in range(n, width, n+1):
        metadata[:2*n+3, c] = True
    if core.shape[1] == width + 1:
        metadata[:-1, width] = True
    colors = np.unique(core[metadata & (core != background)])
    assert len(colors) == 1
    separator = int(colors[0])
    assert separator != background
    # At least one mark per panel in each horizontal registration row,
    # and per internal gap beside each array. No exact stencil required.
    for j in range(k):
        c = j * (n+1)
        for r in (n, 2*n+1):
            assert np.any(core[r, c:c+n] == separator)
    for c in range(n, width, n+1):
        assert np.any(core[:n, c] == separator)
        assert np.any(core[n+1:2*n+1, c] == separator)

    upper, lower, templates, targets, layers = [], [], [], [], []
    occupied = np.zeros((n, n), dtype=bool)
    assignment = np.full((n, n), -1, dtype=np.int8)
    expected_key = np.full(core.shape[1], background, dtype=np.int8)
    for j in range(k):
        c = j * (n+1)
        top = core[:n, c:c+n]
        colors = np.unique(top[top != background])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != separator and label not in upper
        upper.append(label)
        templates.append(top == label)
        swatches = core[2*n+3, [c+o for o in _legend_offsets(n)]]
        assert np.all(swatches == swatches[0]) and swatches[0] != background
        target = int(swatches[0])
        targets.append(target)
        expected_key[[c+o for o in _legend_offsets(n)]] = target
        # Spacer row contains only registered internal or terminal gaps.
        assert np.all(core[2*n+2, c:c+n] == background)
        panel = core[n+1:2*n+1, c:c+n]
        colors = np.unique(panel[panel != background])
        assert len(colors) == 1
        identity = int(colors[0])
        assert identity != separator and identity not in lower
        mask = panel == identity
        assert not np.any(occupied & mask)
        occupied |= mask
        assignment[mask] = identity
        lower.append(identity)
        layers.append(mask)
    assert set(upper) == set(lower)
    assert np.array_equal(core[-1], expected_key)
    stencil = _registration(n, k, background, separator)
    # Limit the stencil comparison to metadata; template colors may coincide
    # numerically with temporary labels in _registration.
    holes = int(np.sum(metadata[:, :width] & (stencil == separator)
                       & (core[:, :width] == background)))
    return dict(n=n, k=k, background=background, separator=separator,
                upper_order=tuple(upper), lower_order=tuple(lower),
                templates=tuple(templates), targets=tuple(targets),
                layers=tuple(layers), assignment=assignment,
                omissions=holes, terminal=core.shape[1] == width + 1)


def _parse_input_uniquely(grid, k=None):
    """Recover the key side, palette roles, scale, count, and all margins.

    Cropping is attempted for each possible background, never by assuming
    that a particular pixel or the most common color is background. Boundary
    registration retains the corners of the two arrays even for sparse motifs.
    """
    candidates = []
    for turns in range(4):
        oriented = np.rot90(grid, turns)
        for background in np.unique(oriented):
            rows, cols = np.nonzero(oriented != background)
            if not len(rows):
                continue
            top, bottom = int(rows.min()), int(rows.max()) + 1
            left, right = int(cols.min()), int(cols.max()) + 1
            height, width = bottom-top, right-left
            if height not in (8, 10, 12, 14):
                continue
            n = (height-4)//2
            count = (width+1)//(n+1)
            if not 2 <= count <= min(8, n*n) or (k is not None and count != k):
                continue
            if width not in (count*(n+1)-1, count*(n+1)):
                continue
            # A reflected scene can put the optional terminal line on the
            # left. Without that line, left/right reversal is an equivalent
            # reading: reversing both macro and micro coordinates commutes
            # with expansion, so prefer the unreflected reading.
            core = oriented[top:bottom, left:right]
            for reflected in (False, True):
                try:
                    parsed = _read_core(np.fliplr(core) if reflected else core,
                                        n, count, int(background))
                except AssertionError:
                    continue
                parsed.update(turns=turns, reflected=reflected, margins=(top, left,
                              oriented.shape[0]-bottom, oriented.shape[1]-right))
                candidates.append(parsed)
                break
    assert len(candidates) == 1, f"expected one registered interpretation, got {len(candidates)}"
    return candidates[0]


def _render_parsed(parsed, correspondence="identity", stamp_turn=0,
                   layer_turn=0, exchange=False):
    """Blockwise reconstruction; deliberately separate from Kronecker generation."""
    n, bg = parsed['n'], parsed['background']
    result = np.full((n*n, n*n), bg, dtype=np.int8)
    upper, lower = parsed['upper_order'], parsed['lower_order']
    shift = upper.index(lower[0])
    for j, (identity, layer) in enumerate(zip(lower, parsed['layers'])):
        index = (upper.index(identity) if correspondence == 'identity' else
                 j if correspondence == 'position' else (shift+j) % parsed['k'])
        template = parsed['templates'][index]
        if exchange:
            layer = parsed['templates'][upper.index(identity)]
            template = parsed['layers'][lower.index(upper[index])]
        layer = _transform(layer, layer_turn)
        tile = np.where(_transform(template, stamp_turn),
                        parsed['targets'][index], bg)
        for r, c in zip(*np.nonzero(layer)):
            result[r*n:(r+1)*n, c*n:(c+1)*n] = tile
    if parsed.get('reflected', False):
        result = np.fliplr(result)
    return np.rot90(result, -parsed['turns']).copy()


def solve(train_pairs, test_input):
    """Predict using only visible demonstrations and a held-out input.

    This is a local consistency solver, not an independent reference solver.
    """
    training = [_parse_input_uniquely(p['input']) for p in train_pairs]
    for p, interpretation in zip(train_pairs, training):
        assert np.array_equal(_render_parsed(interpretation), p['output'])
    query = _parse_input_uniquely(test_input)
    assert {p['turns'] for p in training} == {query['turns']}
    return _render_parsed(query)


def _noncyclic(parsed):
    order = [parsed['upper_order'].index(x) for x in parsed['lower_order']]
    return any(x != (order[0]+i) % len(order) for i, x in enumerate(order))


def _orientation_witness(masks):
    return all(any(not np.array_equal(m, _transform(m, t)) for m in masks)
               for t in range(1, 8))


def _sample_masks(rng, n, k, intensity, witness):
    probabilities = ([.82,.08,.07,.03], [.50,.23,.17,.10], [.30,.35,.25,.10])
    style = int(rng.choice(4, p=probabilities[intensity]))
    masks = []
    for _ in range(k):
        if style in (1, 2):
            density = float(rng.uniform(.65, .91) if style == 1 else rng.uniform(.08, .30))
            mask = rng.random((n, n)) < density
        elif style == 3:
            rr,cc = np.indices((n,n))
            mask = ((rr+cc) % 2 == int(rng.integers(2))) & (rng.random((n,n)) < .70)
        else:
            mask = _motif_candidate(n, rng)
        if not mask.any():
            mask.flat[int(rng.integers(n*n))] = True
        masks.append(mask)
    if not witness and rng.random() < .24:
        a, b = rng.choice(k, 2, replace=False)
        masks[int(b)] = masks[int(a)].copy()
    if witness:
        # Random asymmetric evidence, without reserving an L or any other
        # fixed silhouette in every library. A bounded fallback uses four cells.
        for attempt in range(32):
            if _orientation_witness(masks) and len({m.tobytes() for m in masks}) > 1:
                break
            masks[int(rng.integers(k))] = rng.random((n, n)) < rng.uniform(.3, .8)
            for mask in masks:
                if not mask.any():
                    mask.flat[0] = True
        else:
            masks[0] = np.zeros((n, n), dtype=bool)
            masks[0][0, :3] = True
            masks[0][1, 0] = True
            masks[1] = np.eye(n, dtype=bool)
    return masks


def _sample_assignment(rng, n, labels, intensity):
    k, total = len(labels), n*n
    probabilities = ([.25,.15,.35,.25], [.18,.22,.38,.22], [.20,.30,.30,.20])
    density_mode = int(rng.choice(4, p=probabilities[intensity]))
    if density_mode == 0:
        count = k
    elif density_mode == 1:
        count = int(rng.integers(k, max(k, total//2)+1))
    elif density_mode == 2:
        count = int(rng.integers(k, total+1))
    else:
        count = total
    style = int(rng.integers(0, 7))
    coords = [(r, c) for r in range(n) for c in range(n)]
    if style == 0:  # Scattered.
        positions = rng.permutation(total).tolist()
    elif style == 1:  # Bars, with contiguous runs along a serpentine path.
        positions = [r*n+c for r in range(n)
                     for c in (range(n) if r % 2 == 0 else range(n-1, -1, -1))]
    elif style == 2:  # Successive diagonals produce staircases.
        positions = [r*n+c for r, c in sorted(coords, key=lambda p:(sum(p), p[0]))]
    elif style == 3:  # Compact patches around a variable center.
        cr, cc = rng.integers(0, n, size=2)
        positions = [r*n+c for r,c in sorted(coords,
                     key=lambda p:(abs(p[0]-cr)+abs(p[1]-cc), p))]
    elif style == 4:  # Separated islands, optionally filling their intervening cells.
        positions = [r*n+c for r,c in sorted(coords, key=lambda p:((sum(p)%2), p))]
    elif style == 5:  # Runs around nested borders.
        positions = []
        for lo in range((n+1)//2):
            hi = n-1-lo
            ring = ([(lo,c) for c in range(lo,hi+1)] +
                    [(r,hi) for r in range(lo+1,hi+1)] +
                    ([(hi,c) for c in range(hi-1,lo-1,-1)] if hi>lo else []) +
                    [(r,lo) for r in range(hi-1,lo,-1)])
            positions.extend(r*n+c for r,c in ring)
    else:  # A bar/patch core mixed with scattered sites.
        positions = [r*n+c for r,c in coords]
        cut = int(rng.integers(1, total))
        positions[cut:] = rng.permutation(positions[cut:]).tolist()
    weights = rng.dirichlet(np.full(k, rng.uniform(.15, 1.4)))
    counts = 1 + rng.multinomial(count-k, weights)
    identities = np.repeat(rng.permutation(labels), counts)
    if style == 0:
        rng.shuffle(identities)
    assignment = np.full((n,n), -1, dtype=np.int8)
    assignment.flat[np.array(positions[:count])] = identities
    return _transform(assignment, int(rng.integers(8)))


def _sample_orders(rng, labels, witness):
    upper = [int(x) for x in rng.permutation(labels)]
    k = len(labels)
    mode = int(rng.choice(4, p=[.22, .13, .20, .45]))
    if witness or mode == 2:
        lower = upper.copy()
        a, b = rng.choice(k, 2, replace=False)
        lower[int(a)], lower[int(b)] = lower[int(b)], lower[int(a)]
    elif mode == 0:
        lower = upper.copy()
    elif mode == 1:
        lower = np.roll(upper, int(rng.integers(1,k))).tolist()
    else:
        lower = [int(x) for x in rng.permutation(upper)]
    return upper, lower


def _schedule(rng, train_count, test_count, intensity):
    """Schedules vary independently of presentation, colors, and geometry."""
    regime = int(rng.integers(6))
    if regime == 0:  # Original 3 -> 4 shift with an unseen intermediate scale.
        ns = [3,5] + [int(rng.choice([3,5])) for _ in range(train_count-2)]
        rng.shuffle(ns)
        train = [(n,3) for n in ns]
        test = [(4,4)] + [(int(rng.choice([4,5])),4) for _ in range(test_count-1)]
        rng.shuffle(test)
    elif regime == 1:  # Same scales and library count in both splits.
        n = int(rng.integers(3,6))
        k = int(rng.integers(3, min(7,31//(n+1))+1))
        train, test = [(n,k)]*train_count, [(n,k)]*test_count
    elif regime == 2:  # Mixed training counts, broad tests.
        def draw():
            n = int(rng.integers(3,6))
            return n, int(rng.integers(2,31//(n+1)+1))
        train, test = [draw() for _ in range(train_count)], [draw() for _ in range(test_count)]
        train[0] = (train[0][0], 3)
        train[1] = (train[1][0], int(rng.choice([2,4,5])))
    elif regime == 3:  # Fewer identities in tests, without a required scale shift.
        n = int(rng.integers(3,6))
        train = [(int(rng.integers(3,6)), int(rng.integers(4,6))) for _ in range(train_count)]
        test = [(int(rng.integers(3,6)), int(rng.integers(2,4))) for _ in range(test_count)]
    elif regime == 4:  # Larger libraries after smaller demonstrations.
        train = [(int(rng.integers(3,6)), int(rng.integers(2,4))) for _ in range(train_count)]
        train[0] = (train[0][0],3)
        test = []
        for _ in range(test_count):
            n = int(rng.integers(3,6))
            test.append((n,int(rng.integers(4,31//(n+1)+1))))
    else:  # Explicitly demonstrate compact n=2 registration before using it.
        train = [(2,int(rng.integers(2,5))), (int(rng.integers(3,6)),3)]
        for _ in range(train_count-2):
            n = int(rng.integers(2,6))
            train.append((n,int(rng.integers(2,min(n*n,31//(n+1))+1))))
        test = [(2,int(rng.integers(2,5))) for _ in range(test_count)]
    witnesses = [i for i,(n,k) in enumerate(train) if n >= 3 and k >= 3]
    witness = int(rng.choice(witnesses))
    return train, test, witness


def _generate_pair(rng, regime, params, n, k, witness, previous_roles, fallback=False):
    assert regime in ('source_train', 'target_test')
    intensity = params['intensity']
    if rng.random() < (.45 if intensity == 0 else .15):
        background, separator = 0, 5
    else:
        background, separator = map(int, rng.choice(10,2,replace=False))
    if previous_roles == (background, separator):
        background = (background+1) % 10
        if background == separator:
            background = (background+1) % 10
    labels = [int(x) for x in rng.choice([x for x in range(10)
              if x not in (background,separator)],k,replace=False)]
    masks = _sample_masks(rng,n,k,intensity,witness)
    if witness:
        targets = _sample_targets(rng,labels,background)
    else:
        targets = [int(x) for x in rng.choice([x for x in range(10) if x != background],k)]
        if rng.random() < .30:
            j = int(rng.integers(k))
            targets[j] = labels[j]
        if rng.random() < .24:
            a,b = rng.choice(k,2,replace=False)
            targets[int(b)] = targets[int(a)]
    upper,lower = _sample_orders(rng,labels,witness)
    assignment = _sample_assignment(rng,n,labels,intensity)
    core = _build_input(n,upper,lower,dict(zip(labels,masks)),dict(zip(labels,targets)),
                        assignment,background,separator,0,0)
    # Omit a subset of redundant marks inside the gap columns. Retain one
    # top anchor so margins remain visible even when every motif has an
    # empty top row.
    if not fallback and n >= 3 and rng.random() < .48:
        for c in range(n,core.shape[1],n+1):
            for start in (0,n+1):
                marks = [start+i for i in range(n) if core[start+i,c] == separator]
                chosen = [r for r in marks if rng.random() < .28]
                if len(chosen) == len(marks):
                    chosen = chosen[:-1]
                for r in chosen:
                    core[r,c] = background
        core[0,n] = separator
    if not fallback and core.shape[1] < 30 and rng.random() < .30:
        terminal = np.full((2*n+4,1),background,dtype=np.int8)
        terminal[:n,0] = _separator_marker(n,background,separator)
        terminal[n+1:2*n+1,0] = _separator_marker(n,background,separator)
        terminal[[n,2*n+1,2*n+2],0] = separator
        # The terminal line is redundant and may itself be incomplete.
        terminal[(rng.random(2*n+3) < .20).nonzero()[0],0] = background
        terminal[2*n+1,0] = separator
        core = np.concatenate([core,terminal],axis=1)
    if fallback or rng.random() < (.50 if intensity == 0 else .28):
        top = left = bottom = right = 0
    else:
        top,bottom = map(int,rng.integers(0,5,size=2))
        spare = 30-core.shape[1]
        left = int(rng.integers(0,min(4,spare)+1))
        right = int(rng.integers(0,spare-left+1))
    grid = np.pad(core,((top,bottom),(left,right)),constant_values=background)
    # Ground truth uses independent per-identity Kronecker products. The
    # parser/validator later recovers identities and checks individual blocks.
    output = np.full((n*n,n*n),background,dtype=np.int8)
    for label,mask,target in zip(labels,masks,targets):
        output[np.kron(assignment == label,mask)] = target
    return {'input':_transform(grid,params['presentation']),
            'output':_transform(output,params['presentation'])}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count,test_count = _resolve_counts(seed,num_train,num_test)
    intensity = int(rng.choice(3,p=[.20,.55,.25]))
    params = dict(intensity=intensity, presentation=int(rng.integers(8))
                  if rng.random() < .55 else 0)
    training,tests,witness = _schedule(rng,train_count,test_count,intensity)
    puzzle = {'train':[], 'test':[]}
    seen = set()
    first_roles = None
    for split,schedule in (('train',training),('test',tests)):
        for index,(n,k) in enumerate(schedule):
            for attempt in range(32):
                pair = _generate_pair(rng,'source_train' if split == 'train' else 'target_test',
                       params,n,k,split == 'train' and index == witness,
                       first_roles if split == 'train' and index == 1 else None,
                       fallback=attempt >= 8)
                parsed = _parse_input_uniquely(pair['input'])
                if split == 'train' and index == witness:
                    # The demonstrated axes and array roles must also be
                    # observable, not hidden by accidentally symmetric masks.
                    if any(np.array_equal(pair['output'],_render_parsed(parsed,layer_turn=t))
                           for t in range(1,8)):
                        continue
                    if np.array_equal(pair['output'],_render_parsed(parsed,exchange=True)):
                        continue
                key = _grid_key(pair['input'])
                if key in seen or (_digest(pair['input']),_digest(pair['output'])) in _SOURCE_SIGNATURES:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train' and index == 0:
                    first_roles = parsed['background'],parsed['separator']
                break
            else:
                raise AssertionError('bounded unique-pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


_SOURCE_SIGNATURES = {(entry[2],entry[1]) for entry in _SOURCE}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs,seen_pairs = set(),set()
    parsed_splits = {'train':[], 'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            for a in pair.values():
                assert isinstance(a,np.ndarray) and a.ndim == 2
                assert all(1 <= side <= 30 for side in a.shape)
                assert np.issubdtype(a.dtype,np.integer)
                assert np.all((a>=0)&(a<=9))
            ik,ok = _grid_key(pair['input']),_grid_key(pair['output'])
            assert ik not in seen_inputs and (ik,ok) not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add((ik,ok))
            assert not np.array_equal(pair['input'],pair['output'])
            assert (_digest(pair['input']),_digest(pair['output'])) not in _SOURCE_SIGNATURES
            parsed = _parse_input_uniquely(pair['input'])
            assert pair['output'].shape == (parsed['n']**2,)*2
            assert np.array_equal(pair['output'],_render_parsed(parsed))
            assert set(np.unique(pair['output'])) <= {parsed['background'],*parsed['targets']}
            assert all(np.any(pair['output'] == target) for target in parsed['targets'])
            parsed_splits[split].append(parsed)
    train,test = parsed_splits['train'],parsed_splits['test']
    assert len({p['turns'] for p in train+test}) == 1
    # Every compact-scale test has a demonstration of its dense registration.
    assert not any(p['n'] == 2 for p in test) or any(p['n'] == 2 for p in train)
    assert len({(p['background'],p['separator']) for p in train}) >= 2
    # Demonstrations must visibly refute both column matching and the cyclic
    # shortcut, recolor more than one distinguishable stamp, and determine
    # orientation. These checks inspect resulting arrays, not schedule flags.
    assert any(_noncyclic(p)
               and not np.array_equal(_render_parsed(p),_render_parsed(p,'cyclic'))
               and not np.array_equal(_render_parsed(p),_render_parsed(p,'position'))
               and sum(a != b for a,b in zip(p['upper_order'],p['targets'])) >= 2
               and len(set(p['targets'])) >= 2 for p in train)
    assert all(any(not np.array_equal(_render_parsed(p),_render_parsed(p,stamp_turn=t))
                   for p in train) for t in range(1,8))
    assert all(any(not np.array_equal(_render_parsed(p),_render_parsed(p,layer_turn=t))
                   for p in train) for t in range(1,8))
    assert any(not np.array_equal(_render_parsed(p),_render_parsed(p,exchange=True))
               for p in train)


def measure_puzzle(puzzle):
    """Review measurements recovered exclusively from returned grids."""
    records = []
    for split in ('train','test'):
        for pair in puzzle[split]:
            p = _parse_input_uniquely(pair['input'])
            n,k = p['n'],p['k']
            masks = p['templates']
            counts = [int(np.sum(p['assignment'] == x)) for x in p['upper_order']]
            aligned = sum(a == b for a,b in zip(p['upper_order'],p['lower_order']))
            def components(mask):
                cells = set(zip(*np.nonzero(mask)))
                count = 0
                while cells:
                    stack = [cells.pop()]
                    count += 1
                    while stack:
                        r,c = stack.pop()
                        for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if q in cells:
                                cells.remove(q)
                                stack.append(q)
                return count
            assignment = p['assignment']
            layers = p['layers']
            def has_run(layer):
                return bool(np.any(layer.all(axis=0)) or np.any(layer.all(axis=1)))
            border = np.ones((n,n),dtype=bool)
            border[1:-1,1:-1] = False
            def compact(layer):
                rr,cc = np.nonzero(layer)
                return (len(rr) >= 3 and components(layer) == 1
                        and (int(rr.max()-rr.min())+1)*(int(cc.max()-cc.min())+1)
                        <= 1.5*len(rr))
            stair = np.array([[1,1,0],[0,1,1],[0,0,1]],dtype=bool)
            def staircase(layer):
                return any(np.all(layer[r:r+3,c:c+3][_transform(stair,t)])
                           for r in range(n-2) for c in range(n-2) for t in range(8))
            old_l = np.zeros((n,n),dtype=bool)
            old_l[0,:] = old_l[:,0] = True
            if n >= 4:
                old_l[-1,1] = True
            records.append(dict(split=split,n=n,k=k,
                roles=[p['background'],p['separator']],turns=p['turns'],
                alignment='aligned' if aligned == k else 'partial' if aligned else 'displaced',
                noncyclic=_noncyclic(p), cyclic_wrong=int(np.sum(pair['output'] != _render_parsed(p,'cyclic'))),
                margins=list(p['margins']), terminal=p['terminal'], omissions=p['omissions'],
                occupancy=sum(counts)/(n*n), singletons=all(c == 1 for c in counts),
                use_ratio=max(counts)/min(counts), empty_axis=bool(np.any((assignment == -1).all(axis=0))
                    or np.any((assignment == -1).all(axis=1))),
                layer_bar=any(has_run(m) for m in layers),
                layer_disconnected=any(components(m)>1 for m in layers),
                layer_compact=any(compact(m) for m in layers),
                layer_staircase=any(staircase(m) for m in layers),
                layer_border_run=any(m.sum() >= 3 and components(m) == 1
                                     and not np.any(m & ~border) for m in layers),
                template_density=[float(m.mean()) for m in masks],
                dense_library=all(m.mean() >= .6 for m in masks),
                sparse_library=all(m.mean() <= .35 for m in masks),
                duplicate_silhouettes=len({m.tobytes() for m in masks}) < k,
                old_l_present=any(np.array_equal(m,_transform(old_l,t))
                                  for m in masks for t in range(8)),
                disconnected=any(components(m)>1 for m in masks),
                hollow=any(m[0].all() and m[-1].all() and m[:,0].all()
                           and m[:,-1].all() and not m.all() for m in masks),
                symmetric=any(any(np.array_equal(m,_transform(m,t)) for t in range(1,8)) for m in masks),
                identity_target=any(a == b for a,b in zip(p['upper_order'],p['targets'])),
                shared_target=len(set(p['targets'])) < k,
                target_reuses_separator=p['separator'] in p['targets'],
                target_reuses_role=bool(set(p['targets']) & {*p['upper_order'],p['separator']})))
    return records


def self_test() -> None:
    # All five original inputs, including the omitted middle marker and the
    # irregular terminal line, reconstruct their exact recorded outputs.
    for rows,expected,_ in _SOURCE:
        grid = np.array([[int(c) for c in row] for row in rows],dtype=np.int8)
        assert _digest(_render_parsed(_parse_input_uniquely(grid))) == expected
    first = generate_puzzle(12345,6,3)
    assert _puzzles_equal(first,generate_puzzle(12345,6,3))
    assert not _puzzles_equal(first,generate_puzzle(12346,6,3))
    for seed,a,b in ((101,2,1),(102,10,1),(103,10,4),(201,None,1),
                     (202,None,4),(203,2,None),(204,10,None)):
        puzzle = generate_puzzle(seed,a,b)
        assert a is None or len(puzzle['train']) == a
        assert b is None or len(puzzle['test']) == b
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(2,3),
                (True,1),(3.5,1),(3,False)):
        try:
            generate_puzzle(300,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    records = []
    schedules = dict(source=0,same=0,mixed_training=0,fewer=0,larger=0,compact=0)
    shapes,layouts,motifs = set(),set(),set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training = [{k:v.copy() for k,v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            # The predictor is never passed this pair's output or a seed.
            prediction = solve(training,pair['input'].copy())
            assert np.array_equal(prediction,pair['output'])
        for split in ('train','test'):
            for pair in puzzle[split]:
                parsed = _parse_input_uniquely(pair['input'])
                shapes.add(pair['input'].shape)
                normalized = np.full((parsed['n'],parsed['n']),-1,dtype=np.int8)
                for i,label in enumerate(parsed['upper_order']):
                    normalized[parsed['assignment'] == label] = i
                layouts.add(_grid_key(normalized))
                motifs.update(_grid_key(m) for m in parsed['templates'])
        local = measure_puzzle(puzzle)
        train = [r for r in local if r['split'] == 'train']
        test = [r for r in local if r['split'] == 'test']
        tn,qn = {r['n'] for r in train},{r['n'] for r in test}
        tk,qk = {r['k'] for r in train},{r['k'] for r in test}
        schedules['source'] += tk == {3} and qk == {4} and tn == {3,5} and 4 in qn
        schedules['same'] += tk == qk and tn == qn
        schedules['mixed_training'] += len(tk) > 1
        schedules['fewer'] += max(qk) < min(tk)
        schedules['larger'] += min(qk) > max(tk)
        schedules['compact'] += 2 in tn and 2 in qn
        records.extend(local)
    assert all(value >= 1 for value in schedules.values()), schedules
    assert {r['n'] for r in records} == {2,3,4,5}
    assert {r['k'] for r in records} >= {2,3,4,5,6,7}
    assert {r['alignment'] for r in records} == {'aligned','partial','displaced'}
    assert len(shapes) >= 40 and len(layouts) >= 150 and len(motifs) >= 200
    assert len({tuple(r['roles']) for r in records}) >= 40
    assert {r['turns'] for r in records} == {0,1,2,3}
    for prop in ('terminal','omissions','singletons','empty_axis','dense_library',
                 'sparse_library','duplicate_silhouettes','identity_target','shared_target',
                 'hollow','symmetric','disconnected','layer_bar','layer_disconnected',
                 'layer_compact','layer_staircase','layer_border_run','target_reuses_separator'):
        assert sum(bool(r[prop]) for r in records) >= 5, prop
    assert any(r['occupancy'] <= .16 for r in records)
    assert sum(r['occupancy'] == 1 for r in records) >= 15
    assert any(r['use_ratio'] >= 8 for r in records)
    assert any(r['roles'] == [0,5] for r in records)
    assert sum(not r['old_l_present'] for r in records) >= len(records)//2
    assert sum(r['noncyclic'] and r['k'] >= 4 and r['occupancy'] <= .5 for r in records) >= 10
    for split in ('train','test'):
        part = [r for r in records if r['split'] == split]
        assert sum(any(r['margins']) for r in part) >= 10
        assert sum(not any(r['margins']) for r in part) >= 10
        assert any(r['margins'][0] and r['margins'][1] for r in part)
    # Wrong outputs must be rejected by reconstruction.
    bad = {s:[{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    # Corrupt the original first source input: overlap two layers, then erase
    # one layer's identity entirely. Each must fail input interpretation.
    rows = _SOURCE[0][0]
    original = np.array([[int(c) for c in row] for row in rows],dtype=np.int8)
    for mutation in ('overlap','empty'):
        corrupted = original.copy()
        if mutation == 'overlap':
            corrupted[4,0] = 2  # The yellow layer already occupies macro (0,0).
        else:
            corrupted[4:7,:3] = 0
        try:
            _parse_input_uniquely(corrupted)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid placement layer accepted')
