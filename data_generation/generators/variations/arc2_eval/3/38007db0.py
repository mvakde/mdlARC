"""Select the visible nonmaster panel in each row, preserving its bitmap.

The source summary's missing-panel interpretation is incorrect. Both splits
use the same input-observable selection rule; no completion state is sampled.
"""

import numpy as np


def _checked_count(
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


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")

    return int(seed) % (1 << 64)


def _resolve_counts(
    seed_value: int,
    count_rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        legal_pairs = [
            (train_count, test_count)
            for train_count in range(2, 11)
            for test_count in range(
                1,
                min(4, train_count - 1) + 1,
            )
        ]
        return legal_pairs[seed_value % len(legal_pairs)]

    if num_train is None:
        choices = list(range(max(2, num_test + 1), 11))
        if not choices:
            raise ValueError(
                "no legal num_train exists for the requested num_test"
            )
        num_train = choices[int(count_rng.integers(len(choices)))]

    elif num_test is None:
        choices = list(range(1, min(4, num_train - 1) + 1))
        if not choices:
            raise ValueError(
                "no legal num_test exists for the requested num_train"
            )
        num_test = choices[int(count_rng.integers(len(choices)))]

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test



# Exact source fixtures are used only for regression checks and copy exclusion.
_SOURCE_TEXT = (
    (
        """1111111111111111111
1666661666661666661
1646661646661646461
1644461644461644461
1664661664661664661
1666661666661666661
1111111111111111111
1666661666661666661
1646661646661646661
1644661644461644461
1664661664661664661
1666661666661666661
1111111111111111111
1666661666661666661
1646661646661646661
1644461644461644461
1664661666661664661
1666661666661666661
1111111111111111111""",
        """1111111
1666661
1646461
1644461
1664661
1666661
1111111
1666661
1646661
1644661
1664661
1666661
1111111
1666661
1646661
1644461
1666661
1666661
1111111""",
    ),
    (
        """3333333333333333333333333
3444443444443444443444443
3411143411143411143411143
3441443441443441443441443
3411443411443411143411443
3444443444443444443444443
3333333333333333333333333
3444443444443444443444443
3411143411143411143411143
3441443441443441443441443
3441443411443411443411443
3444443444443444443444443
3333333333333333333333333
3444443444443444443444443
3411143411143411143411443
3441443441443441443441143
3411443411443411443411443
3444443444443444443444443
3333333333333333333333333""",
        """3333333
3444443
3411143
3441443
3411143
3444443
3333333
3444443
3411143
3441443
3441443
3444443
3333333
3444443
3411443
3441143
3411443
3444443
3333333""",
    ),
    (
        """44444444444444444444444444444
48888884888888488888848888884
48118184818818481181848118184
48188184811118481881848188184
48111184818818481111848111184
48188184818118481881848188184
48888884888888488888848888884
44444444444444444444444444444
48888884888888488888848888884
48118184811818481811848118184
48188184818818481881848188184
48111184811118481111848111184
48188184818818481881848188184
48888884888888488888848888884
44444444444444444444444444444
48888884888888488888848888884
48118184818818481181848118184
48188184818818481881848188184
48111184811118481111848111184
48188184818818481881848188184
48888884888888488888848888884
44444444444444444444444444444
48888884888888488888848888884
48118184811818481181848118184
48188184818818481881848188184
48111184811118481111848181184
48188184818818481881848118184
48888884888888488888848888884
44444444444444444444444444444""",
        """44444444
48888884
48188184
48111184
48188184
48181184
48888884
44444444
48888884
48181184
48188184
48111184
48188184
48888884
44444444
48888884
48188184
48188184
48111184
48188184
48888884
44444444
48888884
48118184
48188184
48181184
48118184
48888884
44444444""",
    ),
    (
        """1111111111111111111111111
1333331333331333331333331
1388831388831383831388831
1338331338331338331338331
1388831388831388831388831
1333331333331333331333331
1111111111111111111111111
1333331333331333331333331
1388831388831388831388831
1338331383831338331338331
1388831388831388831388831
1333331333331333331333331
1111111111111111111111111
1333331333331333331333331
1388831388831388831388831
1383331338331338331338331
1388831388831388831388831
1333331333331333331333331
1111111111111111111111111
1333331333331333331333331
1388831388831388831388831
1338331338331338331338331
1388831388831388831388331
1333331333331333331333331
1111111111111111111111111""",
        """1111111
1333331
1383831
1338331
1388831
1333331
1111111
1333331
1388831
1383831
1388831
1333331
1111111
1333331
1388831
1383331
1388831
1333331
1111111
1333331
1388831
1338331
1388331
1333331
1111111""",
    ),
)


def _from_text(text: str) -> np.ndarray:
    return np.array([[int(v) for v in row] for row in text.splitlines()], dtype=np.int8)


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int8).tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair['input']), _grid_key(pair['output'])


_SOURCE_KEYS = frozenset(_grid_key(_from_text(a)) for a, _ in _SOURCE_TEXT)


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ('train', 'test'))


def _assemble_panel_array(panel_rows: list, lattice: int) -> np.ndarray:
    bands, copies = len(panel_rows), len(panel_rows[0])
    ph, pw = panel_rows[0][0].shape
    grid = np.full((bands * (ph + 1) + 1, copies * (pw + 1) + 1),
                   lattice, dtype=np.int8)
    for b, row in enumerate(panel_rows):
        assert len(row) == copies
        for c, panel in enumerate(row):
            assert panel.shape == (ph, pw)
            r0, c0 = 1 + b * (ph + 1), 1 + c * (pw + 1)
            grid[r0:r0 + ph, c0:c0 + pw] = panel
    return grid


def _parse_lattice(grid: np.ndarray) -> dict:
    lattice = int(grid[0, 0])
    rs = np.flatnonzero(np.all(grid == lattice, axis=1))
    cs = np.flatnonzero(np.all(grid == lattice, axis=0))
    assert len(rs) >= 2 and len(cs) >= 2
    assert rs[0] == cs[0] == 0
    assert rs[-1] == grid.shape[0] - 1 and cs[-1] == grid.shape[1] - 1
    assert np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    assert ph >= 1 and pw >= 1
    panels = [[grid[r + 1:s, c + 1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    assert all(np.all(p != lattice) for row in panels for p in row)
    return dict(lattice=lattice, ph=ph, pw=pw, bands=len(panels),
                copies=len(panels[0]), panels=panels)


def _modal_master(panels: list) -> np.ndarray:
    counts, examples = {}, {}
    for row in panels:
        for panel in row:
            key = _grid_key(panel)
            counts[key] = counts.get(key, 0) + 1
            examples[key] = panel
    maximum = max(counts.values())
    winners = [k for k, v in counts.items() if v == maximum]
    assert len(winners) == 1, 'master must have a unique global frequency maximum'
    return examples[winners[0]]


def select_exceptions(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution: no output labels, sampled parameters, or split."""
    info = _parse_lattice(input_grid)
    assert info['copies'] >= 2
    master = _modal_master(info['panels'])
    selected = []
    for row in info['panels']:
        variants = [p for p in row if not np.array_equal(p, master)]
        assert len(variants) == 1, 'every band needs one visible exception'
        selected.append(variants)
    return _assemble_panel_array(selected, info['lattice'])


def solve(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    """Verify the demonstrated convention, then apply it to a fresh input."""
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        assert np.array_equal(select_exceptions(pair['input']), pair['output'])
    return select_exceptions(test_input)


# Independently variable dimensions, constrained only by the 30-cell grid limit.
_GEOMETRIES = tuple(
    (ph, pw, bands, copies)
    for ph in range(3, 14) for pw in range(3, 14)
    if (ph - 2) * (pw - 2) >= 4
    for bands in range(1, 29 // (ph + 1) + 1)
    for copies in range(2, 29 // (pw + 1) + 1)
    if copies >= 3 or bands >= 3
)


def _choose_geometry(rng: np.random.Generator, choices: list, breadth: float) -> tuple:
    # Mix source-sized and wider configurations without indexing either split.
    weights = []
    for ph, pw, bands, copies in choices:
        size_weight = (1.0 if ph in (5, 6) and pw in (5, 6) else breadth)
        square_weight = 2.8 if ph == pw else 1.0
        copy_weight = 0.10 if copies == 2 else 1.0
        weights.append(size_weight * square_weight * copy_weight)
    weights = np.array(weights, dtype=float)
    return choices[int(rng.choice(len(choices), p=weights / weights.sum()))]


def _symmetrize(mask: np.ndarray, symmetry: int) -> np.ndarray:
    if symmetry == 1:
        return mask | mask[::-1, :]
    if symmetry == 2:
        return mask | mask[:, ::-1]
    if symmetry == 3:
        return mask | mask[::-1, :] | mask[:, ::-1] | mask[::-1, ::-1]
    if symmetry == 4:
        return mask | mask[::-1, ::-1]
    if symmetry == 5 and mask.shape[0] == mask.shape[1]:
        return mask | mask.T
    if symmetry == 6 and mask.shape[0] == mask.shape[1]:
        return mask | mask.T[::-1, ::-1]
    return mask


def _sample_master(rng: np.random.Generator, ph: int, pw: int,
                   carrier: int, motif: int, witness: bool) -> np.ndarray:
    h, w = ph - 2, pw - 2
    area = h * w
    minimum = 6 if witness else 1
    style = int(rng.integers(6))
    symmetry = int(rng.integers(1, 7)) if rng.random() < 0.40 else 0
    for attempt in range(48):
        mask = np.zeros((h, w), dtype=bool)
        density = float(rng.uniform(0.18, 0.78))
        if style == 0:  # Unconstrained occupancy, sparse through dense.
            mask = rng.random((h, w)) < density
        elif style == 1:  # Connected silhouette grown one neighbor at a time.
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r, c] = True
            target = max(1, int(area * density))
            for _ in range(target - 1):
                border = [(r, c) for r in range(h) for c in range(w)
                          if not mask[r, c] and any(
                              0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                              for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))]
                if border:
                    mask[border[int(rng.integers(len(border)))]] = True
        elif style == 2:  # A spine with independently sized branches.
            c = int(rng.integers(w))
            mask[:, c] = True
            for r in rng.choice(h, size=int(rng.integers(1, h + 1)), replace=False):
                end = int(rng.integers(w))
                mask[r, min(c, end):max(c, end) + 1] = True
        elif style == 3:  # Hollow frame, variable dimensions and position.
            if h >= 3 and w >= 3:
                rh, rw = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
                r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
                mask[r:r+rh, c] = mask[r:r+rh, c+rw-1] = True
                mask[r, c:c+rw] = mask[r+rh-1, c:c+rw] = True
            else:
                mask = rng.random((h, w)) < density
        elif style == 4:  # Straight lines, stripes, and diagonal runs.
            rr, cc = np.indices((h, w))
            period = int(rng.integers(2, 5))
            direction = int(rng.integers(3))
            position = rr if direction == 0 else cc if direction == 1 else rr + cc
            mask = position % period == int(rng.integers(period))
        else:  # Separated islands; optional dominoes produce varied components.
            rr, cc = np.indices((h, w))
            mask = ((rr + cc) % 2 == int(rng.integers(2))) & (rng.random((h, w)) < 0.8)
            if rng.random() < 0.3:
                mask[:, 1:] |= mask[:, :-1] & (rng.random((h, w-1)) < 0.25)
        if rng.random() < 0.18:
            mask = ~mask
        mask = _symmetrize(mask, symmetry)
        if minimum <= int(mask.sum()) <= area - minimum:
            break
    else:
        # Deterministic bounded fallback, never a failed legal seed.
        mask = np.zeros((h, w), dtype=bool)
        mask.flat[rng.permutation(area)[:area // 2]] = True
    master = np.full((ph, pw), carrier, dtype=np.int8)
    master[1:-1, 1:-1][mask] = motif
    return master


def _variant(rng: np.random.Generator, master: np.ndarray, motif: int,
             mode: str | None = None) -> np.ndarray:
    carrier = int(master[0, 0])
    mask = master[1:-1, 1:-1] == motif
    ink, blank = np.flatnonzero(mask), np.flatnonzero(~mask)
    area = mask.size
    cap = max(1, int(area * 0.45))
    if mode is None:
        mode = str(rng.choice(['add', 'remove', 'exchange', 'scatter', 'shift', 'cut']))
    changed = mask.copy()
    if mode in ('exchange', 'large_exchange'):
        high = min(len(ink) // 2, len(blank) // 2, cap // 2)
        if high:
            low = 3 if mode == 'large_exchange' else 1
            count = int(rng.integers(low, high + 1))
            changed.flat[rng.choice(ink, count, replace=False)] = False
            changed.flat[rng.choice(blank, count, replace=False)] = True
    elif mode in ('add', 'remove'):
        cells = blank if mode == 'add' else ink
        high = min(cap, len(cells) // 2)
        if high:
            count = int(rng.integers(1, high + 1))
            changed.flat[rng.choice(cells, count, replace=False)] = mode == 'add'
    elif mode == 'cut':
        points = np.argwhere(mask)
        anchor = points[int(rng.integers(len(points)))]
        axis = int(rng.integers(2))
        eligible = [tuple(p) for p in points if p[axis] == anchor[axis]]
        count = min(len(eligible), cap, len(ink) // 2)
        if count:
            for i in rng.choice(len(eligible), int(rng.integers(1, count + 1)), replace=False):
                changed[eligible[int(i)]] = False
    elif mode == 'shift':
        axis = int(rng.integers(2))
        step = int(rng.choice([-1, 1]))
        changed = np.roll(mask, step, axis=axis)
        if axis == 0:
            changed[0 if step == 1 else -1, :] = False
        else:
            changed[:, 0 if step == 1 else -1] = False
    else:
        count = int(rng.integers(1, cap + 1))
        cells = rng.choice(area, count, replace=False)
        changed.flat[cells] = ~changed.flat[cells]
    edits = int(np.count_nonzero(mask != changed))
    retained_ink = int(np.count_nonzero(mask & changed))
    retained_carrier = int(np.count_nonzero(~mask & ~changed))
    if (not 1 <= edits <= cap or not 0 < changed.sum() < area
            or retained_ink * 2 < len(ink) or retained_carrier * 2 < len(blank)):
        # A one-cell local change always exists, even in the smallest core.
        changed = mask.copy()
        cells = blank if len(blank) > 1 else ink
        index = int(rng.choice(cells))
        changed.flat[index] = ~changed.flat[index]
    result = np.full(master.shape, carrier, dtype=np.int8)
    result[1:-1, 1:-1][changed] = motif
    return result


def generate_pair(rng: np.random.Generator, regime: str, geometry: tuple,
                  palette: tuple, witness: bool = False) -> dict:
    assert regime in ('source_train', 'target_test')
    ph, pw, bands, copies = geometry
    lattice, carrier, motif = palette
    master = _sample_master(rng, ph, pw, carrier, motif, witness)
    required = {}
    if witness:
        # Locations are random; there is no edit trajectory or fixed multiset.
        positions = rng.choice(bands, 3, replace=False)
        required = dict(zip(map(int, positions), ('add', 'remove', 'large_exchange')))
    columns = rng.integers(copies, size=bands)
    if witness and len(set(map(int, columns))) == 1:
        b = int(rng.integers(bands))
        columns[b] = (columns[b] + int(rng.integers(1, copies))) % copies
    variants = []
    for b in range(bands):
        if b not in required and variants and rng.random() < 0.16:
            variant = variants[int(rng.integers(len(variants)))].copy()
        else:
            variant = _variant(rng, master, motif, required.get(b))
        variants.append(variant)
    if copies == 2 and all(np.array_equal(v, variants[0]) for v in variants):
        # A two-column grid needs cross-band agreement on the common master.
        variant = master.copy()
        core = variant[1:-1, 1:-1]
        for index in range(core.size):
            candidate = master.copy()
            r, c = divmod(index, pw - 2)
            candidate[r+1, c+1] = carrier if master[r+1, c+1] == motif else motif
            original_core = master[1:-1, 1:-1]
            candidate_core = candidate[1:-1, 1:-1]
            retained = all(2 * np.count_nonzero((original_core == color) &
                                               (candidate_core == color)) >=
                           np.count_nonzero(original_core == color)
                           for color in (carrier, motif))
            if not np.array_equal(candidate, variants[0]) and retained:
                variant = candidate
                break
        variants[-1] = variant
    rows = []
    for b in range(bands):
        row = [master.copy() for _ in range(copies)]
        row[int(columns[b])] = variants[b]
        rows.append(row)
    # Ground truth copies the displayed exception, without repairing its cells.
    return {'input': _assemble_panel_array(rows, lattice),
            'output': _assemble_panel_array([[rows[b][int(columns[b])]] for b in range(bands)], lattice)}


def _unique_pair(pair: dict, seen: set) -> dict:
    # Rare collision handling exhausts color permutations, preserving geometry
    # and semantics. There are 720 palettes versus at most 17 excluded inputs.
    if _grid_key(pair['input']) not in seen | _SOURCE_KEYS:
        return pair
    colors = sorted(map(int, np.unique(pair['input'])))
    for a in range(10):
        for b in range(10):
            for c in range(10):
                if len({a, b, c}) != 3:
                    continue
                lookup = np.arange(10, dtype=np.int8)
                lookup[colors] = (a, b, c)
                candidate = {k: lookup[v] for k, v in pair.items()}
                if _grid_key(candidate['input']) not in seen | _SOURCE_KEYS:
                    return candidate
    raise AssertionError('finite palette exhaustion is impossible')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    nt, ne = _resolve_counts(seed_value, rng, num_train, num_test)
    # The only semantic conventions are fixed: horizontal bands, top-to-bottom
    # selection and one output column. These sampling preferences are not rules.
    breadth = float(rng.choice([0.08, 0.35, 1.0]))
    shared_palette = rng.random() < 0.20
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    test_regime = str(rng.choice(['familiar', 'expanded', 'mixed']))
    witness_index = int(rng.integers(nt))
    witness_choices = [g for g in _GEOMETRIES if g[2] >= 3 and g[3] >= 3
                       and (g[0]-2)*(g[1]-2) >= 16]
    witness_geometry = _choose_geometry(rng, witness_choices, breadth)
    geometries = []
    for index in range(nt):
        if index == witness_index:
            geometry = witness_geometry
        else:
            choices = list(_GEOMETRIES)
            if index == (witness_index + 1) % nt:
                choices = [g for g in choices if g[:2] != witness_geometry[:2]
                           and g[2] != witness_geometry[2] and g[3] != witness_geometry[3]]
            geometry = _choose_geometry(rng, choices, breadth)
        geometries.append(geometry)
    for index in range(ne):
        familiar = test_regime == 'familiar' or (test_regime == 'mixed' and rng.random() < 0.45)
        if familiar:
            geometry = geometries[int(rng.integers(nt))]
        else:
            max_ph = max(g[0] for g in geometries[:nt])
            max_bands = max(g[2] for g in geometries[:nt])
            choices = [g for g in _GEOMETRIES if g[0] > max_ph or g[2] > max_bands]
            geometry = _choose_geometry(rng, choices or list(_GEOMETRIES), breadth)
        geometries.append(geometry)
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, geometry in enumerate(geometries):
        split = 'train' if index < nt else 'test'
        if not shared_palette:
            sampled = tuple(map(int, rng.choice(10, 3, replace=False)))
            if index == 1 and sampled[0] == palette[0]:
                choices = [c for c in range(10) if c not in sampled]
                sampled = (int(rng.choice(choices)), sampled[1], sampled[2])
            palette = sampled
        pair = generate_pair(rng, 'source_train' if split == 'train' else 'target_test',
                             geometry, palette, index == witness_index)
        pair = _unique_pair(pair, seen)
        seen.add(_grid_key(pair['input']))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _symmetries(mask: np.ndarray) -> tuple:
    transforms = {'horizontal': mask[::-1, :], 'vertical': mask[:, ::-1],
                  'half_turn': mask[::-1, ::-1]}
    if mask.shape[0] == mask.shape[1]:
        transforms.update(diagonal=mask.T, antidiagonal=mask.T[::-1, ::-1],
                          quarter_turn=np.rot90(mask))
    return tuple(k for k, v in transforms.items() if np.array_equal(mask, v))


def _components(mask: np.ndarray) -> list:
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        group = {remaining.pop()}
        pending = list(group)
        while pending:
            r, c = pending.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    group.add(point)
                    pending.append(point)
        groups.append(group)
    return groups


def _motif_features(mask: np.ndarray) -> dict:
    h, w = mask.shape
    groups = _components(mask)
    holes = sum(all(r not in (0, h-1) and c not in (0, w-1) for r, c in g)
                for g in _components(~mask))
    degrees = [sum(0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                   for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))
               for r, c in np.argwhere(mask)]
    return dict(components=len(groups), holes=holes,
                branched=any(d >= 3 for d in degrees),
                line_like=len(groups) == 1 and len(degrees) >= 3 and max(degrees) <= 2)


def _analyze_pair(pair: dict, regime: str) -> dict:
    assert regime in ('source_train', 'target_test')
    info = _parse_lattice(pair['input'])
    assert np.array_equal(select_exceptions(pair['input']), pair['output'])
    master = _modal_master(info['panels'])
    colors = set(map(int, np.unique(pair['input'])))
    assert len(colors) == 3
    carrier = int(master[0, 0])
    assert len(set(map(int, np.unique(master)))) == 2
    motif = next(iter(colors - {info['lattice'], carrier}))
    columns, edits, additions, removals, variants = [], [], [], [], []
    cuts, shifts = 0, 0
    for row in info['panels']:
        for panel in row:
            assert set(map(int, np.unique(panel))) <= {carrier, motif}
            assert np.all(panel[[0, -1], :] == carrier)
            assert np.all(panel[:, [0, -1]] == carrier)
        c = next(c for c, p in enumerate(row) if not np.array_equal(p, master))
        variant = row[c]
        add = int(np.count_nonzero((master == carrier) & (variant == motif)))
        remove = int(np.count_nonzero((master == motif) & (variant == carrier)))
        columns.append(c)
        edits.append(add + remove)
        additions.append(add)
        removals.append(remove)
        variants.append(_grid_key(variant))
        cells = np.argwhere(master != variant)
        cuts += int(add == 0 and remove >= 2 and
                    (len(set(cells[:, 0])) == 1 or len(set(cells[:, 1])) == 1))
        if add == remove > 0:
            added = set(map(tuple, np.argwhere((master == carrier) & (variant == motif))))
            removed = set(map(tuple, np.argwhere((master == motif) & (variant == carrier))))
            shifts += int(any({(r+dr, c+dc) for r, c in removed} == added
                              for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))))
    return {k: info[k] for k in ('lattice', 'ph', 'pw', 'bands', 'copies')} | dict(
        carrier=carrier, motif=motif, columns=tuple(columns), edits=tuple(edits),
        additions=tuple(additions), removals=tuple(removals), variants=tuple(variants),
        master=_grid_key(master == motif), symmetry=_symmetries(master == motif),
        density=float(np.mean(master[1:-1, 1:-1] == motif)),
        cuts=cuts, shifts=shifts, **_motif_features(master == motif))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair['input'])
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
            assert not np.array_equal(pair['input'], pair['output'])
            measurements[split].append(_analyze_pair(
                pair, 'source_train' if split == 'train' else 'target_test'))
    train = measurements['train']
    changes = [(a, r) for m in train for a, r in zip(m['additions'], m['removals'])]
    assert any(a > 0 and r == 0 for a, r in changes), 'demonstrate added cells'
    assert any(r > 0 and a == 0 for a, r in changes), 'demonstrate removed cells'
    assert any(a == r and a >= 3 for a, r in changes), 'demonstrate a large balanced edit'
    assert any(len(set(m['columns'])) > 1 for m in train), 'rule cannot be a fixed column'
    assert any(m['variants'] != m['variants'][::-1] for m in train), 'demonstrate band order'
    assert len({(m['ph'], m['pw']) for m in train}) >= 2
    assert len({m['bands'] for m in train}) >= 2
    assert len({m['copies'] for m in train}) >= 2
    # Both regimes have exactly the same observable rule. Test geometry may be
    # familiar or expanded; neither a four-band test nor an unseen size is required.
    for pair in puzzle['test']:
        prediction = solve([{k: v.copy() for k, v in p.items()} for p in puzzle['train']],
                           pair['input'].copy())
        assert np.array_equal(prediction, pair['output'])


def self_test() -> None:
    for a, b in _SOURCE_TEXT:
        pair = {'input': _from_text(a), 'output': _from_text(b)}
        _analyze_pair(pair, 'source_train')
        _analyze_pair(pair, 'target_test')
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(200 + 10*nt + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(201, nt, ne)
        assert nt is None or len(p['train']) == nt
        assert ne is None or len(p['test']) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(999, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    records = []
    for p in automatic:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            records.extend((split, _analyze_pair(pair, 'source_train' if split == 'train' else 'target_test'))
                           for pair in p[split])
    all_m = [m for _, m in records]
    for split in ('train', 'test'):
        group = [m for s, m in records if s == split]
        assert len({(m['ph'], m['pw'], m['bands'], m['copies']) for m in group}) >= 20
        assert len({m['bands'] for m in group}) >= 4
        assert len({m['copies'] for m in group}) >= 3
        assert any(m['bands'] == 1 for m in group)
        assert any(m['ph'] != m['pw'] for m in group)
        assert any(len(set(m['columns'])) < m['bands'] for m in group)
        assert any(a == r > 0 for m in group for a, r in zip(m['additions'], m['removals']))
        assert max(e for m in group for e in m['edits']) >= 6
    assert {m['lattice'] for m in all_m} == set(range(10))
    assert len({(m['lattice'], m['carrier'], m['motif']) for m in all_m}) >= 60
    assert len({m['master'] for m in all_m}) >= 100
    assert any(m['copies'] == 2 for m in all_m)
    assert any(m['bands'] > 1 and len(set(m['columns'])) == 1 for m in all_m)
    assert any(len(set(m['variants'])) < m['bands'] for m in all_m)
    assert any('horizontal' in m['symmetry'] and 'vertical' in m['symmetry'] for m in all_m)
    assert sum(not m['symmetry'] for m in all_m) >= 20
    assert min(m['density'] for m in all_m) <= 0.25
    assert max(m['density'] for m in all_m) >= 0.75
    assert sum(m['components'] == 1 for m in all_m) >= 20
    assert sum(m['components'] > 1 for m in all_m) >= 20
    assert sum(m['holes'] > 0 for m in all_m) >= 10
    assert sum(m['branched'] for m in all_m) >= 10
    assert sum(m['line_like'] for m in all_m) >= 10
    assert sum(m['cuts'] for m in all_m) >= 10
    assert sum(m['shifts'] for m in all_m) >= 5
    assert any(m['ph'] != m['pw'] and m['symmetry'] and
               len(set(m['columns'])) < m['bands'] and max(m['edits']) >= 4
               for m in all_m)
    assert any(a == 1 and r == 0 for m in all_m
               for a, r in zip(m['additions'], m['removals']))
    assert any(a == 0 and r == 1 for m in all_m
               for a, r in zip(m['additions'], m['removals']))
    assert len({tuple(sorted(m['edits'])) for m in all_m}) >= 30
    assert any(any(a == b for a, b in zip(m['columns'], m['columns'][1:]))
               for m in all_m)
    palette_counts, familiar, expanded = [], 0, 0
    for p in automatic:
        train = [_analyze_pair(pair, 'source_train') for pair in p['train']]
        tests = [_analyze_pair(pair, 'target_test') for pair in p['test']]
        palette_counts.append(len({m['lattice'] for m in train + tests}))
        geometry = lambda m: (m['ph'], m['pw'], m['bands'], m['copies'])
        familiar += sum(geometry(m) in {geometry(t) for t in train} for m in tests)
        expanded += sum(m['ph'] > max(t['ph'] for t in train) or
                        m['bands'] > max(t['bands'] for t in train) for m in tests)
    assert min(palette_counts) == 1 and sum(n > 1 for n in palette_counts) >= 20
    assert familiar >= 10 and expanded >= 10
    # Negative checks reject absent exceptions, altered selections, and bad framing.
    base = automatic[0]
    for defect in ('missing', 'output_cell', 'border', 'order'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['train'][0]
        if defect == 'missing':
            info = _parse_lattice(pair['input'])
            master = _modal_master(info['panels'])
            for c in range(info['copies']):
                c0 = 1 + c * (info['pw'] + 1)
                pair['input'][1:1+info['ph'], c0:c0+info['pw']] = master
        elif defect == 'output_cell':
            pair['output'][2, 2] = (pair['output'][2, 2] + 1) % 10
        elif defect == 'border':
            pair['output'][0, 0] = (pair['output'][0, 0] + 1) % 10
        else:
            target = next(p for p in bad['train']
                          if _analyze_pair(p, 'source_train')['variants'] !=
                          _analyze_pair(p, 'source_train')['variants'][::-1])
            out = _parse_lattice(target['output'])
            target['output'] = _assemble_panel_array(out['panels'][::-1], out['lattice'])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('validator accepted ' + defect)
