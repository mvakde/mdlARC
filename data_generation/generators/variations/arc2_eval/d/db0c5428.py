"""Spatial center sampling and radial half-turn copying of eight 3x3 tiles."""

import numpy as np


_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)



def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validated_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed_code: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validated_count(num_train, "num_train")
    test_count = _validated_count(num_test, "num_test")

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed_code % 4
        lowest_train = max(2, test_count + 1)
        span = 11 - lowest_train
        train_count = lowest_train + (
            (seed_code // 4 + 3 * seed_code) % span
        )
        return train_count, test_count

    if train_count is None:
        lowest_train = max(2, test_count + 1)
        train_count = int(rng.integers(lowest_train, 11))
        return train_count, test_count

    highest_test = min(4, train_count - 1)
    if highest_test < 1:
        raise ValueError("no legal test count exists for num_train")
    test_count = int(rng.integers(1, highest_test + 1))
    return train_count, test_count


def _grid_value_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel(order="C")),
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        grid.dtype.str,
        grid.tobytes(order="C"),
    )


def _puzzle_signature(puzzle: dict) -> tuple:
    parts = []
    for split in ("train", "test"):
        parts.append(split)
        for pair in puzzle[split]:
            parts.append(_grid_signature(pair["input"]))
            parts.append(_grid_signature(pair["output"]))
    return tuple(parts)


# Exact source fixtures: shape, origin, background, ring, solved footprint.
# Used only for regression and content-based duplicate exclusion.
_SOURCE_FIXTURES = (((18, 18),
  3,
  3,
  8,
  ('664464466',
   '646646646',
   '469696964',
   '466888664',
   '649888946',
   '466888664',
   '469696964',
   '646646646',
   '664464466'),
  ('964888696888469',
   '646888646888646',
   '466888464888664',
   '888664464466888',
   '888646646646888',
   '888469696964888',
   '664466999664466',
   '946649969946649',
   '664466999664466',
   '888469696964888',
   '888646646646888',
   '888664464466888',
   '466888464888664',
   '646888646888646',
   '964888696888469')),
 ((16, 16),
  4,
  3,
  8,
  ('111111111',
   '111111111',
   '117171711',
   '111888111',
   '117888711',
   '111888111',
   '117171711',
   '111111111',
   '111111111'),
  ('711888171888117',
   '111888111888111',
   '111888111888111',
   '888111111111888',
   '888111111111888',
   '888117171711888',
   '111111777111111',
   '711117717711117',
   '111111777111111',
   '888117171711888',
   '888111111111888',
   '888111111111888',
   '111888111888111',
   '111888111888111',
   '711888171888117')),
 ((19, 19),
  5,
  6,
  8,
  ('565555565',
   '655555556',
   '553525355',
   '555888555',
   '552888255',
   '555888555',
   '553525355',
   '655555556',
   '565555565'),
  ('355888525888553',
   '556888555888655',
   '565888555888565',
   '888565555565888',
   '888655555556888',
   '888553525355888',
   '555555323555555',
   '255552252255552',
   '555555323555555',
   '888553525355888',
   '888655555556888',
   '888565555565888',
   '565888555888565',
   '556888555888655',
   '355888525888553')),
 ((17, 17),
  4,
  4,
  8,
  ('377777773',
   '774777477',
   '744717447',
   '777888777',
   '771888177',
   '777888777',
   '744717447',
   '774777477',
   '377777773'),
  ('447888717888744',
   '477888777888774',
   '773888777888377',
   '888377777773888',
   '888774777477888',
   '888744717447888',
   '777777414777777',
   '177771171177771',
   '777777414777777',
   '888744717447888',
   '888774777477888',
   '888377777773888',
   '773888777888377',
   '477888777888774',
   '447888717888744')))

def _apply_rule(
    input_grid: np.ndarray,
    analysis: dict | None = None,
) -> np.ndarray:
    if analysis is None:
        analysis = _analyze_input(input_grid)

    top = analysis["top"]
    left = analysis["left"]
    carrier = analysis["carrier"]
    output_grid = input_grid.copy()

    center_top = top + 3
    center_left = left + 3
    center_tile = np.full((3, 3), carrier, dtype=input_grid.dtype)

    for dr, dc in _DIRECTIONS:
        source_top = center_top + 3 * dr
        source_left = center_left + 3 * dc
        inward_row = 1 - dr
        inward_col = 1 - dc
        center_tile[1 + dr, 1 + dc] = input_grid[
            source_top + inward_row,
            source_left + inward_col,
        ]

    output_grid[
        center_top : center_top + 3,
        center_left : center_left + 3,
    ] = center_tile

    for dr, dc in _DIRECTIONS:
        source_top = center_top + 3 * dr
        source_left = center_left + 3 * dc
        source_tile = input_grid[
            source_top : source_top + 3,
            source_left : source_left + 3,
        ]

        destination_top = center_top + 6 * dr
        destination_left = center_left + 6 * dc
        output_grid[
            destination_top : destination_top + 3,
            destination_left : destination_left + 3,
        ] = source_tile[::-1, ::-1]

    return output_grid

def _check_grid(grid: np.ndarray) -> None:
    _require(isinstance(grid, np.ndarray), "grid must be a NumPy array")
    _require(grid.ndim == 2, "grid must be two-dimensional")
    _require(all(1 <= n <= 30 for n in grid.shape), "invalid grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")


def _analyze_input(grid: np.ndarray) -> dict:
    """Read visible geometry; sample colors need not be unique in their tile."""
    _check_grid(grid)
    background = int(grid[0, 0])  # The input ring always has at least 3 cells of margin.
    occupied = np.argwhere(grid != background)
    _require(len(occupied) == 72, "expected eight full 3x3 tiles")
    top, left = map(int, occupied.min(axis=0))
    bottom, right = map(int, occupied.max(axis=0))
    _require((bottom - top, right - left) == (8, 8), "expected a 9x9 ring")
    _require(3 <= top <= grid.shape[0] - 12, "expansion exceeds row bounds")
    _require(3 <= left <= grid.shape[1] - 12, "expansion exceeds column bounds")
    mask = np.zeros(grid.shape, dtype=bool)
    mask[top:top + 9, left:left + 9] = True
    mask[top + 3:top + 6, left + 3:left + 6] = False
    _require(np.array_equal(mask, grid != background), "malformed ring or noise")
    values, counts = np.unique(grid[mask], return_counts=True)
    carrier = int(values[np.argmax(counts)])
    _require(int(counts.max()) > 36, "carrier must be an unmistakable global majority")
    tiles, samples = [], []
    for dr, dc in _DIRECTIONS:
        r, c = top + 3 * (dr + 1), left + 3 * (dc + 1)
        tile = grid[r:r + 3, c:c + 3].copy()
        sample = int(tile[1 - dr, 1 - dc])
        _require(sample != carrier, "inward cell must visibly mark a detail")
        _require(bool(np.any(tile == carrier)), "carrier must be shared by all tiles")
        tiles.append(tile)
        samples.append(sample)
    return dict(background=background, carrier=carrier, top=top, left=left,
                tiles=tuple(tiles), samples=tuple(samples))


def _source_pairs() -> list:
    pairs = []
    for shape, r, c, bg, ring_rows, output_rows in _SOURCE_FIXTURES:
        a = np.full(shape, bg, dtype=np.int8)
        a[r:r + 9, c:c + 9] = [[int(x) for x in row] for row in ring_rows]
        b = np.full(shape, bg, dtype=np.int8)
        b[r - 3:r + 12, c - 3:c + 12] = [[int(x) for x in row] for row in output_rows]
        pairs.append(dict(input=a, output=b))
    return pairs


_SOURCE_INPUT_KEYS = frozenset(_grid_value_key(p['input']) for p in _source_pairs())
_SOURCE_PAIR_KEYS = frozenset((_grid_value_key(p['input']), _grid_value_key(p['output']))
                              for p in _source_pairs())


def _check_pair(pair: dict) -> dict:
    """Scalar coordinate check independent of the slice-based output builder."""
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair")
    a, b = pair['input'], pair['output']
    _check_grid(a)
    _check_grid(b)
    _require(a.shape == b.shape, "output dimensions changed")
    info = _analyze_input(a)
    r, c = info['top'] + 4, info['left'] + 4
    assigned = np.zeros(a.shape, dtype=bool)
    assigned[r - 1:r + 2, c - 1:c + 2] = True
    _require(b[r, c] == info['carrier'], "incorrect center carrier")
    for dr, dc in _DIRECTIONS:
        _require(b[r + dr, c + dc] == a[r + 2 * dr, c + 2 * dc],
                 "incorrect directional sample")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sr, sc = r + 3 * dr + dy, c + 3 * dc + dx
                tr, tc = r + 6 * dr - dy, c + 6 * dc - dx
                _require(b[tr, tc] == a[sr, sc], "incorrect radial tile half-turn")
                assigned[tr, tc] = True
    _require(int(assigned.sum()) == 81, "incorrect destination geometry")
    _require(np.array_equal(a[~assigned], b[~assigned]), "ring or unused background changed")
    _require(int(np.count_nonzero(a != b)) == 81, "completion must fill exactly 81 cells")
    _require(np.array_equal(np.unique(a), np.unique(b)), "palette changed")
    return info


def _dihedral(tile: np.ndarray) -> tuple:
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _grouped(samples: tuple) -> bool:
    return len({samples[i] for i in (0, 2, 5, 7)}) == 1 and len(
        {samples[i] for i in (1, 3, 4, 6)}) == 1


def _learn_candidates(train_pairs: list) -> tuple:
    """Learn per-direction sample coordinates and D4 operations from labeled training only.

    Allowing a different D4 operation in every direction includes radial reflection
    and is stronger than checking only one global transform for the whole ring.
    """
    positions = [set(range(9)) for _ in _DIRECTIONS]
    operations = [set(range(8)) for _ in _DIRECTIONS]
    non_grouped = False
    for pair in train_pairs:
        info = _analyze_input(pair['input'])
        r, c = info['top'] + 4, info['left'] + 4
        non_grouped |= not _grouped(info['samples'])
        for i, (dr, dc) in enumerate(_DIRECTIONS):
            tile = info['tiles'][i]
            target = pair['output'][r + 6 * dr - 1:r + 6 * dr + 2,
                                    c + 6 * dc - 1:c + 6 * dc + 2]
            value = pair['output'][r + dr, c + dc]
            positions[i].intersection_update(j for j in range(9) if tile.flat[j] == value)
            operations[i].intersection_update(k for k, t in enumerate(_dihedral(tile))
                                               if np.array_equal(t, target))
    _require(all(positions) and all(operations), "training implies contradictory rules")
    return positions, operations, non_grouped


def _unresolved(candidates: tuple, query: np.ndarray) -> tuple:
    positions, operations, non_grouped = candidates
    info = _analyze_input(query)
    directions = set()
    for i, tile in enumerate(info['tiles']):
        sample_values = {int(tile.flat[j]) for j in positions[i]}
        views = _dihedral(tile)
        tile_values = {views[k].tobytes() for k in operations[i]}
        if len(sample_values) != 1 or len(tile_values) != 1:
            directions.add(i)
    return directions, not non_grouped and not _grouped(info['samples'])


def predict_from_training(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    """Prediction check with no test label, seed or generation metadata."""
    candidates = _learn_candidates(train_pairs)
    unresolved, grouped_shortcut = _unresolved(candidates, test_input)
    _require(not unresolved, "training leaves a query-relevant geometric ambiguity")
    _require(not grouped_shortcut, "nongrouped query lacks a nongrouped demonstration")
    info = _analyze_input(test_input)
    r, c = info['top'] + 4, info['left'] + 4
    answer = test_input.copy()
    answer[r, c] = info['carrier']
    for i, (dr, dc) in enumerate(_DIRECTIONS):
        tile = info['tiles'][i]
        answer[r + dr, c + dc] = tile.flat[min(candidates[0][i])]
        answer[r + 6 * dr - 1:r + 6 * dr + 2, c + 6 * dc - 1:c + 6 * dc + 2] = (
            _dihedral(tile)[min(candidates[1][i])])
    return answer


def _repair_training(rng: np.random.Generator, train: list, queries: list) -> None:
    # Teach independent directional colors only when the proposed queries need them.
    if any(not _grouped(_analyze_input(q)['samples']) for q in queries) and all(
            _grouped(_analyze_input(p['input'])['samples']) for p in train):
        pair = train[int(rng.integers(len(train)))]
        info = _analyze_input(pair['input'])
        palette = rng.permutation([x for x in range(10) if x not in
                                   (info['background'], info['carrier'])])
        query = next(q for q in queries if not _grouped(_analyze_input(q)['samples']))
        pattern = _analyze_input(query)['samples']
        color_map = {v: int(palette[j]) for j, v in enumerate(dict.fromkeys(pattern))}
        for i, (dr, dc) in enumerate(_DIRECTIONS):
            pair['input'][info['top'] + 4 + 2 * dr, info['left'] + 4 + 2 * dc] = color_map[pattern[i]]
        pair['output'] = _apply_rule(pair['input'])
    candidates = _learn_candidates(train)
    missing = set().union(*(_unresolved(candidates, q)[0] for q in queries))
    for i in sorted(missing):
        pair = train[int(rng.integers(len(train)))]
        info = _analyze_input(pair['input'])
        dr, dc = _DIRECTIONS[i]
        sample = info['samples'][i]
        auxiliary = int(rng.choice([x for x in range(10) if x not in
                                    (info['background'], info['carrier'], sample)]))
        # A unique inward mark plus one off-axis mark has eight distinct D4 images.
        # Enumerating nine positions is bounded and always supplies such a witness.
        for flat in rng.permutation(9):
            if flat == (1 - dr) * 3 + 1 - dc:
                continue
            tile = np.full((3, 3), info['carrier'], dtype=np.int8)
            tile[1 - dr, 1 - dc] = sample
            tile.flat[flat] = auxiliary
            if len({t.tobytes() for t in _dihedral(tile)}) == 8:
                r, c = info['top'] + 3 * (dr + 1), info['left'] + 3 * (dc + 1)
                pair['input'][r:r + 3, c:c + 3] = tile
                pair['output'] = _apply_rule(pair['input'])
                break
        else:
            raise AssertionError('no asymmetric witness exists')
def _directional_samples(rng: np.random.Generator, details: list, simple: bool,
                         symmetric: bool) -> tuple:
    if simple or symmetric:
        common = rng.random() < 0.55
        return tuple(details[0] if common or dr * dc else details[1]
                     for dr, dc in _DIRECTIONS)
    mode = int(rng.choice(3, p=[0.20, 0.20, 0.60]))
    if mode < 2:
        return tuple(details[0] if mode == 0 or dr * dc else details[1]
                     for dr, dc in _DIRECTIONS)
    count = int(rng.choice(np.arange(2, 9), p=[.16, .23, .23, .10, .09, .07, .12]))
    labels = list(range(count)) + list(rng.integers(count, size=8 - count))
    rng.shuffle(labels)
    return tuple(details[j] for j in labels)


def _orbit(point: tuple, reflection: bool) -> set:
    y, x = point
    result = set()
    for _ in range(4):
        result.add((y, x))
        if reflection:
            result.add((x, y))
        y, x = 8 - x, y
    return result


def _coherent_ring(rng: np.random.Generator, carrier: int, background: int,
                   samples: tuple, details: list, richness: int,
                   reflection: bool) -> np.ndarray:
    ring = np.full((9, 9), carrier, dtype=np.int8)
    ring[3:6, 3:6] = background
    inward = {(4 + 2 * dr, 4 + 2 * dc) for dr, dc in _DIRECTIONS}
    if richness:
        available = {(r, c) for r in range(9) for c in range(9)
                     if not (3 <= r < 6 and 3 <= c < 6)} - inward
        orbits = []
        while available:
            cells = _orbit(min(available), reflection)
            available.difference_update(cells)
            orbits.append(sorted(cells))
        rng.shuffle(orbits)
        budget = int(rng.integers(1, 12 if richness == 1 else 24))
        used = 0
        for cells in orbits:
            if used + len(cells) > budget or rng.random() < .28:
                continue
            color = int(rng.choice(details[:int(rng.integers(1, 9))]))
            for r, c in cells:
                ring[r, c] = color
            used += len(cells)
    for (dr, dc), sample in zip(_DIRECTIONS, samples):
        ring[4 + 2 * dr, 4 + 2 * dc] = sample
    return ring


def _individual_tile(rng: np.random.Generator, carrier: int, sample: int,
                     inward: tuple, colors: list, richness: int) -> np.ndarray:
    tile = np.full((3, 3), carrier, dtype=np.int8)
    tile[inward] = sample
    if not richness or rng.random() < .24:
        return tile
    count = int(rng.integers(1, 4 if richness == 1 else 7))
    motif = int(rng.integers(4))
    # The inward color participates in marks and patches, not just center sampling.
    choices = [sample] + colors[:int(rng.integers(1, len(colors) + 1))]
    if motif == 3:
        tile[2 - inward[0], 2 - inward[1]] = sample
        orbits = [[(0, 0), (2, 2)], [(0, 1), (2, 1)],
                  [(0, 2), (2, 0)], [(1, 0), (1, 2)], [(1, 1)]]
        rng.shuffle(orbits)
        for cells in orbits:
            if inward in cells or count < len(cells):
                continue
            value = int(rng.choice(choices))
            for p in cells:
                tile[p] = value
            count -= len(cells)
        return tile
    positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != inward]
    if motif == 0:  # Repeated marks, optionally multiple colors.
        rng.shuffle(positions)
        selected = positions[:count]
    elif motif == 1:  # A line segment, extended with marks for larger counts.
        axis, fixed = int(rng.integers(2)), int(rng.integers(3))
        rng.shuffle(positions)
        selected = sorted(positions, key=lambda p: p[axis] != fixed)[:count]
    else:  # Connected growth from the inward mark or another random cell.
        start = inward if rng.random() < .65 else positions[int(rng.integers(8))]
        cluster = {start}
        while len(cluster - {inward}) < count:
            frontier = {q for p in cluster for q in
                        ((p[0]-1,p[1]), (p[0]+1,p[1]), (p[0],p[1]-1), (p[0],p[1]+1))
                        if 0 <= q[0] < 3 and 0 <= q[1] < 3} - cluster
            choices_frontier = sorted(frontier)
            cluster.add(choices_frontier[int(rng.integers(len(choices_frontier)))])
        selected = sorted(cluster - {inward})
    color = int(rng.choice(choices))
    for p in selected:
        tile[p] = int(rng.choice(choices)) if motif == 0 and rng.random() < .5 else color
    return tile


def _make_ring(rng: np.random.Generator, background: int, carrier: int,
               details: list, richness: int) -> np.ndarray:
    arrangement = int(rng.choice(4, p=[.22, .13, .20, .45])) if richness else 0
    samples = _directional_samples(rng, details, richness == 0, arrangement == 0)
    if arrangement < 3:
        # D4, C4, or a shared geometric motif with independent sample colors.
        if arrangement == 1:
            samples = _directional_samples(rng, details, False, True)
        return _coherent_ring(rng, carrier, background, samples, details, richness,
                              reflection=arrangement != 1)
    ring = np.full((9, 9), background, dtype=np.int8)
    extras = []
    for (dr, dc), sample in zip(_DIRECTIONS, samples):
        tile = _individual_tile(rng, carrier, sample, (1-dr, 1-dc), details, richness)
        r, c = 3 * (dr + 1), 3 * (dc + 1)
        ring[r:r+3, c:c+3] = tile
        extras.extend((r+y, c+x) for y in range(3) for x in range(3)
                      if (y, x) != (1-dr, 1-dc) and tile[y, x] != carrier)
    # Global carrier clarity permits dense individual tiles. A small reserve also
    # lets coverage repair add distinguishing marks without erasing any witness.
    rng.shuffle(extras)
    for r, c in extras[23:]:
        ring[r, c] = carrier
    return ring


def _layout(rng: np.random.Generator, parameters: dict, regime: str) -> tuple:
    if parameters['test_regime'] == 'size_shift':
        low, high = (15, 23) if regime == 'source_train' else (23, 31)
        h, w = map(int, rng.integers(low, high, size=2))
    elif parameters['nearby_shapes']:
        h, w = (int(np.clip(n + rng.integers(-2, 3), 15, 30))
                for n in parameters['base_shape'])
    else:
        mode = int(rng.choice(4, p=[.58, .22, .08, .12]))
        if mode == 0:
            h, w = map(int, rng.integers(15, 31, size=2))
        elif mode == 1:
            h = w = int(rng.integers(15, 31))
        elif mode == 2:
            h = w = 15
        else:
            h, w = (15, 30) if rng.random() < .5 else (30, 15)
    margins = []
    for n in (h, w):
        spare = n - 15
        mode = int(rng.integers(4))
        offset = (0 if mode == 0 else spare if mode == 1 else spare // 2
                  if mode == 2 else int(rng.integers(spare + 1)))
        margins.append(offset + 3)
    return h, w, margins[0], margins[1]


def _new_pair(rng: np.random.Generator, parameters: dict, regime: str,
              train: list | None = None) -> dict:
    _require(regime in ('source_train', 'target_test'), 'unknown regime')
    test = regime == 'target_test'
    target_mode = parameters['test_regime']
    h, w, r, c = _layout(rng, parameters, regime)
    bg = parameters['background'] if parameters['fixed_background'] else int(rng.integers(10))
    order = [int(x) for x in rng.permutation([x for x in range(10) if x != bg])]
    carrier, details = order[0], order[1:]
    richness = int(rng.choice(3, p=parameters['richness_weights']))
    if test and target_mode == 'simpler':
        richness = 0 if rng.random() < .8 else 1
    if test and target_mode == 'richer':
        richness = 2
    if train and test:
        familiar = train[int(rng.integers(len(train)))]['input']
        info = _analyze_input(familiar)
        if rng.random() < (.8 if target_mode == 'familiar' else .15):
            bg, carrier = info['background'], info['carrier']
            old_details = [int(x) for x in np.unique(familiar) if x not in (bg, carrier)]
            details = old_details + [x for x in range(10) if x not in [bg, carrier] + old_details]
        if target_mode == 'color_shift':
            used = {_analyze_input(p['input'])['carrier'] for p in train}
            choices = [x for x in range(10) if x != bg and x not in used]
            if choices:
                carrier = int(rng.choice(choices))
                details = [int(x) for x in rng.permutation([x for x in range(10)
                                                           if x not in (bg, carrier)])]
        if target_mode != 'size_shift' and rng.random() < (.75 if target_mode == 'familiar' else .12):
            h, w = familiar.shape
            r, c = info['top'], info['left']
    ring = _make_ring(rng, bg, carrier, details, richness)
    a = np.full((h, w), bg, dtype=np.int8)
    a[r:r+9, c:c+9] = ring
    return dict(input=a, output=_apply_rule(a))


def _distinct_pair(pair: dict, seen: set) -> dict:
    if _grid_value_key(pair['input']) not in seen | _SOURCE_INPUT_KEYS:
        return pair
    # Bounded deterministic fallback: 90 distinct background/carrier assignments.
    # At most 17 inputs (13 preceding pairs plus four sources) can be forbidden.
    a = pair['input']
    info = _analyze_input(a)
    order = [info['background'], info['carrier']] + [x for x in range(10)
              if x not in (info['background'], info['carrier'])]
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            target = [bg, carrier] + [x for x in range(10) if x not in (bg, carrier)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[order] = target
            candidate = mapping[a]
            if _grid_value_key(candidate) not in seen | _SOURCE_INPUT_KEYS:
                return dict(input=candidate, output=mapping[pair['output']])
    raise AssertionError('distinct palette fallback exhausted')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    n, m = _resolve_counts(rng, int(seed), num_train, num_test)
    # Only the source geometry and visible-role interpretation are conventions.
    # These puzzle-level sampling preferences never encode an output color map.
    level = int(rng.choice(3, p=[.14, .30, .56]))
    parameters = dict(background=int(rng.integers(10)), fixed_background=rng.random() < .25,
                      nearby_shapes=rng.random() < .25,
                      base_shape=tuple(map(int, rng.integers(15, 31, size=2))),
                      richness_weights=([1., 0., 0.] if level == 0 else [.2, .6, .2]
                                        if level == 1 else [.1, .25, .65]),
                      test_regime=str(rng.choice(['balanced', 'familiar', 'color_shift',
                                                  'size_shift', 'simpler', 'richer'])))
    train = [_new_pair(rng, parameters, 'source_train') for _ in range(n)]
    tests = [_new_pair(rng, parameters, 'target_test', train) for _ in range(m)]
    _repair_training(rng, train, [p['input'] for p in tests])
    rng.shuffle(train)
    puzzle, seen = dict(train=[], test=[]), set()
    for split, pairs in (('train', train), ('test', tests)):
        for pair in pairs:
            pair = _distinct_pair(pair, seen)
            seen.add(_grid_value_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'invalid pair counts')
    seen_inputs, seen_pairs = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check_pair(pair)
            key = _grid_value_key(pair['input']), _grid_value_key(pair['output'])
            _require(key[0] not in seen_inputs and key not in seen_pairs, 'duplicate pair or input')
            _require(key not in _SOURCE_PAIR_KEYS, 'exact source pair')
            seen_inputs.add(key[0])
            seen_pairs.add(key)
    candidates = _learn_candidates(puzzle['train'])
    for pair in puzzle['test']:
        missing, shortcut = _unresolved(candidates, pair['input'])
        _require(not missing, 'training does not resolve test geometry')
        _require(not shortcut, 'independent directional colors are not demonstrated')
def measure_pair(pair: dict) -> dict:
    """Task-specific observables computed exclusively from returned cells."""
    info = _analyze_input(pair['input'])
    bg, carrier = info['background'], info['carrier']
    a = pair['input']
    h, w = a.shape
    r, c = info['top'], info['left']
    ring = a[r:r+9, c:c+9]
    counts = [int(np.count_nonzero(t != carrier)) - 1 for t in info['tiles']]
    repeated, patches, lines, symmetric, dense = 0, 0, 0, 0, 0
    radial_witnesses = 0
    for (dr, dc), t, sample in zip(_DIRECTIONS, info['tiles'], info['samples']):
        repeated += int(np.count_nonzero(t == sample) > 1)
        symmetric += int(np.array_equal(t, t[::-1, ::-1]))
        dense += int(np.count_nonzero(t == carrier) < 5)
        lines += int(any(np.all(row == row[0]) and row[0] != carrier
                         for row in tuple(t) + tuple(t.T)))
        points = set(zip(*np.nonzero(t == sample)))
        connected = False
        while points:
            todo = [points.pop()]
            size = 0
            while todo:
                y, x = todo.pop()
                size += 1
                for q in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if q in points:
                        points.remove(q)
                        todo.append(q)
            connected |= size >= 3
        patches += int(connected)
        if dr == 0 or dc == 0:
            radial = t[::-1, :] if dr else t[:, ::-1]
            radial_witnesses += int(not np.array_equal(radial, t[::-1, ::-1]))
    normalization = {bg: 0, carrier: 1}
    signature = []
    for value in ring.flat:
        normalization.setdefault(int(value), len(normalization))
        signature.append(normalization[int(value)])
    samples = info['samples']
    labels = {}
    sample_pattern = []
    for value in samples:
        labels.setdefault(value, len(labels))
        sample_pattern.append(labels[value])
    return dict(shape=(h, w), layout=(h, w, r, c), background=bg, carrier=carrier,
                palette=tuple(map(int, np.unique(a))), color_count=len(np.unique(a)),
                margins=(r-3, h-r-12, c-3, w-c-12),
                sample_count=len(set(samples)), sample_pattern=tuple(sample_pattern),
                grouped=_grouped(samples), decoration_counts=tuple(counts),
                decoration_total=sum(counts), mixed_decoration=min(counts) == 0 < max(counts),
                auxiliary_count=len(set(map(int, np.unique(a))) - {bg, carrier} - set(samples)),
                repeated_tiles=repeated, sample_patches=patches, line_tiles=lines,
                halfturn_symmetric_tiles=symmetric, dense_tiles=dense,
                radial_witnesses=radial_witnesses,
                quarter_turn=bool(np.array_equal(ring, np.rot90(ring))),
                reflection=bool(np.array_equal(ring, ring.T) and np.array_equal(ring, ring[:, ::-1])),
                structure=tuple(signature))


def _expect_rejection(puzzle: dict) -> None:
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError('validator accepted a corrupted or ambiguous puzzle')


def self_test() -> None:
    sources = _source_pairs()
    for pair in sources:
        _check_pair(pair)
        _require(np.array_equal(_apply_rule(pair['input']), pair['output']),
                 'source regression failed through input analysis')
    _require(np.array_equal(predict_from_training(sources[:3], sources[3]['input']),
                            sources[3]['output']), 'source train-only prediction failed')
    all_features, puzzle_features, signatures = [], [], set()
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        features = {split: [measure_pair(p) for p in puzzle[split]] for split in ('train', 'test')}
        puzzle_features.append(features)
        all_features.extend(features['train'] + features['test'])
        # The predictor is deliberately given only copied training labels and a query.
        train = [{k: a.copy() for k, a in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            _require(np.array_equal(predict_from_training(train, pair['input'].copy()), pair['output']),
                     'training-only prediction disagrees')
    _require(len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3,
             'insufficient puzzle/count diversity')
    _require(len({f['structure'] for f in all_features}) >= 100, 'motifs collapse after recoloring')
    _require(len({f['layout'] for f in all_features}) >= 100, 'insufficient placement diversity')
    _require({f['background'] for f in all_features} == set(range(10)), 'background color omitted')
    _require({f['sample_count'] for f in all_features} == set(range(1, 9)), 'sample counts omitted')
    _require(len({f['sample_pattern'] for f in all_features}) >= 25, 'directional subsets collapsed')
    _require(max(f['color_count'] for f in all_features) == 10, 'palette richness is capped')
    _require(max(f['auxiliary_count'] for f in all_features) >= 3, 'multiple auxiliary colors missing')
    for property_name in ('mixed_decoration', 'repeated_tiles', 'sample_patches', 'line_tiles',
                          'halfturn_symmetric_tiles', 'dense_tiles'):
        _require(sum(bool(f[property_name]) for f in all_features) >= 8,
                 'insufficient coverage: ' + property_name)
    _require(sum(f['quarter_turn'] and f['reflection'] and f['decoration_total'] > 0
                 for f in all_features) >= 8, 'coherent decorated rings missing')
    _require(sum(not f['quarter_turn'] and not f['reflection'] for f in all_features) >= 40,
             'asymmetric rings missing')
    for split in ('train', 'test'):
        rows = [f for p in puzzle_features for f in p[split]]
        _require({n for f in rows for n in f['shape']} == set(range(15, 31)), 'canvas size gap')
        _require(any(f['shape'] == (15, 15) for f in rows), 'tight canvas missing')
        _require(any(max(f['shape']) / min(f['shape']) >= 1.8 for f in rows), 'long rectangles missing')
        _require(any(min(f['shape']) >= 26 for f in rows), 'large canvases missing')
        _require(all(any(f['margins'][side] == 0 for f in rows) for side in range(4)), 'edge placement gap')
    varied_backgrounds = sum(len({f['background'] for f in p['train'] + p['test']}) > 1
                             for p in puzzle_features)
    _require(15 <= varied_backgrounds <= 40, 'fixed/variable background regimes collapsed')
    for name in ('layout', 'sample_pattern', 'decoration_counts', 'color_count'):
        _require(sum(len({f[name] for f in p['train'] + p['test']}) > 1 for p in puzzle_features) >= 20,
                 'within-puzzle variation missing: ' + name)
    familiar_roles = familiar_layouts = new_roles = simpler = richer = 0
    for p in puzzle_features:
        for q in p['test']:
            familiar_roles += q['carrier'] in {f['carrier'] for f in p['train']}
            new_roles += q['carrier'] not in {f['carrier'] for f in p['train']}
            familiar_layouts += q['layout'] in {f['layout'] for f in p['train']}
            score = q['decoration_total'] + q['sample_count']
            simpler += any(score < f['decoration_total'] + f['sample_count'] for f in p['train'])
            richer += any(score > f['decoration_total'] + f['sample_count'] for f in p['train'])
    _require(min(familiar_roles, familiar_layouts, new_roles, simpler, richer) >= 8,
             'test regime variation is missing')
    combined = [f for f in all_features if not f['grouped'] and f['repeated_tiles']
                and f['auxiliary_count'] >= 2 and len(set(f['margins'])) > 1]
    _require(len(combined) >= 4, 'compatible variation axes are not combining')
    orders = {tuple((f['grouped'], f['decoration_total'] == 0) for f in p['train'][:3])
              for p in puzzle_features}
    _require(len(orders) >= 8, 'training has collapsed to a fixed profile order')
    _require(any(all(f['decoration_total'] == 0 for f in p['train'] + p['test'])
                 for p in puzzle_features), 'simple symmetric puzzles have disappeared')
    a, b = generate_puzzle(314159, 6, 2), generate_puzzle(314159, 6, 2)
    _require(_puzzle_signature(a) == _puzzle_signature(b), 'nondeterministic generation')
    requests = ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None))
    for i, (n, m) in enumerate(requests):
        p = generate_puzzle(2000 + i, n, m)
        validate_generated_puzzle(p)
        _require(n is None or len(p['train']) == n, 'requested training count ignored')
        _require(m is None or len(p['test']) == m, 'requested test count ignored')
    for n, m in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (3, True)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for seed in (18, 97, 968):
        p = generate_puzzle(seed, 2, 1)
        _require(np.array_equal(predict_from_training(p['train'], p['test'][0]['input']),
                                p['test'][0]['output']), 'reviewed small-count failure recurred')
    # An explicit adversary: symmetric demonstrations, transverse-asymmetric query.
    base = sources[1]
    train = [{k: (v + delta) % 10 for k, v in base.items()} for delta in (1, 2)]
    query = (base['input'] + 3) % 10
    info = _analyze_input(query)
    query[info['top'], info['left'] + 3] = (info['carrier'] + 2) % 10
    bad = dict(train=train, test=[dict(input=query, output=_apply_rule(query))])
    _expect_rejection(bad)
    _repair_training(np.random.default_rng(999), bad['train'], [query])
    validate_generated_puzzle(bad)
    # Detect wrong center, wrong tile copy, altered original ring and background noise.
    for where in ('center', 'copy', 'ring', 'unused'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
        pair = broken['test'][0]
        info = _analyze_input(pair['input'])
        r, c = info['top'], info['left']
        point = {'center': (r+4, c+4), 'copy': (r-3, c-3),
                 'ring': (r, c), 'unused': (r-3, c)}[where]
        pair['output'][point] = (int(pair['output'][point]) + 1) % 10
        _expect_rejection(broken)
