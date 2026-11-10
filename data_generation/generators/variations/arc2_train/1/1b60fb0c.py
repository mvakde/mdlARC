"""Generate incomplete fourfold figures with one missing quarter-turn."""

import numpy as np


_SOURCE_INPUTS = frozenset({
    "0000000000000111110000011111000000010011000001001100000111110000010011000001001100011111000001111100",
    "0000000000000111000000001100000000100010000011111000001101100000010000000011000000001110000000000000",
    "0000000000000111110000000100000000111001000001010100000111110000010101000011100100000100000001111100",
    "0000100000001111100000001100100000100010000011111100001101100000010010000011000000011111000000010000",
})


def _rotate(cell, turns):
    row, col = cell
    for _ in range(turns):
        row, col = col, -row
    return row, col


def _turn(cell, center2, turns):
    row, col = cell
    dr2, dc2 = _rotate((2 * row - center2[0], 2 * col - center2[1]), turns)
    return (center2[0] + dr2) // 2, (center2[1] + dc2) // 2


def _sector(cell, center2):
    row, col = cell
    dr = 2 * row - center2[0]
    dc = 2 * col - center2[1]
    if dc > abs(dr):
        return 0  # east
    if dr > abs(dc):
        return 1  # south
    if -dc > abs(dr):
        return 2  # west
    if -dr > abs(dc):
        return 3  # north
    return None


def _candidates(foreground, shape):
    """Infer pivots using only the visible cells and strict orbit evidence."""
    height, width = shape
    if not foreground:
        return []
    rows = [cell[0] for cell in foreground]
    cols = [cell[1] for cell in foreground]
    anchors = list(dict.fromkeys((min(foreground), max(foreground),
                                  min(foreground, key=lambda p: p[1]),
                                  max(foreground, key=lambda p: p[1]))))
    found = []
    for cr2 in range(2 * min(rows), 2 * max(rows) + 1):
        for cc2 in range(2 * min(cols) + cr2 % 2,
                         2 * max(cols) + 1, 2):
            center2 = (cr2, cc2)
            plausible = True
            for cell in anchors:
                orbit = {_turn(cell, center2, t) for t in range(4)}
                if (len(orbit) not in (1, 4)
                        or any(not (0 <= r < height and 0 <= c < width)
                               for r, c in orbit)
                        or len(orbit & foreground) < len(orbit) - 1):
                    plausible = False
                    break
            if not plausible:
                continue
            visited = set()
            missing = set()
            direction = None
            for cell in foreground:
                if cell in visited:
                    continue
                orbit = {_turn(cell, center2, t) for t in range(4)}
                if (len(orbit) not in (1, 4)
                        or any(not (0 <= r < height and 0 <= c < width)
                               for r, c in orbit)):
                    plausible = False
                    break
                visited.update(orbit)
                absent = orbit - foreground
                if len(absent) > 1 or (len(orbit) == 1 and absent):
                    plausible = False
                    break
                if absent:
                    one = next(iter(absent))
                    sector = _sector(one, center2)
                    if sector is None or (direction is not None and sector != direction):
                        plausible = False
                        break
                    direction = sector
                    missing.add(one)
            if plausible and len(missing) >= 3 and direction is not None:
                found.append((center2, missing, direction))
    return found


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None:
        low = 2 if num_test is None else max(2, num_test + 1)
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _motif(rng, radius, parity, elaborate):
    """Create an east wedge in doubled coordinates around a zero pivot."""
    hub_level = int(rng.integers(0, 3))
    hub_limit = (2 * hub_level if parity == 0 else 2 * hub_level + 1)
    motif = {(r, c) for r in range(-hub_limit, hub_limit + 1, 2)
             for c in range(-hub_limit, hub_limit + 1, 2)} if parity == 0 else {
                 (r, c) for r in range(-hub_limit, hub_limit + 1, 2)
                 for c in range(-hub_limit, hub_limit + 1, 2)}
    shaft_half = int(rng.integers(0, 3))
    sides = (-1, 1)
    symmetric = bool(rng.random() < 0.24)
    for d in range(1, radius + 1):
        p = 2 * d if parity == 0 else 2 * d - 1
        for q in range(-2 * shaft_half - parity, 2 * shaft_half + parity + 1, 2):
            if p > abs(q):
                motif.add((q, p))
    # Each event varies independently in radial position, reach and side.
    event_count = int(rng.integers(1, 4 if not elaborate else 8))
    for _ in range(event_count):
        d = int(rng.integers(2, radius + 1))
        p = 2 * d if parity == 0 else 2 * d - 1
        max_q = p - 2
        choices = [q for q in range(parity, max_q + 1, 2)
                   if q > 2 * shaft_half + parity]
        if not choices:
            continue
        reach = int(rng.choice(choices))
        chosen_sides = sides if symmetric or rng.random() < 0.35 else (int(rng.choice(sides)),)
        hollow = bool(rng.random() < 0.24)
        for sign in chosen_sides:
            for q in range(2 * shaft_half + parity + 2, reach + 1, 2):
                if not hollow or q == reach or rng.random() < 0.42:
                    motif.add((sign * q, p))
            if elaborate and d < radius and rng.random() < 0.43:
                next_p = p + 2
                if next_p > reach:
                    motif.add((sign * reach, next_p))
    # A detached detail is a coherent fourth-copy feature, not noise.
    if elaborate and radius >= 5 and rng.random() < 0.48:
        d = int(rng.integers(max(4, radius - 2), radius + 1))
        p = 2 * d if parity == 0 else 2 * d - 1
        q = p - 2
        if q > 2 * shaft_half + parity + 2:
            for sign in (sides if symmetric else (int(rng.choice(sides)),)):
                motif.add((sign * q, p))
    return motif


def _margins(rng, contact_count):
    margins = [int(rng.integers(1, 5)) for _ in range(4)]
    for side in rng.choice(4, size=contact_count, replace=False):
        margins[int(side)] = 0
    return margins  # top, right, bottom, left


def _make_pair(rng, colors, missing_direction, contact_count,
               size_regime, pivot_parity, require_asymmetric, used_inputs):
    for _ in range(120):
        radius = int(rng.integers(3, 9 if size_regime == "train_small" else 11))
        parity = (int(rng.integers(0, 2)) if pivot_parity is None
                  else pivot_parity)
        elaborate = bool(rng.random() < (0.68 if size_regime == "test_large" else 0.48))
        motif = _motif(rng, radius, parity, elaborate)
        complete2 = {_rotate(cell, t) for cell in motif for t in range(4)}
        visible2 = {_rotate(cell, t) for cell in motif
                    for t in range(4) if t != missing_direction}
        # Cell coordinates are integral after translating the doubled bounding box.
        min_r2 = min(r for r, _ in complete2)
        max_r2 = max(r for r, _ in complete2)
        min_c2 = min(c for _, c in complete2)
        max_c2 = max(c for _, c in complete2)
        top, right, bottom, left = _margins(rng, contact_count)
        height = (max_r2 - min_r2) // 2 + 1 + top + bottom
        width = (max_c2 - min_c2) // 2 + 1 + left + right
        if height > 30 or width > 30:
            continue
        def place(cell):
            return ((cell[0] - min_r2) // 2 + top,
                    (cell[1] - min_c2) // 2 + left)
        complete = {place(cell) for cell in complete2}
        visible = {place(cell) for cell in visible2}
        missing = complete - visible
        if len(missing) < 3 or len(visible) >= height * width // 2:
            continue
        candidates = _candidates(visible, (height, width))
        if len(candidates) != 1 or candidates[0][1] != missing:
            continue
        center2 = candidates[0][0]
        reflected = ({(center2[0] - r, c) for r, c in missing}
                     if missing_direction in (0, 2) else
                     {(r, center2[1] - c) for r, c in missing})
        if require_asymmetric and reflected == missing:
            continue
        background, source, completion_color = colors
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for r, c in visible:
            input_grid[r, c] = source
            output_grid[r, c] = source
        for r, c in missing:
            output_grid[r, c] = completion_color
        key = (input_grid.shape, input_grid.tobytes())
        if key in used_inputs or (height == 10 and width == 10
                                  and "".join(str(int(x)) for x in input_grid.flat)
                                  in _SOURCE_INPUTS):
            continue
        used_inputs.add(key)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("Could not construct a unique orbit completion")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    completion = int(rng.integers(0, 10))
    direction_mode = int(rng.integers(0, 2))
    stable_direction = int(rng.integers(0, 4))
    pivot_mode = int(rng.integers(0, 3))
    stable_parity = int(rng.integers(0, 2))
    regime_mode = int(rng.integers(0, 3))  # source shift, shared contact, free
    shared_contact = int(rng.integers(0, 5))
    used_inputs = set()
    puzzle = {"train": [], "test": []}
    first_train_role = None
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if direction_mode == 0:
                direction = stable_direction
            elif section == "train" and index < 4:
                direction = (stable_direction + index) % 4
            else:
                direction = int(pair_rng.integers(0, 4))
            if pivot_mode == 0:
                parity = stable_parity
            elif section == "train" and index < 2:
                parity = index
            else:
                parity = None
            if regime_mode == 0:
                contact = 0 if section == "train" else 4
                size_regime = "train_small" if section == "train" else "test_large"
            elif regime_mode == 1:
                contact = shared_contact
                size_regime = "any"
            else:
                contact = int(pair_rng.integers(0, 5))
                size_regime = "any"
            available = [color for color in range(10) if color != completion]
            background, source = (int(x) for x in
                                  pair_rng.choice(available, size=2, replace=False))
            if section == "train" and index == 0:
                first_train_role = (background, source)
            elif section == "train" and index == 1 and (background, source) == first_train_role:
                source = int(pair_rng.choice([color for color in available
                                               if color not in (background, source)]))
            puzzle[section].append(_make_pair(pair_rng,
                                               (background, source, completion),
                                               direction, contact, size_regime,
                                               parity, section == "train" and index == 0,
                                               used_inputs))
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    input_grid = pair["input"]
    output_grid = pair["output"]
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2
    background = int(values[np.argmax(counts)])
    source = int(values[np.argmin(counts)])
    assert counts.max() > counts.min()
    changed = output_grid != input_grid
    assert np.any(changed)
    assert np.all(input_grid[changed] == background)
    completion_values = np.unique(output_grid[changed])
    assert len(completion_values) == 1
    completion = int(completion_values[0])
    assert completion not in (background, source)
    assert np.all(output_grid[~changed] == input_grid[~changed])
    foreground = {tuple(int(x) for x in point) for point in np.argwhere(input_grid == source)}
    candidates = _candidates(foreground, input_grid.shape)
    assert len(candidates) == 1
    center2, missing, direction = candidates[0]
    assert missing == {tuple(int(x) for x in point) for point in np.argwhere(changed)}
    completed = foreground | missing
    assert all({_turn(cell, center2, t) for t in range(4)} <= completed
               for cell in completed)
    edges = (any(r == 0 for r, _ in completed),
             any(c == input_grid.shape[1] - 1 for _, c in completed),
             any(r == input_grid.shape[0] - 1 for r, _ in completed),
             any(c == 0 for _, c in completed))
    return completion, direction, center2, sum(edges), (background, source)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    completion_colors = set()
    train_directions = set()
    test_directions = set()
    train_contacts = []
    test_contacts = []
    roles = []
    train_roles = set()
    asymmetric_train = 0
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            if inp.shape == (10, 10):
                assert "".join(str(int(x)) for x in inp.flat) not in _SOURCE_INPUTS
            completion, direction, center2, contacts, role = _pair_features(pair)
            completion_colors.add(completion)
            roles.append(role)
            if section == "train":
                train_directions.add(direction)
                train_contacts.append(contacts)
                train_roles.add(role)
                missing = {tuple(int(x) for x in point)
                           for point in np.argwhere(inp != out)}
                reflected = ({(center2[0] - r, c) for r, c in missing}
                             if direction in (0, 2) else
                             {(r, center2[1] - c) for r, c in missing})
                asymmetric_train += reflected != missing
            else:
                test_directions.add(direction)
                test_contacts.append(contacts)
    assert len(completion_colors) == 1
    assert all(completion not in role for completion in completion_colors for role in roles)
    assert len(train_roles) >= 2 and asymmetric_train >= 1
    # A stable missing direction is demonstrated, or multiple train examples
    # demonstrate that direction is read locally from the incomplete figure.
    if len(train_directions) == 1:
        assert test_directions == train_directions
    # The supported regimes are a source-like shift, shared contact class,
    # or free margins on both sides of the split.
    assert all(0 <= count <= 4 for count in train_contacts + test_contacts)


def _signature(puzzle):
    return tuple((section, pair["input"].shape, pair["input"].tobytes(),
                  pair["output"].tobytes())
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    train_counts, test_counts, shapes, directions, contacts = set(), set(), set(), set(), set()
    pivot_parities, roles, changes, layouts, signatures = set(), set(), set(), set(), set()
    mixed_direction = mixed_roles = rectangular = 0
    stable_direction = symmetric = disconnected = hollow = 0
    train_interior = train_full = test_interior = test_full = 0
    off_center = odd_square = stable_pivot = mixed_pivot = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        local_directions, local_roles, local_parities = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                color, direction, center, contact, role = _pair_features(pair)
                assert np.array_equal(out[inp != out], np.full(np.count_nonzero(inp != out), color))
                directions.add(direction)
                contacts.add(contact)
                local_directions.add(direction)
                local_roles.add(role)
                shapes.add(inp.shape)
                rectangular += inp.shape[0] != inp.shape[1]
                odd_square += inp.shape[0] == inp.shape[1] and inp.shape[0] % 2 == 1
                off_center += center != (inp.shape[0] - 1, inp.shape[1] - 1)
                pivot_parities.add(center[0] % 2)
                local_parities.add(center[0] % 2)
                roles.add((role, color))
                changes.add(int(np.count_nonzero(inp != out)))
                layouts.add((inp.shape, (inp == role[1]).tobytes()))
                missing = {tuple(int(x) for x in point)
                           for point in np.argwhere(inp != out)}
                reflected = ({(center[0] - r, c) for r, c in missing}
                             if direction in (0, 2) else
                             {(r, center[1] - c) for r, c in missing})
                symmetric += reflected == missing
                remaining = set(missing)
                components = 0
                while remaining:
                    components += 1
                    stack = [remaining.pop()]
                    while stack:
                        r, c = stack.pop()
                        for neighbor in ((r - 1, c), (r + 1, c),
                                         (r, c - 1), (r, c + 1)):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                disconnected += components > 1
                profile = {}
                for r, c in missing:
                    radial, transverse = ((2 * c - center[1], 2 * r - center[0])
                                          if direction in (0, 2) else
                                          (2 * r - center[0], 2 * c - center[1]))
                    profile.setdefault(radial, []).append(transverse)
                hollow += any(max(line) - min(line) > 2 * (len(line) - 1)
                              for line in profile.values())
                if section == "train":
                    train_interior += contact == 0
                    train_full += contact == 4
                else:
                    test_interior += contact == 0
                    test_full += contact == 4
        mixed_direction += len(local_directions) > 1
        stable_direction += len(local_directions) == 1
        mixed_roles += len(local_roles) > 1
        stable_pivot += len(local_parities) == 1
        mixed_pivot += len(local_parities) > 1
        if seed in (0, 7, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and rectangular > 20 and odd_square > 5
    assert off_center > 30
    assert directions == {0, 1, 2, 3} and contacts == {0, 1, 2, 3, 4}
    assert pivot_parities == {0, 1} and len(roles) >= 30
    assert len(changes) >= 12 and len(layouts) >= 100
    assert len(signatures) == 42 and mixed_direction >= 5 and stable_direction >= 5
    assert stable_pivot >= 5 and mixed_pivot >= 5
    assert mixed_roles >= 20 and symmetric >= 10 and disconnected >= 5 and hollow >= 5
    assert min(train_interior, train_full, test_interior, test_full) >= 5
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4),
                                             (None, 1), (None, 4),
                                             (2, None), (10, None))):
        puzzle = generate_puzzle(100 + index, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
        assert _signature(puzzle) == _signature(generate_puzzle(100 + index, ntrain, ntest))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4)):
        try:
            generate_puzzle(300, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
