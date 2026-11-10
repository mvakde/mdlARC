import numpy as np


_DIRECTIONS = ("right", "up", "left", "down")
_MOTIFS = ((-1, 1), (-2, 1), (-1, 2), (-2, 2))

_SOURCE_PAIRS = (
    (
        (14, 14),
        "97777777777777" "77777777777777" "00077777777777" "97000000000000"
        "00077777777777" "77777777777777" "99777777777777" "00077777777777"
        "77000000000000" "00077777777777" "77777777777777" "00077777777777"
        "97000000000000" "00077777777777",
        "97777777777777" "77777777777777" "77777777777000" "77777777777970"
        "77777777777000" "77777777777777" "99777777777777" "77777777777000"
        "77777777777770" "77777777777000" "77777777777777" "77777777777000"
        "77777777777970" "77777777777000",
    ),
    (
        (8, 8),
        "77077707" "77077707" "77077707" "77077707"
        "77077707" "70007000" "70707070" "70909070",
        "70007000" "70707070" "70907070" "77777777"
        "77777777" "77777777" "77777777" "77779777",
    ),
    (
        (16, 16),
        "7777777777777000" "0000000000000079" "7777777777777000"
        "7777777777777777" "7777777777777000" "0000000000000099"
        "7777777777777000" "7777777777777777" "7777777777777777"
        "7777777777777779" "7777777777777000" "0000000000000077"
        "7777777777777000" "7777777777777777" "7777777777777799"
        "7777777777777777",
        "0007777777777777" "0797777777777777" "0007777777777777"
        "7777777777777777" "0007777777777777" "0997777777777777"
        "0007777777777777" "7777777777777777" "7777777777777777"
        "7777777777777779" "0007777777777777" "0777777777777777"
        "0007777777777777" "7777777777777777" "7777777777777799"
        "7777777777777777",
    ),
    (
        (20, 20),
        "90709709070907909097" "70707707070907909097"
        "70007700070007700077" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777"
        "77077770777077770777" "77077770777077770777",
        "97779777777777977797" "77777777777777977797"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "77777777777777777777"
        "77777777777777777777" "70707709070907709077"
        "70707707070907709077" "70007700070007700077",
    ),
)


def _checked_count(value, name, low, high):
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


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _coord(direction, height, width, lane, along, offset):
    if direction == "right":
        return lane + offset, along
    if direction == "left":
        return lane + offset, width - 1 - along
    if direction == "down":
        return along, lane + offset
    return height - 1 - along, lane + offset


def _holder_cells(depth, low, high):
    cells = {(a, low) for a in range(depth + 1)}
    cells.update((a, high) for a in range(depth + 1))
    cells.update((depth, p) for p in range(low, high + 1))
    return cells


def _slot_cells(depth, low, high):
    return [
        (along, offset)
        for along in range(depth)
        for offset in range(low + 1, high)
    ]


def _sample_lanes(rng, motifs, size):
    widths = [high - low + 1 for low, high in motifs]
    minimum = sum(widths) + len(widths) - 1
    if size < minimum:
        raise ValueError("holders do not fit")
    extra = size - minimum
    parts = len(widths) + 1  # two margins and the inter-holder gaps
    weights = rng.dirichlet(np.ones(parts))
    if parts > 1 and rng.random() < 0.48:
        weights[int(rng.integers(parts))] = 0
    if parts > 2 and rng.random() < 0.42:
        weights[int(rng.integers(1, parts - 1))] = 0
    if not np.any(weights):
        weights[0] = 1
    additions = rng.multinomial(extra, weights / weights.sum())
    cursor = int(additions[0])
    lanes = []
    for i, ((low, _), width) in enumerate(zip(motifs, widths)):
        lanes.append(cursor - low)
        cursor += width
        if i + 1 < len(widths):
            cursor += 1 + int(additions[i + 1])
    return lanes


def _choose_size(rng, minimum, regime):
    lower = max(8, minimum)
    if lower > 30:
        raise ValueError("holders exceed ARC dimensions")
    # A tight arrangement is sampled separately from the full size range.
    if rng.random() < 0.32:
        return int(rng.integers(lower, min(30, lower + 3) + 1))
    if regime == "target_test" and rng.random() < 0.30:
        lower = max(lower, 19)
    return int(rng.integers(lower, 31))


def _new_colors(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _choose_motifs(rng, count, pool=None, force_standard=False):
    vocabulary = _MOTIFS if pool is None else tuple(sorted(pool))
    choices = []
    for i in range(count):
        if i == 0 and force_standard:
            choices.append((-1, 1))
        else:
            weights = np.array([0.52 if m == (-1, 1) else 0.16 for m in vocabulary])
            weights /= weights.sum()
            choices.append(vocabulary[int(rng.choice(len(vocabulary), p=weights))])
    rng.shuffle(choices)
    return choices


def _neighbor(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _generate_pair(rng, regime, directions, colors, motif_pool=None,
                   first_training=False):
    background, structural, accent = colors
    if isinstance(directions, str):
        directions = (directions,)
    if first_training:
        count = int(rng.integers(2, 4))
    else:
        count = int(rng.choice((1, 2, 3, 4), p=(.16, .37, .32, .15)))
    if len(directions) == 2:
        count = max(count, 2)
    motifs = _choose_motifs(rng, count, motif_pool, first_training)
    depths = [int(rng.choice((2, 3, 4), p=(.53, .30, .17))) for _ in motifs]
    axis_size = _choose_size(rng, 8, regime)
    perpendicular_size = _choose_size(
        rng, sum(high - low + 1 for low, high in motifs) + count - 1,
        regime,
    )
    horizontal = directions[0] in ("left", "right")
    height, width = ((perpendicular_size, axis_size) if horizontal
                     else (axis_size, perpendicular_size))
    lanes = _sample_lanes(rng, motifs, perpendicular_size)
    if len(directions) == 2:
        assigned = [directions[i % 2] for i in range(count)]
        rng.shuffle(assigned)
    else:
        assigned = [directions[0]] * count

    inp = np.full((height, width), background, dtype=np.int8)
    structure = set()
    original_slots = set()
    destination_holders = set()
    destination_slots = set()
    holders = []
    for lane, depth, motif, direction in zip(lanes, depths, motifs, assigned):
        low, high = motif
        local_holder = _holder_cells(depth, low, high)
        local_slots = _slot_cells(depth, low, high)
        delta = axis_size - 1 - depth
        holder = {_coord(direction, height, width, lane, a, o)
                  for a, o in local_holder}
        rail = {_coord(direction, height, width, lane, a, 0)
                for a in range(depth, axis_size)}
        slots = [(_coord(direction, height, width, lane, a, o),
                  _coord(direction, height, width, lane, a + delta, o))
                 for a, o in local_slots]
        final_holder = {_coord(direction, height, width, lane, a + delta, o)
                        for a, o in local_holder}
        structure.update(holder | rail)
        original_slots.update(first for first, _ in slots)
        destination_holders.update(final_holder)
        destination_slots.update(second for _, second in slots)
        holders.append((holder, slots, direction, depth, low, high))
    if len(structure) != sum(len(_holder_cells(d, l, h)) + axis_size - 1 - d
                             for _, _, _, d, l, h in holders):
        raise ValueError("overlapping rail structures")
    for cell in structure:
        inp[cell] = structural

    payload = set()
    destination_payload = set()
    empty_holders = []
    for index, (_, slots, _, _, _, _) in enumerate(holders):
        probability = float(rng.uniform(.12, .82))
        selected = [pair for pair in slots if rng.random() < probability]
        if first_training and index == 0:
            selected = []
        if first_training and index == 1 and not selected:
            selected = [slots[int(rng.integers(len(slots)))]]
        if not selected:
            empty_holders.append(index)
        for old, new in selected:
            payload.add(old)
            destination_payload.add(new)
            inp[old] = accent

    protected = structure | original_slots | destination_holders | destination_slots
    available = {(r, c) for r in range(height) for c in range(width)} - protected
    near_by_holder = []
    side_by_holder = []
    end_by_holder = []
    for (holder, _, direction, depth, low, high), lane in zip(holders, lanes):
        near = {p for cell in holder for p in _neighbor(cell) if p in available}
        side = {_coord(direction, height, width, lane, a, o)
                for a in range(depth + 1) for o in (low - 1, high + 1)} & available
        end = {_coord(direction, height, width, lane, depth + 1, o)
               for o in (low, high)} & available
        near_by_holder.append(near)
        side_by_holder.append(side)
        end_by_holder.append(end)
    all_near = set().union(*near_by_holder)
    chosen_distractors = set()
    max_count = min(2 + 2 * count, len(available), len(structure) - len(payload) - 1)
    total = int(rng.integers(1, max_count + 1))
    near_target = max(1, min(total, int(round(total * rng.uniform(.60, 1.0)))))
    def take(pool):
        options = tuple(sorted(pool - chosen_distractors))
        if options and len(chosen_distractors) < total:
            chosen_distractors.add(options[int(rng.integers(len(options)))])
    if empty_holders:
        take(near_by_holder[empty_holders[int(rng.integers(len(empty_holders)))]] )
    take(set().union(*side_by_holder))
    if rng.random() < .65:
        take(set().union(*end_by_holder))
    while len(chosen_distractors) < near_target:
        old = len(chosen_distractors)
        take(all_near)
        if len(chosen_distractors) == old:
            break
    while len(chosen_distractors) < total:
        take(available)
    for cell in chosen_distractors:
        inp[cell] = accent

    out = inp.copy()
    for cell in structure | payload:
        out[cell] = background
    for cell in destination_holders:
        out[cell] = structural
    for cell in destination_payload:
        out[cell] = accent
    counts = [int(np.count_nonzero(inp == color)) for color in colors]
    if not (counts[0] > counts[1] > counts[2] > 0):
        raise ValueError("color roles are not distinguishable")
    return inp, out


def _candidate_rails(inp, background, structural, accent):
    height, width = inp.shape
    actual_structure = {tuple(map(int, p)) for p in np.argwhere(inp == structural)}
    candidates = []
    for direction in _DIRECTIONS:
        axis_size = width if direction in ("left", "right") else height
        perpendicular_size = height if direction in ("left", "right") else width
        for lane in range(perpendicular_size):
            line = np.array([inp[_coord(direction, height, width, lane, a, 0)]
                             for a in range(axis_size)])
            if int(line[-1]) != structural:
                continue
            depth = axis_size - 1
            while depth > 0 and int(line[depth - 1]) == structural:
                depth -= 1
            if not 2 <= depth <= 4:
                continue
            def value(along, offset):
                r, c = _coord(direction, height, width, lane, along, offset)
                if not (0 <= r < height and 0 <= c < width):
                    return None
                return int(inp[r, c])
            low = 0
            while value(depth, low - 1) == structural:
                low -= 1
            high = 0
            while value(depth, high + 1) == structural:
                high += 1
            if (low, high) not in _MOTIFS:
                continue
            holder = {_coord(direction, height, width, lane, a, o)
                      for a, o in _holder_cells(depth, low, high)}
            rail = {_coord(direction, height, width, lane, a, 0)
                    for a in range(depth, axis_size)}
            body = holder | rail
            if not body <= actual_structure:
                continue
            slots = [_coord(direction, height, width, lane, a, o)
                     for a, o in _slot_cells(depth, low, high)]
            if any(int(inp[p]) not in (background, accent) for p in slots):
                continue
            payload = {p for p in slots if int(inp[p]) == accent}
            delta = axis_size - 1 - depth
            final_holder = {_coord(direction, height, width, lane, a + delta, o)
                            for a, o in _holder_cells(depth, low, high)}
            final_payload = {_coord(direction, height, width, lane, a + delta, o)
                             for a, o in _slot_cells(depth, low, high)
                             if int(inp[_coord(direction, height, width, lane, a, o)]) == accent}
            candidates.append((body, payload, final_holder, final_payload,
                               {"direction": direction, "lane": lane,
                                "depth": depth, "low": low, "high": high,
                                "payload_count": len(payload)}))
    return candidates, actual_structure


def _solve_input(inp, background, structural, accent):
    candidates, structure = _candidate_rails(inp, background, structural, accent)
    accents = {tuple(map(int, p)) for p in np.argwhere(inp == accent)}
    solutions = []
    def search(remaining, selected, used_payload, final_structure, final_payload):
        if not remaining:
            stationary = accents - used_payload
            if stationary & (final_structure | final_payload):
                return
            out = inp.copy()
            for p in structure | used_payload:
                out[p] = background
            for p in final_structure:
                out[p] = structural
            for p in final_payload:
                out[p] = accent
            solutions.append((out, [item[4] for item in selected],
                              len(used_payload), len(stationary), stationary))
            return
        cell = min(remaining)
        for item in candidates:
            body, payload, new_body, new_payload, _ = item
            if cell not in body or not body <= remaining:
                continue
            if (new_body & (final_structure | final_payload) or
                    new_payload & (final_structure | final_payload) or
                    payload & used_payload):
                continue
            search(remaining - body, selected + [item], used_payload | payload,
                   final_structure | new_body, final_payload | new_payload)
            if len(solutions) > 1:
                return
    search(structure, [], set(), set(), set())
    if len(solutions) != 1:
        raise AssertionError("rail and holder decomposition is not unique")
    return solutions[0]


def _analyze_pair(inp, out, background, structural, accent):
    predicted, holders, payload_count, distractor_count, stationary = _solve_input(
        inp, background, structural, accent)
    assert np.array_equal(predicted, out), "incorrect holder translation"
    return {"directions": tuple(h["direction"] for h in holders),
            "holder_count": len(holders), "holders": holders,
            "payload_count": payload_count, "distractor_count": distractor_count,
            "stationary": stationary}


def _grid_key(grid):
    return grid.shape, grid.dtype.str, grid.tobytes()


def _puzzle_key(puzzle):
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                                 for p in puzzle[section])) for section in ("train", "test"))


def _is_source_pair(inp, out):
    flat_in = "".join(str(int(x)) for x in inp.flat)
    flat_out = "".join(str(int(x)) for x in out.flat)
    return any(inp.shape == shape and flat_in == source_in and flat_out == source_out
               for shape, source_in, source_out in _SOURCE_PAIRS)


def _infer_roles(pair):
    values, counts = np.unique(pair["input"], return_counts=True)
    assert len(values) == 3
    order = np.argsort(-counts)
    roles = tuple(int(values[i]) for i in order)
    assert counts[order[0]] > counts[order[1]] > counts[order[2]] > 0
    return roles


def _predict_from_training(train_pairs, test_input):
    demonstrated_motifs = set()
    demonstrated_axes = set()
    mixed_demonstrated = False
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        info = _analyze_pair(inp, out, *_infer_roles({"input": inp}))
        demonstrated_motifs.update((h["low"], h["high"]) for h in info["holders"])
        demonstrated_axes.update("horizontal" if d in ("left", "right") else "vertical"
                                 for d in info["directions"])
        mixed_demonstrated |= len(set(info["directions"])) > 1
    assert demonstrated_axes == {"horizontal", "vertical"}
    predicted, holders, _, _, _ = _solve_input(
        test_input, *_infer_roles({"input": test_input}))
    assert {(h["low"], h["high"]) for h in holders} <= demonstrated_motifs
    assert len({h["direction"] for h in holders}) == 1 or mixed_demonstrated
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_info, test_info = [], []
    for section, infos in (("train", train_info), ("test", test_info)):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and inp.dtype == out.dtype
            key, pair_key = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(inp, out)
            assert not _is_source_pair(inp, out)
            roles = _infer_roles(pair)
            assert set(int(x) for x in np.unique(out)) == set(roles)
            info = _analyze_pair(inp, out, *roles)
            assert 1 <= info["holder_count"] <= 4
            assert info["distractor_count"] >= 1
            # Every pair teaches that a neighboring exterior accent stays put.
            structural = roles[1]
            holder_pixels = {tuple(map(int, p)) for p in np.argwhere(inp == structural)}
            adjacent = {p for p in info["stationary"]
                        if any(q in holder_pixels for q in _neighbor(p))}
            assert adjacent
            infos.append(info)
    train_motifs = {(h["low"], h["high"]) for info in train_info
                    for h in info["holders"]}
    assert (-1, 1) in train_motifs
    assert any(h["payload_count"] > 0 for info in train_info for h in info["holders"])
    assert any(h["payload_count"] == 0 for info in train_info for h in info["holders"])
    train_axes = {"horizontal" if d in ("left", "right") else "vertical"
                  for info in train_info for d in info["directions"]}
    assert train_axes == {"horizontal", "vertical"}
    trained_mixed = any(len(set(info["directions"])) > 1 for info in train_info)
    for info in test_info:
        assert {(h["low"], h["high"]) for h in info["holders"]} <= train_motifs
        if len(set(info["directions"])) > 1:
            assert trained_mixed


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    base_colors = _new_colors(rng)
    recolor = bool(rng.random() < .38)
    mixed = bool(rng.random() < .26)
    first_axis_horizontal = bool(rng.integers(2))
    first_direction = ("left", "right")[int(rng.integers(2))] if first_axis_horizontal else ("up", "down")[int(rng.integers(2))]
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
    other_axis = ("up", "down") if first_axis_horizontal else ("left", "right")
    train_directions = [(first_direction, opposite[first_direction]) if mixed else (first_direction,),
                        (other_axis[int(rng.integers(2))],)]
    for _ in range(2, num_train):
        direction = _DIRECTIONS[int(rng.integers(4))]
        if mixed and rng.random() < .28:
            train_directions.append((direction, opposite[direction]))
        else:
            train_directions.append((direction,))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, direction in enumerate(train_directions):
        colors = base_colors
        if recolor and index and (index == 1 or rng.random() < .78):
            colors = _new_colors(rng)
            if colors == base_colors:
                colors = (colors[0], colors[2], colors[1])
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            try:
                inp, out = _generate_pair(pair_rng, "source_train", direction, colors,
                                          first_training=(index == 0))
            except ValueError:
                continue
            if _grid_key(inp) in seen or _is_source_pair(inp, out):
                continue
            try:
                _analyze_pair(inp, out, *(_infer_roles({"input": inp})))
            except AssertionError:
                continue
            seen.add(_grid_key(inp))
            puzzle["train"].append({"input": inp, "output": out})
            break
        else:
            raise ValueError("could not generate a distinct training pair")
    train_motifs = {(h["low"], h["high"]) for p in puzzle["train"]
                    for h in _analyze_pair(p["input"], p["output"], *_infer_roles(p))["holders"]}
    demonstrated = [d for direction in train_directions for d in direction]
    for index in range(num_test):
        if rng.random() < .55:
            direction = demonstrated[int(rng.integers(len(demonstrated)))]
        else:
            direction = _DIRECTIONS[int(rng.integers(4))]
        if mixed and rng.random() < .25:
            directions = (direction, opposite[direction])
        else:
            directions = (direction,)
        colors = base_colors
        if recolor and rng.random() < .78:
            colors = _new_colors(rng)
            if colors == base_colors:
                colors = (colors[0], colors[2], colors[1])
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            try:
                inp, out = _generate_pair(pair_rng, "target_test", directions, colors,
                                          motif_pool=train_motifs)
            except ValueError:
                continue
            if _grid_key(inp) in seen or _is_source_pair(inp, out):
                continue
            try:
                _analyze_pair(inp, out, *_infer_roles({"input": inp}))
            except AssertionError:
                continue
            seen.add(_grid_key(inp))
            puzzle["test"].append({"input": inp, "output": out})
            break
        else:
            raise ValueError("could not generate a distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        training_copies = [{"input": pair["input"].copy(),
                            "output": pair["output"].copy()}
                           for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            prediction = _predict_from_training(training_copies, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(1000 + index, ntrain, ntest)
        assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for index, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1100 + index, ntrain, ntest)
        assert (ntrain is None or len(p["train"]) == ntrain)
        assert (ntest is None or len(p["test"]) == ntest)
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1200, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _puzzle_key(generate_puzzle(424242, 7, 3)) == _puzzle_key(generate_puzzle(424242, 7, 3))
    assert _puzzle_key(generate_puzzle(424242, 7, 3)) != _puzzle_key(generate_puzzle(424243, 7, 3))
    signatures, palettes, dimensions, layouts = set(), set(), set(), set()
    train_directions, test_directions = set(), set()
    train_counts, test_counts = set(), set()
    train_motifs, test_motifs = set(), set()
    train_empty = test_empty = near_test = 0
    total = 0
    for p in automatic:
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        for section in ("train", "test"):
            for pair in p[section]:
                total += 1
                inp = pair["input"]
                dimensions.add(inp.shape)
                palettes.add(_infer_roles(pair))
                info = _analyze_pair(inp, pair["output"], *_infer_roles(pair))
                layout = (inp.shape, tuple(sorted((h["direction"], h["lane"], h["depth"], h["low"], h["high"])
                                                  for h in info["holders"])))
                layouts.add(layout)
                (train_counts if section == "train" else test_counts).add(info["holder_count"])
                for h in info["holders"]:
                    (train_directions if section == "train" else test_directions).add(h["direction"])
                    (train_motifs if section == "train" else test_motifs).add((h["low"], h["high"]))
                    if h["payload_count"] == 0:
                        if section == "train": train_empty += 1
                        else: test_empty += 1
                if section == "test":
                    structural = _infer_roles(pair)[1]
                    cells = {tuple(map(int, x)) for x in np.argwhere(inp == structural)}
                    near_test += sum(any(q in cells for q in _neighbor(x)) for x in info["stationary"])
    assert len(signatures) >= 40 and len(palettes) >= 20
    assert len(dimensions) >= 30 and len(layouts) >= int(.90 * total)
    assert train_directions == test_directions == set(_DIRECTIONS)
    assert train_counts == test_counts == {1, 2, 3, 4}
    assert train_motifs == test_motifs == set(_MOTIFS)
    assert train_empty > 0 and test_empty > 0 and near_test > 50

    measured = {name: 0 for name in (
        "down_train", "shared_test_direction", "simple_test", "smaller_test",
        "source_like_test", "wide_both", "single_palette", "recolored",
        "mixed_train", "mixed_test", "tight_gap", "edge_holder",
        "side_accent", "end_accent", "small_train", "small_test",
        "large_train", "large_test", "empty_test", "dense_test",
        "combined_test", "varied_within_puzzle",
    )}
    travel_lengths = {count: set() for count in range(1, 5)}
    for puzzle in automatic:
        all_pairs = puzzle["train"] + puzzle["test"]
        roles = [_infer_roles(pair) for pair in all_pairs]
        measured["recolored" if len(set(roles)) > 1 else "single_palette"] += 1
        train = [(_analyze_pair(pair["input"], pair["output"], *role), pair)
                 for pair, role in zip(puzzle["train"], roles[:len(puzzle["train"])])]
        test = [(_analyze_pair(pair["input"], pair["output"], *role), pair)
                for pair, role in zip(puzzle["test"], roles[len(puzzle["train"]):])]
        if len({pair["input"].shape for pair in all_pairs}) > 1:
            measured["varied_within_puzzle"] += 1
        if any("down" in info["directions"] for info, _ in train):
            measured["down_train"] += 1
        if any(set(t["directions"]) & set(s["directions"])
               for t, _ in test for s, _ in train):
            measured["shared_test_direction"] += 1
        if any(t["holder_count"] < max(s["holder_count"] for s, _ in train)
               for t, _ in test):
            measured["simple_test"] += 1
        if any(max(tp["input"].shape) < max(sp["input"].shape)
               for _, tp in test for _, sp in train):
            measured["smaller_test"] += 1
        if any(all((h["low"], h["high"]) == (-1, 1) for h in t["holders"])
               for t, _ in test):
            measured["source_like_test"] += 1
        if (any(any(h["low"] < -1 or h["high"] > 1 for h in s["holders"])
                for s, _ in train) and
            any(any(h["low"] < -1 or h["high"] > 1 for h in t["holders"])
                for t, _ in test)):
            measured["wide_both"] += 1
        for section, records in (("train", train), ("test", test)):
            for info, pair in records:
                inp = pair["input"]
                height, width = inp.shape
                axis_size = width if info["directions"][0] in ("left", "right") else height
                perpendicular_size = height if info["directions"][0] in ("left", "right") else width
                if len(set(info["directions"])) > 1:
                    measured["mixed_" + section] += 1
                if min(inp.shape) <= 9:
                    measured["small_" + section] += 1
                if max(inp.shape) >= 25:
                    measured["large_" + section] += 1
                intervals = sorted((h["lane"] + h["low"], h["lane"] + h["high"])
                                   for h in info["holders"])
                if any(b[0] - a[1] == 2 for a, b in zip(intervals, intervals[1:])):
                    measured["tight_gap"] += 1
                if intervals[0][0] == 0 or intervals[-1][1] == perpendicular_size - 1:
                    measured["edge_holder"] += 1
                stationary = info["stationary"]
                near_holder = set()
                for holder in info["holders"]:
                    direction = holder["direction"]
                    lane = holder["lane"]
                    depth = holder["depth"]
                    low, high = holder["low"], holder["high"]
                    travel_lengths[info["holder_count"]].add(axis_size)
                    holder_cells = {_coord(direction, height, width, lane, a, o)
                                    for a, o in _holder_cells(depth, low, high)}
                    near_holder |= {p for p in stationary
                                    if any(q in holder_cells for q in _neighbor(p))}
                    side = {_coord(direction, height, width, lane, a, o)
                            for a in range(depth + 1) for o in (low - 1, high + 1)}
                    end = {_coord(direction, height, width, lane, depth + 1, o)
                           for o in (low, high)}
                    if stationary & side:
                        measured["side_accent"] += 1
                    if stationary & end:
                        measured["end_accent"] += 1
                    if section == "test":
                        if holder["payload_count"] == 0:
                            measured["empty_test"] += 1
                        if holder["payload_count"] >= 3:
                            measured["dense_test"] += 1
                assert near_holder
                if (section == "test" and min(inp.shape) <= 12 and
                    any(h["low"] < -1 or h["high"] > 1 for h in info["holders"]) and
                    any(h["payload_count"] == 0 for h in info["holders"])):
                    measured["combined_test"] += 1
    assert measured["down_train"] >= 10
    assert measured["shared_test_direction"] >= 10
    assert measured["simple_test"] >= 10 and measured["smaller_test"] >= 10
    assert measured["source_like_test"] >= 10 and measured["wide_both"] >= 10
    assert measured["single_palette"] >= 8 and measured["recolored"] >= 8
    assert measured["mixed_train"] >= 5 and measured["mixed_test"] >= 1
    assert measured["tight_gap"] >= 50 and measured["edge_holder"] >= 50
    assert measured["side_accent"] >= 50 and measured["end_accent"] >= 20
    assert min(measured["small_train"], measured["small_test"]) >= 5
    assert min(measured["large_train"], measured["large_test"]) >= 20
    assert measured["empty_test"] >= 10 and measured["dense_test"] >= 10
    assert measured["combined_test"] >= 1
    assert measured["varied_within_puzzle"] >= 30
    assert all(len(lengths) >= 10 for lengths in travel_lengths.values())
