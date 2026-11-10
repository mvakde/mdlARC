"""Generate lower-edge ARC beams with minimum rightward obstacle detours."""

import numpy as np


def _validate_count(name, value, low, high):
    if value is not None and (isinstance(value, bool) or
                              not isinstance(value, (int, np.integer)) or
                              not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _resolve_counts(rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _runs(mask, row):
    columns = np.flatnonzero(mask[row])
    if not len(columns):
        return ()
    breaks = np.flatnonzero(np.diff(columns) > 1) + 1
    return tuple((int(part[0]), int(part[-1]) + 1)
                 for part in np.split(columns, breaks))


def _paint(grid, row, left, right, background, beam):
    for col in range(max(0, left), min(grid.shape[1], right)):
        if grid[row, col] == background:
            grid[row, col] = beam


def _solve_grid(input_grid, background, beam):
    """Advance the two-cell front using actual overlap, not run width."""
    output = input_grid.copy()
    height, width = input_grid.shape
    emitter = int(np.flatnonzero(input_grid[-1] == beam)[0])
    obstacles = (input_grid != background) & (input_grid != beam)
    row, left = height - 2, emitter
    _paint(output, row, left, left + 2, background, beam)
    fronts = [(row, left, min(2, width - left))]
    collisions = []
    while row > 0:
        next_row, proposed = row - 1, left + 1
        if proposed >= width:
            break
        candidate = proposed
        while candidate < width and np.any(
            obstacles[next_row, candidate:min(width, candidate + 2)]
        ):
            candidate += 1
        if candidate >= width:
            break
        if candidate != proposed:
            touched = [run for run in _runs(obstacles, next_row)
                       if run[0] < proposed + 2 and run[1] > proposed]
            if not touched:
                raise AssertionError("detour has no contacted obstacle")
            start, end = touched[0]
            collisions.append((next_row, proposed, candidate, start, end))
            _paint(output, row, left, candidate + 1, background, beam)
        _paint(output, next_row, candidate, candidate + 2, background, beam)
        row, left = next_row, candidate
        fronts.append((row, left, min(2, width - left)))
    return output, {"emitter": emitter, "fronts": tuple(fronts),
                    "collisions": tuple(collisions),
                    "clipped": any(count == 1 for _, _, count in fronts),
                    "top": fronts[-1][0] == 0}


def _contact_options(max_width):
    return [(length, offset) for length in range(1, max_width + 1)
            for offset in range(1 - length, 2)]


def _random_contacts(rng, count, max_width):
    contacts = []
    for _ in range(count):
        min_height = max(9, 2 * count + 6)
        remaining = count - len(contacts) - 1
        options = [item for item in _contact_options(max_width)
                   if sum(w + o for w, o in contacts) + sum(item)
                   <= 28 - min_height - remaining]
        contacts.append(options[int(rng.integers(len(options)))])
    return tuple(contacts)


def _add_offpath(grid, background, beam, obstacle, count, rng):
    """Place runs that leave the output beam mask exactly unchanged."""
    added = 0
    for _ in range(count):
        baseline, trace = _solve_grid(grid, background, beam)
        beam_mask = baseline == beam
        height, width = grid.shape
        collision_rows = {r for r, *_ in trace["collisions"]}
        candidates = []
        for row in range(1, height - 1):
            beam_cols = np.flatnonzero(beam_mask[row])
            old = grid[row] == obstacle
            for length in (1, 2, 3):
                for start in range(width - length + 1):
                    stop = start + length
                    if np.any(beam_mask[row, start:stop]):
                        continue
                    if np.any(old[max(0, start - 1):min(width, stop + 1)]):
                        continue
                    if not np.all(grid[row, start:stop] == background):
                        continue
                    trial = grid.copy()
                    trial[row, start:stop] = obstacle
                    changed, _ = _solve_grid(trial, background, beam)
                    if not np.array_equal(changed == beam, beam_mask):
                        continue
                    distance = min((min(abs(start - c), abs(stop - 1 - c))
                                    for c in beam_cols), default=width)
                    weight = 1 + (3 if distance <= 3 else 0)
                    weight += 4 if row in collision_rows else 0
                    candidates.append((row, start, stop, weight))
        if not candidates:
            break
        weights = np.array([c[3] for c in candidates], dtype=float)
        index = int(rng.choice(len(candidates), p=weights / weights.sum()))
        row, start, stop, _ = candidates[index]
        grid[row, start:stop] = obstacle
        added += 1
    return added


def generate_pair(rng, regime, puzzle_parameters, contacts, offpath_count=0,
                  end_mode=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    if any(length not in (1, 2, 3) or offset not in range(1 - length, 2)
           for length, offset in contacts):
        raise ValueError("invalid contact")
    count = len(contacts)
    shifts = sum(length + offset for length, offset in contacts)
    min_height = max(9, 2 * count + 6)
    max_height = min(25, 28 - shifts)
    if min_height > max_height:
        raise ValueError("collision plan cannot fit")
    height = int(rng.integers(min_height, max_height + 1))
    emitter_limit = min(11, 28 - (height - 2) - shifts)
    emitter = int(rng.integers(0, emitter_limit + 1))
    possible_steps = np.arange(2, height - 4)
    steps = sorted(int(s) for s in rng.choice(possible_steps, size=count,
                                               replace=False)) if count else []
    by_step = dict(zip(steps, contacts))
    lefts = [emitter]
    for step in range(1, height - 1):
        length, offset = by_step.get(step, (0, 0))
        lefts.append(lefts[-1] + 1 + length + offset)
    if end_mode is None:
        end_mode = "top" if rng.random() < 0.48 else "clip"
    clip_steps = [step for step in range(max(steps, default=0) + 2,
                                        height - 2)
                  if lefts[step] + 1 >= 5]
    if end_mode == "clip" and not clip_steps:
        end_mode = "top"
    if end_mode == "top":
        width = int(rng.integers(lefts[-1] + 2,
                                 min(30, lefts[-1] + 6) + 1))
    elif end_mode == "clip":
        final_step = clip_steps[int(rng.integers(len(clip_steps)))]
        width = lefts[final_step] + 1
    else:
        raise ValueError("unknown end mode")
    colors = rng.choice(np.arange(10), size=3, replace=False)
    background, beam, obstacle = (int(c) for c in colors)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[-1, emitter] = beam
    for step, (length, offset) in by_step.items():
        row = height - 2 - step
        proposed = lefts[step - 1] + 1
        start = proposed + offset
        grid[row, start:start + length] = obstacle
    _add_offpath(grid, background, beam, obstacle, offpath_count, rng)
    output, trace = _solve_grid(grid, background, beam)
    actual = tuple((end - start, start - proposed)
                   for _, proposed, _, start, end in trace["collisions"])
    if actual != contacts:
        raise AssertionError("constructed contacts changed")
    if end_mode == "clip" and not trace["clipped"]:
        raise AssertionError("expected a clipped final front")
    if end_mode == "top" and not trace["top"]:
        raise AssertionError("expected the beam to reach the top")
    return grid, output


def _grid_key(grid):
    canonical = np.asarray(grid, dtype=np.int16)
    return canonical.shape, canonical.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[section])
                 for section in ("train", "test"))


def _is_source_input(grid, background, beam, obstacle):
    if grid.shape != (10, 10) or background != 0 or beam != 2:
        return False
    emitter = int(np.flatnonzero(grid[-1] == beam)[0])
    coordinates = () if obstacle is None else tuple(map(tuple, np.argwhere(grid == obstacle)))
    originals = {(1, ((6, 3),)), (1, ()),
                 (0, ((5, 4), (5, 5), (7, 1))), (3, ()),
                 (1, ((4, 7), (6, 3), (6, 4)))}
    return (emitter, coordinates) in originals


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape
    height, width = grid.shape
    assert height >= 7 and width >= 4
    background = int(grid[0, 0])
    assert np.all(grid[0] == background)
    emitter_columns = np.flatnonzero(grid[-1] != background)
    assert len(emitter_columns) == 1
    emitter = int(emitter_columns[0])
    beam = int(grid[-1, emitter])
    assert np.count_nonzero(grid == beam) == 1
    others = set(int(v) for v in np.unique(grid)) - {background, beam}
    assert len(others) <= 1
    obstacle = next(iter(others), None)
    obstacle_mask = (grid != background) & (grid != beam)
    assert not np.any(obstacle_mask[0]) and not np.any(obstacle_mask[-1])
    runs = [(row, start, stop) for row in range(1, height - 1)
            for start, stop in _runs(obstacle_mask, row)]
    assert all(1 <= stop - start <= 3 for _, start, stop in runs)
    assert not _is_source_input(grid, background, beam, obstacle)
    expected, trace = _solve_grid(grid, background, beam)
    assert np.array_equal(output, expected)
    assert not np.array_equal(grid, output)
    assert len(trace["fronts"]) >= 3
    assert np.array_equal(output[obstacle_mask], grid[obstacle_mask])
    assert np.all(output[~obstacle_mask & (grid != beam)] != obstacle)
    assert np.array_equal(output[-1], grid[-1])
    contacts, contacted_runs = [], set()
    for row, proposed, shifted, start, end in trace["collisions"]:
        assert (row, start, end) in runs and shifted > proposed
        assert start < proposed + 2 and end > proposed
        assert not np.any(obstacle_mask[row, shifted:min(width, shifted + 2)])
        contacts.append((end - start, start - proposed, shifted - proposed))
        contacted_runs.add((row, start, end))
    return {"trace": trace, "contacts": tuple(contacts),
            "runs": tuple(runs), "offpath": tuple(set(runs) - contacted_runs),
            "colors": (background, beam, obstacle), "emitter": emitter}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[section].append(_inspect_pair(pair))
    train = infos["train"]
    assert any(len(info["contacts"]) == 0 for info in train)
    assert any(any(offset == 0 and shift == 1
                   for _, offset, shift in info["contacts"]) for info in train)
    assert any(any(offset == 1 for _, offset, _ in info["contacts"])
               for info in train)
    max_train_width = max((length for info in train
                           for length, _, _ in info["contacts"]), default=0)
    for info in infos["test"]:
        assert all(length <= max_train_width
                   for length, _, _ in info["contacts"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    puzzle_parameters = {"direction": (-1, 1), "front_width": 2}
    train_plans = [(), ((1, 0), (1, 1))]
    if train_count >= 3:
        train_plans.append(((2, 0),))
    if train_count >= 5:
        train_plans.append(((3, -1), (2, 1)))
    max_width = 3 if train_count >= 5 else (2 if train_count >= 3 else 1)
    while len(train_plans) < train_count:
        n = int(rng.integers(0, 5))
        train_plans.append(_random_contacts(rng, n, max_width))
    rng.shuffle(train_plans)
    test_plans = []
    for _ in range(test_count):
        n = int(rng.choice(5, p=[0.09, 0.19, 0.37, 0.25, 0.10]))
        test_plans.append(_random_contacts(rng, n, max_width))
    puzzle, seen = {"train": [], "test": []}, set()
    for section, plans in (("train", train_plans), ("test", test_plans)):
        for index, contacts in enumerate(plans):
            offpath = int(rng.choice(3, p=[0.2, 0.55, 0.25]))
            if section == "train" and index == 0:
                offpath = max(1, offpath)
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                end_mode = "clip" if rng.random() < 0.5 else "top"
                grid, output = generate_pair(
                    pair_rng, "source_train" if section == "train" else "target_test",
                    puzzle_parameters, contacts, offpath, end_mode)
                background = int(grid[0, 0])
                beam = int(grid[-1, np.flatnonzero(grid[-1] != background)[0]])
                remaining_colors = set(int(value) for value in np.unique(grid))
                remaining_colors.difference_update((background, beam))
                obstacle = next(iter(remaining_colors), None)
                if _is_source_input(grid, background, beam, obstacle):
                    continue
                key = _grid_key(grid)
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input": grid, "output": output})
                    break
            else:
                raise AssertionError("unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for index, counts in enumerate(((None, 1), (None, 4),
                                    (2, None), (10, None))):
        generate_puzzle(2000 + index, *counts)
    assert _puzzle_key(generate_puzzle(314159, 8, 3)) == _puzzle_key(
        generate_puzzle(314159, 8, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) >= 40
    shapes, layouts, emitters, contact_counts, contact_types = (
        set(), set(), set(), set(), set())
    color_sets, clipped, top, offpath, same_row = set(), 0, 0, 0, 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        local_colors = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _inspect_pair(pair)
                grid = pair["input"]
                shapes.add(grid.shape)
                emitters.add(info["emitter"])
                contact_counts.add(len(info["contacts"]))
                contact_types.update(info["contacts"])
                local_colors.add(info["colors"])
                color_sets.add(info["colors"])
                layouts.add((grid.shape, info["emitter"], info["runs"]))
                clipped += int(info["trace"]["clipped"])
                top += int(info["trace"]["top"])
                offpath += len(info["offpath"])
                collision_rows = {row for row, *_ in info["trace"]["collisions"]}
                same_row += sum(row in collision_rows for row, *_ in info["offpath"])
        assert len(local_colors) >= 2
    assert len(shapes) >= 40 and len(layouts) >= 100
    assert len(emitters) >= 8 and len(contact_counts) >= 5
    assert len(contact_types) >= 6 and len(color_sets) >= 50
    assert clipped >= 30 and top >= 30
    assert offpath >= 60 and same_row >= 5
