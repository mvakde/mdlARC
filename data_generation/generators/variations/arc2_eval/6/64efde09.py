"""Asymmetric emitter copies share an input-specified boundary ray code.

Geometry/pooling is the puzzle convention. Palettes, templates and codes are
local to each pair. Construction uses hidden geometry; validation reconstructs
correspondence and the complete answer from the returned input alone.
"""

import numpy as np

_DIRECTIONS = {(-1, 0), (1, 0), (0, -1), (0, 1)}


def _orient(array, transform):
    base = np.fliplr(array) if transform >= 4 else array
    return np.rot90(base, transform % 4).copy()


def _grid_key(array):
    return array.shape, array.astype(np.int8).tobytes()


_SOURCE_INPUTS = (((29, 29), ((0, 4, 1), (0, 5, 1), (1, 0, 9), (1, 4, 4), (1, 5, 1), (2, 4, 4), (2, 5, 2), (2, 18, 3), (2, 19, 2), (2, 20, 2), (2, 21, 2), (2, 22, 2), (2, 23, 1), (2, 24, 1), (3, 4, 4), (3, 5, 2), (3, 18, 3), (3, 19, 4), (3, 20, 4), (3, 21, 4), (3, 22, 4), (3, 23, 4), (3, 24, 1), (4, 4, 4), (4, 5, 2), (5, 4, 4), (5, 5, 2), (6, 4, 3), (6, 5, 3), (10, 24, 3), (10, 25, 3), (11, 24, 2), (11, 25, 4), (11, 28, 7), (12, 24, 2), (12, 25, 4), (13, 24, 2), (13, 25, 4), (14, 24, 2), (14, 25, 4), (15, 24, 1), (15, 25, 4), (16, 24, 1), (16, 25, 1), (22, 9, 1), (22, 10, 1), (23, 9, 4), (23, 10, 1), (24, 9, 4), (24, 10, 2), (25, 9, 4), (25, 10, 2), (26, 9, 4), (26, 10, 2), (27, 9, 4), (27, 10, 2), (28, 9, 3), (28, 10, 3), (28, 21, 6))), ((26, 29), ((0, 18, 4), (0, 19, 4), (1, 18, 2), (1, 19, 4), (2, 18, 2), (2, 19, 3), (3, 18, 2), (3, 19, 3), (4, 18, 2), (4, 19, 3), (5, 18, 2), (5, 19, 3), (6, 18, 2), (6, 19, 4), (7, 1, 4), (7, 2, 3), (7, 3, 3), (7, 4, 3), (7, 5, 3), (7, 6, 4), (7, 7, 4), (8, 1, 2), (8, 2, 2), (8, 3, 2), (8, 4, 2), (8, 5, 2), (8, 6, 2), (8, 7, 4), (15, 12, 4), (15, 13, 3), (15, 14, 3), (15, 15, 3), (15, 16, 3), (15, 17, 4), (15, 18, 4), (16, 12, 2), (16, 13, 2), (16, 14, 2), (16, 15, 2), (16, 16, 2), (16, 17, 2), (16, 18, 4), (18, 22, 4), (18, 23, 4), (19, 22, 4), (19, 23, 2), (19, 28, 6), (20, 22, 3), (20, 23, 2), (21, 22, 3), (21, 23, 2), (22, 22, 3), (22, 23, 2), (23, 22, 3), (23, 23, 2), (24, 22, 4), (24, 23, 2), (25, 3, 5), (25, 13, 1))), ((21, 27), ((0, 5, 1), (0, 6, 4), (0, 19, 6), (1, 0, 3), (1, 5, 1), (1, 6, 4), (2, 5, 1), (2, 6, 2), (2, 14, 1), (2, 15, 1), (2, 16, 1), (2, 17, 1), (2, 18, 1), (2, 19, 1), (2, 20, 1), (2, 21, 1), (2, 22, 1), (3, 5, 1), (3, 6, 2), (3, 14, 4), (3, 15, 2), (3, 16, 2), (3, 17, 2), (3, 18, 2), (3, 19, 2), (3, 20, 2), (3, 21, 4), (3, 22, 4), (4, 5, 1), (4, 6, 2), (5, 0, 9), (5, 5, 1), (5, 6, 2), (6, 5, 1), (6, 6, 2), (7, 5, 1), (7, 6, 2), (8, 5, 1), (8, 6, 4), (9, 16, 4), (9, 17, 1), (10, 16, 4), (10, 17, 1), (11, 16, 2), (11, 17, 1), (12, 16, 2), (12, 17, 1), (13, 16, 2), (13, 17, 1), (14, 16, 2), (14, 17, 1), (15, 16, 2), (15, 17, 1), (16, 16, 2), (16, 17, 1), (17, 16, 4), (17, 17, 1))))


def _cell_for_label(labels, label):
    positions = np.argwhere(labels == label)
    assert positions.shape == (1, 2)
    return int(positions[0, 0]), int(positions[0, 1])


def _ray_for_candidate(
    candidate,
    side,
    slot,
    length,
    height,
    width,
):
    labels = candidate["labels"]

    active = _cell_for_label(labels, side * length + slot)
    inner = _cell_for_label(labels, (1 - side) * length + slot)

    dr = active[0] - inner[0]
    dc = active[1] - inner[1]
    assert abs(dr) + abs(dc) == 1

    row = candidate["r"] + active[0] + dr
    col = candidate["c"] + active[1] + dc

    ray = []
    while 0 <= row < height and 0 <= col < width:
        ray.append((row, col))
        row += dr
        col += dc

    return tuple(ray), (dr, dc)


def _make_candidate(
    template,
    transform,
    row,
    col,
    active_side,
    slots,
    height,
    width,
):
    array = _orient(template, transform)
    labels = _orient(
        np.arange(template.size).reshape(template.shape),
        transform,
    )

    cells = frozenset(
        (row + local_row, col + local_col)
        for local_row in range(array.shape[0])
        for local_col in range(array.shape[1])
    )

    candidate = {
        "arr": array,
        "labels": labels,
        "r": row,
        "c": col,
        "cells": cells,
        "transform": transform,
    }

    rays = {}
    direction = None

    for slot in slots:
        ray, current_direction = _ray_for_candidate(
            candidate,
            active_side,
            slot,
            template.shape[1],
            height,
            width,
        )
        rays[slot] = ray
        if direction is None:
            direction = current_direction
        assert direction == current_direction

    candidate["rays"] = rays
    candidate["ray_cells"] = frozenset(
        position
        for ray in rays.values()
        for position in ray
    )
    candidate["direction"] = direction
    return candidate


def _compatible(candidate, placed):
    for previous in placed:
        if candidate["cells"] & previous["cells"]:
            return False
        if candidate["cells"] & previous["ray_cells"]:
            return False
        if candidate["ray_cells"] & previous["cells"]:
            return False
        if candidate["ray_cells"] & previous["ray_cells"]:
            return False

        for row, col in candidate["cells"]:
            if (
                (row - 1, col) in previous["cells"]
                or (row + 1, col) in previous["cells"]
                or (row, col - 1) in previous["cells"]
                or (row, col + 1) in previous["cells"]
            ):
                return False

    return True


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if seen[row, col]:
                continue
            if int(grid[row, col]) == background:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for dr, dc in _DIRECTIONS:
                    next_row = current_row + dr
                    next_col = current_col + dc

                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if seen[next_row, next_col]:
                        continue
                    if (
                        int(grid[next_row, next_col])
                        == background
                    ):
                        continue

                    seen[next_row, next_col] = True
                    stack.append((next_row, next_col))

            components.append(component)

    return components


def _check_count(name, value, low, high):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _same_puzzle(first, second):
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


def _make_template(rng, length, colors):
    """Mix noisy asymmetric arrays with intentionally readable source-like bars."""
    for attempt in range(64):
        style = int(rng.integers(5))
        a = rng.choice(colors, size=(2, length)).astype(np.int8)
        if style == 1:
            a[0] = colors[0]  # A monochrome long row, as in source train 2.
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[-1]
        elif style == 2:
            a[0] = colors[0]
            a[1] = colors[1]
            a[:, 0] = colors[-1]
            a[0, -2:] = colors[1]
        elif style == 3:
            for row in range(2):
                a[row] = colors[row]
                cuts = sorted(map(int, rng.choice(length, size=2, replace=False)))
                a[row, cuts[0]:cuts[1]+1] = colors[-1-row % 2]
        elif style == 4:
            a[:] = colors[0]
            a[1, 1:] = colors[1]
            a[0, -1] = colors[-1]
        # Add missing colors without destroying the monochrome row.
        positions = list(map(int, rng.permutation(length)))
        for color in colors:
            if color not in a:
                a[1, positions.pop()] = color
        if len({_grid_key(_orient(a, t)) for t in range(8)}) == 8:
            if set(map(int, a.flat)) == set(colors):
                return a
    # Deterministic, asymmetric construction for every supported palette/length.
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    for index, color in enumerate(colors[2:], 2):
        a[1, index] = color
    return a


def _sample_slots(rng, length, count):
    mode = int(rng.integers(4))
    if mode == 0:
        start = int(rng.integers(length-count+1))
        return tuple(range(start, start+count))
    if mode == 1 and count > 1:
        step = int(rng.integers(1, (length-1)//(count-1)+1))
        start = int(rng.integers(length-step*(count-1)))
        return tuple(start+step*i for i in range(count))
    slots = set(map(int, rng.choice(length, count, replace=False)))
    if mode == 2:
        slots.remove(min(slots))
        slots.add(0)
    return tuple(sorted(slots))


def _sample_transforms(rng, count, demonstration):
    base = int(rng.integers(8))
    if demonstration == 'pool':
        # Same direction, opposite longitudinal order. Extra copies may rotate.
        ts = [base, base ^ 4] + list(map(int, rng.integers(8, size=count-2)))
    elif demonstration == 'rotate':
        ts = [base, (base//4)*4 + (base+1)%4]
        ts += list(map(int, rng.integers(8, size=count-2)))
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            ts = [base]*count
        elif mode == 1:
            ts = [base if i % 2 == 0 else (base//4)*4+(base+2)%4
                  for i in range(count)]
        elif mode == 2:
            ts = [base if i % 2 == 0 else base ^ 4 for i in range(count)]
        else:
            ts = list(map(int, rng.integers(8, size=count)))
    rng.shuffle(ts)
    return ts


def _choose_clues(rng, placed, side, slots, height, width, mode):
    length = max(placed[0]['arr'].shape)
    occupied = set().union(*(p['cells'] for p in placed))
    alignments = {}
    for index, emitter in enumerate(placed):
        for local_side in (0, 1):
            for slot in range(length):
                ray, _ = _ray_for_candidate(emitter, local_side, slot,
                                             length, height, width)
                if ray and not occupied.intersection(ray):
                    alignments.setdefault(ray[-1], []).append((index, local_side, slot))
    choices = {}
    for slot in slots:
        choices[slot] = []
        for index, emitter in enumerate(placed):
            endpoint = emitter['rays'][slot][-1]
            if alignments.get(endpoint) != [(index, side, slot)]:
                continue
            if any(max(abs(endpoint[0]-r), abs(endpoint[1]-c)) <= 1
                   for r, c in occupied):
                continue
            choices[slot].append((slot, index, endpoint))
        rng.shuffle(choices[slot])
        if not choices[slot]:
            return None

    preferred = list(map(int, rng.permutation(len(placed))))
    if mode == 'full':
        groups = [[index] for index in preferred]
    elif mode == 'two':
        groups = [preferred[:2], preferred]
    else:
        groups = [preferred]
    nodes = 0
    for group in groups:
        answer = []

        def search(index):
            nonlocal nodes
            nodes += 1
            if nodes > 300:
                return False
            if index == len(slots):
                return mode not in ('split', 'two') or len({x[1] for x in answer}) >= 2
            for entry in choices[slots[index]]:
                if entry[1] not in group:
                    continue
                if any(max(abs(entry[2][0]-old[2][0]),
                           abs(entry[2][1]-old[2][1])) <= 1 for old in answer):
                    continue
                answer.append(entry)
                if search(index+1):
                    return True
                answer.pop()
            return False

        if search(0):
            # Redundant observations are optional and always agree in local coordinates.
            if mode not in ('full', 'split') and rng.random() < .3:
                extras = [x for options in choices.values() for x in options]
                rng.shuffle(extras)
                for entry in extras[:len(slots)+2]:
                    if all(max(abs(entry[2][0]-old[2][0]),
                               abs(entry[2][1]-old[2][1])) > 1 for old in answer):
                        answer.append(entry)
                        if len(answer) >= len(slots)+2:
                            break
            return answer
    return None


def _place_emitters(rng, template, transforms, side, slots, height, width, mode):
    for _ in range(10):
        placed = []
        for transform in transforms:
            ah, aw = _orient(template, transform).shape
            for attempt in range(64):
                row = int(rng.integers(height-ah+1))
                col = int(rng.integers(width-aw+1))
                # Include edge/corner locations, without fixing their emitting direction.
                if rng.random() < .2:
                    if rng.integers(2):
                        row = int(rng.choice([0, 1, height-ah-1, height-ah]))
                    else:
                        col = int(rng.choice([0, 1, width-aw-1, width-aw]))
                candidate = _make_candidate(template, transform, row, col,
                                             side, slots, height, width)
                if any(len(ray) < 2 for ray in candidate['rays'].values()):
                    continue
                if _compatible(candidate, placed):
                    placed.append(candidate)
                    break
            else:
                break
        if len(placed) == len(transforms):
            clues = _choose_clues(rng, placed, side, slots, height, width, mode)
            if clues is not None:
                return placed, clues
    return None


def _sample_shape(rng, length, scale):
    low, high = ((10, 19), (15, 26), (22, 31))[scale]
    low = max(low, length+3)
    height, width = map(int, rng.integers(low, max(low+1, high), size=2))
    if rng.random() < .25:
        width = height
    elif rng.random() < .3:
        if rng.integers(2):
            width = int(rng.integers(max(20, low), 31))
        else:
            height = int(rng.integers(max(20, low), 31))
    # Source dimensions are useful layouts too; only exact source inputs are excluded.
    if rng.random() < .08:
        height, width = ((26, 29), (21, 27), (29, 29))[int(rng.integers(3))]
    return height, width


def _generate_pair(rng, regime, scale, demonstration=None, previous_background=None):
    assert regime in ('source_train', 'target_test')
    for attempt in range(25):
        length = int(rng.integers(4, 13))
        count = int(rng.choice([2, 3, 4, 5, 6], p=[.2, .25, .25, .18, .12]))
        slot_count = int(rng.integers(2 if demonstration else 1, min(5, length)+1))
        slots = _sample_slots(rng, length, slot_count)
        palette = list(map(int, rng.permutation(10)))
        if palette[0] == previous_background:
            palette[0], palette[-1] = palette[-1], palette[0]
        background = palette[0]
        structural_count = int(rng.integers(2, 5))
        structural = palette[1:1+structural_count]
        signal_count = slot_count
        if not demonstration and slot_count > 1 and rng.random() < .3:
            signal_count = int(rng.integers(1, slot_count))
        signals = palette[1+structural_count:1+structural_count+signal_count]
        values = signals + list(map(int, rng.choice(signals, slot_count-signal_count)))
        rng.shuffle(values)
        code = dict(zip(slots, values))
        if demonstration and code == {length-1-s: c for s, c in code.items()}:
            continue
        template = _make_template(rng, length, structural)
        side = int(rng.integers(2))
        transforms = _sample_transforms(rng, count, demonstration)
        height, width = _sample_shape(rng, length, scale)
        mode = 'split' if demonstration == 'pool' else str(rng.choice(['full', 'two', 'spread']))
        if slot_count == 1:
            mode = 'full' if rng.random() < .5 else 'spread'
        # An isolated complete clue code fits one boundary only when slots do not touch.
        if mode == 'full' and any(b-a == 1 for a, b in zip(slots, slots[1:])):
            mode = 'two'
        result = _place_emitters(rng, template, transforms, side, slots,
                                  height, width, mode)
        if result is not None:
            placed, clues = result
            break
    else:
        # Guaranteed spacious fallback with two separated bars and a split code.
        # The second copy reflects or rotates according to the demonstration.
        length, height, width, side = 5, 14, 17, 0
        palette = list(map(int, rng.permutation(10)))
        if palette[0] == previous_background:
            palette[0], palette[-1] = palette[-1], palette[0]
        background = palette[0]
        template = _make_template(rng, length, palette[1:4])
        slots, code = (1, 3), {1: palette[4], 3: palette[5]}
        transforms = (0, 1) if demonstration == 'rotate' else (0, 4)
        positions = ((3, 1), (7, 11)) if demonstration == 'rotate' else ((5, 1), (5, 11))
        placed = [_make_candidate(template, t, r, c, side, slots, height, width)
                  for t, (r, c) in zip(transforms, positions)]
        clues = _choose_clues(rng, placed, side, slots, height, width, 'split')
        assert clues is not None
    input_grid = np.full((height, width), background, dtype=np.int8)
    for emitter in placed:
        r, c = emitter['r'], emitter['c']
        h, w = emitter['arr'].shape
        input_grid[r:r+h, c:c+w] = emitter['arr']
    for slot, _, endpoint in clues:
        input_grid[endpoint] = code[slot]
    output_grid = input_grid.copy()
    for emitter in placed:
        for slot, color in code.items():
            for position in emitter['rays'][slot]:
                output_grid[position] = color
    return {'input': input_grid, 'output': output_grid}


def _decode(input_grid):
    """Reconstruct the rule from pixels, without generation parameters or labels."""
    values, counts = np.unique(input_grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(values[np.argmax(counts)])
    height, width = input_grid.shape
    signals, emitters = [], []
    for component in _components(input_grid, background):
        if len(component) == 1:
            r, c = component[0]
            assert r in (0, height-1) or c in (0, width-1)
            signals.append((r, c, int(input_grid[r, c])))
            continue
        rs, cs = zip(*component)
        r, c = min(rs), min(cs)
        a = input_grid[r:max(rs)+1, c:max(cs)+1]
        assert a.size == len(component) and min(a.shape) == 2
        assert 4 <= max(a.shape) <= 12
        emitters.append({'r': r, 'c': c, 'arr': a, 'cells': frozenset(component)})
    assert 2 <= len(emitters) <= 6 and signals
    canonical = emitters[0]['arr']
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_grid_key(a) for a in forms}) == 8
    structural = frozenset(map(int, np.unique(canonical)))
    assert 2 <= len(structural) <= 4
    assert background not in structural
    assert structural.isdisjoint(color for _, _, color in signals)
    labels = np.arange(canonical.size).reshape(canonical.shape)
    transforms = []
    for emitter in emitters:
        matches = [t for t, a in enumerate(forms) if np.array_equal(a, emitter['arr'])]
        assert len(matches) == 1
        t = matches[0]
        transforms.append(t)
        emitter['labels'] = _orient(labels, t)
    occupied = set().union(*(e['cells'] for e in emitters))
    corridors = {}
    for index, emitter in enumerate(emitters):
        for side in (0, 1):
            for slot in range(length):
                ray, direction = _ray_for_candidate(emitter, side, slot, length, height, width)
                if ray and not occupied.intersection(ray):
                    if all(int(input_grid[p]) == background for p in ray[:-1]):
                        corridors.setdefault(ray[-1], []).append((index, side, slot, direction))
    code, sides, allocations = {}, set(), [set() for _ in emitters]
    observations = []
    for r, c, color in signals:
        matches = corridors.get((r, c), [])
        assert len(matches) == 1, 'ambiguous boundary clue'
        index, side, slot, direction = matches[0]
        assert slot not in code or code[slot] == color, 'conflicting observations'
        code[slot] = color
        sides.add(side)
        allocations[index].add(slot)
        observations.append((index, slot, color))
    assert len(sides) == 1 and 1 <= len(code) <= 5
    side = sides.pop()
    expected, painted, directions, ray_lengths = input_grid.copy(), {}, [], []
    for emitter in emitters:
        for slot, color in code.items():
            ray, direction = _ray_for_candidate(emitter, side, slot, length, height, width)
            assert len(ray) >= 2
            ray_lengths.append(len(ray))
            for p in ray:
                assert p not in occupied and p not in painted, 'intersecting rays or bars'
                assert int(input_grid[p]) in (background, color)
                expected[p] = color
                painted[p] = color
        directions.append(direction)
    assert not np.array_equal(expected, input_grid)
    slots = tuple(sorted(code))
    sequence = tuple(code[s] for s in slots)
    gaps = tuple(b-a for a, b in zip(slots, slots[1:]))
    border_gaps = [min(e['r'], e['c'], height-e['r']-e['arr'].shape[0],
                       width-e['c']-e['arr'].shape[1]) for e in emitters]
    longest_run = 1
    for row in canonical:
        current = 1
        for a, b in zip(row, row[1:]):
            current = current+1 if a == b else 1
            longest_run = max(longest_run, current)
    features = {
        'background': background, 'structural': structural,
        'signals': frozenset(sequence), 'length': length, 'slots': slots,
        'sequence': sequence, 'slot_count': len(slots), 'emitter_count': len(emitters),
        'directions': tuple(directions), 'transforms': tuple(transforms),
        'allocations': tuple(len(s) for s in allocations),
        'clue_count': len(signals), 'adjacent': 1 in gaps,
        'regular': len(gaps) >= 2 and len(set(gaps)) == 1,
        'irregular': len(set(gaps)) > 1, 'end_slot': 0 in slots or length-1 in slots,
        'repeated_color': len(set(sequence)) < len(sequence),
        'full_clues': any(len(s) == len(code) for s in allocations),
        'pooled': all(len(s) < len(code) for s in allocations),
        'redundant': len(signals) > len(code), 'unhinted': 0 in map(len, allocations),
        'monochrome_row': any(len(set(map(int, row))) == 1 for row in canonical),
        'longest_run': longest_run, 'border_gap': min(border_gaps),
        'ray_lengths': tuple(ray_lengths), 'shape': (height, width),
        'asymmetric_code': code != {length-1-s: c for s, c in code.items()},
    }
    return expected, features


def _analyze_pair(input_grid, output_grid, regime):
    assert regime in ('source_train', 'target_test')
    for grid in (input_grid, output_grid):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.all((grid >= 0) & (grid <= 9))
    assert input_grid.shape == output_grid.shape
    for shape, non_background in _SOURCE_INPUTS:
        if input_grid.shape == shape:
            visible = tuple((int(r), int(c), int(input_grid[r, c]))
                            for r, c in np.argwhere(input_grid != 8))
            assert visible != non_background, 'exact source input'
    expected, features = _decode(input_grid)
    assert np.array_equal(expected, output_grid), 'incorrect ray completion'
    return features


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        assert 2 <= len(puzzle['train']) <= 10
        assert 1 <= len(puzzle['test']) <= 4
        assert len(puzzle['train']) > len(puzzle['test'])
        inputs, analyses = set(), {'train': [], 'test': []}
        for split in ('train', 'test'):
            regime = 'source_train' if split == 'train' else 'target_test'
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                f = _analyze_pair(pair['input'], pair['output'], regime)
                key = _grid_key(pair['input'])
                assert key not in inputs
                inputs.add(key)
                analyses[split].append(f)
        train = analyses['train']
        # These are evidence requirements, not split-specific palettes or layouts.
        assert any(f['pooled'] and f['asymmetric_code'] for f in train)
        assert any(len(set(f['transforms'])) > 1 and f['asymmetric_code'] for f in train)
        assert any(any(t >= 4 for t in f['transforms']) and f['asymmetric_code'] for f in train)
        assert any(len({t % 2 for t in f['transforms']}) == 2 for f in train)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('malformed generated puzzle') from exc


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _check_count('num_train', num_train, 2, 10)
    num_test = _check_count('num_test', num_test, 1, 4)
    index = int(seed) % (2**32)
    if num_train is None and num_test is None:
        num_test = 1+index % 4
        minimum = num_test+1
        num_train = minimum+(index//4) % (11-minimum)
    elif num_train is None:
        minimum = num_test+1
        num_train = minimum+index % (11-minimum)
    elif num_test is None:
        num_test = 1+index % min(4, num_train-1)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    # Only geometry/pooling is semantic across pairs. This pedagogical size
    # schedule varies by puzzle; all palettes, templates, sides and codes are local.
    schedule = int(rng.integers(4))  # matched, expanded, smaller test, mixed
    base_scale = int(rng.integers(3))
    puzzle = {'train': [], 'test': []}
    previous_background = None
    for split, count in (('train', num_train), ('test', num_test)):
        for pair_index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            if schedule == 0:
                scale = base_scale
            elif schedule == 1:
                scale = 1 if split == 'train' else 2
            elif schedule == 2:
                scale = 2 if split == 'train' else 0
            else:
                scale = int(pair_rng.integers(3))
            demonstration = None
            if split == 'train' and pair_index < 2:
                demonstration = ('pool', 'rotate')[pair_index]
            pair = _generate_pair(pair_rng, 'source_train' if split == 'train' else 'target_test',
                                  scale, demonstration, previous_background)
            previous_background = _decode(pair['input'])[1]['background']
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _collection_features(puzzles):
    return [[_decode(pair['input'])[1] for split in ('train', 'test')
             for pair in puzzle[split]] for puzzle in puzzles]


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    grouped = _collection_features(puzzles)
    features = [f for group in grouped for f in group]
    tests = [f for p, group in zip(puzzles, grouped) for f in group[len(p['train']):]]
    trains = [f for p, group in zip(puzzles, grouped) for f in group[:len(p['train'])]]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({tuple(_grid_key(x['input']) for x in p['train']+p['test'])
                for p in puzzles}) == 42
    assert len({(_grid_key(p['train'][0]['input'] != group[0]['background']))
                for p, group in zip(puzzles, grouped)}) >= 38
    # Diversity is reconstructed from pixels, not sampled parameter dictionaries.
    for subset in (features, trains, tests):
        assert {f['length'] for f in subset} == set(range(4, 13))
        assert {f['slot_count'] for f in subset} == set(range(1, 6))
        assert {f['emitter_count'] for f in subset} == set(range(2, 7))
        assert {len(f['structural']) for f in subset} == {2, 3, 4}
        assert len({f['shape'] for f in subset}) >= 30
        for property_name in ('adjacent', 'regular', 'irregular', 'end_slot',
                              'repeated_color', 'full_clues', 'pooled',
                              'redundant', 'unhinted', 'monochrome_row'):
            assert sum(f[property_name] for f in subset) >= 3, property_name
        assert any(f['border_gap'] == 0 for f in subset)
        assert any(f['border_gap'] == 1 for f in subset)
        assert any(h == w for h, w in (f['shape'] for f in subset))
        assert any(h >= 1.5*w for h, w in (f['shape'] for f in subset))
        assert any(w >= 1.5*h for h, w in (f['shape'] for f in subset))
        assert any(min(f['shape']) <= 12 for f in subset)
        assert any(f['shape'] in ((26, 29), (21, 27), (29, 29)) for f in subset)
        assert any(len(set(f['directions'])) == 1 for f in subset)
        assert any(len(set(f['directions'])) == 4 for f in subset)
        assert any(len(set(f['transforms'])) < f['emitter_count'] for f in subset)
        assert any(len({t >= 4 for t in f['transforms']}) == 1 for f in subset)
        assert any(f['adjacent'] and f['emitter_count'] >= 5 and f['pooled'] for f in subset)
        assert any(f['length'] % 2 == 0 and f['monochrome_row'] and f['full_clues'] for f in subset)
    for group in grouped:
        assert len({f['background'] for f in group}) >= 2
    for axis in ('length', 'slot_count', 'emitter_count', 'structural', 'signals', 'shape'):
        assert sum(len({f[axis] for f in group}) > 1 for group in grouped) >= 35, axis
    assert min(min(f['ray_lengths']) for f in features) == 2
    assert max(max(f['ray_lengths']) for f in features) >= 20
    # Exercise fresh inference after a whole-grid isometry and per-pair recoloring.
    original = generate_puzzle(7001, 6, 3)
    transformed = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for index, pair in enumerate(original[split]):
            recolor = np.roll(np.arange(10, dtype=np.int8), index+1)
            transformed[split].append({role: _orient(recolor[grid], (index+3) % 8)
                                       for role, grid in pair.items()})
    validate_generated_puzzle(transformed)
    assert _same_puzzle(original, generate_puzzle(7001, 6, 3))
    assert not _same_puzzle(original, generate_puzzle(7002, 6, 3))
    # Returned-pixel corruption must be caught by reconstruction.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('output corruption was accepted')
    for nt, ne in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(8000+(nt or 0)+(ne or 0), nt, ne)
        if nt is not None:
            assert len(p['train']) == nt
        if ne is not None:
            assert len(p['test']) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3),
                   (3, 3), (4, 4), (True, 1), (3.0, 1), (3, False)):
        try:
            generate_puzzle(123, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
