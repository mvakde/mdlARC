import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# FNV-style fingerprints of the five supplied source inputs. Rejecting an
# identical source input also prevents an identical source input/output pair.
_SOURCE_INPUT_FINGERPRINTS = {
    14035972571012050024,
    3372863591258779632,
    6929671316502712388,
    10763929138443479870,
    8050785400168057728,
}


def _is_int(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(seed, num_train, num_test):
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)

    if num_train is not None:
        if not _is_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if not _is_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_test = int(num_test)
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_train = int(num_train)
        possible = min(4, num_train - 1)
        if possible < 1:
            raise ValueError("no legal test count for num_train")
        num_test = 1 + seed % possible
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= train <= 10, "
            "1 <= test <= 4, and train > test"
        )

    return num_train, num_test


def _set_components(cells):
    remaining = set(cells)
    result = []

    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        component = {start}

        while stack:
            r, c = stack.pop()
            for dr, dc in _DIR4:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        result.append(frozenset(component))

    result.sort(key=lambda component: min(component))
    return result


def _grid_components(mask):
    cells = {
        (int(r), int(c))
        for r, c in np.argwhere(mask)
    }
    return _set_components(cells)


def _bbox(cells):
    rows = [point[0] for point in cells]
    cols = [point[1] for point in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _raw_transform(point, op):
    r, c = point

    if op == 0:
        return r, c
    if op == 1:
        return c, -r
    if op == 2:
        return -r, -c
    if op == 3:
        return -c, r
    if op == 4:
        return r, -c
    if op == 5:
        return -c, -r
    if op == 6:
        return -r, c
    if op == 7:
        return c, r

    raise ValueError("dihedral operation must be in range(8)")


def _transform_regions(regions, op):
    raw = [
        [_raw_transform(point, op) for point in region]
        for region in regions
    ]
    all_points = [
        point
        for region in raw
        for point in region
    ]

    min_r = min(point[0] for point in all_points)
    min_c = min(point[1] for point in all_points)

    return tuple(
        frozenset(
            (r - min_r, c - min_c)
            for r, c in region
        )
        for region in raw
    )


def _shape_signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(
        sorted((r - r0, c - c0) for r, c in cells)
    )


def _canonical_shape(cells):
    return min(
        _shape_signature(
            _transform_regions((cells,), op)[0]
        )
        for op in range(8)
    )


def _full_dihedral_orbit(cells):
    return {
        _shape_signature(
            _transform_regions((cells,), op)[0]
        )
        for op in range(8)
    }


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _make_system_shape(rng, forbidden_slot_signatures=(), small=False):
    """Carve a connected separator, then grow a separate exterior region.

    The two target fragments, not a slot-orbit test, define a valid lock.
    Whole-key uniqueness is decided later on the actual canvas.
    """
    for attempt in range(300):
        symmetric = rng.random() < 0.30
        if symmetric:
            n = int(rng.integers(2, 5))
            height = n + int(rng.integers(1, 4))
            width = n
            if rng.random() < 0.5:
                y = int(rng.integers(1, height - n + 1))
                slot = {(y, c) for c in range(n)} | {(y + r, 0) for r in range(n)}
            else:
                y = int(rng.integers(0, height - n))
                slot = {(y + r, c) for r in range(n) for c in range(r + 1)}
        else:
            height = int(rng.integers(3, 7 if small else 9))
            width = int(rng.integers(3, 7 if small else 9))
            y = int(rng.integers(1, height - 1))
            slot = {(y, 0)}
            for c in range(1, width):
                new_y = int(np.clip(y + rng.integers(-2, 3), 0, height - 1))
                slot.update((r, c) for r in range(min(y, new_y), max(y, new_y) + 1))
                y = new_y
            # Unequal widths and optional branches broaden the thin path.
            for _ in range(int(rng.integers(0, width + 1))):
                frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in _DIR4
                                   if 0 <= r + dr < height and 0 <= c + dc < width} - slot)
                if frontier:
                    slot.add(frontier[int(rng.integers(len(frontier)))])
        rectangle = {(r, c) for r in range(height) for c in range(width)}
        target = rectangle - slot
        parts = _set_components(target)
        if len(parts) != 2 or min(map(len, parts)) < 2:
            continue
        if _bbox(target) != (0, height - 1, 0, width - 1):
            continue
        signature = _canonical_shape(slot)
        if signature in forbidden_slot_signatures:
            continue

        relationship = int(rng.integers(3))
        size = (int(rng.integers(2, len(slot))) if relationship == 0 else
                len(slot) if relationship == 1 else
                len(slot) + int(rng.integers(1, 10)))
        size = min(size, 30)
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in _DIR4
                         if (r + dr, c + dc) not in rectangle})
        start = starts[int(rng.integers(len(starts)))]
        handle = {start}
        style = int(rng.integers(3))  # compact growth, bent walk, branching growth
        tip = start
        reach = 5 if small else 7
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in _DIR4
                               if -reach <= r + dr < height + reach
                               and -reach <= c + dc < width + reach}
                              - rectangle - handle)
            if not frontier:
                break
            weights = []
            for r, c in frontier:
                neighbors = sum((r + dr, c + dc) in handle for dr, dc in _DIR4)
                distance = abs(r - start[0]) + abs(c - start[1])
                if style == 0:
                    weight = (neighbors ** 2) / (1 + distance)
                elif style == 1:
                    weight = (10 if abs(r - tip[0]) + abs(c - tip[1]) == 1 else 0.1)
                    weight /= neighbors ** 3
                else:
                    weight = 1 / neighbors
                weights.append(weight)
            probs = np.array(weights, dtype=float)
            tip = frontier[int(rng.choice(len(frontier), p=probs / probs.sum()))]
            handle.add(tip)
        if len(handle) != size:
            continue
        # The incoming key has no preferred axis either. A reused witness
        # retains this orientation while its destination is transformed anew.
        return _transform_regions((target, slot, handle), int(rng.integers(8)))
    return _fallback_shape(int(rng.integers(3)))


def _fallback_shape(index):
    width = 4 + index
    rectangle = {(r, c) for r in range(4) for c in range(width)}
    slot = {(1, width - 2), (1, width - 1)} | {(2, c) for c in range(width)}
    handle = {(2, width), (2, width + 1), (3, width + 1)}
    return tuple(frozenset(x) for x in (rectangle - slot, slot, handle))


def _scene_colors(rng, params, count, first_colors=None):
    if params['stable']:
        palette = params['palette']
    else:
        palette = tuple(int(c) for c in rng.permutation(10))
        if first_colors is not None and palette[0] in first_colors:
            allowed = [c for c in palette if c not in first_colors]
            bg = allowed[0]
            palette = (bg,) + tuple(c for c in palette if c != bg)
    bg, colors = palette[0], palette[1:]
    mode = 0 if params['stable'] else int(rng.integers(4))
    triples = []
    for i in range(count):
        target = colors[0] if mode in (0, 2) else colors[i % len(colors)]
        available = [c for c in colors if c != target]
        if mode in (0, 1):
            key = [colors[-2], colors[-1]]
        else:
            key = [int(c) for c in rng.choice(available, 2, replace=False)]
        if i == 1 and mode == 3:
            # One literal color deliberately has different local roles.
            target = triples[0][1]
            available = [c for c in colors if c != target]
            key = [triples[0][0], int(rng.choice([c for c in available if c != triples[0][0]]))]
        if rng.random() < 0.5:
            key.reverse()
        triples.append((target, *key))
    if first_colors is not None:
        triples[0] = first_colors
    return bg, triples


def _place_items(rng, footprints, height, width):
    """Pack final assemblies and original keys independently, with a moat.

    No system bands, ordering convention, or nearest-destination constraint.
    """
    occupied = set()
    origins = [None] * len(footprints)
    # Large pieces first, random tie breaking; keys and targets are interleaved.
    order = sorted(range(len(footprints)), key=lambda i: -len(footprints[i]) + rng.random())
    for i in order:
        cells = footprints[i]
        _, r1, _, c1 = _bbox(cells)
        if r1 >= height or c1 >= width:
            return None
        options = [(r, c) for r in range(height - r1) for c in range(width - c1)]
        rng.shuffle(options)
        if rng.random() < 0.55:
            options.sort(key=lambda p: not (p[0] in (0, height - r1 - 1)
                                           or p[1] in (0, width - c1 - 1)))
        for r, c in options:
            placed = _shift(cells, r, c)
            if placed & occupied:
                continue
            origins[i] = (r, c)
            occupied.update((y + dy, x + dx) for y, x in placed
                            for dy, dx in ((0, 0),) + _DIR4)
            break
        else:
            return None
    return origins


def _build_pair_once(rng, regime, params, desired_parities, reused=None, fallback=False):
    assert regime in ('source_train', 'target_test')
    count = len(desired_parities)
    shapes, forbidden = [], set()
    for i in range(count):
        shape = (_fallback_shape(i) if fallback else reused[0] if i == 0 and reused else
                 _make_system_shape(rng, forbidden, small=count >= 3))
        shapes.append(shape)
        forbidden.add(_canonical_shape(shape[1]))
    bg, triples = _scene_colors(rng, params, count, reused[1] if reused else None)
    pieces, systems = [], []
    for shape, colors, parity in zip(shapes, triples, desired_parities):
        op = int(rng.integers(4)) + 4 * int(parity)
        target, slot, handle = _transform_regions(shape, op)
        key = _transform_regions(shape[1:], 0)
        pieces.extend((target | slot | handle, key[0] | key[1]))
        systems.append((target, slot, handle, key, colors, op))

    if fallback:
        full_canvas = True
        height = width = 30
        tiles = [(r * 10 + 1, c * 15 + 1) for r in range(3) for c in range(2)]
        rng.shuffle(tiles)
        origins = tiles[:len(pieces)]
    else:
        full_canvas = rng.random() < 0.12
        area = sum(map(len, pieces))
        origins = None
        for attempt in range(12):
            # Both splits draw from the same sizes. Packing selects feasible
            # compact, tall, wide, and full-size canvases from their objects.
            if full_canvas:
                height = width = 30
            elif attempt < 8:
                height = int(rng.integers(10, 31))
                width = int(rng.integers(10, 31))
                if height * width < area * 1.8:
                    continue
            else:
                height = width = 30
            origins = _place_items(rng, pieces, height, width)
            if origins is not None:
                break
        if origins is None:
            raise ValueError('objects do not pack')
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for i, (target, slot, handle, key, colors, op) in enumerate(systems):
        ar, ac = origins[2 * i]
        kr, kc = origins[2 * i + 1]
        tc, pc, hc = colors
        for p in _shift(target, ar, ac):
            inp[p] = out[p] = tc
        for region, color in zip(key, (pc, hc)):
            for p in _shift(region, kr, kc):
                inp[p] = color
        for region, color in zip((slot, handle), (pc, hc) if op < 4 else (hc, pc)):
            for p in _shift(region, ar, ac):
                out[p] = color
    # Crop only unused margins, sometimes keeping padding or a full dimension.
    used = set(map(tuple, np.argwhere((inp != bg) | (out != bg))))
    r0, r1, c0, c1 = _bbox(used)
    if not full_canvas and rng.random() < 0.8:
        margins = [int(rng.integers(0, 5)) for _ in range(4)]
        r0, r1 = max(0, r0 - margins[0]), min(height - 1, r1 + margins[1])
        c0, c1 = max(0, c0 - margins[2]), min(width - 1, c1 + margins[3])
        inp, out = inp[r0:r1 + 1, c0:c1 + 1].copy(), out[r0:r1 + 1, c0:c1 + 1].copy()
    return inp, out


def _background_color(grid):
    values, counts = np.unique(
        grid,
        return_counts=True,
    )
    order = np.argsort(counts)

    assert len(counts) > 0
    assert (
        len(counts) == 1
        or counts[order[-1]] > counts[order[-2]]
    ), "background is not unique"

    return int(values[order[-1]])


def _inspect_input(inp):
    background = _background_color(inp)
    components = _grid_components(inp != background)

    mono = []
    keys = []

    for component in components:
        colors = sorted({
            int(inp[point])
            for point in component
        })

        assert len(colors) in (1, 2), (
            "non-background components must be target "
            "fragments or two-colored keys"
        )

        if len(colors) == 1:
            mono.append({
                "cells": component,
                "color": colors[0],
            })
        else:
            regions = tuple(
                frozenset(
                    point
                    for point in component
                    if int(inp[point]) == color
                )
                for color in colors
            )

            assert all(
                len(_set_components(region)) == 1
                for region in regions
            ), "each key color region must be connected"

            keys.append({
                "cells": component,
                "colors": tuple(colors),
                "regions": regions,
            })

    mono.sort(key=lambda item: min(item["cells"]))
    keys.sort(key=lambda item: min(item["cells"]))

    return {
        "background": background,
        "mono": mono,
        "keys": keys,
    }


def _solve_pair(inp, swap_policy=(0, 1), training_output=None):
    analysis = _inspect_input(inp)
    background = analysis["background"]
    mono = analysis["mono"]
    keys = analysis["keys"]

    assert keys
    assert len(mono) == 2 * len(keys), (
        "each key needs exactly two target fragments"
    )

    target_candidates = []

    for i in range(len(mono)):
        for j in range(i + 1, len(mono)):
            if mono[i]["color"] != mono[j]["color"]:
                continue

            union = (
                set(mono[i]["cells"])
                | set(mono[j]["cells"])
            )
            r0, r1, c0, c1 = _bbox(union)

            rectangle = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
            }
            non_background = {
                point
                for point in rectangle
                if int(inp[point]) != background
            }

            if non_background != union:
                continue

            slot = rectangle - union
            if not slot:
                continue
            if len(_set_components(slot)) != 1:
                continue

            target_candidates.append({
                "parts": (i, j),
                "bbox": (r0, r1, c0, c1),
                "rectangle": frozenset(rectangle),
                "slot": frozenset(slot),
                "target_color": mono[i]["color"],
            })

    assert target_candidates, "no shaped slots found"

    candidates_by_key = []

    for key_index, key in enumerate(keys):
        local_regions = _transform_regions(
            key["regions"],
            0,
        )
        key_candidates = []

        for target_index, target in enumerate(
            target_candidates
        ):
            slot = target["slot"]
            sr0, _, sc0, _ = _bbox(slot)
            tr0, tr1, tc0, tc1 = target["bbox"]

            for plug_index in (0, 1):
                for op in range(8):
                    transformed = _transform_regions(
                        local_regions,
                        op,
                    )
                    plug = transformed[plug_index]
                    pr0, _, pc0, _ = _bbox(plug)
                    dr = sr0 - pr0
                    dc = sc0 - pc0

                    absolute = tuple(
                        frozenset(
                            (r + dr, c + dc)
                            for r, c in region
                        )
                        for region in transformed
                    )

                    if absolute[plug_index] != slot:
                        continue

                    other_index = 1 - plug_index
                    other = absolute[other_index]

                    if any(
                        not (
                            0 <= r < inp.shape[0]
                            and 0 <= c < inp.shape[1]
                        )
                        for r, c in other
                    ):
                        continue

                    if any(
                        int(inp[point]) != background
                        for point in other
                    ):
                        continue

                    if any(
                        tr0 <= r <= tr1
                        and tc0 <= c <= tc1
                        for r, c in other
                    ):
                        continue

                    full = absolute[0] | absolute[1]

                    if swap_policy[int(op >= 4)]:
                        new_colors = key["colors"][::-1]
                    else:
                        new_colors = key["colors"]

                    placed = {}
                    for region_index, region in enumerate(
                        absolute
                    ):
                        for point in region:
                            placed[point] = int(
                                new_colors[region_index]
                            )

                    # Training labels can eliminate competing learned color
                    # conventions; test prediction never supplies this value.
                    if training_output is not None and any(
                        int(training_output[p]) != color for p, color in placed.items()
                    ):
                        continue

                    key_candidates.append({
                        "key_index": key_index,
                        "target_index": target_index,
                        "parts": target["parts"],
                        "bbox": target["bbox"],
                        "rectangle": target["rectangle"],
                        "slot": target["slot"],
                        "op": op,
                        "plug_index": plug_index,
                        "full": frozenset(full),
                        "placed": placed,
                    })

        assert key_candidates, (
            "a key has no legal target placement"
        )
        candidates_by_key.append(key_candidates)

    solutions = []
    solution_outputs = set()
    all_mono = set(range(len(mono)))

    def search(
        key_index,
        used_parts,
        used_rectangles,
        occupied,
        chosen,
    ):
        if len(solutions) > 1:
            return

        if key_index == len(keys):
            if used_parts == all_mono:
                signature = tuple(sorted((p, color) for move in chosen for p, color in move['placed'].items()))
                if signature not in solution_outputs:
                    solution_outputs.add(signature)
                    solutions.append(tuple(chosen))
            return

        for candidate in candidates_by_key[key_index]:
            parts = set(candidate["parts"])

            if parts & used_parts:
                continue

            if any(
                candidate["rectangle"] & rectangle
                for rectangle in used_rectangles
            ):
                continue

            if candidate["full"] & occupied:
                continue

            search(
                key_index + 1,
                used_parts | parts,
                used_rectangles
                + (candidate["rectangle"],),
                occupied | set(candidate["full"]),
                chosen + [candidate],
            )

    search(
        0,
        set(),
        tuple(),
        set(),
        [],
    )

    assert len(solutions) == 1, (
        "the global key-to-slot solution is not unique"
    )

    moves = solutions[0]
    out = np.full(
        inp.shape,
        background,
        dtype=inp.dtype,
    )

    # Preserve all fixed target fragments.
    for item in mono:
        for point in item["cells"]:
            out[point] = item["color"]

    # Original key locations remain background; only their solved
    # destinations are painted.
    for move in moves:
        for point, color in move["placed"].items():
            assert int(out[point]) == background, (
                "solved keys overlap"
            )
            out[point] = color

    analysis["target_candidates"] = target_candidates
    return out, moves, analysis


def _array_signature(array):
    canonical = np.ascontiguousarray(
        array,
        dtype=np.int16,
    )
    return canonical.shape, canonical.tobytes()


def _fingerprint(array):
    value = 1469598103934665603
    sequence = [
        array.shape[0],
        array.shape[1],
    ]
    sequence.extend(int(cell) for cell in array.flat)

    for cell in sequence:
        value ^= cell + 1
        value = (
            value * 1099511628211
        ) & ((1 << 64) - 1)

    return value


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _generate_pair(rng, regime, params, parities, reused=None, witness_parity=None):
    for attempt in range(100):
        try:
            inp, out = _build_pair_once(rng, regime, params, parities, reused,
                                        fallback=attempt >= 80)
            solved, moves, analysis = _solve_pair(inp)
            if witness_parity is not None and not any(
                int(m['op'] >= 4) == witness_parity
                and len(analysis['keys'][m['key_index']]['regions'][0])
                != len(analysis['keys'][m['key_index']]['regions'][1])
                for m in moves
            ):
                continue
            if (np.array_equal(solved, out)
                    and sorted(int(m['op'] >= 4) for m in moves) == sorted(parities)):
                return {'input': inp, 'output': out}
        except (AssertionError, ValueError):
            continue
    raise AssertionError('bounded construction exhausted including safe fallback')


def _operation_schedule(rng, count):
    mode = int(rng.integers(3 if count > 1 else 2))
    if mode < 2:
        return [mode] * count
    result = [0, 1] + [int(rng.integers(2)) for _ in range(count - 2)]
    rng.shuffle(result)
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The only learned convention is rigid fitting with reflection-local color
    # exchange. Palette stability and count regimes control sampling difficulty,
    # and never supply information needed to infer an output.
    params = {'stable': rng.random() < 0.25,
              'palette': tuple(int(c) for c in rng.permutation(10))}
    regime = int(rng.integers(4))
    if regime == 0:
        train_sizes, test_sizes = [1] * num_train, [2] * num_test
    elif regime == 1:
        n = int(rng.integers(1, 4))
        train_sizes, test_sizes = [n] * num_train, [n] * num_test
    else:
        train_sizes = [int(rng.integers(1, 4 if regime == 2 else 3)) for _ in range(num_train)]
        test_sizes = [int(rng.integers(1 if regime == 2 else 2, 4)) for _ in range(num_test)]
        if len(set(train_sizes)) == 1:
            train_sizes[-1] = 1 + train_sizes[-1] % (3 if regime == 2 else 2)
        if num_test > 1 and len(set(test_sizes)) == 1:
            test_sizes[-1] = (1 + test_sizes[-1] % 3 if regime == 2 else 5 - test_sizes[-1])
    reused = None
    if rng.random() < 0.45:
        for _ in range(40):
            shape = _make_system_shape(rng, small=max(train_sizes[:2]) >= 3)
            if len(shape[1]) != len(shape[2]):
                break
        else:
            shape = _fallback_shape(0)
        _, colors = _scene_colors(rng, params, 1)
        reused = (shape, colors[0])
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, sizes in (('train', train_sizes), ('test', test_sizes)):
        for i, n in enumerate(sizes):
            parities = _operation_schedule(rng, n)
            # Two witnesses suffice at the legal (2,1) boundary. Shuffle all
            # training pairs afterward so demonstration order carries no rule.
            if split == 'train' and i < 2:
                parities[0] = i
            for _ in range(10):
                pair = _generate_pair(rng, 'source_train' if split == 'train' else 'target_test',
                                      params, parities, reused if split == 'train' and i < 2 else None,
                                      i if split == 'train' and i < 2 else None)
                signature = _array_signature(pair['input'])
                if signature not in seen and _fingerprint(pair['input']) not in _SOURCE_INPUT_FINGERPRINTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('could not construct distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _learn_color_rule(train_pairs):
    compatible = []
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for pair in train_pairs:
            try:
                solved, _, _ = _solve_pair(pair['input'], policy, pair['output'])
                if not np.array_equal(solved, pair['output']):
                    break
            except AssertionError:
                break
        else:
            compatible.append(policy)
    assert compatible == [(0, 1)], 'training must uniquely establish the parity-to-color rule'
    return compatible[0]


def _predict_from_training(train_pairs, test_input):
    policy = _learn_color_rule(train_pairs)
    return _solve_pair(test_input, policy)[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, train_parities, test_parities = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ik, ok = _array_signature(inp), _array_signature(out)
            assert ik not in seen_inputs and (ik, ok) not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            assert _fingerprint(inp) not in _SOURCE_INPUT_FINGERPRINTS, 'source input copied'
            solved, moves, analysis = _solve_pair(inp)
            assert np.array_equal(solved, out), 'output must be the unique complete colored solution'
            assert set(np.unique(inp)) == set(np.unique(out))
            assert 1 <= len(moves) <= 3
            assert len(analysis['mono']) == 2 * len(moves)
            for move in moves:
                # All tests operate in the same family; count and geometry may
                # increase, but neither split is recognized by palette or size.
                key = analysis['keys'][move['key_index']]
                assert len(key['colors']) == 2 and analysis['background'] not in key['colors']
                assert len(_set_components(move['slot'])) == 1
                target = move['rectangle'] - move['slot']
                assert len(_set_components(target)) == 2
                assert _bbox(target) == move['bbox']
                assert all(int(out[p]) == int(inp[p]) for p in target)
                assert all(int(out[p]) == analysis['background'] for p in key['cells'])
                (train_parities if split == 'train' else test_parities).add(int(move['op'] >= 4))
    assert train_parities == {0, 1}, 'training must demonstrate preservation and exchange'
    assert test_parities <= train_parities
    _learn_color_rule(puzzle['train'])


def _review_features(inp):
    """Measure actual reconstructed objects, never generation-time parameters."""
    _, moves, analysis = _solve_pair(inp)
    features = {'count': len(moves), 'shape': inp.shape, 'background': analysis['background'],
                'palette': tuple(int(c) for c in np.unique(inp)),
                'operations': tuple(sorted(m['op'] for m in moves)), 'assemblies': []}
    target_colors = {item['color'] for item in analysis['mono']}
    key_colors = {color for key in analysis['keys'] for color in key['colors']}
    features['local_reuse'] = bool(target_colors & key_colors)
    features['target_palettes'] = len(target_colors)
    features['key_palettes'] = len({key['colors'] for key in analysis['keys']})
    features['shared_key_color'] = len(key_colors) < 2 * len(moves)
    features['equal_area_decoys'] = any(len(a['slot']) == len(b['slot'])
                                      and _canonical_shape(a['slot']) != _canonical_shape(b['slot'])
                                      for i, a in enumerate(moves) for b in moves[i + 1:])
    centers = [((m['bbox'][0] + m['bbox'][1]) / 2, (m['bbox'][2] + m['bbox'][3]) / 2) for m in moves]
    key_centers = []
    for move in moves:
        kr0, kr1, kc0, kc1 = _bbox(analysis['keys'][move['key_index']]['cells'])
        key_centers.append(((kr0 + kr1) / 2, (kc0 + kc1) / 2))
    features['reversed_spatial_order'] = any(
        (centers[i][axis] - centers[j][axis]) * (key_centers[i][axis] - key_centers[j][axis]) < 0
        for i in range(len(moves)) for j in range(i + 1, len(moves)) for axis in (0, 1)
    )
    for i, move in enumerate(moves):
        key = analysis['keys'][move['key_index']]
        pi = move['plug_index']
        plug, handle = key['regions'][pi], key['regions'][1 - pi]
        r0, r1, c0, c1 = move['bbox']
        kr0, kr1, kc0, kc1 = _bbox(key['cells'])
        center = ((kr0 + kr1) / 2, (kc0 + kc1) / 2)
        distance = lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1])
        exterior = move['full'] - move['slot']
        contacts = sum(any((r + dr, c + dc) in move['slot'] for dr, dc in _DIR4) for r, c in exterior)
        attachment_sides = tuple(sorted({(dr, dc) for r, c in move['slot'] for dr, dc in _DIR4
                                         if (r + dr, c + dc) in exterior}))
        attachment_positions = tuple(sorted((r - r0, c - c0) for r, c in move['slot']
                                           if any((r + dr, c + dc) in exterior for dr, dc in _DIR4)))
        hr0, hr1, hc0, hc1 = _bbox(handle)
        degrees = [sum((r + dr, c + dc) in handle for dr, dc in _DIR4) for r, c in handle]
        branch = max(degrees) >= 3
        features['assemblies'].append({
            'plug_size': len(plug), 'handle_size': len(handle),
            'size_relation': int(np.sign(len(plug) - len(handle))), 'plug_color_index': pi,
            'slot_shape': _canonical_shape(plug), 'handle_shape': _canonical_shape(handle),
            'slot_orbit': len(_full_dihedral_orbit(plug)), 'target_shape': (r1 - r0 + 1, c1 - c0 + 1),
            'edge': r0 == 0 or c0 == 0 or r1 == inp.shape[0] - 1 or c1 == inp.shape[1] - 1,
            'corner': (r0 == 0 or r1 == inp.shape[0] - 1) and (c0 == 0 or c1 == inp.shape[1] - 1),
            'key_edge': kr0 == 0 or kc0 == 0 or kr1 == inp.shape[0] - 1 or kc1 == inp.shape[1] - 1,
            'relative_direction': (int(np.sign(center[0] - centers[i][0])), int(np.sign(center[1] - centers[i][1]))),
            'column_overlap': max(c0, kc0) <= min(c1, kc1),
            'row_overlap': max(r0, kr0) <= min(r1, kr1),
            'nearer_wrong_target': any(distance(p) < distance(centers[i]) for j, p in enumerate(centers) if i != j),
            'cross_midpoint': (center[0] < inp.shape[0] / 2) != (centers[i][0] < inp.shape[0] / 2),
            'handle_density': len(handle) / ((hr1 - hr0 + 1) * (hc1 - hc0 + 1)),
            'handle_branch': branch, 'attachment_contacts': contacts,
            'bent_arm': max(degrees) <= 2 and degrees.count(1) == 2 and hr0 < hr1 and hc0 < hc1,
            'handle_extent': max(hr1 - hr0 + 1, hc1 - hc0 + 1),
            'attachment_sides': attachment_sides, 'attachment_positions': attachment_positions,
        })
    return features


def self_test() -> None:
    train_counts, test_counts, grids, slots, handles, dimensions = set(), set(), set(), set(), set(), set()
    relations, operations, counts, directions, target_shapes, plug_indices = set(), set(), set(), set(), set(), set()
    attachment_sides, attachment_positions, handle_extents = set(), set(), set()
    observed = {k: 0 for k in ('symmetric', 'reuse', 'edge', 'corner', 'nearer_wrong', 'equal_area',
                             'column_overlap', 'key_edge', 'branch', 'combined', 'within_counts',
                             'within_palettes', 'within_backgrounds', 'stable_palettes', 'same_counts',
                             'one_to_two', 'cross_midpoint', 'within_relations', 'within_test_counts',
                             'multi_target_palette', 'multi_key_palette', 'overlapping_keys',
                             'shared_targets', 'compact_handle', 'extended_handle', 'wide', 'tall',
                             'full_size', 'simple_stable', 'short_bent_arm', 'long_bent_arm', 'reversed_order')}
    operation_regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        features = [_review_features(p['input']) for s in ('train', 'test') for p in puzzle[s]]
        train_n = {f['count'] for f in features[:len(puzzle['train'])]}
        test_n = {f['count'] for f in features[len(puzzle['train']):]}
        observed['within_counts'] += len(train_n) > 1
        observed['within_test_counts'] += len(test_n) > 1
        observed['same_counts'] += len(train_n | test_n) == 1
        observed['one_to_two'] += train_n == {1} and test_n == {2}
        observed['within_palettes'] += len({f['palette'] for f in features}) > 1
        observed['within_backgrounds'] += len({f['background'] for f in features}) > 1
        observed['stable_palettes'] += len({f['palette'] for f in features}) == 1
        observed['simple_stable'] += len({f['palette'] for f in features}) == 1 and max(train_n | test_n) <= 2
        observed['within_relations'] += len({a['size_relation'] for f in features for a in f['assemblies']}) > 1
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grids.add(_array_signature(pair['input']))
                if split == 'test':
                    predicted = _predict_from_training(puzzle['train'], pair['input'].copy())
                    assert np.array_equal(predicted, pair['output'])
        for f in features:
            dimensions.add(f['shape'])
            counts.add(f['count'])
            operations.update(f['operations'])
            operation_regimes.add(tuple(sorted({int(op >= 4) for op in f['operations']})))
            observed['reuse'] += f['local_reuse']
            observed['equal_area'] += f['equal_area_decoys']
            observed['reversed_order'] += f['reversed_spatial_order']
            observed['combined'] += f['local_reuse'] and any(a['slot_orbit'] < 8 for a in f['assemblies'])
            observed['multi_target_palette'] += f['target_palettes'] > 1
            observed['multi_key_palette'] += f['key_palettes'] > 1
            observed['overlapping_keys'] += f['count'] > 1 and f['shared_key_color']
            observed['shared_targets'] += f['count'] > 1 and f['target_palettes'] == 1
            observed['wide'] += f['shape'][1] > 2 * f['shape'][0]
            observed['tall'] += f['shape'][0] > 2 * f['shape'][1]
            observed['full_size'] += f['shape'] == (30, 30)
            for a in f['assemblies']:
                slots.add(a['slot_shape'])
                handles.add(a['handle_shape'])
                relations.add(a['size_relation'])
                plug_indices.add(a['plug_color_index'])
                target_shapes.add(a['target_shape'])
                directions.add(a['relative_direction'])
                attachment_sides.update(a['attachment_sides'])
                attachment_positions.add(a['attachment_positions'])
                handle_extents.add(a['handle_extent'])
                observed['short_bent_arm'] += a['bent_arm'] and a['handle_size'] <= 5
                observed['long_bent_arm'] += a['bent_arm'] and a['handle_size'] >= 8
                observed['compact_handle'] += a['handle_density'] >= 0.8
                observed['extended_handle'] += a['handle_density'] < 0.4
                for feature, metric in [('symmetric', 'slot_orbit'), ('edge', 'edge'), ('corner', 'corner'),
                                        ('nearer_wrong', 'nearer_wrong_target'), ('column_overlap', 'column_overlap'),
                                        ('key_edge', 'key_edge'), ('branch', 'handle_branch'), ('cross_midpoint', 'cross_midpoint')]:
                    observed[feature] += a[metric] < 8 if feature == 'symmetric' else a[metric]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(grids) >= 150 and len(dimensions) >= 60
    assert len(slots) >= 60 and len(handles) >= 80 and len(target_shapes) >= 20
    assert {(4, 4), (3, 5), (5, 4), (3, 6)} <= target_shapes
    assert relations == {-1, 0, 1} and plug_indices == {0, 1}
    assert operations == set(range(8)) and counts == {1, 2, 3}
    assert operation_regimes == {(0,), (1,), (0, 1)}
    assert len(directions) >= 6
    assert attachment_sides == set(_DIR4) and len(attachment_positions) >= 12
    assert max(handle_extents) >= 6
    assert all(value > 0 for value in observed.values()), observed

    for nt, ne in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(101, nt, ne)
        validate_generated_puzzle(p)
        assert nt is None or len(p['train']) == nt
        assert ne is None or len(p['test']) == ne
    explicit = generate_puzzle(9001, 6, 2)
    assert _puzzles_equal(explicit, generate_puzzle(9001, 6, 2))
    assert not _puzzles_equal(explicit, generate_puzzle(9002, 6, 2))
    reordered = {'train': list(reversed(explicit['train'])), 'test': explicit['test']}
    validate_generated_puzzle(reordered)
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(5, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in explicit[s]] for s in explicit}
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    # A bar slot admits incompatible reflected colored outputs in open space.
    ambiguous = np.zeros((12, 14), dtype=np.int8)
    ambiguous[4, 2:5] = 2
    ambiguous[6, 2:5] = 2
    ambiguous[2, 9:12] = 3
    ambiguous[2:4, 12] = 4
    try:
        _solve_pair(ambiguous)
    except AssertionError as exc:
        assert 'global key-to-slot solution is not unique' in str(exc)
    else:
        raise AssertionError('competing colored outputs accepted')
    # Explicitly exercise the bounded fallback with each assembly count and
    # parity regime, without relying on rare random packing failures.
    rng = np.random.default_rng(71)
    params = {'stable': False, 'palette': tuple(range(10))}
    for parities in ([0], [1], [0, 0], [1, 1], [0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0]):
        inp, out = _build_pair_once(rng, 'target_test', params, parities, fallback=True)
        assert np.array_equal(_solve_pair(inp)[0], out)
