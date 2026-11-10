"""Compositional generators for the two-colour diagonal L-echo task."""

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _cells(obj):
    r, c = obj["top"]
    return {(r + y, c + x) for y in (0, 1) for x in (0, 1)
            if 2 * y + x != obj["missing"]}


def _echo_cells(obj):
    mr, mc = divmod(obj["missing"], 2)
    dy, dx = (1 if mr else -1), (1 if mc else -1)
    if obj["role"] == 1:
        dy, dx = -dy, -dx
    return {(r + dy, c + dx) for r, c in _cells(obj)}


def _inside(cell, height, width):
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _compatible(obj, objects, height, width):
    source = _cells(obj)
    if any(not _inside(cell, height, width) for cell in source):
        return False
    echo = _echo_cells(obj)
    for other in objects:
        old_source, old_echo = _cells(other), _echo_cells(other)
        if any(abs(r - y) + abs(c - x) <= 1
               for r, c in source for y, x in old_source):
            return False
        if source & old_echo or echo & old_source:
            return False
    return True


def _render(height, width, objects, colors, background):
    inp = np.full((height, width), background, dtype=np.int8)
    for obj in objects:
        for r, c in _cells(obj):
            inp[r, c] = colors["source"][obj["role"]]
    out = inp.copy()
    for role in (0, 1):
        for obj in objects:
            if obj["role"] == role:
                for cell in _echo_cells(obj):
                    if _inside(cell, height, width) and inp[cell] == background:
                        out[cell] = colors["echo"][role]
    return inp, out


def _features(objects, height, width):
    echoes = [set(), set()]
    partial = False
    for obj in objects:
        cells = _echo_cells(obj)
        visible = {p for p in cells if _inside(p, height, width)}
        partial |= 0 < len(visible) < 3
        echoes[obj["role"]].update(visible)
    return partial, echoes[0] & echoes[1], echoes


def _patterns(roles, overlap, orientations):
    """Enumerate local, unobstructed echo overlaps without an anchor lattice."""
    result = []
    for m0 in orientations:
        for m1 in orientations:
            first = {"role": roles[0], "missing": m0, "top": (5, 5)}
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    second = {"role": roles[1], "missing": m1,
                              "top": (5 + dr, 5 + dc)}
                    if not _compatible(second, [first], 20, 20):
                        continue
                    if len(_echo_cells(first) & _echo_cells(second)) == overlap:
                        result.append((m0, m1, dr, dc))
    return result


def _add_pattern(rng, objects, height, width, roles, overlap, orientations):
    patterns = _patterns(roles, overlap, orientations)
    if not patterns:
        return None
    for index in rng.permutation(len(patterns)):
        m0, m1, dr, dc = patterns[int(index)]
        # Random translation over all positions, including borders.
        for _ in range(24):
            r = int(rng.integers(0, height - 1))
            c = int(rng.integers(0, width - 1))
            first = {"role": roles[0], "missing": m0, "top": (r, c)}
            second = {"role": roles[1], "missing": m1,
                      "top": (r + dr, c + dc)}
            if (_compatible(first, objects, height, width)
                    and _compatible(second, objects + [first], height, width)
                    and any(_inside(p, height, width)
                            for p in _echo_cells(first) & _echo_cells(second))):
                return [first, second]
    return None


def _add_edge(rng, objects, height, width, orientations):
    candidates = []
    for role in (0, 1):
        for missing in orientations:
            for r in range(height - 1):
                for c in range(width - 1):
                    if r not in (0, height - 2) and c not in (0, width - 2):
                        continue
                    obj = {"role": role, "missing": missing, "top": (r, c)}
                    visible = sum(_inside(p, height, width)
                                  for p in _echo_cells(obj))
                    if visible in (1, 2) and _compatible(obj, objects, height, width):
                        candidates.append(obj)
    if candidates:
        return candidates[int(rng.integers(len(candidates)))]
    return None


def _add_random(rng, objects, height, width, orientations, fixed_role=None,
                cluster=False):
    for _ in range(240):
        role = fixed_role if fixed_role is not None else int(rng.integers(0, 2))
        missing = orientations[int(rng.integers(len(orientations)))]
        if cluster and objects and rng.random() < 0.8:
            anchor = objects[int(rng.integers(len(objects)))]["top"]
            r = anchor[0] + int(rng.integers(-5, 6))
            c = anchor[1] + int(rng.integers(-5, 6))
        else:
            r = int(rng.integers(0, height - 1))
            c = int(rng.integers(0, width - 1))
        obj = {"role": role, "missing": missing, "top": (r, c)}
        if _compatible(obj, objects, height, width):
            return obj
    return None


def _shape(rng, scenario, regime):
    if scenario == 0:  # Original-style scale shift, with independent axes.
        low, high = (8, 14) if regime == "train" else (17, 29)
    elif scenario == 1:  # Comparable train and test canvases.
        low, high = 8, 27
    else:  # Large demonstrations, sometimes with a smaller test.
        low, high = (14, 31) if regime == "train" else (8, 23)
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _target_count(rng, height, width, minimum):
    density = float(rng.uniform(0.014, 0.058))
    maximum = 23
    desired = int(round(height * width * density)) + int(rng.integers(-1, 2))
    return max(minimum, min(maximum, desired))


def _make_pair(rng, colors, background, shape, orientations, regime,
               collision=0, clipping=False, same_merge=False,
               allowed_roles=(0, 1), cluster=False):
    height, width = shape
    for attempt in range(70):
        objects = []
        if collision:
            pair = _add_pattern(rng, objects, height, width, (0, 1),
                                collision, orientations)
            if pair is None:
                continue
            objects.extend(pair)
        if same_merge:
            role = int(rng.integers(0, 2))
            pair = _add_pattern(rng, objects, height, width, (role, role),
                                2, orientations)
            if pair is not None:
                objects.extend(pair)
        if clipping:
            edge = _add_edge(rng, objects, height, width, orientations)
            if edge is None:
                continue
            objects.append(edge)
        minimum = len(objects)
        target = _target_count(rng, height, width, minimum)
        # Dense placement has a bounded fallback to a lower object count.
        for _ in range(target - minimum):
            role = (allowed_roles[0] if len(allowed_roles) == 1 else None)
            obj = _add_random(rng, objects, height, width, orientations,
                              role, cluster)
            if obj is None:
                break
            objects.append(obj)
        if not objects:
            continue
        inp, out = _render(height, width, objects, colors, background)
        partial, collisions, echoes = _features(objects, height, width)
        if np.array_equal(inp, out):
            continue
        if collision and not collisions:
            continue
        if clipping and not partial:
            continue
        if regime == "train" and collision == 1 and (
                not echoes[0] - echoes[1] or not echoes[1] - echoes[0]):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not place a valid L-echo pair")


def _array_key(grid):
    return grid.shape, bytes(grid.astype(np.uint8, copy=False).ravel())


def _objects(inp, background, source):
    objects = []
    for role, color in enumerate(source):
        remaining = {tuple(p) for p in np.argwhere(inp == color)}
        while remaining:
            start = remaining.pop()
            component, pending = {start}, [start]
            while pending:
                r, c = pending.pop()
                for neighbor in ((r - 1, c), (r + 1, c),
                                 (r, c - 1), (r, c + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        pending.append(neighbor)
            _require(len(component) == 3, "source component is not three cells")
            r0, c0 = min(r for r, _ in component), min(c for _, c in component)
            normalized = {(r - r0, c - c0) for r, c in component}
            corners = {(0, 0), (0, 1), (1, 0), (1, 1)}
            _require(normalized < corners and len(corners - normalized) == 1,
                     "source component is not an L")
            mr, mc = next(iter(corners - normalized))
            objects.append({"role": role, "top": (r0, c0),
                            "missing": 2 * mr + mc})
    _require(int(np.count_nonzero(inp != background)) == 3 * len(objects),
             "input contains unrecognized colors")
    for i, obj in enumerate(objects):
        _require(_compatible(obj, objects[:i] + objects[i + 1:],
                             *inp.shape), "source objects touch or block echoes")
    return objects


_SOURCE_INPUTS = (
    ("77777777", "77772777", "75772277", "75577777",
     "77777777", "77777777", "77777277", "77772277"),
    ("77777777", "75777557", "75577757", "77777777",
     "77777777", "72277277", "77277227", "77777777"),
    ("7777777777777777", "7577777777777777", "7557777727755777",
     "7777755722775777", "7777775777777777", "7777777777772277",
     "7775775577777277", "7775577577777777", "7777777777722777",
     "7777722772772777", "7757772772277777", "7755777777777227",
     "7777777775777727", "7727722775577777", "7722772777772777",
     "7777777777772277"),
)


def _source_copy(inp):
    return tuple("".join(str(int(v)) for v in row) for row in inp) in _SOURCE_INPUTS


def _infer_background(inp):
    values, counts = np.unique(inp, return_counts=True)
    index = int(np.argmax(counts))
    _require(sum(counts == counts[index]) == 1, "background is ambiguous")
    return int(values[index])


def _solve(inp, background, source, echo):
    objects = _objects(inp, background, source)
    return _render(*inp.shape, objects,
                   {"source": source, "echo": echo}, background)[1], objects


def _analyze(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys must be train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "pair collections must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    records, seen = [], set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys must be input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer)
                         and all(1 <= n <= 30 for n in grid.shape)
                         and bool(np.all((grid >= 0) & (grid <= 9))),
                         "invalid ARC grid")
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "shape mismatch or unchanged output")
            key = _array_key(inp)
            _require(key not in seen and not _source_copy(inp),
                     "duplicate input or source-task copy")
            seen.add(key)
            background = _infer_background(inp)
            records.append((regime, inp, out, background))
    backgrounds = {r[3] for r in records}
    sources = {int(v) for regime, inp, _, bg in records if regime == "train"
               for v in np.unique(inp) if int(v) != bg}
    echoes = {int(v) for regime, _, out, _ in records if regime == "train"
              for v in np.unique(out)} - backgrounds - sources
    _require(len(sources) == len(echoes) == 2
             and not (backgrounds & (sources | echoes))
             and not (sources & echoes), "ambiguous color dictionary")
    candidates = []
    for source in ((min(sources), max(sources)), (max(sources), min(sources))):
        for echo in ((min(echoes), max(echoes)), (max(echoes), min(echoes))):
            try:
                if all(np.array_equal(_solve(inp, bg, source, echo)[0], out)
                       for regime, inp, out, bg in records if regime == "train"):
                    candidates.append((source, echo))
            except AssertionError:
                pass
    _require(len(candidates) == 1, "training does not uniquely reveal the rule")
    source, echo = candidates[0]
    analyzed, train_orientations = [], [set(), set()]
    train_clean = [set(), set()]
    train_collision = False
    for regime, inp, out, bg in records:
        _require(bg not in sources | echoes, "background collides with a role color")
        _require({int(v) for v in np.unique(inp)} <= {bg, *sources},
                 "input contains an unknown color")
        expected, objects = _solve(inp, bg, source, echo)
        _require(np.array_equal(out, expected), "incorrect L echo output")
        partial, collisions, projected = _features(objects, *inp.shape)
        if regime == "train":
            train_collision |= bool(collisions)
            for obj in objects:
                train_orientations[obj["role"]].add(obj["missing"])
            for role in (0, 1):
                train_clean[role].update(projected[role] - projected[1 - role])
        analyzed.append((regime, inp, out, bg, objects, partial, collisions))
    _require(train_collision and all(train_clean),
             "training must show collision priority and both clean echoes")
    for regime, _, _, _, objects, _, _ in analyzed:
        if regime == "test":
            for obj in objects:
                observed = train_orientations[obj["role"]]
                _require(obj["missing"] in observed or len(observed) >= 2,
                         "test orientation is unsupported by training")
    return analyzed, source, echo


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _analyze(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    chosen = [int(v) for v in rng.choice(10, size=4, replace=False)]
    colors = {"source": tuple(chosen[:2]), "echo": tuple(chosen[2:])}
    backgrounds = [v for v in range(10) if v not in chosen]
    rng.shuffle(backgrounds)
    scenario = int(rng.integers(0, 3))
    pool_size = int(rng.choice((1, 2, 3, 4), p=(0.17, 0.25, 0.25, 0.33)))
    orientations = tuple(int(v) for v in rng.choice(4, size=pool_size,
                                                    replace=False))
    collision_index = int(rng.integers(num_train))
    clip_index = int(rng.integers(num_train)) if rng.random() < 0.72 else -1
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            background = backgrounds[(index if regime == "train"
                                      else num_train + index) % len(backgrounds)]
            for _ in range(24):
                shape = _shape(pair_rng, scenario, regime)
                if regime == "train":
                    collision = 1 if index == collision_index else 0
                    clipping = index == clip_index
                    allowed_roles = ((int(pair_rng.integers(2)),)
                                     if collision == 0 and not clipping
                                     and pair_rng.random() < 0.28 else (0, 1))
                else:
                    collision = int(pair_rng.choice((0, 1, 3), p=(0.60, 0.25, 0.15)))
                    clipping = clip_index >= 0 and pair_rng.random() < 0.45
                    allowed_roles = ((int(pair_rng.integers(2)),)
                                     if collision == 0 and not clipping
                                     and pair_rng.random() < 0.22 else (0, 1))
                same_merge = (regime == "test" and collision == 0
                              and pair_rng.random() < 0.24)
                try:
                    pair = _make_pair(pair_rng, colors, background, shape,
                                      orientations, regime, collision, clipping,
                                      same_merge, allowed_roles,
                                      cluster=pair_rng.random() < 0.47)
                except RuntimeError:
                    continue
                if _array_key(pair["input"]) not in seen and not _source_copy(pair["input"]):
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen.add(_array_key(pair["input"]))
            puzzle[regime].append(pair)
    # The sampled orientation pool is a puzzle-level convention only for
    # construction. Restrict unseen test orientations when demonstrations
    # have shown just one orientation of a class.
    observed = [set(), set()]
    for pair in puzzle["train"]:
        bg = _infer_background(pair["input"])
        for obj in _objects(pair["input"], bg, colors["source"]):
            observed[obj["role"]].add(obj["missing"])
    for index, pair in enumerate(puzzle["test"]):
        bg = _infer_background(pair["input"])
        objs = _objects(pair["input"], bg, colors["source"])
        if any(obj["missing"] not in observed[obj["role"]]
               and len(observed[obj["role"]]) < 2 for obj in objs):
            # Regenerate with the demonstrated orientation subset.
            allowed = tuple(sorted(set.intersection(*(observed[r] for r in (0, 1)))))
            if not allowed:
                allowed = tuple(sorted(observed[0] | observed[1]))
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(32):
                shape = _shape(pair_rng, scenario, "test")
                try:
                    replacement = _make_pair(pair_rng, colors, bg, shape,
                                             allowed, "test", 0, False)
                except RuntimeError:
                    continue
                new_bg = _infer_background(replacement["input"])
                new_objs = _objects(replacement["input"], new_bg,
                                    colors["source"])
                if (all(o["missing"] in observed[o["role"]]
                        or len(observed[o["role"]]) >= 2 for o in new_objs)
                        and _array_key(replacement["input"]) not in seen):
                    seen.remove(_array_key(pair["input"]))
                    seen.add(_array_key(replacement["input"]))
                    puzzle["test"][index] = replacement
                    break
            else:
                raise RuntimeError("could not support test orientations")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_array_key(p["input"]), _array_key(p["output"]))
                 for regime in ("train", "test") for p in puzzle[regime])


def _check_rule_direct(inp, out, bg, objects, echo):
    """A second output construction used only by self_test."""
    expected = inp.copy()
    for role in (0, 1):
        for obj in objects:
            if obj["role"] != role:
                continue
            top_r, top_c = obj["top"]
            mr, mc = divmod(obj["missing"], 2)
            dy, dx = 2 * mr - 1, 2 * mc - 1
            if role == 1:
                dy, dx = -dy, -dx
            for y in (0, 1):
                for x in (0, 1):
                    if (y, x) == (mr, mc):
                        continue
                    row, col = top_r + y + dy, top_c + x + dx
                    if (0 <= row < inp.shape[0] and 0 <= col < inp.shape[1]
                            and inp[row, col] == bg):
                        expected[row, col] = echo[role]
    _require(np.array_equal(expected, out), "independent rule check failed")


def self_test() -> None:
    signatures, shapes, layouts, palettes = set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    backgrounds_vary = 0
    staggered = 0
    full_overlap = 0
    single_class = 0
    partial = 0
    interior_test = 0
    collision_free_test = 0
    same_merge_two = 0
    multi_collision = 0
    close_test = 0
    offset_13 = 0
    one_l = 0
    square8 = 0
    square16 = 0
    large_train = 0
    small_test = 0
    wide_or_tall = 0
    clip_without_collision = 0
    collision_without_clip = 0
    common_orientations = 0
    all_orientations = 0
    heldout = 0
    collision_indices = set()
    larger_test_puzzles = 0
    smaller_test_puzzles = 0
    partial_by_role = [0, 0]
    fully_clipped_reverse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        analyzed, source, echo = _analyze(puzzle)
        signatures.add(_puzzle_key(puzzle))
        palettes.add((source, echo))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        backgrounds_vary += len({x[3] for x in analyzed}) > 1
        train = analyzed[:len(puzzle["train"])]
        test = analyzed[len(puzzle["train"]):]
        trained = [set(), set()]
        for pair in train:
            for obj in pair[4]:
                trained[obj["role"]].add(obj["missing"])
        common_orientations += len(trained[0] | trained[1]) == 1
        all_orientations += len(trained[0] | trained[1]) == 4
        heldout += any(obj["missing"] not in trained[obj["role"]]
                       for pair in test for obj in pair[4])
        collision_indices.add(next(i for i, pair in enumerate(train)
                                   if pair[6]))
        larger_test_puzzles += any(pair[1].shape[0] > max(x[1].shape[0] for x in train)
                                   and pair[1].shape[1] > max(x[1].shape[1] for x in train)
                                   for pair in test)
        smaller_test_puzzles += any(pair[1].shape[0] < min(x[1].shape[0] for x in train)
                                    and pair[1].shape[1] < min(x[1].shape[1] for x in train)
                                    for pair in test)
        for regime, inp, out, bg, objects, clipped, collisions in analyzed:
            _check_rule_direct(inp, out, bg, objects, echo)
            for obj in objects:
                visible_count = sum(_inside(cell, *inp.shape)
                                    for cell in _echo_cells(obj))
                if visible_count in (1, 2):
                    partial_by_role[obj["role"]] += 1
                if obj["role"] == 1 and visible_count == 0:
                    fully_clipped_reverse += 1
            shapes.add(inp.shape)
            layouts.add((inp.shape, tuple(sorted((o["role"], o["top"],
                                                   o["missing"]) for o in objects))))
            partial += clipped
            single_class += len({o["role"] for o in objects}) == 1
            one_l += len(objects) == 1
            square8 += inp.shape == (8, 8)
            square16 += inp.shape == (16, 16)
            large_train += regime == "train" and max(inp.shape) >= 24
            small_test += regime == "test" and max(inp.shape) <= 14
            wide_or_tall += max(inp.shape) >= 2 * min(inp.shape)
            clip_without_collision += bool(clipped and not collisions)
            collision_without_clip += bool(collisions and not clipped)
            if regime == "test":
                close_test += any(abs(a["top"][0] - b["top"][0]) <= 2
                                  and abs(a["top"][1] - b["top"][1]) <= 2
                                  for i, a in enumerate(objects)
                                  for b in objects[i + 1:])
            cross_events = 0
            for i, a in enumerate(objects):
                for b in objects[i + 1:]:
                    overlap = (_echo_cells(a) & _echo_cells(b))
                    visible = {p for p in overlap if _inside(p, *inp.shape)}
                    if a["role"] == b["role"]:
                        same_merge_two += len(visible) == 2
                    if a["role"] != b["role"]:
                        if visible:
                            cross_events += 1
                            dr = abs(a["top"][0] - b["top"][0])
                            dc = abs(a["top"][1] - b["top"][1])
                            staggered += dr > 0 and dc > 0
                            offset_13 += (dr, dc) in ((1, 3), (3, 1))
                            full_overlap += len(visible) == 3
            multi_collision += cross_events >= 2
            if regime == "test":
                interior_test += not clipped
                collision_free_test += not collisions
    _require(len(signatures) == 42 and len(shapes) >= 35
             and len(layouts) >= 160 and len(palettes) >= 30,
             "structural diversity is insufficient")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "pair-count diversity is insufficient")
    _require(backgrounds_vary == 42 and staggered > 0
             and full_overlap > 0 and single_class > 0
             and partial > 0 and interior_test > 0
             and collision_free_test > 0,
             "reviewed variation is insufficient")
    _require(same_merge_two > 0 and multi_collision > 0
             and close_test > 0 and offset_13 > 0
             and one_l > 0 and square8 > 0 and square16 > 0
             and large_train > 0 and small_test > 0 and wide_or_tall > 0
             and clip_without_collision > 0 and collision_without_clip > 0
             and common_orientations > 0 and all_orientations > 0
             and heldout > 0 and len(collision_indices) >= 3
             and larger_test_puzzles > 0 and smaller_test_puzzles > 0
             and all(partial_by_role) and fully_clipped_reverse > 0,
             "a review-specific construction is absent")
    for seed, nt, ne in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, 7, 3), (201, None, 1), (202, None, 4),
                         (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "train count ignored")
        if ne is not None:
            _require(len(puzzle["test"]) == ne, "test count ignored")
    a = generate_puzzle(999, 8, 3)
    _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(999, 8, 3))
             and _puzzle_key(a) != _puzzle_key(generate_puzzle(1000, 8, 3)),
             "seed determinism failed")
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
