"""Generate paired binary-mask intersection puzzles."""

import numpy as np


_SOURCE = (
    (("1105010", "0015111", "1105010"), ("020", "002", "020")),
    (("1005010", "0105111", "1005000"), ("000", "020", "000")),
    (("0015000", "1105101", "0115101"), ("000", "200", "002")),
    (("1015101", "0105101", "1015010"), ("202", "000", "000")),
)


def _signature(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


_SOURCE_KEYS = frozenset(
    ((_signature(np.array([[int(c) for c in row] for row in a])),
      _signature(np.array([[int(c) for c in row] for row in b]))))
    for a, b in _SOURCE
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt is None and nq is None:
        nq = 1 + seed % 4
        nt = nq + 1 + (seed // 4) % (10 - nq)
    elif nt is None:
        nt = max(2, nq + 1) + seed % (11 - max(2, nq + 1))
    elif nq is None:
        nq = 1 + seed % min(4, nt - 1)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return nt, nq


def _dimensions(rng, orientation):
    tier = int(rng.choice(4, p=[0.24, 0.39, 0.24, 0.13]))
    if tier == 0:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    elif tier == 1:
        h, w = int(rng.integers(3, 11)), int(rng.integers(3, 11))
    elif tier == 2:
        h, w = int(rng.integers(7, 15)), int(rng.integers(7, 15))
    elif orientation == "vertical":
        h, w = int(rng.integers(15, 31)), int(rng.integers(2, 15))
    else:
        h, w = int(rng.integers(2, 15)), int(rng.integers(15, 31))
    if orientation == "vertical":
        w = min(w, 14)
    else:
        h = min(h, 14)
    return h, w


def _select(rng, shape, count, allowed=None, style=None):
    h, w = shape
    n = h * w
    if allowed is None:
        allowed = np.ones(n, dtype=bool)
    else:
        allowed = np.asarray(allowed, dtype=bool).reshape(n)
    indices = np.flatnonzero(allowed)
    count = min(max(0, int(count)), len(indices))
    result = np.zeros(n, dtype=bool)
    if count == 0:
        return result.reshape(shape)
    if count == len(indices):
        result[indices] = True
        return result.reshape(shape)
    style = style or str(rng.choice(["scatter", "run", "rectangle", "patch", "components"]))
    if style == "scatter":
        chosen = rng.choice(indices, size=count, replace=False)
    elif style == "patch":
        chosen_set = {int(rng.choice(indices))}
        while len(chosen_set) < count:
            frontier = set()
            for index in chosen_set:
                row, col = divmod(index, w)
                for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    ni = nr * w + nc
                    if 0 <= nr < h and 0 <= nc < w and allowed[ni] and ni not in chosen_set:
                        frontier.add(ni)
            choices = sorted(frontier) if frontier else [int(i) for i in indices if i not in chosen_set]
            chosen_set.add(int(rng.choice(choices)))
        chosen = list(chosen_set)
    else:
        rows, cols = np.divmod(indices, w)
        if style == "run":
            horizontal = bool(rng.integers(0, 2))
            anchor = int(rng.integers(0, h if horizontal else w))
            center = int(rng.integers(0, w if horizontal else h))
            distance = (abs(rows - anchor) * 4 + abs(cols - center) / max(2, w)) if horizontal else (abs(cols - anchor) * 4 + abs(rows - center) / max(2, h))
        elif style == "rectangle":
            cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
            ry, rx = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
            distance = np.maximum(abs(rows - cy) / ry, abs(cols - cx) / rx)
        else:
            centers = rng.choice(indices, size=min(len(indices), int(rng.integers(2, 5))), replace=False)
            distance = np.min([abs(rows - center // w) + abs(cols - center % w) for center in centers], axis=0)
        score = distance + rng.random(len(indices)) * 0.22
        chosen = indices[np.argsort(score)[:count]]
    result[chosen] = True
    return result.reshape(shape)


def _amount(rng, area, allow_zero=True):
    fraction = float(rng.choice([0.025, 0.08, 0.18, 0.32, 0.5, 0.7, 0.88, 1.0]))
    value = int(round(area * fraction * float(rng.uniform(0.78, 1.22))))
    return min(area, max(0 if allow_zero else 1, value))


def _overlap(rng, shape, mode, count=None):
    h, w = shape
    area = h * w
    if mode == "row":
        count = min(w, count if count is not None else int(rng.integers(1, w + 1)))
        allowed = np.zeros(shape, dtype=bool)
        allowed[int(rng.integers(0, h)), :] = True
        return _select(rng, shape, count, allowed, "run")
    if mode == "column":
        count = min(h, count if count is not None else int(rng.integers(1, h + 1)))
        allowed = np.zeros(shape, dtype=bool)
        allowed[:, int(rng.integers(0, w))] = True
        return _select(rng, shape, count, allowed, "run")
    if count is None:
        count = _amount(rng, area)
    return _select(rng, shape, count, style={"block": "rectangle", "scattered": "scatter", "patch": "patch"}.get(mode, mode))


def _masks(rng, shape, kind, forced=None):
    h, w = shape
    area = h * w
    if kind in ("informative", "overlap", "original_shift"):
        if forced is not None:
            both = forced.copy()
        else:
            mode = "row" if kind == "original_shift" else str(rng.choice(["row", "column", "block", "scattered", "patch"]))
            count = 2 if kind == "original_shift" else None
            if kind == "informative":
                count = int(rng.integers(1, min(area - 3, max(2, area // 3)) + 1))
            both = _overlap(rng, shape, mode, count)
        available = ~both
        left_count = max(1, min(int(available.sum()) - 2, _amount(rng, area) - int(both.sum()))) if kind == "informative" else min(int(available.sum()), max(0, _amount(rng, area) - int(both.sum())))
        left_only = _select(rng, shape, left_count, available)
        available &= ~left_only
        right_count = max(1, min(int(available.sum()) - 1, _amount(rng, area) - int(both.sum()))) if kind == "informative" else min(int(available.sum()), max(0, _amount(rng, area) - int(both.sum())))
        right_only = _select(rng, shape, right_count, available)
        return both | left_only, both | right_only
    if kind == "identical":
        mask = _select(rng, shape, _amount(rng, area))
        return mask, mask.copy()
    if kind == "disjoint":
        first = _select(rng, shape, min(area, _amount(rng, area)))
        second = _select(rng, shape, _amount(rng, area), ~first)
        return first, second
    if kind == "containment":
        large = _select(rng, shape, _amount(rng, area, False))
        small = _select(rng, shape, int(rng.integers(0, int(large.sum()) + 1)), large)
        return (small, large) if bool(rng.integers(0, 2)) else (large, small)
    if kind == "empty":
        other = _select(rng, shape, _amount(rng, area))
        return (np.zeros(shape, dtype=bool), other) if bool(rng.integers(0, 2)) else (other, np.zeros(shape, dtype=bool))
    if kind == "full":
        other = _select(rng, shape, _amount(rng, area))
        return (np.ones(shape, dtype=bool), other) if bool(rng.integers(0, 2)) else (other, np.ones(shape, dtype=bool))
    return (_select(rng, shape, _amount(rng, area)),
            _select(rng, shape, _amount(rng, area)))


def _assemble(first, second, orientation, background, active, divider, highlight):
    panel_a = np.where(first, active, background).astype(np.int8)
    panel_b = np.where(second, active, background).astype(np.int8)
    output = np.where(first & second, highlight, background).astype(np.int8)
    if orientation == "vertical":
        line = np.full((first.shape[0], 1), divider, dtype=np.int8)
        inp = np.concatenate((panel_a, line, panel_b), axis=1)
    else:
        line = np.full((1, first.shape[1]), divider, dtype=np.int8)
        inp = np.concatenate((panel_a, line, panel_b), axis=0)
    return {"input": inp, "output": output}


def _split(inp):
    h, w = inp.shape
    found = []
    if w >= 5 and w % 2:
        c = w // 2
        line = inp[:, c]
        if np.all(line == line[0]) and not np.any(inp[:, :c] == line[0]) and not np.any(inp[:, c + 1:] == line[0]):
            found.append(("vertical", inp[:, :c], inp[:, c + 1:], int(line[0])))
    if h >= 5 and h % 2:
        r = h // 2
        line = inp[r, :]
        if np.all(line == line[0]) and not np.any(inp[:r, :] == line[0]) and not np.any(inp[r + 1:, :] == line[0]):
            found.append(("horizontal", inp[:r, :], inp[r + 1:, :], int(line[0])))
    _check(len(found) == 1, "input must have one uniquely colored central divider")
    return found[0]


def _grid(grid):
    _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
    _check(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
    _check(all(1 <= n <= 30 for n in grid.shape), "grid dimension outside ARC bounds")
    _check(bool(np.all((0 <= grid) & (grid <= 9))), "grid color outside ARC palette")


def _learn(train):
    highlights, backgrounds, actives = set(), set(), set()
    cases = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        _, a, b, divider = _split(inp)
        _check(a.shape == out.shape == b.shape, "output size differs from panel")
        panel_colors = {int(v) for v in np.unique(np.concatenate((a, b)))}
        _check(len(panel_colors) <= 2 and divider not in panel_colors, "panel roles are invalid")
        output_colors = {int(v) for v in np.unique(out)}
        if len(panel_colors) == 2 and len(output_colors) == 2:
            common = panel_colors & output_colors
            if len(common) == 1:
                bg = next(iter(common))
                active = next(iter(panel_colors - {bg}))
                highlight = next(iter(output_colors - {bg}))
                _check(highlight not in panel_colors and highlight != divider, "highlight role collides")
                _check(np.array_equal(out, np.where((a == active) & (b == active), highlight, bg)), "training output breaks intersection rule")
                backgrounds.add(bg)
                actives.add(active)
                highlights.add(highlight)
                for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    if np.any((a == (active if left else bg)) & (b == (active if right else bg))):
                        cases.add((left, right))
    _check(len(highlights) == 1, "training does not establish a unique highlight")
    _check(backgrounds and actives, "training does not expose both panel roles")
    _check(len(backgrounds) == 1 or len(actives) == 1, "both panel roles vary")
    _check(cases == {(0, 0), (0, 1), (1, 0), (1, 1)}, "training does not expose the AND truth table")
    return backgrounds, actives, next(iter(highlights))


def _predict(roles, inp):
    backgrounds, actives, highlight = roles
    _, a, b, divider = _split(inp)
    panel_colors = {int(v) for v in np.unique(np.concatenate((a, b)))}
    _check(len(panel_colors) <= 2 and divider not in panel_colors and highlight not in panel_colors, "invalid panel palette")
    candidates = []
    for bg in backgrounds:
        for active in actives:
            if bg != active and panel_colors <= {bg, active} and divider not in (bg, active, highlight):
                candidates.append(np.where((a == active) & (b == active), highlight, bg).astype(np.int8))
    _check(bool(candidates), "test uses a role absent from demonstrations")
    _check(all(np.array_equal(candidates[0], p) for p in candidates[1:]), "test output is ambiguous")
    return candidates[0]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0, 2**63 - 1)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(v) for v in rng.permutation(10)]
    color_mode = str(rng.choice(["stable", "active", "background"], p=[0.32, 0.35, 0.33]))
    role_count = min(train_count, 3) if color_mode != "stable" else 1
    if color_mode == "active":
        bgs, acts, highlight = [colors[0]], colors[2:2 + role_count], colors[1]
        dividers = colors[2 + role_count:5 + role_count]
    elif color_mode == "background":
        bgs, acts, highlight = colors[2:2 + role_count], [colors[0]], colors[1]
        dividers = colors[2 + role_count:5 + role_count]
    else:
        bgs, acts, highlight = [colors[0]], [colors[1]], colors[2]
        dividers = colors[3:6]
    layout_mode = str(rng.choice(["uniform", "transfer", "mixed"], p=[0.2, 0.24, 0.56]))
    initial_orientation = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
    uniform_shape = _dimensions(rng, initial_orientation)
    puzzle = {"train": [], "test": []}
    seen = set()
    repeated = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if layout_mode == "uniform":
                orientation, shape = initial_orientation, uniform_shape
            elif layout_mode == "transfer":
                orientation = initial_orientation if split == "train" else ("horizontal" if initial_orientation == "vertical" else "vertical")
                shape = _dimensions(rng, orientation)
            else:
                orientation = (initial_orientation if index == 0 else ("horizontal" if initial_orientation == "vertical" else "vertical")) if split == "train" and index < 2 else str(rng.choice(["vertical", "horizontal"]))
                shape = _dimensions(rng, orientation)
                if split == "train" and index == 1 and shape == uniform_shape:
                    shape = _dimensions(rng, orientation)
            if split == "train":
                bg = bgs[index % len(bgs)]
                active = acts[index % len(acts)]
            else:
                bg = int(rng.choice(bgs))
                active = int(rng.choice(acts))
            divider = int(rng.choice(dividers))
            kind = "informative" if split == "train" and index < max(len(bgs), len(acts), 2) else str(rng.choice(
                ["independent", "overlap", "disjoint", "identical", "containment", "empty", "full", "original_shift"],
                p=[0.31, 0.22, 0.10, 0.09, 0.10, 0.06, 0.06, 0.06] if split == "train" else [0.25, 0.21, 0.12, 0.10, 0.11, 0.07, 0.07, 0.07]))
            forced = None
            if split == "train" and index == train_count - 1 and repeated is not None and len(bgs) == 1 and train_count >= 4 and rng.random() < 0.5:
                orientation, shape, forced = repeated
                kind = "overlap"
            for attempt in range(80):
                first, second = _masks(rng, shape, kind, forced)
                if kind == "informative" and len({(bool(x), bool(y)) for x, y in zip(first.flat, second.flat)}) < 4:
                    continue
                pair = _assemble(first, second, orientation, bg, active, divider, highlight)
                key = (_signature(pair["input"]), _signature(pair["output"]))
                if key[0] not in seen and key not in _SOURCE_KEYS:
                    break
            else:
                raise RuntimeError("could not generate a distinct pair")
            seen.add(key[0])
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                repeated = (orientation, shape, first & second)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys are invalid")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "pair counts invalid")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys invalid")
            inp, out = pair["input"], pair["output"]
            _grid(inp)
            _grid(out)
            _, a, b, divider = _split(inp)
            _check(a.shape == b.shape == out.shape, "output must match panel dimensions")
            _check(divider not in np.unique(out), "divider appears in output")
            key = (_signature(inp), _signature(out))
            _check(key[0] not in seen_inputs and key not in seen_pairs and key not in _SOURCE_KEYS, "duplicate or source pair")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
    roles = _learn(puzzle["train"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(np.array_equal(_predict(roles, pair["input"]), pair["output"]), "pair breaks learned intersection rule")


def _fingerprint(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
    return count


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    dimensions, orientations, overlap_counts, palettes = set(), set(), set(), set()
    row_pairs = column_pairs = empty_pairs = identical_pairs = 0
    mixed_orientations = mixed_dimensions = 0
    color_modes = set()
    varied_dividers = uniform_layouts = test_original_shift = 0
    vertical_to_horizontal = horizontal_to_vertical = 0
    sparse_panels = dense_panels = unequal_pairs = full_panels = 0
    contained_pairs = repeated_results = multicomponent_overlaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        local_shapes, local_orientations = set(), set()
        train_orientations, test_orientations, divider_colors = set(), set(), set()
        outputs = {}
        roles = _learn(puzzle["train"])
        color_modes.add("active" if len(roles[1]) > 1 else "background" if len(roles[0]) > 1 else "stable")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                orientation, a, b, divider = _split(inp)
                divider_colors.add(divider)
                local_orientations.add(orientation)
                (train_orientations if split == "train" else test_orientations).add(orientation)
                local_shapes.add(out.shape)
                dimensions.add(out.shape)
                orientations.add(orientation)
                palettes.add(tuple(int(v) for v in np.unique(inp)))
                _check(np.array_equal(_predict(roles, inp), out), "training-based prediction failed")
                overlap = out == roles[2]
                n = int(overlap.sum())
                colors = {int(v) for v in np.unique(np.concatenate((a, b)))}
                candidates = [(bg, active) for bg in roles[0] for active in roles[1] if colors <= {bg, active}]
                _check(bool(candidates), "panel roles cannot be reconstructed")
                _, active = candidates[0]
                left, right = a == active, b == active
                area = left.size
                sparse_panels += int(left.mean() < 0.1) + int(right.mean() < 0.1)
                dense_panels += int(left.mean() > 0.7) + int(right.mean() > 0.7)
                unequal_pairs += int(abs(left.mean() - right.mean()) > 0.4)
                full_panels += int(left.sum() == area) + int(right.sum() == area)
                contained_pairs += int(np.all(left <= right) or np.all(right <= left))
                multicomponent_overlaps += int(_component_count(overlap) >= 2)
                output_key = _signature(out)
                repeated_results += int(output_key in outputs and outputs[output_key] != _signature(inp))
                outputs[output_key] = _signature(inp)
                overlap_counts.add(n)
                row_pairs += int(n >= 2 and np.max(overlap.sum(axis=1)) >= 2)
                column_pairs += int(n >= 2 and np.max(overlap.sum(axis=0)) >= 2)
                empty_pairs += int(n == 0)
                identical_pairs += int(np.array_equal(a, b))
                if split == "test":
                    test_original_shift += int(n == 2 and np.max(overlap.sum(axis=1)) == 2)
        mixed_orientations += int(len(local_orientations) > 1)
        mixed_dimensions += int(len(local_shapes) > 1)
        varied_dividers += int(len(divider_colors) > 1)
        uniform_layouts += int(len(local_orientations) == 1 and len(local_shapes) == 1)
        vertical_to_horizontal += int(train_orientations == {"vertical"} and test_orientations == {"horizontal"})
        horizontal_to_vertical += int(train_orientations == {"horizontal"} and test_orientations == {"vertical"})
    _check(len(train_counts) >= 4 and len(test_counts) >= 3 and len(fingerprints) >= 40, "automatic-seed diversity failed")
    _check(len(dimensions) >= 35 and orientations == {"horizontal", "vertical"} and len(palettes) >= 30, "layout or palette diversity failed")
    _check(mixed_orientations >= 12 and mixed_dimensions >= 20, "within-puzzle layout diversity failed")
    _check(len(overlap_counts) >= 10 and row_pairs >= 10 and column_pairs >= 10 and empty_pairs >= 5 and identical_pairs >= 5, "mask diversity failed")
    _check(color_modes == {"active", "background", "stable"} and varied_dividers >= 15, "color-role diversity failed")
    _check(uniform_layouts >= 2 and vertical_to_horizontal >= 2 and horizontal_to_vertical >= 2, "layout regime diversity failed")
    _check(sparse_panels >= 30 and dense_panels >= 30 and unequal_pairs >= 20, "mask density diversity failed")
    _check(full_panels >= 5 and contained_pairs >= 15 and repeated_results >= 2 and multicomponent_overlaps >= 10 and test_original_shift >= 2, "boundary and test-regime diversity failed")
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(123456, *counts)
        b = generate_puzzle(123456, *counts)
        _check(_fingerprint(a) == _fingerprint(b), "requested counts are nondeterministic")
        validate_generated_puzzle(a)
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(10, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
