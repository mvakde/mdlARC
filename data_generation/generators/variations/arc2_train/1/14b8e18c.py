import numpy as np
from collections import Counter


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_INPUTS = {
    ("7777777777", "7666677767", "7677677677", "7677677777", "7666677777", "7777777777", "7777777777", "7767766667", "7677766667", "7777777777"),
    ("7777777777", "7777777777", "7766677677", "7767677677", "7766677677", "7777777777", "7777777777", "7766677777", "7777777777", "7777777777"),
    ("7777777777", "7888888887", "7877777787", "7877777787", "7877887787", "7877887787", "7877777787", "7877777787", "7888888887", "7777777777"),
    ("7777777777", "7666776677", "7676776677", "7666777777", "7777777777", "7777766677", "7667767677", "7667766677", "7777777777", "7666666667"),
}


def _counts(rng, seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    answer = []
    for raw_r, raw_c in np.argwhere(mask):
        start = (int(raw_r), int(raw_c))
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        cells = set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        answer.append(cells)
    return answer


def _record(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    full = len(cells) == h * w
    border = len(cells) == 2 * h + 2 * w - 4 and all(r in (r0, r1) or c in (c0, c1) for r, c in cells)
    eligible = h == w and h >= 2 and (full or border)
    return {"cells": cells, "r0": r0, "r1": r1, "c0": c0, "c1": c1, "height": h, "width": w,
            "kind": ("solid" if full else "frame") if eligible else None, "side": h if eligible else None,
            "eligible": eligible, "full": full, "border": border}


def _analyze(grid, background):
    return [_record(c) for c in _components(grid != background)]


def _sites(record):
    a, b, x, y = record["r0"], record["r1"], record["c0"], record["c1"]
    return {(a - 1, x), (a - 1, y), (b + 1, x), (b + 1, y),
            (a, x - 1), (b, x - 1), (a, y + 1), (b, y + 1)}


def _square(kind, side):
    if kind == "solid":
        return {(r, c) for r in range(side) for c in range(side)}
    return {(r, c) for r in range(side) for c in range(side)
            if r in (0, side - 1) or c in (0, side - 1)}


def _distractor(rng, category=None):
    if category is None:
        category = ("line", "filled_rect", "hollow_rect", "broken_frame", "irregular", "singleton", "diagonal", "tee")[int(rng.integers(0, 8))]
    if category == "singleton":
        return {(0, 0)}
    if category == "diagonal":
        return {(0, 0), (1, 1)}
    if category == "line":
        length = int(rng.integers(2, 11))
        return {(0, c) for c in range(length)} if rng.random() < .5 else {(r, 0) for r in range(length)}
    if category in ("filled_rect", "hollow_rect"):
        h = int(rng.integers(2 if category == "filled_rect" else 3, 7))
        w = int(rng.integers(2 if category == "filled_rect" else 3, 9))
        if h == w:
            w = w + 1 if w < 9 else w - 1
        return {(r, c) for r in range(h) for c in range(w)
                if category == "filled_rect" or r in (0, h - 1) or c in (0, w - 1)}
    if category == "broken_frame":
        side = int(rng.integers(3, 8))
        cells = _square("frame", side)
        edge = int(rng.integers(0, 4))
        pos = int(rng.integers(1, side - 1))
        cells.remove(((0, pos), (side - 1, pos), (pos, 0), (pos, side - 1))[edge])
        return cells
    if category == "irregular":
        side = int(rng.integers(3, 8))
        cells = {(r, 0) for r in range(side)} | {(side - 1, c) for c in range(side)}
        if rng.random() < .5:
            cells |= {(r, int(rng.integers(1, side))) for r in range(1, side - 1)}
        return _rotate(cells, int(rng.integers(0, 4)))
    width = int(rng.integers(3, 8))
    height = int(rng.integers(2, 7))
    cells = {(0, c) for c in range(width)} | {(r, width // 2) for r in range(height)}
    return _rotate(cells, int(rng.integers(0, 4)))


def _rotate(cells, turns):
    result = set(cells)
    for _ in range(turns):
        result = {(c, -r) for r, c in result}
        mr = min(r for r, _ in result)
        mc = min(c for _, c in result)
        result = {(r - mr, c - mc) for r, c in result}
    return result


def _try_add(state, cells, target=False):
    h, w = state["shape"]
    occupied = state["occupied"]
    reserved = state["reserved"]
    if any(not (0 <= r < h and 0 <= c < w) or (r, c) in occupied or (r, c) in reserved for r, c in cells):
        return False
    if any((r + dr, c + dc) in occupied for r, c in cells for dr, dc in _NEIGHBORS):
        return False
    marks = _sites(_record(cells)) if target else set()
    if any(not (0 <= r < h and 0 <= c < w) or (r, c) in occupied or (r, c) in cells for r, c in marks):
        return False
    if target and not _record(cells)["eligible"]:
        return False
    if not target and any(record["eligible"] for record in (_record(c) for c in _components_from_cells(cells))):
        return False
    state["objects"].append((cells, target))
    occupied.update(cells)
    reserved.update(marks)
    return True


def _components_from_cells(cells):
    remaining = set(cells)
    answer = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    stack.append(nxt)
        answer.append(component)
    return answer


def _place(state, rng, local, target=False, bounds=None):
    h, w = state["shape"]
    bh = max(r for r, _ in local) + 1
    bw = max(c for _, c in local) + 1
    if bounds is None:
        bounds = (0, h - bh, 0, w - bw)
    rlo, rhi, clo, chi = bounds
    rlo, clo = max(0, rlo), max(0, clo)
    rhi, chi = min(h - bh, rhi), min(w - bw, chi)
    if rlo > rhi or clo > chi:
        return None
    positions = [(r, c) for r in range(rlo, rhi + 1) for c in range(clo, chi + 1)]
    for idx in rng.permutation(len(positions)):
        top, left = positions[int(idx)]
        translated = {(top + r, left + c) for r, c in local}
        if _try_add(state, translated, target):
            return (top, left)
    return None


def _new_state(h, w):
    return {"shape": (h, w), "objects": [], "occupied": set(), "reserved": set()}


def _choose_dimensions(rng, profile):
    if profile == "intro":
        lo, hi = 12, 23
    elif profile == "mixed":
        lo, hi = 15, 27
    else:
        special = rng.random()
        if special < .055:
            return 10, 10
        if special < .075:
            return 30, 30
        band = int(rng.choice(3, p=[.29, .46, .25]))
        lo, hi = ((6, 12), (11, 23), (22, 31))[band]
    h = int(rng.integers(lo, hi))
    w = int(rng.integers(lo, hi))
    return h, w


def _choose_side(rng, h, w, kind):
    maximum = min(h, w) - 2
    choices = [s for s in range(2, min(14, maximum) + 1)]
    if not choices:
        return 0
    weights = np.array([6 if s <= 3 else 5 if s <= 5 else 3 if s <= 8 else 1 for s in choices], float)
    if kind == "solid":
        weights *= np.array([1 if s <= 6 else .25 for s in choices])
    weights /= weights.sum()
    return int(rng.choice(choices, p=weights))


def _add_nested(state, rng, mandatory_solid=None):
    h, w = state["shape"]
    if mandatory_solid is not None:
        inner_side = mandatory_solid
    elif min(h, w) >= 11 and rng.random() < .55:
        inner_side = int(rng.integers(6, min(8, min(h, w) - 5) + 1))
    else:
        inner_side = int(rng.integers(2, 5))
    outer_side = int(rng.integers(inner_side + 4, min(15, min(h, w) - 1) + 1)) if min(h, w) >= inner_side + 5 else 0
    if not outer_side:
        return 0
    outer = _place(state, rng, _square("frame", outer_side), True)
    if outer is None:
        return 0
    r0, c0 = outer
    inner_kind = "solid" if mandatory_solid is not None or (inner_side < 6 and rng.random() < .62) or (inner_side >= 6 and rng.random() < .20) else "frame"
    inner = _place(state, rng, _square(inner_kind, inner_side), True,
                   (r0 + 2, r0 + outer_side - inner_side - 2,
                    c0 + 2, c0 + outer_side - inner_side - 2))
    if inner is None:
        return -1
    count = 2
    if inner_kind == "frame" and inner_side >= 6 and rng.random() < .82:
        rs, cs = inner
        if _place(state, rng, _square("solid", 2), True,
                  (rs + 2, rs + inner_side - 4, cs + 2, cs + inner_side - 4)) is not None:
            count += 1
    if outer_side >= 10 and rng.random() < .55:
        extra_kind = "solid" if rng.random() < .6 else "frame"
        if _place(state, rng, _square(extra_kind, 2), True,
                  (r0 + 2, r0 + outer_side - 4, c0 + 2, c0 + outer_side - 4)) is not None:
            count += 1
    if outer_side >= 9 and rng.random() < .4:
        _place(state, rng, _distractor(rng, "singleton"), False,
               (r0 + 1, r0 + outer_side - 2, c0 + 1, c0 + outer_side - 2))
    return count


def _add_shared(state, rng, mandatory_solid=None):
    h, w = state["shape"]
    side1 = int(rng.integers(2, 5))
    side2 = mandatory_solid if mandatory_solid is not None else int(rng.integers(2, 5))
    if min(h, w) < side1 + side2 + 2:
        return 0
    positions = [(r, c) for r in range(1, h - side1 - side2)
                 for c in range(1, w - side1 - side2)]
    for idx in rng.permutation(len(positions)):
        r, c = positions[int(idx)]
        first = {(r + dr, c + dc) for dr, dc in _square("frame", side1)}
        second = {(r + side1 + dr, c + side1 + dc) for dr, dc in _square("solid", side2)}
        trial = {"shape": state["shape"], "objects": list(state["objects"]),
                 "occupied": set(state["occupied"]), "reserved": set(state["reserved"])}
        if _try_add(trial, first, True) and _try_add(trial, second, True):
            state.update(trial)
            return 2
    return 0


def _construct(rng, profile, mode, target_goal, highlight, background):
    h, w = _choose_dimensions(rng, profile)
    state = _new_state(h, w)
    count = 0
    if profile == "intro":
        side = int(rng.integers(3, 7))
        if _place(state, rng, _square("frame", side), True) is None:
            return None
        count = 1
    elif profile == "mixed":
        side = int(rng.integers(3, 6))
        if mode == "nested":
            count = _add_nested(state, rng, side)
        elif mode == "shared":
            count = _add_shared(state, rng, side)
        if count <= 0:
            count = 0
            state = _new_state(h, w)
            if _place(state, rng, _square("solid", side), True) is None:
                return None
            count = 1
            if _place(state, rng, _square("frame", int(rng.integers(3, 7))), True) is not None:
                count += 1
        if not any(_record(c)["kind"] == "solid" and _record(c)["side"] >= 3 for c, t in state["objects"] if t):
            return None
    elif target_goal and mode == "nested":
        count = _add_nested(state, rng)
        if count < 0:
            return None
    elif target_goal and mode == "shared":
        count = _add_shared(state, rng)
    while count < target_goal:
        kind = "frame" if rng.random() < .52 else "solid"
        side = _choose_side(rng, h, w, kind)
        if not side or _place(state, rng, _square(kind, side), True) is None:
            break
        count += 1
    if count < target_goal:
        return None
    if profile == "intro":
        distractor_plan = ["hollow_rect" if rng.random() < .5 else "broken_frame"]
        if rng.random() < .6:
            distractor_plan.append("singleton" if rng.random() < .5 else "line")
    elif profile == "mixed":
        distractor_plan = ["irregular" if rng.random() < .5 else "broken_frame"]
        if rng.random() < .55:
            distractor_plan.append("filled_rect")
    else:
        n = int(rng.choice(5, p=[.23, .30, .25, .16, .06]))
        if target_goal == 0:
            n = max(1, n)
        distractor_plan = [None] * n
    for category in distractor_plan:
        placed = False
        for _ in range(5):
            shape = _distractor(rng, category)
            if _place(state, rng, shape) is not None:
                placed = True
                break
        if not placed and profile in ("intro", "mixed"):
            return None
    if len(state["occupied"]) * 2 >= h * w:
        return None
    palette = [x for x in range(10) if x not in (background, highlight)]
    if len(state["objects"]) > 1 and rng.random() < .63:
        colors = [int(x) for x in rng.choice(palette, size=min(len(state["objects"]), int(rng.integers(2, 5))), replace=False)]
        assignments = [colors[i % len(colors)] for i in range(len(state["objects"]))]
        rng.shuffle(assignments)
    else:
        assignments = [int(rng.choice(palette))] * len(state["objects"])
    grid = np.full((h, w), background, dtype=np.int8)
    for (cells, _), color in zip(state["objects"], assignments):
        for r, c in cells:
            grid[r, c] = color
    records = _analyze(grid, background)
    eligible = [rec for rec in records if rec["eligible"]]
    if len(eligible) != count:
        return None
    sites = set().union(*(_sites(rec) for rec in eligible)) if eligible else set()
    if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != background for r, c in sites):
        return None
    output = grid.copy()
    for r, c in sites:
        output[r, c] = highlight
    return {"input": grid, "output": output}


def _pair_key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, seed, num_train, num_test)
    highlight = int(rng.integers(0, 10))
    backgrounds = [int(x) for x in rng.choice([x for x in range(10) if x != highlight], size=2, replace=False)]
    intro_goal = int(rng.choice((1, 2, 3), p=[.55, .32, .13]))
    mixed_goal = int(rng.choice((2, 3, 4), p=[.48, .37, .15]))
    if mixed_goal == intro_goal:
        mixed_goal += 1
    puzzle = {"train": [], "test": []}
    seen = set()
    interactions = set()
    for section, total in (("train", ntrain), ("test", ntest)):
        for index in range(total):
            profile = "intro" if section == "train" and index == 0 else "mixed" if section == "train" and index == 1 else "free"
            if profile == "mixed":
                mode = str(rng.choice(("nested", "shared", "separate"), p=[.43, .36, .21]))
                goal = mixed_goal
            else:
                modes = ("nested", "shared", "separate") if section == "train" else tuple(sorted(interactions)) + ("separate", "separate")
                mode = str(rng.choice(modes)) if profile == "free" else "separate"
                if profile == "intro":
                    goal = intro_goal
                elif section == "train":
                    goal = int(rng.choice((0, 1, 2, 3, 4, 5), p=[.10, .23, .25, .22, .14, .06]))
                else:
                    goal = int(rng.choice((1, 2, 3, 4, 5, 6), p=[.13, .18, .24, .23, .16, .06]))
                if goal < 2:
                    mode = "separate"
            background = backgrounds[index] if section == "train" and index < 2 else int(rng.choice([x for x in range(10) if x != highlight]))
            for _ in range(110):
                local = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _construct(local, profile, mode, goal, highlight, background)
                if pair is None:
                    continue
                key = _pair_key(pair["input"])
                if key in seen or _is_source_input(pair["input"]):
                    continue
                if section == "train" and index == 1:
                    first = puzzle["train"][0]["input"]
                    first_colors = set(int(x) for x in np.unique(first) if int(x) != backgrounds[0])
                    current_colors = set(int(x) for x in np.unique(pair["input"]) if int(x) != background)
                    if first_colors == current_colors:
                        continue
                if section == "test":
                    trial_targets = [r for r in _analyze(pair["input"], background) if r["eligible"]]
                    if (_has_nested(trial_targets) and "nested" not in interactions) or (_has_shared(trial_targets) and "shared" not in interactions):
                        continue
                seen.add(key)
                puzzle[section].append(pair)
                if section == "train":
                    records = [r for r in _analyze(pair["input"], background) if r["eligible"]]
                    if _has_nested(records):
                        interactions.add("nested")
                    if _has_shared(records):
                        interactions.add("shared")
                break
            else:
                raise RuntimeError(f"could not construct {section} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _is_source_input(grid):
    if grid.shape != (10, 10):
        return False
    return tuple("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUTS


def _has_nested(records):
    return any(a is not b and a["kind"] == "frame" and a["r0"] < b["r0"] and b["r1"] < a["r1"] and a["c0"] < b["c0"] and b["c1"] < a["c1"] for a in records for b in records)


def _has_shared(records):
    return any(not _sites(a).isdisjoint(_sites(b)) for i, a in enumerate(records) for b in records[i + 1:])


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _role_info(pair):
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() > inp.size / 2
    differences = inp != out
    assert all(inp[r, c] == bg for r, c in np.argwhere(differences))
    changed_colors = {int(x) for x in np.unique(out[differences])}
    assert len(changed_colors) <= 1
    color = next(iter(changed_colors)) if changed_colors else None
    if color is not None:
        assert color not in values
    return bg, color


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    highlight = None
    training_features = {"frame": False, "large_solid": False, "nested": False, "shared": False, "negative": False}
    pair_analysis = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and not _is_source_input(inp)
            key = _pair_key(inp)
            assert key not in seen
            seen.add(key)
            bg, mark = _role_info(pair)
            records = _analyze(inp, bg)
            assert all(len({int(inp[r, c]) for r, c in record["cells"]}) == 1 for record in records)
            targets = [r for r in records if r["eligible"]]
            expected_sites = set().union(*(_sites(r) for r in targets)) if targets else set()
            assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and inp[r, c] == bg for r, c in expected_sites)
            actual_sites = {(int(r), int(c)) for r, c in np.argwhere(inp != out)}
            assert actual_sites == expected_sites
            if actual_sites:
                assert mark is not None
                if highlight is None:
                    highlight = mark
                assert mark == highlight
            else:
                assert np.array_equal(inp, out) and section == "train"
            if section == "test":
                assert targets
            if section == "train":
                training_features["frame"] |= any(r["kind"] == "frame" for r in targets)
                training_features["large_solid"] |= any(r["kind"] == "solid" and r["side"] >= 3 for r in targets)
                training_features["nested"] |= _has_nested(targets)
                training_features["shared"] |= _has_shared(targets)
                training_features["negative"] |= any(not r["eligible"] for r in records)
            pair_analysis.append((section, inp, bg, targets, records))
    assert highlight is not None and training_features["frame"] and training_features["large_solid"] and training_features["negative"]
    for _, inp, _, _, _ in pair_analysis:
        assert highlight not in inp
    for section, _, _, targets, _ in pair_analysis:
        if section == "test":
            assert not _has_nested(targets) or training_features["nested"]
            assert not _has_shared(targets) or training_features["shared"]


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    learned = {int(color) for pair in train_pairs
               for color in np.unique(pair["output"][pair["input"] != pair["output"]])}
    assert len(learned) == 1
    highlight = next(iter(learned))
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert highlight not in values
    answer = test_input.copy()
    for cells in _components(test_input != background):
        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells)
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)
        side = bottom - top + 1
        if side < 2 or side != right - left + 1:
            continue
        filled = len(cells) == side * side
        perimeter = {(r, c) for r in range(top, bottom + 1)
                     for c in range(left, right + 1)
                     if r in (top, bottom) or c in (left, right)}
        if not filled and cells != perimeter:
            continue
        extensions = ((top - 1, left), (top - 1, right),
                      (bottom + 1, left), (bottom + 1, right),
                      (top, left - 1), (bottom, left - 1),
                      (top, right + 1), (bottom, right + 1))
        for r, c in extensions:
            assert 0 <= r < answer.shape[0] and 0 <= c < answer.shape[1]
            assert test_input[r, c] == background
            answer[r, c] = highlight
    return answer


def self_test() -> None:
    puzzles = [generate_puzzle(s) for s in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _equal_puzzles(generate_puzzle(1), generate_puzzle(2))
    dimensions, counts, kinds, colors, regimes = set(), set(), set(), set(), set()
    review = Counter()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first_two = puzzle["train"][:2]
        first_roles = [_role_info(pair)[0] for pair in first_two]
        assert first_roles[0] != first_roles[1]
        first_foregrounds = [{int(x) for x in np.unique(pair["input"]) if int(x) != bg}
                             for pair, bg in zip(first_two, first_roles)]
        assert first_foregrounds[0] != first_foregrounds[1]
        split_counts = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg, mark = _role_info(pair)
                records = _analyze(inp, bg)
                targets = [r for r in records if r["eligible"]]
                negatives = [r for r in records if not r["eligible"]]
                split_counts[section].append(len(targets))
                sites = set().union(*(_sites(r) for r in targets)) if targets else set()
                assert sites == {(int(r), int(c)) for r, c in np.argwhere(inp != out)}
                if section == "test":
                    assert np.array_equal(_predict_from_training(puzzle["train"], inp), out)
                dimensions.add(inp.shape)
                counts.add(len(targets))
                kinds.update((r["kind"], r["side"]) for r in targets)
                colors.add((bg, mark))
                regimes.add((section, len(targets), _has_nested(targets), _has_shared(targets)))
                foreground_colors = {int(x) for x in np.unique(inp) if int(x) != bg}
                target_colors = {int(inp[next(iter(r["cells"]))]) for r in targets}
                negative_colors = {int(inp[next(iter(r["cells"]))]) for r in negatives}
                density = np.count_nonzero(inp != bg) / inp.size
                review["multicolor"] += len(foreground_colors) >= 2
                review["multicolor_targets"] += len(target_colors) >= 2
                review["same_role_color"] += bool(target_colors & negative_colors)
                review["different_role_color"] += bool(target_colors and negative_colors and target_colors.isdisjoint(negative_colors))
                review["small"] += min(inp.shape) <= 9
                review["large"] += max(inp.shape) >= 25
                review["source_scale"] += inp.shape == (10, 10)
                review["max_board"] += inp.shape == (30, 30)
                review["rectangular"] += inp.shape[0] != inp.shape[1]
                review["dense"] += density >= .30
                review["compact_multi"] += min(inp.shape) <= 12 and len(targets) >= 2
                review["many_targets"] += len(targets) >= 4
                review["all_frames"] += bool(targets) and all(r["kind"] == "frame" for r in targets)
                review["all_solids"] += bool(targets) and all(r["kind"] == "solid" for r in targets)
                review["mixed"] += {r["kind"] for r in targets} == {"solid", "frame"}
                review["no_distractors"] += bool(targets) and not negatives
                review["unchanged"] += not targets
                review[section + "_nested"] += _has_nested(targets)
                review[section + "_shared"] += _has_shared(targets)
                review["nested_multicolor"] += _has_nested(targets) and len(foreground_colors) >= 2
                review["shared_negative"] += _has_shared(targets) and bool(negatives)
                review["diagonal_target_contact"] += any(any(abs(r - rr) == 1 and abs(c - cc) == 1 for r, c in a["cells"] for rr, cc in b["cells"]) for i, a in enumerate(targets) for b in targets[i + 1:])
                if section == "train":
                    review["separate_solid"] += any(r["kind"] == "solid" and not any(f["kind"] == "frame" and f["r0"] < r["r0"] and r["r1"] < f["r1"] and f["c0"] < r["c0"] and r["c1"] < f["c1"] for f in targets) for r in targets)
                for target in targets:
                    review["side_six"] += target["side"] == 6
                    review["large_solid"] += target["kind"] == "solid" and target["side"] >= 3
                    review["edge_highlight"] += any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1) for r, c in _sites(target))
                    if target["kind"] == "frame":
                        inside = [other for other in targets if other is not target and target["r0"] < other["r0"] and other["r1"] < target["r1"] and target["c0"] < other["c0"] and other["c1"] < target["c1"]]
                        review["multiple_inside"] += len(inside) >= 2
                        review["off_center_nesting"] += any(other["r0"] - target["r0"] != target["r1"] - other["r1"] or other["c0"] - target["c0"] != target["c1"] - other["c1"] for other in inside)
                        review["deep_nesting"] += any(other["kind"] == "frame" and any(third is not other and other["r0"] < third["r0"] and third["r1"] < other["r1"] and other["c0"] < third["c0"] and third["c1"] < other["c1"] for third in targets) for other in inside)
                        review["negative_inside"] += any(target["r0"] < other["r0"] and other["r1"] < target["r1"] and target["c0"] < other["c0"] and other["c1"] < target["c1"] for other in negatives)
                singleton_cells = [next(iter(r["cells"])) for r in negatives if len(r["cells"]) == 1]
                review["singletons"] += len(singleton_cells)
                review["diagonal_singletons"] += any(abs(r - rr) == 1 and abs(c - cc) == 1 for i, (r, c) in enumerate(singleton_cells) for rr, cc in singleton_cells[i + 1:])
                for negative in negatives:
                    h, w = negative["height"], negative["width"]
                    review["vertical_lines"] += w == 1 and h >= 2
                    review["horizontal_lines"] += h == 1 and w >= 2
                    review["filled_nonsquare"] += h != w and negative["full"] and min(h, w) >= 2
                    review["hollow_nonsquare"] += h != w and negative["border"] and min(h, w) >= 3
                    border = {(r, c) for r in range(negative["r0"], negative["r1"] + 1) for c in range(negative["c0"], negative["c1"] + 1) if r in (negative["r0"], negative["r1"]) or c in (negative["c0"], negative["c1"])}
                    nearly_frame = h == w and h >= 3 and negative["cells"] < border and len(border - negative["cells"]) == 1
                    review["broken_frames"] += nearly_frame
                    review["square_bbox_irregular"] += h == w and h >= 3 and not nearly_frame
        review["count_increase"] += max(split_counts["test"]) > max(split_counts["train"])
        review["matched_or_simpler"] += min(split_counts["test"]) <= max(split_counts["train"])
        review["train_count_varied"] += len(set(split_counts["train"])) >= 2
        review["test_count_varied"] += len(set(split_counts["test"])) >= 2
    assert len(dimensions) >= 60 and len(counts) >= 5 and len(kinds) >= 12
    assert len(colors) >= 20 and len(regimes) >= 10
    assert review["train_count_varied"] >= 35 and review["test_count_varied"] >= 15
    assert review["count_increase"] >= 5 and review["matched_or_simpler"] >= 20
    for property_name in ("multicolor", "multicolor_targets", "same_role_color", "different_role_color",
                          "small", "large", "source_scale", "max_board", "rectangular", "dense", "compact_multi", "many_targets",
                          "all_frames", "all_solids", "mixed", "no_distractors", "unchanged",
                          "train_nested", "test_nested", "train_shared", "test_shared", "nested_multicolor",
                          "shared_negative", "diagonal_target_contact", "separate_solid", "side_six", "large_solid", "edge_highlight", "multiple_inside", "off_center_nesting",
                          "deep_nesting", "negative_inside", "singletons", "diagonal_singletons",
                          "vertical_lines", "horizontal_lines", "filled_nonsquare", "hollow_nonsquare",
                          "broken_frames", "square_bbox_irregular"):
        assert review[property_name] > 0, property_name
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, a, b)
        validate_generated_puzzle(p)
        if a is not None:
            assert len(p["train"]) == a
        if b is not None:
            assert len(p["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_train": 2.0}):
        try:
            generate_puzzle(10, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {kwargs}")
