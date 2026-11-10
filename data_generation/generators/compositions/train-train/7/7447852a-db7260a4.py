"""Leaky accordion reservoirs: periodic activation, triangular containment, flow union.

Only NumPy and the Python standard library are used.  The solver reparses the
visible geometry; no generation descriptors or puzzle-level state are retained.
"""

from collections import defaultdict
import numpy as np


HOST_SEEDS = (0, 552763724, 771460163, 795485579, 95771222, 230514934,
              937161696, 165644606, 361586062, 157491213, 961541635,
              563739513, 726322337, 513752826, 229338809, 583453392,
              588896093, 769510536, 261388731, 260565160, 781921775,
              452557806, 914160884, 706256836, 27912220, 473495077,
              855041012, 879183296, 270852759, 748083411, 684414155,
              889910355, 285377170, 97182019, 145963337, 174375355,
              621956987, 451242086, 444864502, 954651418, 50314688,
              793621075)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
          "grid dimensions must be 1..30")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must use integers")
    _need(np.all((grid >= 0) & (grid <= 9)), "colors must be 0..9")
    return grid


def _path_row(top, left, depth, col):
    return top + min(col - left, left + 2 * depth - col)


def _interior(top, left, depth):
    # A V with depth d has d^2 background cells above its diagonal floor.
    for row in range(top, top + depth):
        inset = row - top
        for col in range(left + inset + 1, left + 2 * depth - inset):
            yield row, col


def _parse(input_grid):
    grid = _checked_grid(input_grid)
    height, width = grid.shape
    background = int(grid[0, 0])
    colors = set(int(v) for v in np.unique(grid))
    _need(len(colors) == 3, "need background, wall, and source colors")
    marker_cols = np.flatnonzero(grid[0] != background)
    _need(len(marker_cols) == 1, "one top-row source marker required")
    source_col = int(marker_cols[0])
    source = int(grid[0, source_col])
    _need(0 < source_col < width - 1 and np.count_nonzero(grid == source) == 1,
          "source must be an interior-column singleton")
    wall = next(iter(colors - {background, source}))
    _need(height >= 6 and np.all(grid[1] == background), "blank source clearance")
    _need(np.all(grid[-1] == background), "bottom spill row must be clear")

    tiers = []
    covered = np.zeros_like(grid, dtype=bool)
    cursor = 2
    while cursor < height - 1:
        rows = np.flatnonzero(np.any(grid[cursor:-1] == wall, axis=1))
        if not len(rows):
            break
        top = cursor + int(rows[0])
        peaks = np.flatnonzero(grid[top] == wall)
        _need(len(peaks) >= 2, "tier needs at least two intact peaks")
        steps = np.diff(peaks)
        _need(np.all(steps == steps[0]) and steps[0] >= 4 and steps[0] % 2 == 0,
              "equally spaced peaks must define complete V-shaped bays")
        depth = int(steps[0]) // 2
        _need(top + depth < height - 1, "tier floor must be above spill row")
        bays = []
        for left in peaks[:-1]:
            left = int(left)
            right = left + 2 * depth
            missing = []
            for col in range(left, right + 1):
                row = _path_row(top, left, depth, col)
                covered[row, col] = True
                value = int(grid[row, col])
                _need(value in (background, wall), "unexpected floor color")
                if value == background:
                    _need(left < col < right, "peaks cannot be broken")
                    missing.append(col)
            _need(len(missing) <= 1, "a bay has at most one drain")
            _need(all(grid[r, c] == background for r, c in _interior(top, left, depth)),
                  "input bay interiors must be empty")
            bays.append((left, right, None if not missing else missing[0]))
        tiers.append(dict(top=top, depth=depth, bays=bays))
        _need(np.all(grid[top + depth + 1] == background), "tiers need a blank row")
        cursor = top + depth + 2
    _need(bool(tiers), "no reservoir tiers")
    _need(np.all(grid[~covered & (grid != source)] == background),
          "all wall cells must belong to the reconstructed zigzags")
    anchors = [i for i, (left, right, _) in enumerate(tiers[0]["bays"])
               if left < source_col < right]
    _need(len(anchors) == 1, "source must align with a top bay, never a peak")
    return grid, (background, wall, source), source_col, tiers, anchors[0]


def _flow(tiers, anchor, cadence=3, ignore_drains=False):
    """Propagate sets of origins down a DAG of bays, coalescing on arrival."""
    active = [defaultdict(set) for _ in tiers]
    hops = {}
    for index in range(anchor, len(tiers[0]["bays"]), cadence):
        active[0][index].add(index)
        hops[0, index] = 1
    visited, sinks, edges = {}, set(), []
    spill = False
    for level, tier in enumerate(tiers):
        for index, origins in sorted(active[level].items()):
            visited[level, index] = frozenset(origins)
            gap = tier["bays"][index][2]
            if gap is None or ignore_drains:
                sinks.add((level, index))
                continue
            destination = None
            for lower in range(level + 1, len(tiers)):
                bays = tiers[lower]["bays"]
                if gap < bays[0][0] or gap > bays[-1][1]:
                    continue  # A miss may still reach a later tier.
                hits = [j for j, (left, right, _) in enumerate(bays) if left < gap < right]
                _need(len(hits) == 1, "a falling signal must not strike a peak")
                destination = (lower, hits[0])
                active[lower][hits[0]].update(origins)
                hops[destination] = max(hops.get(destination, 0), hops[level, index] + 1)
                break
            edges.append(((level, index), destination, gap))
            if destination is None:
                spill = True
    return dict(visited=visited, sinks=sinks, edges=edges, spill=spill,
                merged=any(len(origins) > 1 for origins in visited.values()), hops=hops)


def _render_answer(grid, roles, source_col, tiers, flow):
    background, _, source = roles
    answer = grid.copy()
    answer[0, source_col] = background
    for level, index in flow["sinks"]:
        tier = tiers[level]
        left = tier["bays"][index][0]
        for row, col in _interior(tier["top"], left, tier["depth"]):
            answer[row, col] = source
    if flow["spill"]:
        answer[-1, :] = source
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible source, complete peaks, diagonal floors, and drains."""
    grid, roles, source_col, tiers, anchor = _parse(input_grid)
    return _render_answer(grid, roles, source_col, tiers, _flow(tiers, anchor))


def _tier(rng, width, top, depth, count, left=None, broken_probability=0.45):
    if left is None:
        left = int(rng.integers(width - 2 * depth * count))
    bays = []
    for index in range(count):
        lft = left + 2 * depth * index
        gap = int(rng.integers(lft + 1, lft + 2 * depth)) if rng.random() < broken_probability else None
        bays.append((lft, lft + 2 * depth, gap))
    return dict(top=top, depth=depth, bays=bays)


def _set_gap(tier, index, gap):
    left, right, _ = tier["bays"][index]
    tier["bays"][index] = (left, right, gap)


def _draw(width, height, tiers, source_col, roles):
    background, wall, source = roles
    grid = np.full((height, width), background, dtype=np.int8)
    for tier in tiers:
        for left, right, gap in tier["bays"]:
            for col in range(left, right + 1):
                if col != gap:
                    grid[_path_row(tier["top"], left, tier["depth"], col), col] = wall
    grid[0, source_col] = source
    return grid


def _make_pair(rng, roles, mode, levels=2, allow_spill=True, top_depth=None):
    # Rejection handles only valid geometric compositions, not hidden semantics.
    for _attempt in range(700):
        depth = int(rng.choice((2, 3))) if top_depth is None else top_depth
        count = int(rng.integers(4, 8 if depth == 2 else 5))
        if depth == 2 and mode in ("retain", "merge"):
            count = max(count, 5)  # Teach a marker anchor beyond bay zero.
        if mode == "merge":
            count = min(count, 6)  # Exactly two initially active bays.
        minimum_width = 2 * depth * count + 1
        width = int(rng.integers(minimum_width, 31))
        top = int(rng.integers(2, 5))
        first = _tier(rng, width, top, depth, count)
        anchor = int(rng.integers(count - 3))
        if depth == 2 and mode in ("retain", "merge"):
            anchor = int(rng.integers(1, count - 3))
        selected = list(range(anchor, count, 3))
        source_left, source_right, _ = first["bays"][anchor]
        source_col = int(rng.integers(source_left + 1, source_right))
        if mode == "retain":
            for index in range(count):
                _set_gap(first, index, None)
        else:
            for index in selected:
                left, right, _ = first["bays"][index]
                gap = int(rng.integers(left + 1, right))
                _set_gap(first, index, gap)
            if mode == "spill":
                _set_gap(first, selected[-1], None)
            elif mode == "mixed" and rng.random() < 0.55:
                _set_gap(first, selected[int(rng.integers(len(selected)))], None)
        tiers = [first]
        if mode == "merge":
            for index in range(count):
                if index not in selected:
                    _set_gap(first, index, None)
            a, b = selected
            gap_a = first["bays"][a][1] - 1
            gap_b = first["bays"][b][0] + 1
            _set_gap(first, a, gap_a)
            _set_gap(first, b, gap_b)
            low_depth = (gap_b - gap_a) // 2 + 1
            low_depth += int(rng.integers(0, min(2, (width - 1) // 2 - low_depth) + 1))
            lo = max(0, gap_b - 2 * low_depth + 1)
            hi = min(gap_a - 1, width - 2 * low_depth - 1)
            if lo > hi:
                continue
            lower = _tier(rng, width, top + depth + int(rng.integers(2, 4)),
                          low_depth, 1, left=int(rng.integers(lo, hi + 1)),
                          broken_probability=0)
            tiers.append(lower)
        else:
            for level in range(1, levels):
                previous = tiers[-1]
                lower_top = previous["top"] + previous["depth"] + int(rng.integers(2, 4))
                low_depth = int(rng.integers(2, 5 if levels >= 3 else 7))
                low_count = int(rng.integers(1, min(6, (width - 1) // (2 * low_depth)) + 1))
                if mode == "retain":
                    probability = 0.5
                elif mode == "spill" or (mode in ("cascade", "deep") and level < levels - 1):
                    probability = 1.0
                elif mode in ("cascade", "deep"):
                    probability = 0.0
                else:
                    probability = float(rng.choice((0.2, 0.5, 0.8)))
                tiers.append(_tier(rng, width, lower_top, low_depth, low_count,
                                   broken_probability=probability))
        height = tiers[-1]["top"] + tiers[-1]["depth"] + int(rng.integers(2, 5))
        if height > 30:
            continue
        try:
            flow = _flow(tiers, anchor)
        except AssertionError:
            continue
        if mode == "merge" and not (flow["merged"] and flow["sinks"] == {(1, 0)}):
            continue
        if mode == "spill" and not (flow["spill"] and any(l > 0 for l, _ in flow["visited"])):
            continue
        if mode in ("cascade", "deep") and not any(flow["hops"][node] == levels for node in flow["sinks"]):
            continue
        if mode == "mixed" and not any(l >= 1 for l, _ in flow["visited"]):
            continue
        if not allow_spill and flow["spill"]:
            continue
        grid = _draw(width, height, tiers, source_col, roles)
        answer = _render_answer(grid, roles, source_col, tiers, flow)
        # Check reconstruction against the construction, rather than generating
        # labels by calling solve().  This also detects any tier parsing mismatch.
        parsed = _parse(grid)
        _need(parsed[3] == tiers and parsed[4] == anchor, "geometry reconstruction failed")
        return {"input": grid, "output": answer}
    raise RuntimeError(f"could not fit a valid {mode} scene")


def _counts_and_seed(seed, num_train, num_test):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(seed), int(num_train), int(num_test)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(int(v) for v in grid.flat)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed, num_train, num_test = _counts_and_seed(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base_roles = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    stable_palette = bool(rng.integers(2))
    # Both top depths are demonstrated before they are recombined in tests.
    depths = [2, 3]
    rng.shuffle(depths)
    plans = [("retain", 2, depths[0]), ("merge", 2, depths[1])]
    if num_train >= 3:
        plans.append(("spill", 2, int(rng.choice(depths))))
    if num_train >= 4:
        plans.append(("cascade", 3, int(rng.choice(depths))))
    while len(plans) < num_train:
        mode = str(rng.choice(("retain", "merge", "spill", "cascade", "mixed")))
        plans.append((mode, 3 if mode == "cascade" else int(rng.integers(2, 4)), int(rng.choice(depths))))
    rng.shuffle(plans)
    test_plans = []
    for index in range(num_test):
        levels = 4 if num_train >= 4 and index == 0 else 3
        mode = "deep" if index == 0 or num_train == 2 or rng.random() < 0.6 else "mixed"
        test_plans.append((mode, levels, int(rng.choice(depths))))
    result, seen = {"train": [], "test": []}, set()
    for split, settings in (("train", plans), ("test", test_plans)):
        for mode, levels, depth in settings:
            for _ in range(100):
                roles = base_roles if stable_palette else tuple(int(c) for c in rng.choice(10, 3, replace=False))
                pair = _make_pair(rng, roles, mode, levels, allow_spill=num_train >= 3, top_depth=depth)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _inspect(pair):
    grid, roles, source_col, tiers, anchor = _parse(pair["input"])
    flow = _flow(tiers, anchor)
    expected = _render_answer(grid, roles, source_col, tiers, flow)
    answer = _checked_grid(pair["output"])
    _need(np.array_equal(expected, answer), "output violates routing or whole-bay fill")
    background, wall, source = roles
    _need(np.array_equal(answer == wall, grid == wall), "wall cells changed")
    _need(answer[0, source_col] == background, "source was not erased")
    for level, tier in enumerate(tiers):
        for index, (left, right, gap) in enumerate(tier["bays"]):
            cells = list(_interior(tier["top"], left, tier["depth"]))
            color = source if (level, index) in flow["sinks"] else background
            _need(len(cells) == tier["depth"] ** 2, "incorrect triangular area")
            _need(all(answer[r, c] == color for r, c in cells), "partial or unrelated bay fill")
            if gap is not None:
                row = _path_row(tier["top"], left, tier["depth"], gap)
                _need(answer[row, gap] == background, "drain was painted")
    _need(np.all(answer[-1] == (source if flow["spill"] else background)), "wrong spill convention")
    return grid, roles, source_col, tiers, anchor, flow


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _need(all(isinstance(puzzle[k], list) for k in puzzle), "sections must be lists")
    _counts_and_seed(0, len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair keys required")
            info = _inspect(pair)
            key = _key(info[0])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            _need(np.array_equal(solve(info[0]), pair["output"]), "solver disagreement")
            infos[split].append(info)
    train = infos["train"]
    # Inspect evidence in grids; do not trust the generation modes.
    _need({info[3][0]["depth"] for info in train} >= {2, 3}, "both top geometries must be taught")
    _need(any(info[4] > 0 for info in train), "marker-relative origin must be taught")
    _need(any(len(f["sinks"]) >= 2 and all(l == 0 for l, _ in f["sinks"])
              and not f["spill"] for *_, f in train), "missing periodic retention demonstration")
    _need(any(f["merged"] and any(l > 0 for l, _ in f["sinks"])
              for *_, f in train), "missing convergent leak demonstration")
    # Counterfactual ablations must change a training answer for each parent.
    no_a, all_bays, no_b = False, False, False
    for grid, roles, col, tiers, anchor, flow in train:
        correct = _render_answer(grid, roles, col, tiers, flow)
        for cadence in (len(tiers[0]["bays"]) + 1, 1):
            # An ablated route may hit a peak that the real route avoids. Use
            # only geometrically valid witnesses; the all-intact demo always is.
            try:
                ablated_flow = _flow(tiers, anchor, cadence=cadence)
            except AssertionError:
                continue
            alternative = _render_answer(grid, roles, col, tiers, ablated_flow)
            if cadence == 1:
                all_bays |= not np.array_equal(correct, alternative)
            else:
                no_a |= not np.array_equal(correct, alternative)
        alternative = _render_answer(grid, roles, col, tiers, _flow(tiers, anchor, ignore_drains=True))
        no_b |= not np.array_equal(correct, alternative)
    _need(no_a and all_bays and no_b, "training does not distinguish parent ablations")
    if len(train) >= 3:
        _need(any(info[-1]["spill"] for info in train), "missing spill demonstration")
    else:
        _need(not any(info[-1]["spill"] for info in infos["test"]), "untaught spill behavior tested")
    if len(train) >= 4:
        _need(any(len(info[3]) >= 3 and any(info[-1]["hops"][node] >= 3 for node in info[-1]["sinks"])
                  for info in train), "missing multi-leak demonstration")
    _need(all(len(info[3]) >= 3 for info in infos["test"]), "test must recombine at deeper tiers")
    first_test = infos["test"][0][-1]
    required_hops = 4 if len(train) >= 4 else 3
    _need(any(first_test["hops"][node] >= required_hops for node in first_test["sinks"]),
          "first test needs an actual longer leak chain")


def _signature(puzzle):
    return tuple(_key(pair[name]) for split in ("train", "test")
                 for pair in puzzle[split] for name in ("input", "output"))


def self_test():
    """Bounded structural, ablation, determinism, count, and 42-seed checks."""
    signatures, shapes, structures, palettes = set(), set(), set(), set()
    depths, counts, gaps, sinks, anchors, tier_counts = set(), set(), set(), set(), set(), set()
    spill_pairs = merging_pairs = skips = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _need(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterminism")
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, roles, col, tiers, anchor, flow = _inspect(pair)
                # Lists and NumPy arrays are both valid interfaces.
                _need(np.array_equal(solve(grid.tolist()), pair["output"]), "list solve failed")
                shapes.add(grid.shape)
                palettes.add(roles)
                mapping = {c: i for i, c in enumerate(roles)}
                structures.add((grid.shape, tuple(mapping[int(v)] for v in grid.flat)))
                anchors.add(anchor)
                tier_counts.add(len(tiers))
                spill_pairs += flow["spill"]
                merging_pairs += flow["merged"]
                sinks.update(l for l, _ in flow["sinks"])
                for tier in tiers:
                    depths.add(tier["depth"])
                    counts.add(len(tier["bays"]))
                    for left, right, gap in tier["bays"]:
                        if gap is not None:
                            gaps.add(gap - (left + right) // 2)
                skips += sum(dst is not None and dst[0] > src[0] + 1
                             for src, dst, _ in flow["edges"])
                before = grid.copy()
                solve(grid)
                _need(np.array_equal(before, grid), "solve mutated input")
    _need(len(signatures) == 42, "duplicate puzzles across seeds")
    _need(len(shapes) >= 35 and len(structures) >= 150 and len(palettes) >= 35,
          "too little geometric or palette variation")
    _need({2, 3, 4}.issubset(tier_counts) and {0, 1, 2, 3}.issubset(sinks), "depth variation missing")
    _need(len(depths) >= 7 and len(counts) >= 6 and len(anchors) >= 3,
          "basin structure or phase variation missing")
    _need(min(gaps) < 0 < max(gaps) and 0 in gaps, "drain offsets missing")
    _need(spill_pairs >= 30 and merging_pairs >= 42 and skips > 0, "flow variation missing")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(9201 + train * 10 + test, train, test)
            _need((len(p["train"]), len(p["test"])) == (train, test), "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                color_normalized_inputs=len(structures), palettes=len(palettes),
                depths=sorted(depths), basin_counts=sorted(counts),
                tier_counts=sorted(tier_counts), anchor_indices=sorted(anchors),
                spill_pairs=spill_pairs, merging_pairs=merging_pairs, skipped_tiers=skips)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
