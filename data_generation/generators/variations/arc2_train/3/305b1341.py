"""Ordered legend, nonrecursive square-stamp ARC puzzle generator."""

import hashlib
import numpy as np

SOURCE_HASHES = {
    "23674996f1474d5c44b86ca074e30ecfb64a2df71a18387230149ff0dd01c723d",
    "6f30683e6dc51efef405d180ba0ade6db609cab08f1c81943a270c596dc6fcbd",
    "4ea8ecf948b3a9d1387f1e7ff19c996741b8e070c1c38350b3d5890e2aa9f69c",
    "6196586c1b3eb5a802a2bdea3c57c509ddcf5350521d165d8e0d909dd0a494c0",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 to 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 to 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size // 2, "background is not identifiable")
    return bg


def _parse_legend(grid):
    _require(grid.ndim == 2 and min(grid.shape) >= 8, "grid is too small")
    bg = _background(grid)
    length = 0
    while (length < grid.shape[0] and int(grid[length, 0]) != bg
           and int(grid[length, 1]) != bg):
        length += 1
    _require(1 <= length <= 5 and length < grid.shape[0], "bad legend length")
    _require(int(grid[length, 0]) == bg and int(grid[length, 1]) == bg,
             "ambiguous legend termination")
    pairs = [(int(grid[r, 0]), int(grid[r, 1])) for r in range(length)]
    cores = [core for core, _ in pairs]
    _require(len(set(cores)) == length, "repeated core")
    _require(all(core != halo and core != bg and halo != bg for core, halo in pairs),
             "legend colors collide")
    payload = grid.copy()
    payload[:length, :2] = bg
    _require({int(x) for x in np.unique(payload)} <= {bg, *cores},
             "unkeyed payload color")
    _require(all(np.any(payload == core) for core in cores), "empty layer")
    return pairs, payload, bg


def _footprint(shape, centers):
    mask = np.zeros(shape, dtype=bool)
    for row, col in centers:
        row, col = int(row), int(col)
        mask[max(0, row - 1):min(shape[0], row + 2),
             max(0, col - 1):min(shape[1], col + 2)] = True
    return mask


def _proposals(grid):
    pairs, payload, bg = _parse_legend(grid)
    centers, proposals = [], []
    for core, halo in pairs:
        points = np.argwhere(payload == core)
        proposal = np.full(grid.shape, -1, dtype=np.int16)
        proposal[_footprint(grid.shape, points)] = halo
        proposal[points[:, 0], points[:, 1]] = core
        centers.append(points)
        proposals.append(proposal)
    return pairs, payload, bg, centers, proposals


def _compose(proposals, bg, order=None):
    output = np.full(proposals[0].shape, bg, dtype=np.int8)
    if order is None:
        order = range(len(proposals) - 1, -1, -1)
    for rank in order:
        proposal = proposals[rank]
        output[proposal >= 0] = proposal[proposal >= 0]
    return output


def _transform_input(grid):
    _, _, bg, _, proposals = _proposals(grid)
    return _compose(proposals, bg)


def _rule_candidates(train):
    """Infer radius and legend direction using training grids alone."""
    answers = []
    for radius in (0, 1, 2):
        for top_first in (False, True):
            agrees = True
            for pair in train:
                grid, expected = pair["input"], pair["output"]
                pairs, _, bg, centers, _ = _proposals(grid)
                prediction = np.full(grid.shape, bg, dtype=np.int8)
                ranks = (range(len(pairs) - 1, -1, -1) if top_first
                         else range(len(pairs)))
                for rank in ranks:
                    core, halo = pairs[rank]
                    for row, col in centers[rank]:
                        row, col = int(row), int(col)
                        prediction[max(0, row - radius):min(grid.shape[0], row + radius + 1),
                                   max(0, col - radius):min(grid.shape[1], col + radius + 1)] = halo
                    prediction[centers[rank][:, 0], centers[rank][:, 1]] = core
                if not np.array_equal(prediction, expected):
                    agrees = False
                    break
            if agrees:
                answers.append((radius, top_first))
    return answers


def _sample_palette(rng, count, bg, mode=None):
    available = [x for x in range(10) if x != bg]
    cores = [int(x) for x in rng.choice(available, size=count, replace=False)]
    extras = [x for x in available if x not in cores]
    if mode is None:
        modes = ["distinct", "chain", "cycle", "reverse", "shared", "free"]
        if count >= 3:
            modes.append("front_back")
        mode = str(rng.choice(modes if 2 * count <= 9 else modes[1:]))
    if mode == "distinct":
        halos = [int(x) for x in rng.choice(extras, size=count, replace=False)]
    elif mode == "chain":
        halos = cores[1:] + [int(rng.choice(extras))]
    elif mode == "cycle":
        halos = cores[1:] + cores[:1]
    elif mode == "reverse":
        halos = [int(rng.choice(extras))] + cores[:-1]
    elif mode == "shared":
        halos = [int(rng.choice(extras))] * count
    elif mode == "front_back":
        halos = [cores[-1]] + [int(rng.choice([x for x in available if x != core]))
                               for core in cores[1:]]
    else:
        halos = [int(rng.choice([x for x in available if x != core])) for core in cores]
    return list(zip(cores, halos))


def _sample_local_mask(rng, height, width, complexity):
    rows = int(rng.integers(2, min(6, max(2, (height - 1) // 2)) + 1))
    cols = int(rng.integers(2, min(6, max(2, (width - 1) // 2)) + 1))
    styles = ["block", "block", "frame", "notch", "open", "bend",
              "diagonal", "cross", "scatter", "line"]
    if complexity == 0:
        styles += ["block", "block", "frame", "line"]
    if complexity == 2:
        styles += ["notch", "open", "bend", "diagonal", "scatter"]
    style = str(rng.choice(styles))
    cells = {(i, j) for i in range(rows) for j in range(cols)}
    if style in ("frame", "open"):
        cells = {(i, j) for i, j in cells if i in (0, rows - 1) or j in (0, cols - 1)}
        if style == "open":
            side = int(rng.integers(4))
            if side == 0:
                cells -= {(0, j) for j in range(1, cols - 1)}
            elif side == 1:
                cells -= {(rows - 1, j) for j in range(1, cols - 1)}
            elif side == 2:
                cells -= {(i, 0) for i in range(1, rows - 1)}
            else:
                cells -= {(i, cols - 1) for i in range(1, rows - 1)}
    elif style == "notch":
        cr, cc = int(rng.choice([0, rows - 1])), int(rng.choice([0, cols - 1]))
        cells.remove((cr, cc))
        if rows >= 4 and cols >= 4 and rng.random() < 0.6:
            cells -= {(cr, cc + (1 if cc == 0 else -1)),
                      (cr + (1 if cr == 0 else -1), cc)}
    elif style == "bend":
        br, bc = int(rng.integers(rows)), int(rng.integers(cols))
        cells = {(br, j) for j in range(cols)} | {(i, bc) for i in range(rows)}
        if rng.random() < 0.5:
            cells = {(i, j) for i, j in cells if i <= br or j >= bc}
    elif style == "diagonal":
        cells = {(i, min(cols - 1, i * cols // rows)) for i in range(rows)}
        if rng.random() < 0.5:
            cells |= {(rows - 1, j) for j in range(cols)}
    elif style == "cross":
        cells = {(rows // 2, j) for j in range(cols)} | {(i, cols // 2) for i in range(rows)}
    elif style == "scatter":
        count = int(rng.integers(2, max(3, rows * cols // 2 + 1)))
        cells = {(int(k) // cols, int(k) % cols)
                 for k in rng.choice(rows * cols, size=count, replace=False)}
    elif style == "line":
        if rng.random() < 0.5:
            line_row = int(rng.integers(rows))
            cells = {(line_row, j) for j in range(cols)}
        else:
            line_col = int(rng.integers(cols))
            cells = {(i, line_col) for i in range(rows)}
    if rng.random() < 0.5:
        cells = {(j, i) for i, j in cells}
    min_r, min_c = min(r for r, _ in cells), min(c for _, c in cells)
    return {(2 * (r - min_r), 2 * (c - min_c)) for r, c in cells}


def _axis(rng, maximum):
    if maximum <= 0:
        return 0
    if rng.random() < 0.36:
        return int(rng.choice(sorted({0, 1, max(0, maximum - 1), maximum})))
    return int(rng.integers(maximum + 1))


def _place_mask(rng, local, height, width, legend_length, existing, relation):
    max_row = height - 1 - max(r for r, _ in local)
    max_col = width - 1 - max(c for _, c in local)
    if min(max_row, max_col) < 0:
        return None
    old_masks = [_footprint((height, width), np.array(list(old))) for old in existing]
    for _ in range(80):
        if existing and relation in ("overlap", "contain", "chain"):
            target = existing[int(rng.integers(len(existing)))] if relation != "chain" else existing[-1]
            rr, cc = [r for r, _ in target], [c for _, c in target]
            jitter = 2 if relation == "contain" else 5
            row = int(np.clip((min(rr) + max(rr)) // 2 - max(r for r, _ in local) // 2
                              + int(rng.integers(-jitter, jitter + 1)), 0, max_row))
            col = int(np.clip((min(cc) + max(cc)) // 2 - max(c for _, c in local) // 2
                              + int(rng.integers(-jitter, jitter + 1)), 0, max_col))
        else:
            row, col = _axis(rng, max_row), _axis(rng, max_col)
        points = {(row + r, col + c) for r, c in local}
        if any(r <= legend_length and c < 2 for r, c in points):
            continue
        if any(points & old for old in existing):
            continue
        if existing and relation != "random":
            mask = _footprint((height, width), np.array(list(points)))
            overlaps = any(np.any(mask & old) for old in old_masks)
            if relation == "separate" and overlaps:
                continue
            if relation in ("overlap", "contain", "chain") and not overlaps:
                continue
        return points
    return None


def _add_patch(rng, points, height, width, legend_length, occupied):
    for _ in range(45):
        nr, nc = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        local = {(2 * i, 2 * j) for i in range(nr) for j in range(nc)}
        r0 = _axis(rng, height - 1 - max(r for r, _ in local))
        c0 = _axis(rng, width - 1 - max(c for _, c in local))
        patch = {(r0 + r, c0 + c) for r, c in local}
        if any(r <= legend_length and c < 2 for r, c in patch) or patch & occupied:
            continue
        if min(max(abs(r - q), abs(c - d)) for r, c in patch for q, d in points) < 4:
            continue
        return points | patch
    return points


def _all_layers_visible(grid, output):
    pairs, _, _, centers, proposals = _proposals(grid)
    higher = np.zeros(grid.shape, dtype=bool)
    for rank, (core, halo) in enumerate(pairs):
        point_mask = np.zeros(grid.shape, dtype=bool)
        point_mask[centers[rank][:, 0], centers[rank][:, 1]] = True
        visible_core = point_mask & ~higher & (output == core)
        visible_halo = (proposals[rank] == halo) & ~higher & (output == halo)
        if not np.any(visible_core) or not np.any(visible_halo):
            return False
        higher |= proposals[rank] >= 0
    return True


def _priority_evidence(grid, output):
    _, _, bg, centers, proposals = _proposals(grid)
    if len(proposals) < 2:
        return False
    # The rejected shortcut gives spatially upper/left masks priority.
    order = sorted(range(len(proposals)),
                   key=lambda rank: (int(centers[rank][:, 0].min()),
                                     int(centers[rank][:, 1].min())), reverse=True)
    return not np.array_equal(_compose(proposals, bg, order), output)


def _nonrecursive_evidence(grid, output):
    pairs, payload, bg, _, proposals = _proposals(grid)
    cores = {core for core, _ in pairs}
    for rank, (_, halo) in enumerate(pairs):
        if halo not in cores:
            continue
        visible = np.argwhere((proposals[rank] == halo) & (output == halo) & (payload == bg))
        for row, col in visible:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = int(row + dr), int(col + dc)
                    if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and output[rr, cc] == bg:
                        return True
    return False


def _pair_hash(pair):
    a = pair["input"].astype(np.uint8)
    b = pair["output"].astype(np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _make_pair(rng, layer_count, background, regime, complexity, palette_mode=None):
    for _ in range(240):
        canvas_choice = rng.random()
        if canvas_choice < 0.045:
            height, width = 30, 30
        elif canvas_choice < 0.135:
            height, width = 20, 20
        else:
            low = 12 if layer_count <= 2 else 16
            height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
            if rng.random() < 0.3:
                if rng.random() < 0.5:
                    height, width = int(rng.integers(low, 18)), int(rng.integers(23, 31))
                else:
                    height, width = int(rng.integers(23, 31)), int(rng.integers(low, 18))
        pairs = _sample_palette(rng, layer_count, background, palette_mode)
        choices = ["random", "overlap", "contain", "chain", "separate"]
        weights = ([0.22, 0.35, 0.13, 0.15, 0.15] if regime == "source_train"
                   else [0.18, 0.37, 0.17, 0.18, 0.10])
        relation = str(rng.choice(choices, p=weights))
        layer_order = list(rng.permutation(layer_count))
        shapes = [None] * layer_count
        placed = []
        for rank in layer_order:
            local = _sample_local_mask(rng, height, width, complexity)
            points = _place_mask(rng, local, height, width, layer_count, placed,
                                 "random" if not placed else relation)
            if points is None:
                break
            if complexity > 0 and rng.random() < (0.12 if complexity == 1 else 0.26):
                occupied = set().union(*placed) if placed else set()
                points = _add_patch(rng, points, height, width, layer_count, occupied)
            shapes[rank] = points
            placed.append(points)
        if any(shape is None for shape in shapes):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for rank, (core, halo) in enumerate(pairs):
            grid[rank, 0], grid[rank, 1] = core, halo
            for row, col in shapes[rank]:
                grid[row, col] = core
        try:
            _, _, bg, _, proposals = _proposals(grid)
        except AssertionError:
            continue
        output = _compose(proposals, bg)
        if not _all_layers_visible(grid, output) or np.array_equal(grid, output):
            continue
        pair = {"input": grid, "output": output}
        if _pair_hash(pair) not in SOURCE_HASHES:
            return pair
    raise ValueError("could not construct a valid composition")


def _validate_generated_puzzle_impl(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_max, has_priority, has_nonrecursive = 0, False, False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "each pair must contain input and output")
            inp, out = pair["input"], pair["output"]
            for name, grid in (("input", inp), ("output", out)):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a 2D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), f"{name} must have integer cells")
                _require(all(1 <= d <= 30 for d in grid.shape) and np.all((grid >= 0) & (grid <= 9)),
                         f"{name} has illegal dimensions or colors")
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "dimensions changed or pair is unchanged")
            legend, _, bg, centers, proposals = _proposals(inp)
            _require(np.array_equal(_compose(proposals, bg), out),
                     "output disagrees with ordered nonrecursive stamping")
            _require(_all_layers_visible(inp, out), "a layer is invisible")
            for points in centers:
                for i, (r, c) in enumerate(points):
                    _require(all(max(abs(int(r - q)), abs(int(c - d))) >= 2
                                 for q, d in points[i + 1:]),
                             "same-layer centers are adjacent")
            input_key, pair_key = _grid_key(inp), _pair_hash(pair)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair or input")
            _require(pair_key not in SOURCE_HASHES, "original source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_max = max(train_max, len(legend))
                has_priority |= _priority_evidence(inp, out)
                has_nonrecursive |= _nonrecursive_evidence(inp, out)
            else:
                _require(len(legend) <= train_max,
                         "test uses more layers than training demonstrated")
    _require(has_priority, "training lacks visible legend-order evidence")
    _require(has_nonrecursive, "training lacks visible nonrecursive reuse evidence")
    _require(_rule_candidates(train) == [(1, True)],
             "training does not identify the one-cell top-first rule")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Swatch orientation, priority, and radius are the shared conventions.
    # Each input supplies its own complete palette and layer identities.
    max_layers = int(rng.choice([2, 3, 4, 5], p=[0.16, 0.43, 0.31, 0.10]))
    complexity = int(rng.integers(3))
    preferred_bg = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen = set()

    def add_pair(split, index):
        if split == "train":
            if index == 0:
                count = max_layers
            elif index == 1:
                count = int(rng.integers(2, max_layers + 1))
            else:
                count = int(rng.integers(1, max_layers + 1))
        else:
            count = int(rng.integers(1, max_layers + 1))
            if max_layers >= 3 and rng.random() < 0.55:
                count = int(rng.integers(3, max_layers + 1))
        bg = preferred_bg if rng.random() < 0.65 else int(rng.integers(10))
        mode = "chain" if split == "train" and index == 1 else None
        for _ in range(140):
            pair = _make_pair(rng, count, bg,
                              "source_train" if split == "train" else "target_test",
                              complexity, mode)
            key = _grid_key(pair["input"])
            if key in seen:
                continue
            if split == "train" and index == 0 and not _priority_evidence(pair["input"], pair["output"]):
                continue
            if split == "train" and index == 1 and not _nonrecursive_evidence(pair["input"], pair["output"]):
                continue
            seen.add(key)
            puzzle[split].append(pair)
            return
        raise ValueError("could not construct a distinctive training pair")

    for index in range(num_train):
        add_pair("train", index)
    for index in range(num_test):
        add_pair("test", index)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and
               all(np.array_equal(x[part], y[part])
                   for x, y in zip(a[split], b[split]) for part in ("input", "output"))
               for split in ("train", "test"))


def _normalized_masks(grid):
    _, _, _, centers, _ = _proposals(grid)
    masks = []
    for points in centers:
        origin = points.min(axis=0)
        masks.append(tuple(sorted((int(r - origin[0]), int(c - origin[1])) for r, c in points)))
    return tuple(masks)


def self_test() -> None:
    train_counts, test_counts, dimensions = set(), set(), set()
    masks, backgrounds, layer_counts, reuse_patterns = set(), set(), set(), set()
    small = large = priority = nonrecursive = 0
    review = {name: 0 for name in (
        "one_layer", "four_plus", "twenty", "full_canvas_train", "nonzero_bg", "bg_changes",
        "reuse_changes", "distinct_palette", "reused_palette",
        "test_separated", "test_threeway", "back_middle", "back_front",
        "winner_below", "winner_right", "winner_inside", "winner_around",
        "clipped", "edge_touch")}
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_bgs, train_reuse = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                legend, _, bg, centers, proposals = _proposals(grid)
                _require(np.array_equal(_compose(proposals, bg), out),
                         "stamp transformation failed")
                dimensions.add(grid.shape)
                backgrounds.add(bg)
                layer_counts.add(len(legend))
                masks.update(_normalized_masks(grid))
                signature = tuple(next((j for j, (core, _) in enumerate(legend)
                                        if core == halo), -1) for _, halo in legend)
                reuse_patterns.add(signature)
                puzzle_bgs.add(bg)
                if split == "train":
                    train_reuse.add(signature)
                review["one_layer"] += len(legend) == 1
                review["four_plus"] += len(legend) >= 4
                review["twenty"] += grid.shape == (20, 20)
                review["full_canvas_train"] += split == "train" and grid.shape == (30, 30)
                review["nonzero_bg"] += bg != 0
                review["distinct_palette"] += all(x == -1 for x in signature)
                review["reused_palette"] += any(x >= 0 for x in signature)
                footprints = [proposal >= 0 for proposal in proposals]
                overlaps = [(i, j) for i in range(len(legend))
                            for j in range(i + 1, len(legend))
                            if np.any(footprints[i] & footprints[j])]
                triples = any(np.any(footprints[i] & footprints[j] & footprints[k])
                              for i in range(len(legend)) for j in range(i + 1, len(legend))
                              for k in range(j + 1, len(legend)))
                if split == "test":
                    review["test_separated"] += not overlaps
                    review["test_threeway"] += triples
                    review["back_middle"] += (1, 2) in overlaps
                    review["back_front"] += (0, 2) in overlaps
                for i, j in overlaps:
                    hi, lo = centers[i], centers[j]
                    hr0, hr1 = int(hi[:, 0].min()), int(hi[:, 0].max())
                    hc0, hc1 = int(hi[:, 1].min()), int(hi[:, 1].max())
                    lr0, lr1 = int(lo[:, 0].min()), int(lo[:, 0].max())
                    lc0, lc1 = int(lo[:, 1].min()), int(lo[:, 1].max())
                    review["winner_below"] += hr0 > lr0
                    review["winner_right"] += hc0 > lc0
                    review["winner_inside"] += hr0 >= lr0 and hr1 <= lr1 and hc0 >= lc0 and hc1 <= lc1
                    review["winner_around"] += hr0 <= lr0 and hr1 >= lr1 and hc0 <= lc0 and hc1 >= lc1
                review["clipped"] += any(np.any((points[:, 0] == 0) |
                                                  (points[:, 0] == grid.shape[0] - 1) |
                                                  (points[:, 1] == 0) |
                                                  (points[:, 1] == grid.shape[1] - 1))
                                          for points in centers)
                review["edge_touch"] += any(np.any((points[:, 0] == 1) |
                                                     (points[:, 0] == grid.shape[0] - 2) |
                                                     (points[:, 1] == 1) |
                                                     (points[:, 1] == grid.shape[1] - 2))
                                             for points in centers)
                small += min(grid.shape) <= 16
                large += max(grid.shape) >= 28
                if split == "train":
                    priority += _priority_evidence(grid, out)
                    nonrecursive += _nonrecursive_evidence(grid, out)
        review["bg_changes"] += len(puzzle_bgs) > 1
        review["reuse_changes"] += len(train_reuse) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(len(dimensions) >= 35 and small >= 8 and large >= 20,
             "canvas sizes lack diversity")
    _require(len(masks) >= 50, "mask shapes lack diversity")
    _require(len(backgrounds) >= 6 and len(layer_counts) >= 4 and len(reuse_patterns) >= 8,
             "color or layer roles lack diversity")
    _require(priority >= 42 and nonrecursive >= 42,
             "training lacks priority or nonrecursive evidence")
    minimums = {
        "one_layer": 10, "four_plus": 10, "twenty": 5, "full_canvas_train": 2,
        "nonzero_bg": 100, "bg_changes": 20, "reuse_changes": 20,
        "distinct_palette": 40, "reused_palette": 40,
        "test_separated": 10, "test_threeway": 5,
        "back_middle": 10, "back_front": 10,
        "winner_below": 30, "winner_right": 30,
        "winner_inside": 5, "winner_around": 5,
        "clipped": 30, "edge_touch": 30,
    }
    _require(all(review[key] >= minimum for key, minimum in minimums.items()),
             f"review features lack diversity: {review}")
    _require(not _puzzles_equal(puzzles[0], puzzles[1]), "seeds repeated a puzzle")
    explicit = generate_puzzle(731, 7, 3)
    _require(_puzzles_equal(explicit, generate_puzzle(731, 7, 3)),
             "explicit generation is not deterministic")
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + offset, train, test)
        _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test,
                 "explicit counts were ignored")
    for offset, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                     {"num_train": 2}, {"num_train": 10})):
        puzzle = generate_puzzle(1100 + offset, **kwargs)
        validate_generated_puzzle(puzzle)
        for name, value in kwargs.items():
            split = "train" if name == "num_train" else "test"
            _require(len(puzzle[split]) == value, "partial count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
