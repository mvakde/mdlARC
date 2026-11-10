import numpy as np


_SOURCE_INPUT_TEXTS = {
    (
        "4713777964777777/"
        "7773777459770007/"
        "2773777000770407/"
        "3333777000770017/"
        "7777777650770207/"
        "7777777777770007/"
        "7770000900577777/"
        "7770090290077700/"
        "7772000000077714/"
        "7770010400877700/"
        "7777777777777710/"
        "0000000170000700/"
        "9509008679008760/"
        "0426000070605700/"
        "2000600070000700/"
        "7777777770220708"
    ),
    (
        "7973777777777777/"
        "7773770007770007/"
        "5763771007776047/"
        "3333770507770007/"
        "7777770007770907/"
        "7777772067770007/"
        "7777770007770807/"
        "7777777777770007/"
        "7777777777777777/"
        "7000100077777777/"
        "7000005077777777/"
        "7090800077700009/"
        "7000004077706000/"
        "7020400077700000/"
        "7777777777700510/"
        "7777777777780000"
    ),
    (
        "7673777777777777/"
        "1773777080005000/"
        "7783777050060000/"
        "3333777400000800/"
        "7777777900600040/"
        "7000007000059050/"
        "7522007900008000/"
        "7001007100200006/"
        "7005507777777777/"
        "7000907770900777/"
        "7000607770000777/"
        "7777777778000777/"
        "7761000070000700/"
        "7702000070000700/"
        "7708000077777700/"
        "7777777777777700"
    ),
    (
        "7273770007777777/"
        "7753770067000007/"
        "7473770017405007/"
        "3333770007408007/"
        "7777770007777777/"
        "9077770007900000/"
        "0870070507000650/"
        "0070072047020000/"
        "0070070007004080/"
        "0070570107094050/"
        "6070570097777777/"
        "0075070017700000/"
        "5075070907700901/"
        "0070570987710400/"
        "5070470007777777/"
        "0070270057777777"
    ),
}


def _validate_count_argument(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_count_argument(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_count_argument(
        num_test, "num_test", 1, 4
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        span = 11 - lower
        num_train = lower + (seed // 4) % span
        return num_train, num_test

    if num_train is None:
        lower = max(2, num_test + 1)
        span = 11 - lower
        num_train = lower + seed % span
        return num_train, num_test

    maximum_test = min(4, num_train - 1)
    num_test = 1 + seed % maximum_test
    return num_train, num_test


def _canonical(grid, corner):
    """A view with the selected corner at the upper left."""
    if corner & 2:
        grid = grid[::-1, :]
    if corner & 1:
        grid = grid[:, ::-1]
    return grid


def _boxes_have_background_gap(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _analyze_input(grid):
    """Recover every role and record from visible cells, without layout metadata."""
    selectors = []
    for corner in range(4):
        view = _canonical(grid, corner)
        frame = int(view[3, 3])
        if not (np.all(view[3, :4] == frame) and np.all(view[:4, 3] == frame)):
            continue
        values, counts = np.unique(view[:3, :3], return_counts=True)
        padding = values[counts == 6]
        if len(padding) != 1:
            continue
        background = int(padding[0])
        query = frozenset(int(x) for x in values if x != background)
        if (len(query) == 3 and frame != background and frame not in query
                and np.count_nonzero(grid == frame) == 7):
            selectors.append((corner, background, frame, query))
    assert len(selectors) == 1, "The selector must be uniquely recognizable"
    corner, background, frame, query = selectors[0]
    view = _canonical(grid, corner)
    mask = view != background
    mask[:4, :4] = False
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for r, c in queue:
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
        rows, cols = zip(*queue)
        box = (min(rows), max(rows)+1, min(cols), max(cols)+1)
        r0, r1, c0, c1 = box
        area = (r1-r0)*(c1-c0)
        assert len(queue) == area, "Records must be solid rectangles"
        colors, counts = np.unique(view[r0:r1, c0:c1], return_counts=True)
        components.append({"bbox": tuple(int(x) for x in box), "area": int(area),
                           "counts": {int(c): int(n) for c, n in zip(colors, counts)}})
    assert components
    # Pool evidence across records. Neither a per-record majority nor a mark
    # in every record is required. The common filler is a non-query color
    # present in every record and uniquely most frequent in the pooled data.
    common = set.intersection(*(set(c["counts"]) for c in components))
    common -= set(query) | {background, frame}
    totals = {c: sum(obj["counts"].get(c, 0) for obj in components) for c in common}
    assert totals
    greatest = max(totals.values())
    modes = [c for c, n in totals.items() if n == greatest]
    assert len(modes) == 1, "Common filler must be inferable"
    filler = modes[0]
    boxes = [(0, 4, 0, 4)]
    for obj in components:
        assert background not in obj["counts"] and frame not in obj["counts"]
        assert all(_boxes_have_background_gap(obj["bbox"], other) for other in boxes)
        boxes.append(obj["bbox"])
        obj["labels"] = frozenset(set(obj["counts"]) - {filler})
        obj["marks"] = obj["area"] - obj["counts"][filler]
        obj["positive"] = query <= obj["labels"]
    return {"corner": corner, "background": background, "frame": frame,
            "filler": filler, "query": query, "components": components}


def _solve_grid(grid):
    info = _analyze_input(grid)
    output = grid.copy()
    view = _canonical(output, info["corner"])
    for obj in info["components"]:
        if obj["positive"]:
            r0, r1, c0, c1 = obj["bbox"]
            view[r0:r1, c0:c1] = info["background"]
    return output


def _cuts(box, minimum):
    r0, r1, c0, c1 = box
    choices = []
    for gap in (1, 2, 3):
        for cut in range(r0+1, r1-gap):
            if (cut-r0)*(c1-c0) >= minimum and (r1-cut-gap)*(c1-c0) >= minimum:
                choices.append(((r0, cut, c0, c1), (cut+gap, r1, c0, c1)))
        for cut in range(c0+1, c1-gap):
            if (r1-r0)*(cut-c0) >= minimum and (r1-r0)*(c1-cut-gap) >= minimum:
                choices.append(((r0, r1, c0, cut), (r0, r1, cut+gap, c1)))
    return choices


def _partition(rng, height, width, count, minimum):
    """Random recursive cuts; failed small packings expand deterministically."""
    for expansion in range(20):
        h, w = min(30, height+expansion), min(30, width+expansion)
        for attempt in range(12):
            if int(rng.integers(2)):
                leaves = [(0, h, 5, w), (5, h, 0, 4)]
            else:
                leaves = [(5, h, 0, w), (0, 4, 5, w)]
            leaves = [b for b in leaves if (b[1]-b[0])*(b[3]-b[2]) >= minimum]
            if count == 1:
                leaves = [leaves[int(rng.integers(len(leaves)))]]
            while len(leaves) < count:
                possibilities = [(i, _cuts(b, minimum)) for i, b in enumerate(leaves)]
                possibilities = [(i, cuts) for i, cuts in possibilities if cuts]
                if not possibilities:
                    break
                # Larger free regions are more likely to split, but there are
                # no fixed separator coordinates or mandatory object slots.
                weights = np.array([(leaves[i][1]-leaves[i][0])*(leaves[i][3]-leaves[i][2])
                                    for i, _ in possibilities], dtype=float)
                i, cuts = possibilities[int(rng.choice(len(possibilities), p=weights/weights.sum()))]
                chosen = cuts[int(rng.integers(len(cuts)))]
                leaves.pop(i)
                leaves.extend(chosen)
            if len(leaves) == count:
                order = rng.permutation(count)
                return h, w, [leaves[int(i)] for i in order]
    # A balanced partition of the largest legal canvas is the deterministic
    # fallback. For at most twelve records of minimum area twelve it has ample
    # capacity; it does not impose slots on the ordinary sampling path.
    leaves = [(0, 30, 5, 30), (5, 30, 0, 4)]
    if count == 1:
        leaves = leaves[:1]
    while len(leaves) < count:
        candidates = [(i, cut) for i, box in enumerate(leaves) for cut in _cuts(box, minimum)]
        assert candidates
        i, cut = max(candidates, key=lambda item: min((b[1]-b[0])*(b[3]-b[2]) for b in item[1]))
        leaves.pop(i)
        leaves.extend(cut)
    return 30, 30, leaves


def _shape_choices(height, width, minimum):
    return [(h, w) for h in range(1, height+1) for w in range(1, width+1)
            if h*w >= minimum]


def _crop_box(rng, leaf, minimum, bias, shape=None):
    r0, r1, c0, c1 = leaf
    if shape is None:
        options = _shape_choices(r1-r0, c1-c0, minimum)
        kind = rng.choice(("full", "large", "square", "wide", "tall", "compact", bias))
        if kind == "full":
            options = [(r1-r0, c1-c0)]
        elif kind == "large":
            options = [(h, w) for h, w in options if h*w >= (r1-r0)*(c1-c0)*0.6]
        elif kind == "square":
            deviation = min(abs(h-w) for h, w in options)
            options = [(h, w) for h, w in options if abs(h-w) == deviation]
        elif kind in ("wide", "tall"):
            selected = [(h, w) for h, w in options if (w >= 2*h if kind == "wide" else h >= 2*w)]
            options = selected or options
        elif kind == "compact":
            options = [(h, w) for h, w in options if h*w <= max(minimum*2, min(a*b for a, b in options))]
        shape = options[int(rng.integers(len(options)))]
    h, w = shape
    def offset(slack):
        return int(rng.choice((0, slack, int(rng.integers(slack+1)))))
    row, col = r0+offset(r1-r0-h), c0+offset(c1-c0-w)
    return row, row+h, col, col+w


def _layout(rng, height, width, count, minimum, bias, match_sizes):
    h, w, leaves = _partition(rng, height, width, count, minimum)
    shared = None
    if match_sizes:
        for i in range(count):
            for j in range(i+1, count):
                a, b = leaves[i], leaves[j]
                choices = _shape_choices(min(a[1]-a[0], b[1]-b[0]),
                                         min(a[3]-a[2], b[3]-b[2]), minimum)
                if choices:
                    shared = choices[int(rng.integers(len(choices)))]
                    leaves = [a, b] + [leaf for k, leaf in enumerate(leaves) if k not in (i, j)]
                    break
            if shared:
                break
    boxes = [_crop_box(rng, leaf, minimum, bias, shared if shared and i < 2 else None)
             for i, leaf in enumerate(leaves)]
    return h, w, boxes


def _random_marks(rng, query, extras, area, positive):
    if positive:
        labels = list(query)
        n = int(rng.integers(len(extras)+1))
        labels += list(rng.permutation(extras)[:n])
    else:
        if rng.random() < 0.15:
            return []
        k = int(rng.integers(3))
        labels = list(rng.permutation(query)[:k])
        labels += list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
    if not labels:
        return []
    # Keep the filler visibly supported, including minority-filler records.
    # Start with each chosen identity once; trim optional extras in small boxes.
    while len(labels)+2 > area:
        labels.pop()
    marks = [int(c) for c in labels]
    target = int(rng.integers(len(marks), max(len(marks), int(area*rng.uniform(.15, .73)))+1))
    while len(marks) < target:
        counts = {c: marks.count(c) for c in labels}
        options = [c for c in labels if max(max(counts.values()), counts[c]+1) < area-len(marks)-1]
        if not options:
            break
        marks.append(int(rng.choice(options)))
    return marks


def _make_pair(rng, colors, query, corner, shape, package, bias):
    background, frame, filler = colors[:3]
    extras = [c for c in colors[3:] if c not in query]
    q = list(query)
    x = extras
    if package == "identity":
        # The first pair of records is matched in mark/color count, and in
        # dimensions whenever the free regions admit a common size.
        special = [q, [q[1], q[2], x[0]],
                   [q[0], q[2], x[0], x[1], q[0], x[0]],
                   [q[0], q[1], q[0], q[1], q[0], x[0], x[0]]]
    elif package == "extras":
        used = x[:int(rng.integers(1, min(3, len(x))+1))]
        special = [q+used+[q[0], used[0]],
                   [q[0], q[1]]+used+[q[0], q[1], used[0]], []]
    else:
        special = []
    if special:
        count = len(special)+int(rng.integers(6))
        minimum = 12
        mode = "mixed"
    else:
        count = int(rng.integers(1, 13))
        minimum = 8
        mode = str(rng.choice(("none", "all", "mixed"), p=(.14, .14, .72)))
    h, w, boxes = _layout(rng, *shape, count, minimum, bias, package == "identity")
    grid = np.full((h, w), background, dtype=np.int8)
    grid[3, :4] = frame
    grid[:4, 3] = frame
    positions = rng.choice(9, 3, replace=False)
    for pos, c in zip(positions, rng.permutation(q)):
        grid[int(pos)//3, int(pos)%3] = c
    output = grid.copy()
    # Truth is painted from membership in the actual record's marks. The
    # validator later reconstructs records and roles independently from cells.
    for index, box in enumerate(boxes):
        r0, r1, c0, c1 = box
        area = (r1-r0)*(c1-c0)
        if index < len(special):
            marks = special[index]
        else:
            positive = mode == "all" or (mode == "mixed" and rng.random() < .5)
            marks = _random_marks(rng, q, x, area, positive)
        grid[r0:r1, c0:c1] = filler
        positions = rng.choice(area, len(marks), replace=False)
        for position, color in zip(positions, rng.permutation(marks)):
            grid[r0+int(position)//(c1-c0), c0+int(position)%(c1-c0)] = color
        output[r0:r1, c0:c1] = grid[r0:r1, c0:c1]
        if set(q) <= set(marks):
            output[r0:r1, c0:c1] = background
    return {"input": _canonical(grid, corner).copy(),
            "output": _canonical(output, corner).copy()}


def _array_key(array):
    return array.shape, array.astype(np.int8, copy=False).tobytes()


def _grid_text(grid):
    return "/".join("".join(str(int(c)) for c in row) for row in grid)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Puzzle conventions: the same three-identity predicate and L enclosure,
    # a source-like/moderate/combined variation regime, and optional geometric
    # or familiar-color shifts. Literal role assignments are locally readable.
    regime = str(rng.choice(("source", "moderate", "combined"), p=(.20, .35, .45)))
    vary_roles = regime == "combined" or (regime == "moderate" and rng.random() < .55)
    move_corner = regime == "combined" or (regime == "moderate" and rng.random() < .5)
    shift = str(rng.choice(("shared", "shared", "tall", "wide", "recombine")))
    base_colors = [int(c) for c in rng.permutation(10)]
    active = int(rng.integers(5, 8))
    familiar = base_colors[3:3+active]
    base_query = [int(c) for c in rng.choice(familiar, 3, replace=False)]
    puzzle = {"train": [], "test": []}
    training_corners, training_queries = [], []
    observed_queries = set()
    seen = set()
    previous_roles = None
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            package = ("identity" if index == 0 else "extras" if index == 1 else "free") if split == "train" else "free"
            if vary_roles:
                colors = [int(c) for c in rng.permutation(10)]
                if previous_roles == tuple(colors[:3]):
                    colors[0], colors[1] = colors[1], colors[0]
                lower = 5 if package == "identity" else 4 if package == "extras" else 3
                colors = colors[:3+int(rng.integers(lower, 8))]
                query = [int(c) for c in rng.choice(colors[3:], 3, replace=False)]
            else:
                colors = base_colors[:3]+familiar
                query = base_query if index == 0 and split == "train" else [int(c) for c in rng.choice(familiar, 3, replace=False)]
                if split == "train" and index == 1:
                    if shift == "recombine":
                        new = [c for c in familiar if c not in observed_queries]
                        query = (new+base_query)[:3]
                    elif rng.random() < .40:
                        query = base_query
                if split == "test" and shift == "recombine":
                    eligible = [(a, b, c) for a in familiar for b in familiar for c in familiar
                                if a < b < c and {a, b, c} <= observed_queries
                                and frozenset((a, b, c)) not in training_queries]
                    if eligible:
                        query = list(eligible[int(rng.integers(len(eligible)))])
            previous_roles = tuple(colors[:3])
            if move_corner:
                if split == "train":
                    corner = 0 if index == 0 else int(rng.integers(1, 4)) if index == 1 else int(rng.integers(4))
                else:
                    corner = int(rng.choice(training_corners))
            else:
                corner = 0
            if regime == "source":
                shape = (16, 16)
            else:
                low = 12 if package != "free" else 8
                high = 25 if regime == "moderate" else 31
                shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
                if rng.random() < .15:
                    shape = (16, 16)
            bias = "tall" if split == "test" and shift == "tall" else "wide" if split == "test" and shift == "wide" else "compact" if shift == "tall" else "large"
            for attempt in range(8):
                pair = _make_pair(rng, colors, query, corner, shape, package, bias)
                key = _array_key(pair["input"])
                if key not in seen and _grid_text(pair["input"]) not in _SOURCE_INPUT_TEXTS:
                    break
            else:
                raise AssertionError("Duplicate sampling exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_corners.append(corner)
                training_queries.append(frozenset(query))
                observed_queries.update(query)
    # Teaching witnesses are distributed in random pair order, not a deletion
    # schedule that a solver can exploit.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    records = [(info, obj) for info in infos for obj in info["components"]]
    positives = [(i, c) for i, c in records if c["positive"]]
    negatives = [(i, c) for i, c in records if not c["positive"]]
    assert positives and negatives
    assert any(c["marks"] == 3 for _, c in positives)
    assert any(len(c["labels"]) >= 4 and c["marks"] > 3 for _, c in negatives)
    assert any(not c["labels"] for _, c in negatives)
    assert any(c["labels"]-i["query"] for i, c in positives)
    for group in (positives, negatives):
        assert any(any(c["counts"].get(q, 0) > 1 for q in i["query"])
                   and any(c["counts"][x] > 1 for x in c["labels"]-i["query"])
                   for i, c in group)
    assert any(all(any(not c["positive"] and c["labels"] & i["query"] == i["query"]-{q}
                      for c in i["components"]) for q in i["query"]) for i in infos)
    # A tied mark count and tied distinct-color count defeat either scalar
    # feature even when a different threshold is chosen per puzzle.
    assert any(p["marks"] == n["marks"] and len(p["labels"]) == len(n["labels"])
               for _, p in positives for _, n in negatives)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and min(inp.shape) >= 8
            key = _array_key(inp)
            assert key not in seen
            seen.add(key)
            assert _grid_text(inp) not in _SOURCE_INPUT_TEXTS
            info = _analyze_input(inp)
            assert np.array_equal(_solve_grid(inp), out), "Incorrect whole-record deletion"
            infos[split].append(info)
    _training_coverage(infos["train"])
    corners = {i["corner"] for i in infos["train"]}
    assert all(i["corner"] in corners for i in infos["test"])
    train_roles = {(i["background"], i["frame"], i["filler"]) for i in infos["train"]}
    test_roles = {(i["background"], i["frame"], i["filler"]) for i in infos["test"]}
    assert len(train_roles) > 1 or test_roles <= train_roles


def _measure(puzzles):
    """Task-specific observations reconstructed from the returned grids."""
    values = {name: set() for name in ("heights", "widths", "shapes", "roles", "corners", "palettes",
              "candidates", "deletions", "survivors", "layouts", "queries", "train_counts", "test_counts",
              "selector_patterns", "separations", "densities")}
    counts = {name: 0 for name in ("pairs", "objects", "exact_positive", "extra_positive", "several_extra_positive",
              "empty_negative", "zero_query_negative", "one_query_negative", "two_query_negative",
              "dense", "positive_dense", "negative_dense", "positive_repeat_query", "negative_repeat_query",
              "positive_repeat_extra", "negative_repeat_extra", "all_match", "no_match", "matched_geometry",
              "vary_roles", "vary_corners", "repeat_queries", "source_like", "background_not_dominant",
              "combined_pairs", "positive_edge", "negative_edge", "positive_span", "negative_span")}
    geometry = {f"{split}_{status}_{axis}": set() for split in ("train", "test")
                for status in ("positive", "negative") for axis in ("height", "width", "area", "aspect", "marks", "colors")}
    within = []
    for puzzle in puzzles:
        values["train_counts"].add(len(puzzle["train"]))
        values["test_counts"].add(len(puzzle["test"]))
        local_roles, local_corners, local_queries, local_shapes = set(), set(), [], set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                h, w = grid.shape
                roles = tuple(info[k] for k in ("background", "frame", "filler"))
                objects = info["components"]
                positives = [c for c in objects if c["positive"]]
                negatives = [c for c in objects if not c["positive"]]
                values["heights"].add(h); values["widths"].add(w); values["shapes"].add((h, w))
                values["roles"].add(roles); values["corners"].add(info["corner"])
                values["palettes"].add(len(np.unique(grid)))
                values["candidates"].add(len(objects)); values["deletions"].add(len(positives))
                values["survivors"].add(len(negatives)); values["queries"].add(tuple(sorted(info["query"])))
                values["layouts"].add((h, w, info["corner"], tuple(c["bbox"] for c in objects)))
                selector = _canonical(grid, info["corner"])[:3, :3]
                values["selector_patterns"].add(tuple(int(x) for x in np.flatnonzero(selector.ravel() != info["background"])))
                for j, first in enumerate(objects):
                    for second in objects[j+1:]:
                        a, b = first["bbox"], second["bbox"]
                        values["separations"].add(max(0, a[0]-b[1], b[0]-a[1], a[2]-b[3], b[2]-a[3]))
                local_roles.add(roles); local_corners.add(info["corner"])
                local_queries.append(info["query"]); local_shapes.add((h, w))
                counts["pairs"] += 1
                counts["all_match"] += not negatives
                counts["no_match"] += not positives
                counts["background_not_dominant"] += int(np.count_nonzero(grid == info["background"]) < max(np.bincount(grid.ravel(), minlength=10)))
                counts["combined_pairs"] += info["corner"] != 0 and (h, w) != (16, 16) and len(np.unique(grid)) < 10
                for p in positives:
                    pr0, pr1, pc0, pc1 = p["bbox"]
                    counts["matched_geometry"] += any(p["marks"] == n["marks"] and len(p["labels"]) == len(n["labels"])
                        and (pr1-pr0, pc1-pc0) == (n["bbox"][1]-n["bbox"][0], n["bbox"][3]-n["bbox"][2]) for n in negatives)
                for c in objects:
                    counts["objects"] += 1
                    status = "positive" if c["positive"] else "negative"
                    r0, r1, c0, c1 = c["bbox"]
                    oh, ow = r1-r0, c1-c0
                    observations = {"height": oh, "width": ow, "area": c["area"], "marks": c["marks"],
                                    "colors": len(c["labels"]), "aspect": "square" if oh == ow else "wide" if ow > oh else "tall"}
                    for axis, value in observations.items():
                        geometry[f"{split}_{status}_{axis}"].add(value)
                    counts[status+"_edge"] += r0 == 0 or c0 == 0 or r1 == h or c1 == w
                    counts[status+"_span"] += oh == h or ow == w
                    dense = c["marks"]*2 >= c["area"]
                    values["densities"].add(round(c["marks"]/c["area"], 4))
                    counts["dense"] += dense
                    counts[status+"_dense"] += dense
                    counts[status+"_repeat_query"] += any(c["counts"].get(q, 0) > 1 for q in info["query"])
                    counts[status+"_repeat_extra"] += any(c["counts"][x] > 1 for x in c["labels"]-info["query"])
                    if c["positive"]:
                        counts["exact_positive"] += c["marks"] == 3
                        counts["extra_positive"] += bool(c["labels"]-info["query"])
                        counts["several_extra_positive"] += len(c["labels"]-info["query"]) >= 2
                    else:
                        counts["empty_negative"] += not c["labels"]
                        counts[("zero", "one", "two")[len(c["labels"] & info["query"])]+"_query_negative"] += 1
        counts["vary_roles"] += len(local_roles) > 1
        counts["vary_corners"] += len(local_corners) > 1
        counts["repeat_queries"] += len(set(local_queries)) < len(local_queries)
        counts["source_like"] += local_shapes == {(16, 16)} and local_corners == {0}
        within.append({"roles": len(local_roles), "corners": len(local_corners), "shapes": len(local_shapes)})
    return {"values": {k: sorted(v) for k, v in values.items()}, "counts": counts,
            "geometry": {k: sorted(v) for k, v in geometry.items()}, "within_puzzle": within}


def _selector_counterfactual(puzzle):
    for pair in puzzle["train"]:
        info = _analyze_input(pair["input"])
        options = [c for c in info["components"] if not c["positive"] and len(c["labels"]) == 3
                   and len(c["labels"] & info["query"]) == 2]
        exact = [c for c in info["components"] if c["positive"] and c["marks"] == 3]
        if options and exact:
            changed = pair["input"].copy()
            view = _canonical(changed, info["corner"])
            points = list(zip(*np.nonzero(view[:3, :3] != info["background"])))
            for point, color in zip(points, sorted(options[0]["labels"])):
                view[point] = color
            prediction = _solve_grid(changed)
            solved_view = _canonical(prediction, info["corner"])
            for c in exact:
                r0, r1, c0, c1 = c["bbox"]
                assert np.array_equal(solved_view[r0:r1, c0:c1], view[r0:r1, c0:c1])
            r0, r1, c0, c1 = options[0]["bbox"]
            assert np.all(solved_view[r0:r1, c0:c1] == info["background"])
            # Moving query cells within the selector has no spatial meaning.
            moved = pair["input"].copy()
            moved_view = _canonical(moved, info["corner"])
            moved_view[:3, :3] = np.roll(moved_view[:3, :3].copy(), 1, axis=0)
            expected = pair["output"].copy()
            _canonical(expected, info["corner"])[:3, :3] = moved_view[:3, :3]
            assert np.array_equal(_solve_grid(moved), expected)
            return
    raise AssertionError("No selector-only counterfactual witness")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    def key(puzzle):
        return tuple(tuple((_array_key(p["input"]), _array_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))
    assert len({key(p) for p in puzzles}) == 42
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _selector_counterfactual(puzzle)
        if seed < 3:
            assert key(generate_puzzle(seed)) == key(puzzle)
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(431, a, b)
        validate_generated_puzzle(p)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
        assert key(p) == key(generate_puzzle(431, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(42, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # The source's sparse/empty and minority-filler examples must be readable.
    originals = [np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
                 for text in sorted(_SOURCE_INPUT_TEXTS)]
    analyzed = [_analyze_input(grid) for grid in originals]
    assert any(not c["labels"] for i in analyzed for c in i["components"])
    assert any(c["marks"]*2 > c["area"] for i in analyzed for c in i["components"])
    metrics = _measure(puzzles)
    v, c, g = metrics["values"], metrics["counts"], metrics["geometry"]
    assert len(v["train_counts"]) >= 4 and len(v["test_counts"]) >= 3
    assert len(v["heights"]) >= 15 and len(v["widths"]) >= 15 and (16, 16) in v["shapes"]
    assert len(v["shapes"]) > 100 and len(v["layouts"]) > 250
    assert v["corners"] == [0, 1, 2, 3] and len(v["roles"]) >= 60
    assert all({roles[i] for roles in v["roles"]} == set(range(10)) for i in range(3))
    assert len(v["selector_patterns"]) >= 70 and min(v["separations"]) == 1 and max(v["separations"]) >= 5
    assert min(v["densities"]) == 0 and max(v["densities"]) > .60
    assert len(v["palettes"]) >= 4 and len(v["candidates"]) >= 10
    assert min(v["candidates"]) == 1 and max(v["candidates"]) == 12
    assert len(v["deletions"]) >= 8 and len(v["survivors"]) >= 8
    assert c["source_like"] >= 2 and c["vary_roles"] >= 15 and c["vary_corners"] >= 15
    assert c["repeat_queries"] >= 8 and c["matched_geometry"] >= 30
    for name in ("dense", "positive_dense", "negative_dense", "all_match", "no_match", "combined_pairs",
                 "background_not_dominant", "several_extra_positive", "positive_span", "negative_span"):
        assert c[name] >= 5, (name, c[name])
    for split in ("train", "test"):
        for status in ("positive", "negative"):
            prefix = f"{split}_{status}_"
            assert g[prefix+"aspect"] == ["square", "tall", "wide"]
            assert min(g[prefix+"height"]) == 1 and min(g[prefix+"width"]) == 1
            assert max(g[prefix+"area"]) >= 80
    # Mutation checks target each structural/output invariant explicitly.
    for corruption in ("partial_deletion", "changed_survivor", "selector_padding"):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        for pair in bad["train"]:
            info = _analyze_input(pair["input"])
            if any(c["positive"] for c in info["components"]) and any(not c["positive"] for c in info["components"]):
                break
        if corruption == "selector_padding":
            view = _canonical(pair["input"], info["corner"])
            row, col = np.argwhere(view[:3, :3] == info["background"])[0]
            view[row, col] = info["filler"]
        else:
            obj = next(c for c in info["components"] if c["positive"] == (corruption == "partial_deletion"))
            r0, _, c0, _ = obj["bbox"]
            view = _canonical(pair["output"], info["corner"])
            view[r0, c0] = info["filler"] if corruption == "partial_deletion" else info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted puzzle accepted")
