"""A record disappears when the selector colors share a vertical mirror witness.

Only standard-library modules and NumPy are used. The solver reads visible cells;
construction metadata, the seed, and training outputs never enter inference.
"""
from itertools import combinations
import numpy as np


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return array.astype(np.int8, copy=False)


def _canonical(grid, corner):
    if corner & 2:
        grid = grid[::-1, :]
    if corner & 1:
        grid = grid[:, ::-1]
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _gap(a, b):
    # Half-open boxes with at least one complete background row or column.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _axis_scores(block, color):
    """Score counts supported cells, not pairs. Axis is twice its column."""
    mask = (block == color).astype(np.int16)
    scores = np.zeros(2 * block.shape[1] - 1, dtype=int)
    for row in mask:
        scores += np.convolve(row, row)
    return scores


def _certificates(block, query):
    eligible = set(range(2 * block.shape[1] - 1))
    for color in query:
        eligible.intersection_update(map(int, np.flatnonzero(_axis_scores(block, color))))
    return eligible


def _best_pruned(block, query, filler):
    """Deliberately simpler alternative, used only for counterexample checks."""
    result = np.full_like(block, filler)
    for color in query:
        scores = _axis_scores(block, color)
        winners = np.flatnonzero(scores == scores.max())
        if scores.max() == 0 or len(winners) != 1:
            return None
        axis = int(winners[0])
        for r, c in np.argwhere(block == color):
            partner = axis - int(c)
            if 0 <= partner < block.shape[1] and block[r, partner] == color:
                result[r, c] = color
    return result


def _strongest_first_fails(block, query, certificates):
    """Whether all maxima of total selected-color support miss a query color."""
    scores = sum(_axis_scores(block, q) for q in query)
    return bool(certificates) and max(scores) > max(scores[a] for a in certificates)


def _analyze(input_grid):
    grid = _grid(input_grid)
    if min(grid.shape) < 5:
        raise ValueError("The corner selector needs a 4x4 enclosure and a gap")
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
        query = frozenset(int(c) for c in values if c != background)
        if (len(query) == 3 and frame not in query and frame != background
                and np.count_nonzero(grid == frame) == 7):
            selectors.append((corner, background, frame, query))
    if len(selectors) != 1:
        raise ValueError("Expected one uniquely recognizable corner selector")
    corner, background, frame, query = selectors[0]
    view = _canonical(grid, corner)
    mask = view != background
    mask[:4, :4] = False
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    records = []
    boxes = [(0, 4, 0, 4)]
    filler = None
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    cells.append(point)
        rows, cols = zip(*cells)
        box = (min(rows), max(rows)+1, min(cols), max(cols)+1)
        r0, r1, c0, c1 = box
        block = view[r0:r1, c0:c1]
        if min(block.shape) < 3 or len(cells) != block.size:
            raise ValueError("Records must be filled rectangles of height and width >= 3")
        if any(not _gap(box, other) for other in boxes):
            raise ValueError("Record boxes need a background gap")
        boxes.append(box)
        local_filler = int(block[0, 0])
        if not (np.all(block[0] == local_filler) and np.all(block[-1] == local_filler)
                and np.all(block[:, 0] == local_filler) and np.all(block[:, -1] == local_filler)):
            raise ValueError("The complete perimeter must identify the common filler")
        if local_filler in query or local_filler in (background, frame):
            raise ValueError("Filler, background, frame and label roles must be distinct")
        if filler is not None and local_filler != filler:
            raise ValueError("All records must use the same visible filler")
        filler = local_filler
        colors, counts = np.unique(block, return_counts=True)
        labels = frozenset(int(c) for c in colors if c != filler)
        if background in labels or frame in labels:
            raise ValueError("Structural colors cannot occur as marks")
        axes = _certificates(block, query)
        records.append({"box": box, "block": block, "labels": labels,
                        "counts": {int(c): int(n) for c, n in zip(colors, counts)},
                        "axes": axes, "positive": bool(axes),
                        "marks": int(np.count_nonzero(block != filler))})
    if not records:
        raise ValueError("At least one record is required")
    return {"corner": corner, "background": background, "frame": frame,
            "filler": filler, "query": query, "records": records}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase whole records with a shared axis; retain everything else verbatim."""
    grid = _grid(input_grid)
    info = _analyze(grid)
    output = grid.copy()
    view = _canonical(output, info["corner"])
    for record in info["records"]:
        if record["positive"]:
            r0, r1, c0, c1 = record["box"]
            view[r0:r1, c0:c1] = info["background"]
    return output


def _minimum(kind):
    return {"matched_positive": (5, 9), "matched_negative": (5, 9),
            "missing_exact": (5, 9), "missing_dense": (5, 9),
            "missing": (5, 8), "nonmax": (7, 13), "self": (5, 7),
            "positive": (5, 3), "disagree": (5, 8),
            "random_negative": (3, 5), "empty": (3, 3)}[kind]


def _extras(block, rng, colors, amount):
    points = [(r, c) for r in range(1, block.shape[0]-1)
              for c in range(1, block.shape[1]-1) if block[r, c] == block[0, 0]]
    if not points or not colors:
        return
    for index in rng.permutation(len(points))[:min(amount, len(points))]:
        block[points[int(index)]] = int(rng.choice(colors))


def _make_record(rng, shape, kind, query, extras, filler, missing_index=0):
    h, w = shape
    block = np.full(shape, filler, dtype=np.int8)
    q = list(map(int, rng.permutation(query)))
    extra = list(map(int, rng.permutation(extras)))
    rows = list(map(int, rng.permutation(np.arange(1, h-1))))
    if kind in ("matched_positive", "matched_negative", "missing_exact", "missing_dense", "missing"):
        shift = int(rng.integers(w-7))
        if kind == "matched_negative":
            pairs = ((1, 2), (3, 4), (5, 6))
        else:
            pairs = ((1, 6), (2, 5), (3, 4))
        # Missing witnesses refer to the unshuffled query order.
        labels = q.copy()
        if kind.startswith("missing"):
            absent = int(query[missing_index])
            labels[labels.index(absent)] = extra[0]
        for row, color, (left, right) in zip(rows, labels, pairs):
            block[row, left+shift] = block[row, right+shift] = color
        if kind == "missing_dense":
            # At least four identities and >6 marks, while one Q is absent.
            point = next((r, c) for r in range(1, h-1) for c in range(1, w-1)
                         if block[r, c] == filler)
            block[point] = extra[1]
            _extras(block, rng, labels+extra[:2], max(3, (h-2)*(w-2)//2))
        elif kind == "missing":
            _extras(block, rng, labels+extra, int(rng.integers(0, (h-2)*(w-2)//2+1)))
    elif kind == "nonmax":
        # Common axis 4 is weaker than each color's unique best axis.
        # Competitor pairs are nonoverlapping, and survive separate pruning.
        shift = int(rng.integers(w-12))
        for row, color in zip(rows[:3], q):
            block[row, 1+shift] = block[row, 3+shift] = color
        for color, (left, right) in zip(q, ((6, 10), (4, 8), (9, 11))):
            for row in rows[3:5]:
                block[row, left+shift] = block[row, right+shift] = color
        # This singleton raises aggregate Q support at axis 16 to seven,
        # above the common witness's six, without contributing the third Q.
        # Thus choosing a globally strongest axis before testing membership
        # also fails, independently of the irrelevant extra-color clutter.
        block[rows[0], 8+shift] = q[0]
        # Extras cannot vote, even when they have stronger geometric evidence.
        _extras(block, rng, extra, max(1, int(rng.integers(1, (h-2)*(w-2)//2+1))))
    elif kind == "self":
        shift = int(rng.integers(w-6))
        for row, color, cols in zip(rows, q, ((1, 5), (2, 4), (3,))):
            for col in cols:
                block[row, col+shift] = color
        _extras(block, rng, extra, int(rng.integers((h-2)*(w-2)//3+1)))
    elif kind == "disagree":
        shift = int(rng.integers(w-7))
        for color, cols in zip(q, ((1, 2), (3, 4), (5, 6))):
            row = int(rng.integers(1, h-1))
            for col in cols:
                block[row, col+shift] = color
            # Repetition inside disjoint column intervals preserves disagreement.
            for r in rows[:int(rng.integers(len(rows)+1))]:
                for col in cols:
                    if rng.random() < .7:
                        block[r, col+shift] = color
        _extras(block, rng, extra, int(rng.integers((h-2)*(w-2)//2+1)))
    elif kind == "positive":
        alternatives = []
        for axis in range(2, 2*w-3):
            orbits = [(r, c, axis-c) for r in range(1, h-1) for c in range(1, w-1)
                      if c <= axis-c < w-1]
            if len(orbits) >= 3:
                alternatives.append((axis, orbits))
        axis, orbits = alternatives[int(rng.integers(len(alternatives)))]
        # First prefer off-axis evidence, while supporting thin integral cases.
        pairs = [o for o in orbits if o[1] != o[2]]
        pool = pairs if len(pairs) >= 3 and rng.random() < .8 else orbits
        for i, color in zip(rng.permutation(len(pool))[:3], q):
            r, c, other = pool[int(i)]
            block[r, c] = block[r, other] = color
        # Added unmatched Q marks and extras cannot remove an existing witness.
        _extras(block, rng, q+extra, int(rng.integers((h-2)*(w-2)*2//3+1)))
    elif kind == "random_negative":
        for _ in range(40):
            block[:] = filler
            # These three singletons initially disagree; then add varied clutter.
            cols = rng.choice(np.arange(1, w-1), 3, replace=False)
            for color, col in zip(q, cols):
                block[int(rng.integers(1, h-1)), int(col)] = color
            _extras(block, rng, q+extra, int(rng.integers((h-2)*(w-2)*2//3+1)))
            if not _certificates(block, query):
                break
        else:
            block[:] = filler
            for color, col in zip(q, (1, 2, 3)):
                block[1, col] = color
    elif kind != "empty":
        raise AssertionError("Unknown record kind")
    positive = kind in ("matched_positive", "nonmax", "self", "positive")
    assert bool(_certificates(block, query)) == positive
    return block, positive


def _sizes(rng, kinds, package):
    sizes = []
    for kind in kinds:
        mh, mw = _minimum(kind)
        h = mh + int(rng.integers(0, 5))
        w = mw + int(rng.integers(0, 5))
        if kind == "positive" and rng.random() < .3:
            w, h = 3, int(rng.integers(5, 15))
        if kind == "random_negative" and rng.random() < .25:
            h, w = 3, int(rng.integers(5, 18))
        sizes.append((h, w))
    if package == "identity":
        sizes[1] = sizes[0]
    return sizes


def _random_pack(rng, shape, sizes):
    h, w = shape
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][0]*sizes[i][1], float(rng.random())))
    boxes = [(0, 4, 0, 4)]
    result = [None] * len(sizes)
    for i in order:
        rh, rw = sizes[i]
        possible = [(r, r+rh, c, c+rw) for r in range(h-rh+1) for c in range(w-rw+1)
                    if all(_gap((r, r+rh, c, c+rw), other) for other in boxes)]
        if not possible:
            return None
        # Either class can touch a grid edge; membership does not choose positions.
        edge = [b for b in possible if b[0] == 0 or b[2] == 0 or b[1] == h or b[3] == w]
        pool = edge if edge and rng.random() < .25 else possible
        box = pool[int(rng.integers(len(pool)))]
        result[i] = box
        boxes.append(box)
    return result


def _layout(rng, kinds, package):
    sizes = _sizes(rng, kinds, package)
    count = len(kinds)
    low = 21 if count >= 5 else 16 if package != "free" else 10
    initial = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
    for phase in range(3):
        shape = initial if phase == 0 else (30, 30)
        if phase == 2:
            sizes = [_minimum(k) for k in kinds]
        for _ in range(8):
            boxes = _random_pack(rng, shape, sizes)
            if boxes is not None:
                return shape, boxes
    # Guaranteed shelf fallback using minimum sizes, well below the 30x30 cap.
    # Ordinary layouts have independently sampled coordinates, not these shelves.
    boxes = [None] * count
    row, col, shelf_height = 5, 0, 0
    order = sorted(range(count), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    for i in order:
        rh, rw = sizes[i]
        if col + rw > 30:
            row += shelf_height + 1
            col, shelf_height = 0, 0
        assert row + rh <= 30
        boxes[i] = (row, row+rh, col, col+rw)
        col += rw + 1
        shelf_height = max(shelf_height, rh)
    return (30, 30), boxes


def _free_kinds(rng, count, is_test):
    mode = str(rng.choice(("all", "none", "mixed"), p=(.12, .12, .76)))
    positives = ("positive", "self", "nonmax")
    negatives = ("missing", "disagree", "random_negative", "empty")
    choices = positives if mode == "all" else negatives if mode == "none" else positives+negatives
    kinds = [str(rng.choice(choices)) for _ in range(count)]
    # Limit the large 7x13 construction to two free records. Smaller witnesses
    # then guarantee minimum-size packing even at seven records on a 30x30 grid.
    large = 0
    for i, kind in enumerate(kinds):
        if kind == "nonmax":
            large += 1
            if large > 2:
                kinds[i] = str(rng.choice(("positive", "self")))
    if is_test and kinds[0] in ("missing", "empty"):
        kinds[0] = "disagree" if mode == "none" else str(rng.choice(("positive", "random_negative")))
    return kinds


def _make_pair(rng, query, label_pool, roles, corner, package, is_test):
    if package == "identity":
        kinds = ["matched_positive", "matched_negative", "missing_exact", "missing_dense", "missing"]
        kinds += _free_kinds(rng, int(rng.integers(3)), False)
    elif package == "interaction":
        kinds = ["nonmax", "self", "empty"]
        kinds += _free_kinds(rng, int(rng.integers(3)), False)
    else:
        kinds = _free_kinds(rng, int(rng.integers(1, 8)), is_test)
    shape, boxes = _layout(rng, kinds, package)
    background, frame, filler = roles
    extras = [c for c in label_pool if c not in query]
    inp = np.full(shape, background, dtype=np.int8)
    inp[3, :4] = frame
    inp[:4, 3] = frame
    for point, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        inp[int(point)//3, int(point)%3] = color
    out = inp.copy()
    # Truth comes from the construction type, then solve/validator recompute it.
    for index, (kind, box) in enumerate(zip(kinds, boxes)):
        r0, r1, c0, c1 = box
        missing = index-2 if package == "identity" and 2 <= index <= 4 else int(rng.integers(3))
        block, positive = _make_record(rng, (r1-r0, c1-c0), kind, query, extras, filler, missing)
        inp[r0:r1, c0:c1] = block
        out[r0:r1, c0:c1] = background if positive else block
    return {"input": _canonical(inp, corner).copy(), "output": _canonical(out, corner).copy()}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    structural = palette[:3]
    labels = palette[3:3+int(rng.integers(5, 8))]
    # Test queries recombine familiar identities but are reserved from training.
    first = tuple(labels[:3])
    second = tuple(labels[2:5])
    familiar = labels[:5]
    other = [q for q in combinations(familiar, 3) if frozenset(q) not in (frozenset(first), frozenset(second))]
    order = rng.permutation(len(other))
    test_queries = [other[int(i)] for i in order[:num_test]]
    reserved = {frozenset(q) for q in test_queries}
    train_options = [q for q in combinations(familiar, 3) if frozenset(q) not in reserved]
    queries = [first, second] + [train_options[int(rng.integers(len(train_options)))] for _ in range(num_train-2)]
    vary_roles = bool(rng.integers(2))
    moved_corners = bool(rng.integers(2))
    first_corner = int(rng.integers(4))
    second_corner = (first_corner + int(rng.integers(1, 4))) % 4 if moved_corners else first_corner
    train_corners = [first_corner, second_corner]
    train_corners += [int(rng.integers(4)) if moved_corners else first_corner for _ in range(num_train-2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            query = queries[i] if split == "train" else test_queries[i]
            corner = train_corners[i] if split == "train" else int(rng.choice(train_corners))
            roles = tuple(map(int, rng.permutation(structural))) if vary_roles else tuple(structural)
            if split == "train" and i == 1 and vary_roles and roles == first_roles:
                roles = (roles[1], roles[0], roles[2])
            if split == "train" and i == 0:
                first_roles = roles
            package = "identity" if split == "train" and i == 0 else "interaction" if split == "train" and i == 1 else "free"
            for _ in range(8):
                pair = _make_pair(rng, query, labels, roles, corner, package, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not sample a distinct input after bounded retries")
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    pairs = [(i, r) for i in infos for r in i["records"]]
    positive = [(i, r) for i, r in pairs if r["positive"]]
    negative = [(i, r) for i, r in pairs if not r["positive"]]
    assert positive and negative
    # Scalar counts, literal containment and independent per-color symmetry fail.
    assert any(i["query"] <= r["labels"] for i, r in negative)
    assert any(len(r["labels"]) >= 4 and r["marks"] > 6 for _, r in negative)
    assert any(not r["labels"] for _, r in negative)
    assert any(r["labels"] - i["query"] for i, r in positive)
    assert any(all(any(not r["positive"] and r["labels"] & i["query"] == i["query"]-{q}
                       for r in i["records"]) for q in i["query"]) for i in infos)
    assert any(p["block"].shape == n["block"].shape and p["marks"] == n["marks"] == 6
               and p["labels"] == n["labels"] == i["query"]
               and all(p["counts"][q] == n["counts"][q] == 2 for q in i["query"])
               for i in infos for p in i["records"] if p["positive"]
               for n in i["records"] if not n["positive"])
    assert any(r["positive"] and (pruned := _best_pruned(r["block"], i["query"], i["filler"])) is not None
               and not _certificates(pruned, i["query"]) for i, r in positive)
    assert any(_strongest_first_fails(r["block"], i["query"], r["axes"]) for i, r in positive)
    assert any(len(r["labels"]) == 3 and len(r["labels"] & i["query"]) == 2
               and _certificates(r["block"], r["labels"]) for i, r in negative)
    assert any(any(r["counts"].get(q, 0) == 1 for q in i["query"]) for i, r in positive)
    assert {a % 2 for _, r in positive for a in r["axes"]} == {0, 1}
    assert any(any(a != r["block"].shape[1]-1 for a in r["axes"]) for _, r in positive)
    # No common occupied column, yet off-axis pairs certify this record.
    assert any(not set.intersection(*(set(map(int, np.argwhere(r["block"] == q)[:, 1]))
                                      for q in i["query"])) for i, r in positive)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            infos[split].append(info)
            assert np.array_equal(solve(inp), out), "Output violates shared-axis whole-record deletion"
            assert np.all((inp == out) | (out == info["background"])), "Deletion only"
            actual = _canonical(out, info["corner"])
            original = _canonical(inp, info["corner"])
            assert np.array_equal(actual[:4, :4], original[:4, :4]), "Selector must be copied"
            for record in info["records"]:
                r0, r1, c0, c1 = record["box"]
                piece = actual[r0:r1, c0:c1]
                assert np.all(piece == info["background"]) if record["positive"] else np.array_equal(piece, record["block"])
    _training_coverage(infos["train"])
    train_queries = {i["query"] for i in infos["train"]}
    assert len({i["query"] for i in infos["test"]}) == len(infos["test"])
    familiar = set.union(*(set(q) for q in train_queries))
    corners = {i["corner"] for i in infos["train"]}
    for info in infos["test"]:
        assert info["query"] not in train_queries and info["query"] <= familiar
        assert info["corner"] in corners
        assert any(info["query"] <= r["labels"] for r in info["records"])


def _selector_counterfactual(puzzle):
    for pair in puzzle["train"]:
        info = _analyze(pair["input"])
        positives = [r for r in info["records"] if r["positive"] and r["labels"] == info["query"]]
        negatives = [r for r in info["records"] if not r["positive"] and len(r["labels"]) == 3
                     and len(r["labels"] & info["query"]) == 2 and _certificates(r["block"], r["labels"])]
        if not positives or not negatives:
            continue
        changed = pair["input"].copy()
        view = _canonical(changed, info["corner"])
        points = np.argwhere(view[:3, :3] != info["background"])
        for point, color in zip(points, sorted(negatives[0]["labels"])):
            view[tuple(point)] = color
        solved = _canonical(solve(changed), info["corner"])
        for r in positives:
            r0, r1, c0, c1 = r["box"]
            assert np.array_equal(solved[r0:r1, c0:c1], r["block"])
        r0, r1, c0, c1 = negatives[0]["box"]
        assert np.all(solved[r0:r1, c0:c1] == info["background"])
        return
    raise AssertionError("Missing selector-only counterfactual")


def _check_rule_examples():
    # Literal specification examples: known certificates, rather than inferred
    # construction labels. The last negative has equal column sums but the
    # supposed partners occupy different rows and cannot support one another.
    examples = (
        ("000000000/010000100/002002000/000330000/000000000", {7}),
        ("000000000/011000000/000220000/000003300/000000000", set()),
        ("000000000/010000100/002002000/000440000/000000000", set()),
        ("0000000/0100010/0020200/0003000/0000000", {6}),
        ("00000/01010/02020/03030/00000", {2, 4, 6}),
        ("000000000/010030000/002000100/000302000/000000000", set()),
    )
    for text, axes in examples:
        block = np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
        assert _certificates(block, (1, 2, 3)) == axes
        inp = np.full((block.shape[0]+5, max(5, block.shape[1])), 8, dtype=np.int8)
        inp[3, :4] = inp[:4, 3] = 9
        inp[0, 0], inp[1, 1], inp[2, 2] = 1, 2, 3
        inp[5:, :block.shape[1]] = block
        expected = inp.copy()
        if axes:
            expected[5:, :block.shape[1]] = 8
        for corner in range(4):
            original = _canonical(inp, corner).copy()
            before = original.copy()
            assert np.array_equal(solve(original), _canonical(expected, corner))
            assert np.array_equal(original, before)
            assert np.array_equal(solve(original.tolist()), _canonical(expected, corner))
    # Filler is identified by the perimeter even when it is a minority. This
    # also exercises a record spanning the full width of its canvas.
    dense = np.full((25, 20), 8, dtype=np.int8)
    dense[3, :4] = dense[:4, 3] = 9
    dense[0, 0], dense[1, 1], dense[2, 2] = 1, 2, 3
    dense[5:] = 0
    for r in range(6, 24):
        dense[r, 1:-1] = 1 + r % 3
    expected = dense.copy()
    expected[5:] = 8
    assert np.array_equal(solve(dense), expected)


def self_test():
    """Bounded 42-seed structural, semantic, counterfactual and diversity checks."""
    shapes, roles, corners, counts, geometries, layouts, axes, fingerprints = (set() for _ in range(8))
    _check_rule_examples()
    tally = {k: 0 for k in ("pairs", "records", "deleted", "raw_membership_negative",
                            "nonmax_positive", "aggregate_nonmax_positive", "axis_cell_positive",
                            "empty", "edge", "all", "none")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _selector_counterfactual(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ("input", "output"))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze(grid)
                shapes.add(grid.shape)
                roles.add((info["background"], info["frame"], info["filler"]))
                corners.add(info["corner"])
                counts.add(len(info["records"]))
                layouts.add((grid.shape, info["corner"], tuple(r["box"] for r in info["records"])))
                tally["pairs"] += 1
                positives = sum(r["positive"] for r in info["records"])
                tally["all"] += positives == len(info["records"])
                tally["none"] += positives == 0
                for r in info["records"]:
                    geometries.add(r["block"].shape)
                    axes.update(a % 2 for a in r["axes"])
                    tally["records"] += 1
                    tally["deleted"] += r["positive"]
                    tally["raw_membership_negative"] += not r["positive"] and info["query"] <= r["labels"]
                    pruned = _best_pruned(r["block"], info["query"], info["filler"])
                    tally["nonmax_positive"] += r["positive"] and pruned is not None and not _certificates(pruned, info["query"])
                    tally["aggregate_nonmax_positive"] += _strongest_first_fails(r["block"], info["query"], r["axes"])
                    tally["axis_cell_positive"] += r["positive"] and any(r["counts"].get(q, 0) == 1 for q in info["query"])
                    tally["empty"] += not r["labels"]
                    r0, r1, c0, c1 = r["box"]
                    tally["edge"] += r0 == 0 or c0 == 0 or r1 == grid.shape[0] or c1 == grid.shape[1]
        # Relabeling roles, including color 0, cannot alter decisions.
        pair = puzzle["test"][0]
        permutation = np.random.default_rng(9000+seed).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        info = _analyze(pair["input"])
        moved = pair["input"].copy()
        expected = pair["output"].copy()
        interior = _canonical(moved, info["corner"])
        interior[:3, :3] = np.roll(interior[:3, :3].copy(), 1, axis=0)
        _canonical(expected, info["corner"])[:3, :3] = interior[:3, :3]
        assert np.array_equal(solve(moved), expected)
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(layouts) >= 180
    assert len(geometries) >= 35 and corners == {0, 1, 2, 3} and axes == {0, 1}
    assert {1, 3, 5, 7} <= counts
    assert all({r[i] for r in roles} == set(range(10)) for i in range(3))
    assert all(tally[k] >= 42 for k in ("raw_membership_negative", "nonmax_positive",
                                       "aggregate_nonmax_positive", "axis_cell_positive", "empty"))
    assert tally["edge"] > 20 and tally["all"] > 0 and tally["none"] > 0
    # Check every legal explicit count pair, and reject types as well as ranges.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(12000+a*10+b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Reject partial erasure and deletion of a nonqualifying record.
    for wanted in (True, False):
        bad = generate_puzzle(0, 2, 1)
        for pair in bad["train"]:
            info = _analyze(pair["input"])
            records = [r for r in info["records"] if r["positive"] == wanted]
            if records:
                r0, _, c0, _ = records[0]["box"]
                _canonical(pair["output"], info["corner"])[r0, c0] = info["filler"] if wanted else info["background"]
                break
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Invalid output accepted")
    return {"seeds": 42, "grid_shapes": len(shapes), "record_shapes": len(geometries),
            "layouts": len(layouts), "role_assignments": len(roles), "record_counts": sorted(counts),
            "axis_parities": sorted(axes), "tally": {k: int(v) for k, v in tally.items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
