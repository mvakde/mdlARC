"""Joint periodic repair and four-connected cavity checksums.

The sampler constructs clean templates first. The decoder reads only its input;
it does not consult the sampler, a seed, examples, files, or a color dictionary.
"""
from itertools import combinations, product

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                other = r + dr, c + dc
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        result.append(cells)
    return result


def _count(mask, diagonal=False):
    return len(_components(mask, diagonal))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _repair_suffix(suffix, background, carrier, checksum):
    """Enumerate minimum-distance majority/tie templates, then check topology."""
    h, length = suffix.shape
    holes = suffix == background
    allowance = max(2, suffix.size // 8)
    best_cost, supports, answer = allowance + 1, [], None
    for width in range(3, length // 3 + 1):
        unit = np.zeros((h, width), dtype=bool)
        tied = []
        cost = 0
        valid = True
        all_ties = 0
        for c in range(width):
            observations = holes[:, c::width]
            counts = observations.sum(axis=1)
            n = observations.shape[1]
            cost += int(np.minimum(counts, n - counts).sum())
            for r in range(h):
                boundary = r in (0, h - 1) or c in (0, width - 1)
                if 2 * counts[r] == n:
                    all_ties += 1
                    if not boundary:
                        tied.append((r, c))
                elif 2 * counts[r] > n:
                    if boundary:
                        valid = False
                    unit[r, c] = True
        if not valid or cost > allowance or cost > best_cost:
            continue
        # Each tie costs at least two edits (three observations cannot tie).
        # The sparse error budget bounds enumeration, without a hidden tie cap.
        for values in product((False, True), repeat=len(tied)):
            candidate = unit.copy()
            for cell, value in zip(tied, values):
                candidate[cell] = value
            if _count(candidate) != checksum:
                continue
            repaired_holes = candidate[:, np.arange(length) % width]
            repaired = np.where(repaired_holes, background, carrier).astype(np.int8)
            if cost < best_cost:
                best_cost, answer, supports = cost, repaired, []
            if cost == best_cost:
                supports.append((width, candidate.copy(), tuple(tied), all_ties, repaired))
    if answer is None:
        raise ValueError("no supported periodic template has the tab's cavity count")
    if any(not np.array_equal(item[4], answer) for item in supports):
        raise ValueError("checksum leaves conflicting minimum-distance repairs")
    width, unit, tied, all_ties, _ = supports[0]
    tie_mask = np.zeros_like(holes)
    for r, c in tied:
        tie_mask[r, c::width] = True
    different = suffix != answer
    copies = length // width
    info = {
        "width": width, "unit": unit, "tied": tied,
        "max_ties": max(len(item[2]) for item in supports),
        "supported_widths": tuple(item[0] for item in supports),
        "errors": best_cost, "strict_changes": int((different & ~tie_mask).sum()),
        "tied_changes": int((different & tie_mask).sum()),
        "partial": length % width,
        "partial_changes": int(different[:, copies * width:].sum()),
        "no_clean_copy": all(not np.array_equal(holes[:, j * width:(j + 1) * width], unit)
                             for j in range(copies)),
        "count_correct_damaged_copy": any(
            _count(holes[:, j * width:(j + 1) * width]) == checksum
            and not np.array_equal(holes[:, j * width:(j + 1) * width], unit)
            for j in range(copies)),
        "cavity_sizes": tuple(sorted(map(len, _components(unit)))),
        "connectivity_witness": _count(unit, True) < checksum,
    }
    return answer, info


def _interpret(grid, background, marker, boxes):
    ribbons = []
    previous_end = -1
    for r0, r1, c0, c1 in sorted(boxes):
        if r0 <= previous_end:
            raise ValueError("ribbons need a separating background row")
        previous_end = r1
        box = grid[r0:r1, c0:c1]
        h, total = box.shape
        if not 4 <= h <= 8:
            raise ValueError("invalid ribbon thickness")
        carrier = int(box[0, 0])
        if carrier in (background, marker):
            raise ValueError("carrier has a reserved role")
        spines = np.flatnonzero(np.all(box == marker, axis=0))
        if len(spines) != 1:
            raise ValueError("a ribbon needs one marker spine")
        spine = int(spines[0])
        if not 3 <= spine <= 6 or total - spine - 1 < 9:
            raise ValueError("invalid tab or suffix dimensions")
        tab, suffix = box[:, :spine], box[:, spine + 1:]
        if not set(map(int, np.unique(tab))) <= {marker, carrier}:
            raise ValueError("tab uses other colors")
        if not (np.all(tab[0] == carrier) and np.all(tab[-1] == carrier)
                and np.all(tab[:, 0] == carrier) and np.all(tab[:, -1] == carrier)):
            raise ValueError("tab perimeter is damaged")
        dots = tab == marker
        if np.any(dots[:-1] & dots[1:]) or np.any(dots[:, :-1] & dots[:, 1:]):
            raise ValueError("counter dots must be four-neighbor isolated")
        if not set(map(int, np.unique(suffix))) <= {background, carrier}:
            raise ValueError("suffix is not binary")
        if not np.all(suffix[[0, -1]] == carrier):
            raise ValueError("suffix cross-axis edges must be intact")
        checksum = int(dots.sum())
        repaired, info = _repair_suffix(suffix, background, carrier, checksum)
        info.update({"box": (r0, r1, c0, c1), "spine": spine,
                     "carrier": carrier, "checksum": checksum,
                     "suffix": suffix.copy(), "repaired": repaired,
                     "thickness": h, "length": suffix.shape[1]})
        ribbons.append(info)
    if not 1 <= len(ribbons) <= 3:
        raise ValueError("invalid ribbon count")
    output = grid.copy()
    for info in ribbons:
        r0, r1, c0, c1 = info["box"]
        output[r0:r1, c0 + info["spine"] + 1:c1] = info["repaired"]
    return output, ribbons


def _decode(input_grid):
    original = _grid(input_grid)
    interpretations = []
    for vertical in (False, True):
        grid = original.T if vertical else original
        colors = list(map(int, np.unique(grid)))
        for background in colors:
            components = _components(grid != background)
            if not 1 <= len(components) <= 3:
                continue
            boxes = []
            for cells in components:
                rs, cs = zip(*cells)
                boxes.append((min(rs), max(rs) + 1, min(cs), max(cs) + 1))
            if any(not 4 <= b - a <= 8 or d - c < 13 for a, b, c, d in boxes):
                continue
            for marker in colors:
                if marker == background:
                    continue
                try:
                    output, ribbons = _interpret(grid, background, marker, boxes)
                except ValueError:
                    continue
                interpretations.append((output.T.copy() if vertical else output,
                                        {"vertical": vertical, "background": background,
                                         "marker": marker, "ribbons": ribbons}))
    if len(interpretations) != 1:
        raise ValueError("input must have one background, spine and orientation interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely supported repair derived entirely from this input."""
    return _decode(input_grid)[0]


def _unique_checksum(mask, pivots):
    target = _count(mask)
    for values in product((False, True), repeat=len(pivots)):
        other = mask.copy()
        for cell, value in zip(pivots, values):
            other[cell] = value
        if _count(other) == target and not np.array_equal(other, mask):
            return False
    return True


def _sample_mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(6))
    if mode == 0:
        # Whole rectangular cavities and symmetric rings are welcome.
        mask[1:-1, 1:-1] = True
    elif mode == 1:
        # A random connected growing hole, not a catalog of stencils.
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        mask[r, c] = True
        for _ in range(int(rng.integers((h - 2) * (w - 2)))):
            neighbors = [(a, b) for a in range(1, h - 1) for b in range(1, w - 1)
                         if not mask[a, b] and any(mask[a + dr, b + dc] for dr, dc in _FOUR)]
            if not neighbors:
                break
            mask[neighbors[int(rng.integers(len(neighbors)))]] = True
    elif mode != 2:
        mask[1:-1, 1:-1] = rng.random((h - 2, w - 2)) < rng.uniform(0.2, 0.7)
    return mask


def _length_choices(w, pivots, partial=False):
    choices = []
    for copies in range(3, 7):
        for tail in range(w):
            length = copies * w + tail
            if length + 4 > 30 or partial and tail == 0:
                continue
            if all((copies + int(c < tail)) % 2 == 0
                   and copies + int(c < tail) >= 4 for r, c in pivots):
                choices.append(length)
    return choices


def _corrupt(rng, mask, pivots, length, every_copy=False, endpoint_cell=None, extras=None):
    h, w = mask.shape
    noisy = mask[:, np.arange(length) % w].copy()
    cost = 0
    for r, c in pivots:
        positions = list(range(c, length, w))
        chosen = rng.choice(positions, len(positions) // 2, replace=False)
        noisy[r, chosen] = ~noisy[r, chosen]
        cost += len(chosen)
    groups = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
              if (r, c) not in pivots]
    budget = max(2, noisy.size // 8) - cost
    if every_copy:
        extra_count = length // w
    elif extras is not None:
        extra_count = extras
    else:
        extra_count = int(rng.integers(min(5, len(groups), budget) + 1))
    extra_count = max(extra_count, int(endpoint_cell is not None))
    if extra_count > min(len(groups), budget):
        return None
    rng.shuffle(groups)
    if endpoint_cell is not None:
        groups.remove(endpoint_cell)
        groups.insert(0, endpoint_cell)
    for j, (r, c) in enumerate(groups[:extra_count]):
        positions = list(range(c, length, w))
        if endpoint_cell == (r, c):
            position = (length // w) * w + c
        elif every_copy:
            position = j * w + c
        else:
            position = int(rng.choice(positions))
        noisy[r, position] = ~noisy[r, position]
    return noisy


def _admit_lane(mask, noisy, pivots):
    if noisy is None or _count(~noisy) != 1:
        return False
    desired = mask[:, np.arange(noisy.shape[1]) % mask.shape[1]]
    try:
        repaired, info = _repair_suffix(np.where(noisy, 0, 1), 0, 1, _count(mask))
    except ValueError:
        return False
    return (np.array_equal(repaired == 0, desired)
            and info["max_ties"] >= len(pivots))


def _random_lane(rng, ties=1, partial=False):
    for _ in range(120):
        h, w = int(rng.integers(4, 9)), int(rng.integers(3, 7))
        mask = _sample_mask(rng, h, w)
        if _count(~mask) != 1:
            continue
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        pivot_sets = list(combinations(cells, ties))
        rng.shuffle(pivot_sets)
        for pivots in pivot_sets[:32]:
            if not _unique_checksum(mask, pivots):
                continue
            lengths = _length_choices(w, pivots, partial)
            rng.shuffle(lengths)
            for length in lengths[:3]:
                # Check that a tab carrying this count can fit too.
                capacity = max(((h - 2) * (cw - 2) + 1) // 2
                               for cw in range(3, min(6, 29 - length) + 1))
                if _count(mask) > capacity:
                    continue
                noisy = _corrupt(rng, mask, pivots, length)
                if _admit_lane(mask, noisy, pivots):
                    return {"mask": mask, "noisy": noisy, "pivots": pivots}
    # Constructive fallback: removing isolated holes strictly decreases count.
    mask = np.zeros((7, 5), dtype=bool)
    mask[1, 1] = mask[3, 3] = True
    pivots = ((1, 1),) if ties == 1 else ((1, 1), (3, 3))
    length = 21 if partial else 20
    noisy = _corrupt(rng, mask, pivots, length, extras=0)
    if not _admit_lane(mask, noisy, pivots):
        raise RuntimeError("constructive lane fallback did not admit a unique repair")
    return {"mask": mask, "noisy": noisy, "pivots": pivots}


def _contrast_lanes(rng):
    for _ in range(80):
        h, w = int(rng.integers(5, 9)), int(rng.integers(5, 7))
        mask = np.zeros((h, w), dtype=bool)
        r = int(rng.integers(1, h - 1))
        c = int(rng.integers(1, w - 3))
        mask[r, c:c + 3] = True
        pivot = (r, c + 1)
        other = mask.copy()
        other[pivot] = False
        noisy = _corrupt(rng, mask, (pivot,), 4 * w, every_copy=True)
        if _admit_lane(mask, noisy, (pivot,)) and _admit_lane(other, noisy, (pivot,)):
            return [{"mask": m, "noisy": noisy.copy(), "pivots": (pivot,)}
                    for m in (mask, other)]
    mask = np.zeros((6, 5), dtype=bool)
    mask[2, 1:4] = True
    pivot = (2, 2)
    other = mask.copy()
    other[pivot] = False
    noisy = mask[:, np.arange(20) % 5].copy()
    noisy[2, [2, 7]] = False
    for j, (r, c) in enumerate(((1, 1), (1, 2), (1, 3), (3, 1))):
        noisy[r, j * 5 + c] = ~noisy[r, j * 5 + c]
    if not (_admit_lane(mask, noisy, (pivot,)) and _admit_lane(other, noisy, (pivot,))):
        raise RuntimeError("constructive contrast fallback failed")
    return [{"mask": m, "noisy": noisy.copy(), "pivots": (pivot,)} for m in (mask, other)]


def _connectivity_lane(rng):
    for _ in range(80):
        h, w = int(rng.integers(5, 9)), int(rng.integers(5, 7))
        mask = np.zeros((h, w), dtype=bool)
        r = int(rng.integers(1, h - 2))
        c = int(rng.integers(3, w - 1))
        mask[r, c] = mask[r + 1, c - 1] = mask[r + 1, c - 2] = True
        if rng.integers(2):
            mask = mask[::-1].copy()
            r = h - 1 - r
            endpoint = (r - 1, c - 2)
        else:
            endpoint = (r + 1, c - 2)
        pivots = ((r, c),)
        tail = int(rng.integers(c - 1, c + 1))
        length = 4 * w + tail
        if length + 4 > 30:
            continue
        noisy = _corrupt(rng, mask, pivots, length, endpoint_cell=endpoint)
        if _admit_lane(mask, noisy, pivots):
            return {"mask": mask, "noisy": noisy, "pivots": pivots}
    # This geometry admits a final partial copy containing a damaged hole.
    mask = np.zeros((6, 5), dtype=bool)
    mask[1, 3] = mask[2, 2] = mask[2, 1] = True
    pivots = ((1, 3),)
    noisy = _corrupt(rng, mask, pivots, 22, endpoint_cell=(2, 1), extras=1)
    if not _admit_lane(mask, noisy, pivots):
        raise RuntimeError("connectivity fallback failed")
    return {"mask": mask, "noisy": noisy, "pivots": pivots}


def _tab(rng, h, max_width, checksum, carrier, marker):
    eligible = [w for w in range(3, min(6, max_width) + 1)
                if ((h - 2) * (w - 2) + 1) // 2 >= checksum]
    if not eligible:
        raise ValueError("counter cannot fit")
    w = int(rng.choice(eligible))
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    chosen = None
    for _ in range(16):
        rng.shuffle(cells)
        trial = []
        for r, c in cells:
            if all(abs(r - a) + abs(c - b) != 1 for a, b in trial):
                trial.append((r, c))
                if len(trial) == checksum:
                    break
        if len(trial) >= checksum:
            chosen = trial[:checksum]
            break
    if chosen is None:
        # Any rectangle's larger checkerboard parity guarantees capacity.
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                 if (r + c) % 2 == 0]
        rng.shuffle(cells)
        chosen = cells[:checksum]
    tab = np.full((h, w), carrier, dtype=np.int8)
    for cell in chosen:
        tab[cell] = marker
    return tab


def _scene(rng, lanes, vertical, background, marker, shared_carrier=False):
    pool = list(map(int, rng.permutation([c for c in range(10) if c not in (background, marker)])))
    pool = pool[:int(rng.integers(1, 9))]
    same = int(rng.choice(pool))
    objects = []
    for lane in lanes:
        h, w = lane["mask"].shape
        length = lane["noisy"].shape[1]
        carrier = same if shared_carrier else int(rng.choice(pool))
        tab = _tab(rng, h, 29 - length, _count(lane["mask"]), carrier, marker)
        spine = np.full((h, 1), marker, dtype=np.int8)
        clean = lane["mask"][:, np.arange(length) % w]
        objects.append((np.concatenate((tab, spine, np.where(lane["noisy"], background, carrier)), axis=1),
                        np.concatenate((tab, spine, np.where(clean, background, carrier)), axis=1)))
    height = sum(obj[0].shape[0] for obj in objects)
    gaps = [1] * (len(objects) - 1)
    for i in range(len(gaps)):
        extra = int(rng.integers(min(2, 30 - height - sum(gaps)) + 1))
        gaps[i] += extra
    content_h = height + sum(gaps)
    min_w = max(obj[0].shape[1] for obj in objects)
    canvas_w = min_w + int(rng.integers(min(4, 30 - min_w) + 1))
    extra_h = int(rng.integers(min(4, 30 - content_h) + 1))
    top = int(rng.integers(extra_h + 1))
    inp = np.full((content_h + extra_h, canvas_w), background, dtype=np.int8)
    out = inp.copy()
    row = top
    for i, (a, b) in enumerate(objects):
        left = int(rng.integers(canvas_w - a.shape[1] + 1))
        inp[row:row + a.shape[0], left:left + a.shape[1]] = a
        out[row:row + b.shape[0], left:left + b.shape[1]] = b
        row += a.shape[0] + (gaps[i] if i < len(gaps) else 0)
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp.astype(np.int8), "output": out.astype(np.int8)}


def _valid_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name + " has an illegal count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _valid_count(num_train, 2, 10, "num_train")
    ne = _valid_count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mixed = bool(rng.random() < 0.65)
    base_axis = bool(rng.integers(2))
    varied_palette = bool(rng.random() < 0.8)
    base_roles = list(map(int, rng.choice(10, 2, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            vertical = (base_axis if not mixed or split == "train" and i == 0
                        else not base_axis if split == "train" and i == 1
                        else bool(rng.integers(2)))
            for attempt in range(40):
                if split == "train" and i == 0:
                    lanes = _contrast_lanes(rng)
                elif split == "train" and i == 1:
                    lanes = [_connectivity_lane(rng)]
                    if rng.integers(2):
                        lanes.extend(_random_lane(rng) for _ in range(2))
                else:
                    lanes = [_random_lane(rng, 1 if split == "train" else 2,
                                          partial=bool(rng.integers(2)))
                             for _ in range(int(rng.integers(1, 4)))]
                background, marker = (list(map(int, rng.choice(10, 2, replace=False)))
                                      if varied_palette else base_roles)
                pair = _scene(rng, lanes, vertical, background, marker,
                              shared_carrier=split == "train" and i == 0)
                if _key(pair["input"]) in used:
                    continue
                try:
                    prediction, analysis = _decode(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(prediction, pair["output"]):
                    continue
                used.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("pair construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _contrast_witness(ribbons):
    for a, b in combinations(ribbons, 2):
        area_inversion = ((a["checksum"] - b["checksum"])
                          * (int(a["unit"].sum()) - int(b["unit"].sum())) < 0)
        if (np.array_equal(a["suffix"], b["suffix"])
                and a["checksum"] != b["checksum"]
                and not np.array_equal(a["repaired"], b["repaired"])
                and area_inversion
                and a["tied_changes"] and b["tied_changes"]
                and a["strict_changes"] and b["strict_changes"]
                and a["no_clean_copy"] and b["no_clean_copy"]
                and (a["count_correct_damaged_copy"] or b["count_correct_damaged_copy"])):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            try:
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                prediction, info = _decode(inp)
            except ValueError as exc:
                raise AssertionError("malformed or ambiguous generated pair") from exc
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            assert out.shape == inp.shape and np.array_equal(prediction, out), "incorrect repair"
            assert not np.array_equal(inp, out), "generated pair must need repair"
            for ribbon in info["ribbons"]:
                assert ribbon["errors"] > 0
                assert _count(ribbon["unit"]) == ribbon["checksum"]
                assert _count(~ribbon["unit"]) == 1, "clean foreground disconnected"
                assert ribbon["max_ties"] >= (2 if split == "test" else 1)
            analyses[split].append(info)
    train = [r for a in analyses["train"] for r in a["ribbons"]]
    assert any(_contrast_witness(a["ribbons"]) for a in analyses["train"]), "missing checksum contrast"
    assert any(r["connectivity_witness"] for r in train), "missing four/eight witness"
    assert any(r["partial"] and r["partial_changes"] for r in train), "missing endpoint repair"
    assert any(r["strict_changes"] for r in train), "repetition correction not demonstrated"
    return None


def _signature(puzzle):
    return tuple(_key(np.asarray(pair[role], dtype=np.int8))
                 for split in ("train", "test") for pair in puzzle[split]
                 for role in ("input", "output"))


def self_test():
    """Bounded 42-seed structural and semantic checks; no external oracle."""
    shapes, geometries, palettes, axes, counts, widths, areas, masks = (set() for _ in range(8))
    signatures = set()
    totals = {k: 0 for k in ("pairs", "ribbons", "partial", "zero", "multi_cell",
                              "connectivity", "no_clean_copy", "two_ties", "irregular")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                predicted, info = _decode(pair["input"])
                assert np.array_equal(predicted, pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated input"
                shapes.add(pair["input"].shape)
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                axes.add(info["vertical"])
                geometries.add((info["vertical"], tuple((r["thickness"], r["width"], r["length"])
                                                       for r in info["ribbons"])))
                totals["pairs"] += 1
                for r in info["ribbons"]:
                    widths.update(r["supported_widths"])
                    counts.add(r["checksum"])
                    masks.add((r["unit"].shape, r["unit"].tobytes()))
                    totals["ribbons"] += 1
                    totals["partial"] += bool(r["partial"])
                    totals["zero"] += r["checksum"] == 0
                    totals["multi_cell"] += any(n > 1 for n in r["cavity_sizes"])
                    totals["connectivity"] += r["connectivity_witness"]
                    totals["no_clean_copy"] += r["no_clean_copy"]
                    totals["two_ties"] += r["max_ties"] >= 2
                    areas.update(r["cavity_sizes"])
                    for cavity in _components(r["unit"]):
                        rs, cs = zip(*cavity)
                        totals["irregular"] += len(cavity) < (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
    assert len(signatures) == 42 and axes == {False, True}
    assert len(shapes) >= 25 and len(geometries) >= 100 and len(palettes) >= 30
    assert {3, 4, 5, 6} <= widths and {0, 1, 2, 3} <= counts
    assert len(masks) >= 50 and max(areas) >= 5
    assert all(totals[k] > 0 for k in totals)
    assert _signature(generate_puzzle(731, 2, 1)) == _signature(generate_puzzle(731, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Remove the contrast demonstration while retaining individually solved pairs.
    broken["train"] = [sample["train"][i] for i in range(2)
                       if not _contrast_witness(_decode(sample["train"][i]["input"])[1]["ribbons"])]
    other = generate_puzzle(1, 2, 1)
    broken["train"].append(next(p for p in other["train"]
                                if not _contrast_witness(_decode(p["input"])[1]["ribbons"])))
    broken["test"] = sample["test"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("training without a checksum contrast accepted")
    # Two independent tied singleton locations and a one-cavity checksum leave
    # two equally supported answers. The decoder must not pick by scan order.
    ambiguous = np.ones((5, 24), dtype=np.int8)
    ambiguous[:, 3] = 2
    ambiguous[1, 1] = 2
    for j in (0, 1):
        ambiguous[1, 4 + j * 5 + 1] = 0
    for j in (2, 3):
        ambiguous[3, 4 + j * 5 + 3] = 0
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting checksum repairs were arbitrarily resolved")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "geometries": len(geometries), "palettes": len(palettes),
            "unit_masks": len(masks), "unit_widths": sorted(widths),
            "checksum_counts": sorted(counts), "cavity_areas": sorted(areas), "totals": totals}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
