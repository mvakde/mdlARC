"""Generate colored cardinal-ray compaction puzzles."""
import hashlib
import numpy as np

STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MASKS = tuple(tuple(i for i in range(4) if bits & (1 << i))
              for bits in range(16) if 2 <= bits.bit_count() <= 4)
SOURCE_HASHES = {
    "02ff96ed054377e39fbf21799cf7684bd21dd1c1916ff88908dd81a47af69b08",
    "b9915d10ccdfc856600c232583accffb976825192e5472e9e056b0fd60471c5a",
    "968bc35eeb86ea3e69eb2465c9e0babcad05e69f0efa52b8cc80860dcb0702a0",
    "102950bd3711dc2776e86a1247bfdf91d4ba8b3cfe1c4d27614a68b7e1126b16",
}


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def source_copy(grid):
    return grid.shape == (15, 15) and hashlib.sha256(
        ''.join(str(int(x)) for x in grid.flat).encode()
    ).hexdigest() in SOURCE_HASHES


def key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def counts(rng, nt, ne):
    for name, value, low, high in (("num_train", nt, 2, 10), ("num_test", ne, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")
    if nt is None and ne is None:
        if rng.random() < 0.34:
            return 3, 1
        nt = int(rng.choice((2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10)))
    if nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    if ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ne)


def sample_mask(rng, regime, pool):
    if regime == "target_test" and pool and rng.random() < 0.42:
        return pool[int(rng.integers(len(pool)))]
    if regime == "target_test" and pool and rng.random() < 0.48:
        novel = [m for m in MASKS if m not in pool]
        if novel:
            return novel[int(rng.integers(len(novel)))]
    size = int(rng.choice((2, 3, 4), p=(0.48, 0.39, 0.13)))
    return tuple(sorted(int(x) for x in rng.choice(4, size=size, replace=False)))


def sample_shape(rng, groups):
    band = int(rng.choice(4, p=(0.22, 0.28, 0.32, 0.18)))
    low, high = ((7, 11), (12, 16), (17, 23), (24, 30))[band]
    low = max(low, (7, 10, 12)[groups - 1])
    high = max(low, high)
    h, w = (int(rng.integers(low, high + 1)) for _ in range(2))
    if band == 1 and rng.random() < 0.25:
        h = w = 15
    return h, w


def make_pair(rng, colors, masks, cross, mode):
    groups = len(masks)
    background, anchors, markers = colors
    for _ in range(1600):
        h, w = sample_shape(rng, groups)
        centers = []
        for _ in range(groups):
            options = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                       if all(r != ar and c != ac and abs(r-ar) + abs(c-ac) >= 5
                              for ar, ac in centers)]
            if not options:
                break
            centers.append(options[int(rng.integers(len(options)))])
        if len(centers) != groups:
            continue
        compact = set(centers)
        targets = {}
        for role, mask in enumerate(masks):
            ar, ac = centers[role]
            for direction in mask:
                dr, dc = STEPS[direction]
                cell = (ar + dr, ac + dc)
                if cell in compact or not (0 <= cell[0] < h and 0 <= cell[1] < w):
                    break
                compact.add(cell)
                targets[(role, direction)] = cell
            else:
                continue
            break
        if len(targets) != sum(map(len, masks)):
            continue
        tasks = list(targets)
        cross_task = None
        cross_distance = None
        if cross:
            options = []
            for role, direction in tasks:
                ar, ac = centers[role]
                dr, dc = STEPS[direction]
                for other, (br, bc) in enumerate(centers):
                    if other == role:
                        continue
                    d = (br-ar)*dr + (bc-ac)*dc
                    cell = (ar+dr*d, ac+dc*d)
                    if d >= 2 and cell not in compact:
                        options.append(((role, direction), d))
            if not options:
                continue
            cross_task, cross_distance = options[int(rng.integers(len(options)))]
        adjacent_task = None
        if mode == "adjacent":
            options = [t for t in tasks if t != cross_task]
            adjacent_task = options[int(rng.integers(len(options)))]
        equal_d = int(rng.integers(2, 6)) if mode == "equal" else None
        order = [tasks[int(i)] for i in rng.permutation(len(tasks))]
        if cross_task is not None:
            order.remove(cross_task)
            order.insert(0, cross_task)
        occupied = set(centers)
        placed = {}
        for task in order:
            role, direction = task
            ar, ac = centers[role]
            dr, dc = STEPS[direction]
            maximum = ar if dr < 0 else h-1-ar if dr > 0 else ac if dc < 0 else w-1-ac
            if task == cross_task:
                distances = [cross_distance]
            elif task == adjacent_task:
                distances = [1]
            elif equal_d is not None:
                distances = [equal_d]
            else:
                distances = [int(x) for x in rng.permutation(np.arange(2, maximum + 1))]
            for d in distances:
                if d > maximum:
                    continue
                cell = (ar+dr*d, ac+dc*d)
                if cell in occupied or (cell in compact and cell != targets[task]):
                    continue
                if task != cross_task and any(
                    cell[0] == br or cell[1] == bc
                    for other, (br, bc) in enumerate(centers) if other != role
                ):
                    continue
                occupied.add(cell)
                placed[task] = (cell, d)
                break
            if task not in placed:
                break
        if len(placed) != len(tasks) or all(d == 1 for _, d in placed.values()):
            continue
        if mode == "uneven" and len({d for _, d in placed.values()}) < 2:
            continue
        inp = np.full((h, w), background, dtype=np.int8)
        out = np.full((h, w), background, dtype=np.int8)
        for role, cell in enumerate(centers):
            inp[cell] = anchors[role]
            out[cell] = anchors[role]
        for (role, direction), (cell, _) in placed.items():
            inp[cell] = markers[role]
            out[targets[(role, direction)]] = markers[role]
        if source_copy(inp):
            continue
        return inp, out
    raise ValueError("could not place a non-conflicting pair")


def grid_info(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    freq = {int(c): int(n) for c, n in zip(values, frequencies)}
    background = max(freq, key=freq.get)
    need(freq[background] > max((n for c, n in freq.items() if c != background), default=0),
         "background not uniquely identifiable")
    anchors = [c for c, n in freq.items() if c != background and n == 1]
    markers = [c for c, n in freq.items() if c != background and 2 <= n <= 4]
    need(len(anchors) in (1, 2, 3) and len(markers) == len(anchors)
         and len(freq) == 1 + 2*len(anchors), "invalid group colors")
    positions = {c: tuple(int(x) for x in np.argwhere(grid == c)[0]) for c in anchors}
    return background, positions, markers


def ray(anchor, cell):
    ar, ac = anchor
    r, c = cell
    if ar == r and ac != c:
        return (1 if c > ac else 3), abs(c-ac)
    if ac == c and ar != r:
        return (2 if r > ar else 0), abs(r-ar)
    return None


def pair_data(pair, learned=None):
    inp, out = pair["input"], pair["output"]
    background, anchors, markers = grid_info(inp)
    ic, inc = np.unique(inp, return_counts=True)
    oc, onc = np.unique(out, return_counts=True)
    need(np.array_equal(ic, oc) and np.array_equal(inc, onc), "palette or counts changed")
    need(all(out[pos] == color for color, pos in anchors.items()), "anchor moved")
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    for color, pos in anchors.items():
        expected[pos] = color
    mapping, masks, distances = {}, [], []
    ambiguous = False
    for marker in markers:
        cells = [tuple(int(x) for x in row) for row in np.argwhere(inp == marker)]
        candidates = []
        for color, anchor in anchors.items():
            rays = [ray(anchor, cell) for cell in cells]
            if all(x is not None for x in rays) and len({x[0] for x in rays}) == len(rays):
                candidates.append((color, rays))
        out_cells = {tuple(int(x) for x in row) for row in np.argwhere(out == marker)}
        matching = [(color, rays) for color, rays in candidates if
                    {tuple(anchors[color][axis] + STEPS[direction][axis] for axis in (0, 1))
                     for direction, _ in rays} == out_cells]
        need(len(matching) == 1, "marker output is not uniquely paired")
        color, rays = matching[0]
        ambiguous |= any(
            cell[0] == other_anchor[0] or cell[1] == other_anchor[1]
            for cell in cells for other_color, other_anchor in anchors.items()
            if other_color != color
        )
        if learned is not None and len(candidates) > 1:
            need(learned.get(marker) == color, "cross-axis color pairing absent from training")
        mapping[marker] = color
        masks.append(tuple(sorted(d for d, _ in rays)))
        for direction, distance in rays:
            target = tuple(anchors[color][axis] + STEPS[direction][axis] for axis in (0, 1))
            need(0 <= target[0] < inp.shape[0] and 0 <= target[1] < inp.shape[1],
                 "adjacent output out of bounds")
            need(expected[target] == background, "output collision")
            expected[target] = marker
            distances.append(distance)
    need(np.array_equal(expected, out), "output is not cardinal compaction")
    need(any(d > 1 for d in distances), "unchanged pair")
    return {"background": background, "anchors": anchors, "markers": markers,
            "mapping": mapping, "masks": tuple(masks), "distances": tuple(distances),
            "ambiguous": ambiguous, "shape": inp.shape}


def predict_input(test_input, learned):
    """Use only a test input and color associations shown by training outputs."""
    background, anchors, markers = grid_info(test_input)
    predicted = np.full(test_input.shape, background, dtype=test_input.dtype)
    for color, position in anchors.items():
        predicted[position] = color
    for marker in markers:
        cells = [tuple(int(x) for x in row) for row in np.argwhere(test_input == marker)]
        candidates = []
        for color, anchor in anchors.items():
            rays = [ray(anchor, cell) for cell in cells]
            if all(x is not None for x in rays) and len({x[0] for x in rays}) == len(rays):
                candidates.append((color, rays))
        if len(candidates) == 1:
            color, rays = candidates[0]
        else:
            matches = [(color, rays) for color, rays in candidates
                       if learned.get(marker) == color]
            need(len(matches) == 1, "test pairing is not inferable from training")
            color, rays = matches[0]
        for direction, _ in rays:
            target = tuple(anchors[color][axis] + STEPS[direction][axis]
                           for axis in (0, 1))
            need(0 <= target[0] < predicted.shape[0]
                 and 0 <= target[1] < predicted.shape[1]
                 and predicted[target] == background, "predicted output collision")
            predicted[target] = marker
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                need(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid type")
                need(all(1 <= x <= 30 for x in grid.shape), "invalid grid shape")
                need(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                need(bool(np.all((0 <= grid) & (grid <= 9))), "grid has illegal colors")
            need(inp.shape == out.shape and inp.dtype == out.dtype, "input/output mismatch")
            need(not np.array_equal(inp, out), "unchanged pair")
            k = key(inp)
            need(k not in seen, "duplicate input")
            seen.add(k)
            need(not source_copy(inp), "original source input copied")
            if split == "train":
                training.append(pair_data(pair))
    learned, conflicts = {}, set()
    for data in training:
        for marker, anchor in data["mapping"].items():
            if marker in learned and learned[marker] != anchor:
                conflicts.add(marker)
            else:
                learned[marker] = anchor
    for marker in conflicts:
        del learned[marker]
    need(any(len(d["markers"]) >= 2 for d in training), "training lacks multiple groups")
    need(any(len(mask) == 4 for d in training for mask in d["masks"]),
         "training lacks a four-direction group")
    for pair in puzzle["test"]:
        pair_data(pair, learned)
        need(np.array_equal(predict_input(pair["input"], learned), pair["output"]),
             "test output cannot be recovered from training and test input")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ne = counts(rng, num_train, num_test)
    max_groups = 3 if rng.random() < 0.52 else 2
    stable = bool(rng.random() < 0.62)
    palette = [int(x) for x in rng.choice(10, size=1+2*max_groups, replace=False)]
    pair_seeds = [int(x) for x in rng.integers(
        0, np.iinfo(np.uint64).max, size=nt+ne, dtype=np.uint64)]
    puzzle, seen, pool = {"train": [], "test": []}, set(), []
    for index, pair_seed in enumerate(pair_seeds):
        prng = np.random.default_rng(pair_seed)
        regime = "source_train" if index < nt else "target_test"
        if index == 0:
            groups = 2
        elif index == 1 and max_groups == 3:
            groups = 3
        elif index == 2 and nt >= 3:
            groups = 1
        else:
            groups = int(prng.choice(range(1, max_groups+1),
                         p=(0.22, 0.53, 0.25) if max_groups == 3 else (0.26, 0.74)))
        if stable:
            roles = [int(x) for x in prng.choice(max_groups, size=groups, replace=False)]
            if index == 0:
                roles = [0, 1]
            if index == 1 and max_groups == 3:
                roles = [0, 1, 2]
            colors = (palette[0], [palette[1+r] for r in roles],
                      [palette[1+max_groups+r] for r in roles])
        else:
            local = [int(x) for x in prng.choice(10, size=1+2*groups, replace=False)]
            colors = (local[0], local[1:1+groups], local[1+groups:])
        masks = [sample_mask(prng, regime, pool) for _ in range(groups)]
        if index == 0:
            masks[0] = (0, 1, 2, 3)
            if len(masks[1]) == 4:
                masks[1] = (0, 2)
        if index == 1 and all(len(mask) != 3 for mask in masks):
            masks[0] = tuple(sorted(int(x) for x in prng.choice(4, size=3, replace=False)))
        if regime == "source_train":
            pool.extend(masks)
        cross = stable and groups >= 2 and (index == 0 or prng.random() < 0.38)
        mode = str(prng.choice(("uneven", "equal", "adjacent"), p=(0.53, 0.23, 0.24)))
        for _ in range(8):
            inp, out = make_pair(prng, colors, masks, cross, mode)
            k = key(inp)
            if k not in seen:
                seen.add(k)
                break
        else:
            raise ValueError("could not generate unique input")
        puzzle["train" if index < nt else "test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, shapes, palettes, masks = set(), set(), set(), set(), set()
    groups_seen, distances, backgrounds, inputs = set(), set(), set(), set()
    equal_pairs = adjacent_pairs = cross_pairs = boundary_pairs = source_shapes = 0
    short_puzzles = changed_backgrounds = changed_groups = 0
    opposite_pairs = small_pairs = large_pairs = three_cross = cross_adjacent = 0
    reused_test_masks = novel_test_masks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        short_puzzles += (len(puzzle["train"]), len(puzzle["test"])) == (3, 1)
        train_data = [pair_data(pair) for pair in puzzle["train"]]
        test_data = [pair_data(pair) for pair in puzzle["test"]]
        train_masks = {mask for data in train_data for mask in data["masks"]}
        changed_backgrounds += len({data["background"] for data in train_data + test_data}) > 1
        changed_groups += len({len(data["anchors"]) for data in train_data + test_data}) > 1
        reused_test_masks += any(mask in train_masks for data in test_data for mask in data["masks"])
        novel_test_masks += any(mask not in train_masks for data in test_data for mask in data["masks"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = pair_data(pair)
                inp = pair["input"]
                inputs.add(key(inp))
                shapes.add(data["shape"])
                palettes.add(tuple(sorted(int(x) for x in np.unique(inp))))
                masks.update(data["masks"])
                groups_seen.add(len(data["anchors"]))
                distances.update(data["distances"])
                backgrounds.add(data["background"])
                equal_pairs += len(set(data["distances"])) == 1
                adjacent_pairs += 1 in data["distances"]
                cross_pairs += data["ambiguous"]
                opposite_pairs += any(mask in ((0, 2), (1, 3)) for mask in data["masks"])
                small_pairs += max(inp.shape) <= 11
                large_pairs += max(inp.shape) >= 24
                three_cross += len(data["anchors"]) == 3 and data["ambiguous"]
                cross_adjacent += data["ambiguous"] and 1 in data["distances"]
                source_shapes += data["shape"] == (15, 15)
                for r, c in data["anchors"].values():
                    boundary_pairs += r in (1, inp.shape[0]-2) or c in (1, inp.shape[1]-2)
    need(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    need(groups_seen == {1, 2, 3}, "group count diversity too low")
    need(len(shapes) >= 45 and source_shapes > 0, "shape diversity too low")
    need(len(palettes) >= 25 and len(backgrounds) >= 7, "color diversity too low")
    need(len(masks) >= 9, "direction diversity too low")
    need(len(distances) >= 12, "distance diversity too low")
    need(min(equal_pairs, adjacent_pairs, cross_pairs, boundary_pairs) > 0,
         "a reviewed regime is absent")
    need(short_puzzles >= 5 and changed_backgrounds > 0 and changed_groups >= 30,
         "puzzle-level regime diversity too low")
    need(min(opposite_pairs, small_pairs, large_pairs, three_cross, cross_adjacent) > 0,
         "a combined review regime is absent")
    need(reused_test_masks > 0 and novel_test_masks > 0,
         "test masks lack reused or held-out combinations")
    need(len(inputs) >= 160, "input diversity too low")
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000+i, nt, ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            need(len(puzzle["train"]) == nt, "train count not honored")
        if ne is not None:
            need(len(puzzle["test"]) == ne, "test count not honored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": 2.0}):
        try:
            generate_puzzle(4000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    a, b = generate_puzzle(987654321, 7, 3), generate_puzzle(987654321, 7, 3)
    need(all(np.array_equal(pa[k], pb[k]) for split in ("train", "test")
             for pa, pb in zip(a[split], b[split]) for k in ("input", "output")),
         "generation is not deterministic")
