"""Generate keyed local-exterior-fill ARC puzzles."""

import hashlib
import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EDGES = ("top", "bottom", "left", "right")
_SOURCE_PAIR_DIGESTS = {
    "8888f0fa21862acefcb9b9ecb3c81af61b2f8e3f3adf436f71ff0889288aecb3",
    "caa7914ffdfda745b914554f5a3293421efe3cee103be32d14514ac09d894642",
    "222c2af5dc805ec4931cecdc1329a8044efb47cf92ded294f4f61b386f1d9d40",
}


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = [(r, c)]
        seen[r, c] = True
        for x, y in queue:
            for dx, dy in _DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]
                        and mask[nx, ny] and not seen[nx, ny]):
                    seen[nx, ny] = True
                    queue.append((nx, ny))
        found.append(queue)
    return found


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _rect(bbox, height, width):
    r0, r1, c0, c1 = bbox
    return max(0, r0 - 1), min(height - 1, r1 + 1), max(0, c0 - 1), min(width - 1, c1 + 1)


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _topology(mask):
    exterior, holes = set(), set()
    for part in _components(~mask):
        boundary = any(r in (0, mask.shape[0] - 1)
                       or c in (0, mask.shape[1] - 1) for r, c in part)
        (exterior if boundary else holes).update(part)
    interior_open = {(r, c) for r, c in exterior
                     if 0 < r < mask.shape[0] - 1 and 0 < c < mask.shape[1] - 1}
    return holes, interior_open


def _make_stencil(rng, kind):
    if kind == "mixed":
        h = int(rng.integers(5, 7))
        w = int(rng.integers(5, 7))
    else:
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
    if kind == "irregular":
        for _ in range(160):
            mask = rng.random((h, w)) < float(rng.uniform(0.57, 0.8))
            if (mask[0].any() and mask[-1].any() and mask[:, 0].any()
                    and mask[:, -1].any() and 6 <= mask.sum() < h * w - 1
                    and len(_components(mask)) == 1):
                return mask
        return _make_stencil(rng, "open")
    mask = np.ones((h, w), dtype=bool)
    if kind == "hole":
        mask[1:-1, 1:-1] = False
        if h >= 6 and w >= 6 and rng.random() < 0.5:
            mask[1, 1:-1] = True
            mask[1, w // 2] = False
    elif kind == "open":
        middle = w // 2
        mask[0, middle] = False
        mask[1, middle] = False
        if h > 4 and rng.random() < 0.75:
            mask[2, middle] = False
        if w > 4 and rng.random() < 0.55:
            mask[1, middle - 1] = False
    elif kind == "mixed":
        middle = w // 2
        mask[0, middle] = False
        mask[1, middle] = False
        mask[h - 2, middle] = False
        if w == 6 and rng.random() < 0.5:
            mask[h - 2, middle - 1] = False
    elif kind == "simple":
        style = int(rng.integers(0, 5))
        if style == 1:
            mask[0, w // 2] = False
        elif style == 2:
            mask[1:-1, :] = False
            mask[:, w // 2] = True
        elif style == 3:
            mask[:, :] = False
            mask[:, w // 2] = True
            mask[h // 2, :] = True
        elif style == 4:
            mask[:, :] = False
            mask[:, 0] = True
            mask[-1, :] = True
    else:
        raise ValueError("unknown stencil kind")
    for _ in range(int(rng.integers(0, 4))):
        mask = np.rot90(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    assert len(_components(mask)) == 1
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    holes, opening = _topology(mask)
    assert (kind != "hole" or holes) and (kind != "open" or opening)
    assert kind != "mixed" or (holes and opening)
    return mask.copy()


def _validate_count(name, value, lo, hi):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} through {hi}")


def _resolve_counts(rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _candidate(mask, row, col, height, width):
    h, w = mask.shape
    r0, r1 = max(0, row), min(height, row + h)
    c0, c1 = max(0, col), min(width, col + w)
    if r0 >= r1 or c0 >= c1:
        return None
    visible = mask[r0 - row:r1 - row, c0 - col:c1 - col]
    if visible.sum() < 5 or len(_components(visible)) != 1:
        return None
    rr, cc = np.nonzero(visible)
    cells = tuple((int(r0 + r), int(c0 + c)) for r, c in zip(rr, cc))
    bbox = _bbox(cells)
    if bbox[1] - bbox[0] < 1 or bbox[3] - bbox[2] < 1:
        return None
    rect = _rect(bbox, height, width)
    if _overlap(rect, (height - 5, height - 1, 0, 1)):
        return None
    clipped = row < 0 or col < 0 or row + h > height or col + w > width
    edges = set()
    if bbox[0] == 0:
        edges.add("top")
    if bbox[1] == height - 1:
        edges.add("bottom")
    if bbox[2] == 0:
        edges.add("left")
    if bbox[3] == width - 1:
        edges.add("right")
    return cells, bbox, rect, clipped, edges


def _candidate_lists(mask, height, width):
    regular, edges, clipped = [], {edge: [] for edge in _EDGES}, {edge: [] for edge in _EDGES}
    for row in range(-1, height - mask.shape[0] + 2):
        for col in range(-1, width - mask.shape[1] + 2):
            item = _candidate(mask, row, col, height, width)
            if item is None:
                continue
            if item[3]:
                for edge in item[4]:
                    clipped[edge].append(item)
            else:
                regular.append(item)
                for edge in item[4]:
                    edges[edge].append(item)
    return regular, edges, clipped


def _place(rng, masks, labels, height, width, edge, body_clip):
    choices = [_candidate_lists(mask, height, width) for mask in masks]
    forced_label = None
    if edge is not None:
        possible = [i for i in (0, 1) if not body_clip or labels.count(i) >= 2]
        if not possible:
            return None
        forced_label = possible[int(rng.integers(len(possible)))]
        forced_choices = choices[forced_label][2 if body_clip else 1][edge]
        if not forced_choices:
            return None
    else:
        forced_choices = ()
    order = list(labels)
    if forced_label is not None:
        order.remove(forced_label)
        order.insert(0, forced_label)
        if body_clip:
            second = order.index(forced_label, 1)
            order.insert(1, order.pop(second))
    for _ in range(90):
        selected = []
        good = True
        for index, label in enumerate(order):
            pool = forced_choices if index == 0 and forced_label is not None else choices[label][0]
            if not pool:
                good = False
                break
            found = None
            for j in rng.permutation(len(pool)):
                item = pool[int(j)]
                if all(not _overlap(item[2], previous[1][2]) for previous in selected):
                    found = item
                    break
            if found is None:
                good = False
                break
            selected.append((label, found))
        if good:
            return selected
    return None


def _palette(rng, first_train_roles=None):
    background = int(rng.choice(np.arange(10), p=[0.59] + [0.41 / 9] * 9))
    available = [x for x in range(10) if x != background]
    roles = [int(x) for x in rng.choice(available, size=4, replace=False)]
    if first_train_roles is not None and rng.random() < 0.55:
        former_fill = [x for x in first_train_roles[2:] if x != background]
        if former_fill:
            chosen = former_fill[int(rng.integers(len(former_fill)))]
            roles = [chosen] + [x for x in roles if x != chosen]
            for x in available:
                if len(roles) == 4:
                    break
                if x not in roles:
                    roles.append(x)
            roles = roles[:4]
            if rng.random() < 0.5:
                roles[0], roles[1] = roles[1], roles[0]
    return background, tuple(roles)


def _flood(grid, rect, background):
    r0, r1, c0, c1 = rect
    seen = set()
    queue = []
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if grid[r, c] == background and (r, c) not in seen:
                seen.add((r, c))
                queue.append((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if grid[r, c] == background and (r, c) not in seen:
                seen.add((r, c))
                queue.append((r, c))
    for r, c in queue:
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if (r0 <= nr <= r1 and c0 <= nc <= c1
                    and grid[nr, nc] == background and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def _decode(grid):
    height, width = grid.shape
    assert height >= 10 and width >= 10
    background = int(grid[height - 3, 0])
    motifs = (int(grid[height - 5, 0]), int(grid[height - 2, 0]))
    fills = (int(grid[height - 5, 1]), int(grid[height - 2, 1]))
    assert len(set((background,) + motifs + fills)) == 5
    key = np.full((5, 2), background, dtype=grid.dtype)
    key[0:2, 0], key[0:2, 1] = motifs[0], fills[0]
    key[3:5, 0], key[3:5, 1] = motifs[1], fills[1]
    assert np.array_equal(grid[-5:, :2], key)
    assert set(int(x) for x in np.unique(grid)) == set((background,) + motifs + fills)
    assert all(int(np.count_nonzero(grid == color)) == 2 for color in fills)
    objects = []
    for label, color in enumerate(motifs):
        cells = grid == color
        cells[height - (5 if label == 0 else 2):height - (3 if label == 0 else 0), 0] = False
        parts = _components(cells)
        assert parts
        for part in parts:
            bbox = _bbox(part)
            rect = _rect(bbox, height, width)
            assert not _overlap(rect, (height - 5, height - 1, 0, 1))
            mask = np.zeros((bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1), dtype=bool)
            for r, c in part:
                mask[r - bbox[0], c - bbox[2]] = True
            assert len(part) >= 5 and mask.shape[0] >= 2 and mask.shape[1] >= 2
            objects.append((label, bbox, rect, mask))
    for i, a in enumerate(objects):
        for b in objects[:i]:
            assert not _overlap(a[2], b[2]), "local fill rectangles overlap"
    return background, motifs, fills, objects


def _derive_output(grid, decoded=None):
    if decoded is None:
        decoded = _decode(grid)
    background, _, fills, objects = decoded
    output = grid.copy()
    for label, _, rect, _ in objects:
        for r, c in _flood(grid, rect, background):
            output[r, c] = fills[label]
    return output


def _generate_pair(rng, regime, index, first_roles, first_hole_row, growth_mode):
    if growth_mode and regime == "source_train":
        count = int(rng.integers(3 if index < 2 else 2, 5))
    elif growth_mode:
        count = int(rng.integers(4, 7))
    elif regime == "source_train":
        count = int(rng.integers(3 if index < 2 else 2, 6))
    else:
        count = int(rng.integers(2, 7))
    if regime == "source_train" and index < 2:
        kinds = ["open", "open"]
        kinds[first_hole_row if index == 0 else 1 - first_hole_row] = "hole"
    else:
        kinds = [str(x) for x in rng.choice(("hole", "open", "mixed", "simple", "irregular"), size=2,
                                           p=(0.20, 0.22, 0.18, 0.20, 0.20))]
    masks = [_make_stencil(rng, kind) for kind in kinds]
    n0 = int(rng.integers(1, count))
    labels = [0] * n0 + [1] * (count - n0)
    labels = [labels[int(i)] for i in rng.permutation(count)]
    minimum = 16 if count <= 3 else 20 if count == 4 else 23 if count == 5 else 26
    edge = None
    if rng.random() < (0.48 if regime == "source_train" else 0.65):
        edge = _EDGES[int(rng.integers(4))]
    body_clip = edge is not None and count >= 3 and rng.random() < (0.35 if regime == "source_train" else 0.48)
    positions = None
    for attempt in range(12):
        if growth_mode and regime == "source_train":
            low, high = max(19, minimum), 25
        elif growth_mode:
            low, high = max(27, minimum), 31
        else:
            low, high = min(30, minimum + min(attempt // 3, 4)), 31
        height = int(rng.integers(low, high))
        width = int(rng.integers(low, high))
        positions = _place(rng, masks, labels, height, width, edge, body_clip)
        if positions is not None:
            break
    if positions is None:
        height = width = 30
        edge = None
        body_clip = False
        positions = _place(rng, masks, labels, height, width, edge, body_clip)
    if positions is None:
        raise RuntimeError("could not pack motif regions")
    background, roles = _palette(rng, first_roles if regime == "target_test" else None)
    motifs, fills = roles[:2], roles[2:]
    grid = np.full((height, width), background, dtype=np.int8)
    for label, item in positions:
        for r, c in item[0]:
            grid[r, c] = motifs[label]
    grid[-5:-3, 0], grid[-5:-3, 1] = motifs[0], fills[0]
    grid[-2:, 0], grid[-2:, 1] = motifs[1], fills[1]
    output = _derive_output(grid)
    return {"input": grid, "output": output}, roles


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    first_hole_row = int(rng.integers(2))
    growth_mode = bool(rng.random() < 0.27)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    train, test = [], []
    first_roles = None
    for index in range(train_count):
        pair, roles = _generate_pair(np.random.default_rng(int(pair_seeds[index])),
                                     "source_train", index, None, first_hole_row, growth_mode)
        train.append(pair)
        if index == 0:
            first_roles = roles
    for index in range(test_count):
        pair, _ = _generate_pair(np.random.default_rng(int(pair_seeds[train_count + index])),
                                 "target_test", index, first_roles, first_hole_row, growth_mode)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_digest(grid, output):
    content = b""
    for array in (grid, output):
        content += bytes(array.shape) + array.astype(np.uint8).tobytes()
    return hashlib.sha256(content).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        inputs, pairs = set(), set()
        training_holes = training_open = False
        row_evidence = [{"hole": False, "open": False} for _ in range(2)]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                grid, output = pair["input"], pair["output"]
                for array in (grid, output):
                    assert isinstance(array, np.ndarray) and array.ndim == 2
                    assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                    assert np.issubdtype(array.dtype, np.integer)
                    assert bool(np.all((0 <= array) & (array <= 9)))
                assert grid.shape == output.shape and not np.array_equal(grid, output)
                insig, outsig = _signature(grid), _signature(output)
                assert insig not in inputs and (insig, outsig) not in pairs
                assert _source_digest(grid, output) not in _SOURCE_PAIR_DIGESTS
                inputs.add(insig)
                pairs.add((insig, outsig))
                decoded = _decode(grid)
                assert np.array_equal(output, _derive_output(grid, decoded))
                background, motifs, fills, objects = decoded
                assert set(label for label, _, _, _ in objects) == {0, 1}
                assert np.array_equal(grid[-5:, :2], output[-5:, :2])
                for label in (0, 1):
                    complete = [mask for item_label, bbox, _, mask in objects
                                if item_label == label and bbox[0] > 0
                                and bbox[1] < grid.shape[0] - 1 and bbox[2] > 0
                                and bbox[3] < grid.shape[1] - 1]
                    assert all(np.array_equal(complete[0], mask) for mask in complete[1:])
                for label, bbox, rect, mask in objects:
                    holes, opening = _topology(mask)
                    if section == "train":
                        training_holes |= bool(holes)
                        training_open |= bool(opening)
                        row_evidence[label]["hole"] |= bool(holes)
                        row_evidence[label]["open"] |= bool(opening)
                    for r, c in holes:
                        assert grid[bbox[0] + r, bbox[2] + c] == background
                        assert output[bbox[0] + r, bbox[2] + c] == background
                    for r, c in opening:
                        assert output[bbox[0] + r, bbox[2] + c] == fills[label]
                    assert any(output[r, c] == fills[label]
                               for r, c in _flood(grid, rect, background))
        assert training_holes and training_open
        assert all(all(evidence.values()) for evidence in row_evidence)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _puzzles_equal(a, b):
    return all(len(a[section]) == len(b[section]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[section], b[section])
                   for k in ("input", "output")) for section in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((101, None, None), (102, 6, 3), (103, None, 1),
                              (104, None, 4), (105, 2, None), (106, 10, None),
                              (107, 2, 1), (108, 10, 1), (109, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(222, 5, 2), generate_puzzle(222, 5, 2))
    assert not _puzzles_equal(generate_puzzle(222, 5, 2), generate_puzzle(223, 5, 2))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(333, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid count request: {kwargs}")
    train_counts, test_counts, signatures, dimensions, masks, palettes = (set() for _ in range(6))
    topologies, edges, body_sizes, object_counts, backgrounds = (set() for _ in range(5))
    varied_shapes = varied_palettes = varied_counts = varied_dimensions = 0
    body_clips = role_switches = smaller_tests = uneven = solid = sparse = asymmetric = 0
    nonzero_and_clip = 0
    for seed in range(60000, 60042):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_masks, pair_palettes, pair_counts, pair_dimensions = [], [], [], []
        first_roles = _decode(puzzle["train"][0]["input"])[1:3]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                decoded = _decode(grid)
                background, motif, fill, objects = decoded
                signatures.add(_signature(grid))
                dimensions.add(grid.shape)
                backgrounds.add(background)
                palettes.add((background,) + motif + fill)
                object_counts.add(len(objects))
                pair_palettes.append((background,) + motif + fill)
                pair_counts.append(len(objects))
                pair_dimensions.append(grid.shape)
                representatives = []
                for label in (0, 1):
                    class_masks = [obj[3] for obj in objects if obj[0] == label]
                    main = max(class_masks, key=lambda mask: mask.size)
                    representatives.append((main.shape, main.tobytes()))
                    body_clips += any(mask.size < main.size for mask in class_masks)
                pair_masks.append(tuple(representatives))
                uneven += sum(obj[0] == 0 for obj in objects) != sum(obj[0] == 1 for obj in objects)
                solid += any(obj[3].all() for obj in objects)
                sparse += any(obj[3].mean() <= 0.5 for obj in objects)
                asymmetric += any(
                    not np.array_equal(obj[3], obj[3][:, ::-1])
                    and not np.array_equal(obj[3], obj[3][::-1, :])
                    for obj in objects
                )
                if section == "test":
                    role_switches += bool(set(motif) & set(first_roles[1]))
                    smaller_tests += grid.size < max(p["input"].size for p in puzzle["train"])
                if background != 0 and any(
                    mask.size < max(x[3].size for x in objects if x[0] == label)
                    for label, _, _, mask in objects
                ):
                    nonzero_and_clip += 1
                for _, bbox, _, mask in objects:
                    masks.add((mask.shape, mask.tobytes()))
                    holes, opening = _topology(mask)
                    topologies.add((bool(holes), bool(opening)))
                    body_sizes.add(mask.shape)
                    for name, condition in (("top", bbox[0] == 0),
                                            ("bottom", bbox[1] == grid.shape[0] - 1),
                                            ("left", bbox[2] == 0),
                                            ("right", bbox[3] == grid.shape[1] - 1)):
                        if condition:
                            edges.add(name)
        varied_shapes += len(set(pair_masks)) > 1
        varied_palettes += len(set(pair_palettes)) > 1
        varied_counts += len(set(pair_counts)) > 1
        varied_dimensions += len(set(pair_dimensions)) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 180 and len(dimensions) >= 45
    assert len(masks) >= 45 and len(palettes) >= 60
    assert topologies == {(False, False), (False, True), (True, False), (True, True)}
    assert edges == set(_EDGES) and len(body_sizes) >= 8
    assert len(object_counts) >= 4 and len(backgrounds) >= 5
    assert min(varied_shapes, varied_palettes, varied_counts, varied_dimensions) >= 25
    assert body_clips >= 5 and role_switches >= 10 and smaller_tests >= 5
    assert uneven >= 20 and solid >= 5 and sparse >= 20 and asymmetric >= 20
    assert nonzero_and_clip >= 1
