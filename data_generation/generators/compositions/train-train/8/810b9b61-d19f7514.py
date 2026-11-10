"""ARC family: rotate, group, and synchronize pictures in sealed windows.

Only the standard library and NumPy are used. Generation constructs outputs
from its intended pictures; solve reconstructs everything from the input.
"""

from collections import defaultdict
import numpy as np


_AREA_FAMILIES = (((2, 4), (4, 2), (1, 8), (8, 1)),
                  ((3, 3), (1, 9), (9, 1)),
                  ((2, 5), (5, 2)),
                  ((3, 4), (4, 3), (2, 6), (6, 2)),
                  ((3, 5), (5, 3)),
                  ((4, 4), (2, 8), (8, 2)))
_SHAPES = tuple(shape for bucket in _AREA_FAMILIES for shape in bucket)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return array.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        todo, cells = [(r, c)], set()
        seen[r, c] = True
        while todo:
            y, x = todo.pop()
            cells.add((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _perimeter(h, w):
    # Cyclic order, also used to remove contiguous pieces from open outlines.
    return ([(0, c) for c in range(w)]
            + [(r, w - 1) for r in range(1, h)]
            + [(h - 1, c) for c in range(w - 2, -1, -1)]
            + [(r, 0) for r in range(h - 2, 0, -1)])


def _parse(input_grid):
    """Recognize outline/picture roles and all local coordinate systems."""
    grid = _grid(input_grid)
    colors = set(map(int, np.unique(grid))) - {0}
    if len(colors) != 2 or not np.any(grid == 0):
        raise ValueError("need black background and two visible nonzero roles")
    candidates = []
    for ink in sorted(colors):
        signal = next(iter(colors - {ink}))
        records, accounted = [], set()
        valid = True
        for cells in _components(grid == ink):
            rows, cols = zip(*cells)
            top, left = min(rows), min(cols)
            h, w = max(rows) - top + 1, max(cols) - left + 1
            local = {(r - top, c - left) for r, c in cells}
            border = set(_perimeter(h, w))
            if h < 3 or w < 3 or not local <= border:
                valid = False
                break
            inside = grid[top + 1:top + h - 1, left + 1:left + w - 1]
            if np.any(inside == ink):
                valid = False
                break
            # TL, TR, BR, BL: rotating counterclockwise by this index
            # moves that corner to TL.
            corners = ((top - 1, left - 1), (top - 1, left + w),
                       (top + h, left + w), (top + h, left - 1))
            marked = [k for k, (r, c) in enumerate(corners)
                      if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                      and grid[r, c] == signal]
            if len(marked) != 1:
                valid = False
                break
            k = marked[0]
            mask = np.rot90(inside == signal, k).copy()
            records.append({"box": (top, left, h, w), "turn": k,
                            "mask": mask, "closed": local == border,
                            "gap": len(border - local), "beacon": corners[k]})
            accounted.add(corners[k])
            accounted.update((top + 1 + int(r), left + 1 + int(c))
                             for r, c in zip(*np.nonzero(inside == signal)))
        if (valid and records and any(o["closed"] for o in records)
                and accounted == set(zip(*np.nonzero(grid == signal)))):
            # No nested or overlapping window boxes.
            boxes = [o["box"] for o in records]
            for i, (t, l, h, w) in enumerate(boxes):
                for y, x, a, b in boxes[:i]:
                    if not (t + h <= y or y + a <= t or l + w <= x or x + b <= l):
                        valid = False
            if valid:
                candidates.append((grid, ink, signal, records))
    if len(candidates) != 1:
        raise ValueError("outline role or corner registration is ambiguous")
    return candidates[0]


def _groups(records, include_open=False):
    groups = defaultdict(list)
    for obj in records:
        if obj["closed"] or include_open:
            groups[obj["mask"].shape].append(obj)
    return groups


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Union pictures of equally shaped, corner-registered closed windows."""
    grid, _, signal, records = _parse(input_grid)
    output = grid.copy()
    for members in _groups(records).values():
        shared = np.logical_or.reduce([o["mask"] for o in members])
        for obj in members:
            t, l, h, w = obj["box"]
            output[t + 1:t + h - 1, l + 1:l + w - 1] = np.where(
                np.rot90(shared, -obj["turn"]), signal, 0)
    return output


def _scores(rng, shape):
    rows, cols = np.indices(shape)
    noise = rng.random(shape)
    style = int(rng.integers(0, 4))
    if style == 0:
        return noise
    if style == 1:
        return rng.random((shape[0], 1)) + .25 * noise
    if style == 2:
        return rng.random((1, shape[1])) + .25 * noise
    cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
    return -((rows - cy) ** 2 + (cols - cx) ** 2) + noise


def _pictures(rng, shape, count, forbidden=()):
    """Every peer has an exclusive pixel; one pixel belongs to every peer."""
    size = int(np.prod(shape))
    banned = set(forbidden)
    remaining = [i for i in range(size) if i not in banned]
    # At least two absent coordinates, independent of density or split.
    if len(banned) < 2:
        banned.update(map(int, rng.choice(remaining, size=2 - len(banned), replace=False)))
    available = np.array([i for i in range(size) if i not in banned], dtype=int)
    length = int(rng.integers(count + 1, len(available) + 1))
    scores = _scores(rng, shape).ravel()
    chosen = available[np.argsort(scores[available])[-length:]]
    rng.shuffle(chosen)
    pictures = [np.zeros(size, dtype=bool) for _ in range(count)]
    for i in range(count):
        pictures[i][chosen[i]] = True
    for picture in pictures:
        picture[chosen[count]] = True
    for cell in chosen[count + 1:]:
        owners = rng.choice(count, size=int(rng.integers(1, count + 1)), replace=False)
        for owner in owners:
            pictures[int(owner)][cell] = True
    pictures = [p.reshape(shape) for p in pictures]
    return pictures, np.logical_or.reduce(pictures)


def _random_picture(rng, shape, required=None):
    picture = rng.random(shape) < rng.uniform(.15, .65)
    if required is not None:
        picture[required] = True
    if not np.any(picture):
        picture.flat[int(rng.integers(picture.size))] = True
    return picture


def _object(mask, turn, target=None, gap=0):
    return {"mask": mask.copy(), "turn": int(turn),
            "target": mask.copy() if target is None else target.copy(), "gap": int(gap)}


def _plan(rng, shapes, index, split, origin):
    a, b, c = shapes
    singleton = None
    if split == "train" and index < 2:
        active = [a if index == 0 else b]
        counts = [2 if index == 0 else 3]
        turns = [[origin, (origin + 2) % 4] if index == 0 else
                 [(origin + 1) % 4, (origin + 3) % 4, origin]]
        singleton = b if index == 0 else a
    elif split == "test" or rng.random() < .45:
        active = [a, b]
        counts = [2, int(rng.integers(2, 5))]
        if rng.random() < .5:
            counts.reverse()
        turns = [list(map(int, rng.integers(0, 4, size=n))) for n in counts]
        singleton = c if rng.random() < .3 else None
    else:
        active = [shapes[int(rng.integers(0, 2))]]
        counts = [int(rng.integers(2, 5))]
        turns = [list(map(int, rng.integers(0, 4, size=counts[0])))]
        singleton = next(s for s in (a, b) if s != active[0]) if rng.random() < .65 else None
    # A singleton supplies a visible counterexample to broadcasting across sizes.
    cross_hole = None
    if singleton is not None:
        common_h = min(singleton[0], active[0][0])
        common_w = min(singleton[1], active[0][1])
        cross_hole = (int(rng.integers(common_h)), int(rng.integers(common_w)))
    objects, unions = [], {}
    for j, (shape, n, rotations) in enumerate(zip(active, counts, turns)):
        forbidden = () if j or cross_hole is None else (np.ravel_multi_index(cross_hole, shape),)
        for _ in range(100):
            pictures, union = _pictures(rng, shape, n, forbidden)
            # In the first teaching pair, opposite beacons must give distinct
            # physical target pictures. This guarantees that ignoring the
            # beacon cannot accidentally fit both recipients.
            if not (split == "train" and index == 0) or not np.array_equal(union, np.rot90(union, 2)):
                break
        else:
            raise RuntimeError("could not expose rotation registration")
        unions[shape] = union
        for picture, k in zip(pictures, rotations):
            objects.append(_object(picture, k, union))
    if singleton is not None:
        objects.append(_object(_random_picture(rng, singleton, cross_hole),
                               int(rng.integers(0, 4))))
    # An open peer contains a poison pixel absent from the closed peers' union.
    broken_count = 1 if index < 2 or split == "test" else int(rng.integers(1, 3))
    for _ in range(broken_count):
        shape = active[int(rng.integers(len(active)))]
        holes = np.argwhere(~unions[shape])
        poison = tuple(map(int, holes[int(rng.integers(len(holes)))]))
        gap = 1 if split == "train" and index < 2 else int(rng.integers(1, 4))
        objects.append(_object(_random_picture(rng, shape, poison),
                               int(rng.integers(0, 4)), gap=gap))
    return objects


def _empty_positions(occupied, h, w):
    prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, objects):
    sizes = []
    for obj in objects:
        h, w = np.rot90(obj["mask"], -obj["turn"]).shape
        sizes.append((h + 4, w + 4))  # perimeter plus a ring for the beacon
    area = sum(h * w for h, w in sizes)
    for attempt in range(160):
        if attempt >= 80:
            height, width = 30, 30
        else:
            height, width = map(int, rng.integers(15, 31, size=2))
        if area > .83 * height * width or any(h > height or w > width for h, w in sizes):
            continue
        occupied = np.zeros((height, width), dtype=bool)
        order = sorted(range(len(objects)),
                       key=lambda i: sizes[i][0] * sizes[i][1] + rng.uniform(0, 15), reverse=True)
        positions = [None] * len(objects)
        for i in order:
            h, w = sizes[i]
            available = _empty_positions(occupied, h, w)
            if len(available) == 0:
                break
            r, c = map(int, available[int(rng.integers(len(available)))])
            positions[i] = (r, c)
            occupied[r:r + h, c:c + w] = True
        else:
            return (height, width), positions
    return None


def _render(rng, objects, palette):
    packed = _pack(rng, objects)
    if packed is None:
        return None
    shape, positions = packed
    ink, signal = palette
    inp = np.zeros(shape, dtype=np.int8)
    out = np.zeros(shape, dtype=np.int8)
    for obj, (r, c) in zip(objects, positions):
        picture = np.rot90(obj["mask"], -obj["turn"])
        target = np.rot90(obj["target"], -obj["turn"])
        h, w = picture.shape[0] + 2, picture.shape[1] + 2
        border = _perimeter(h, w)
        removed = set()
        if obj["gap"]:
            for _ in range(100):
                start = int(rng.integers(len(border)))
                removed = {border[(start + j) % len(border)] for j in range(obj["gap"])}
                kept = set(border) - removed
                if ({y for y, _ in kept} >= {0, h - 1}
                        and {x for _, x in kept} >= {0, w - 1}):
                    break
            else:
                raise RuntimeError("cannot make a connected open outline")
        top, left = r + 1, c + 1
        for y, x in border:
            if (y, x) not in removed:
                inp[top + y, left + x] = out[top + y, left + x] = ink
        inp[top + 1:top + h - 1, left + 1:left + w - 1] = np.where(picture, signal, 0)
        out[top + 1:top + h - 1, left + 1:left + w - 1] = np.where(target, signal, 0)
        corners = ((r, c), (r, c + w + 1), (r + h + 1, c + w + 1), (r + h + 1, c))
        y, x = corners[obj["turn"]]
        inp[y, x] = out[y, x] = signal
    # Remove random amounts of completely blank margin. No clipping of objects.
    rows, cols = np.nonzero(inp)
    top = int(rng.integers(0, int(rows.min()) + 1))
    left = int(rng.integers(0, int(cols.min()) + 1))
    bottom = shape[0] - int(rng.integers(0, shape[0] - int(rows.max())))
    right = shape[1] - int(rng.integers(0, shape[1] - int(cols.max())))
    return {"input": inp[top:bottom, left:right].copy(),
            "output": out[top:bottom, left:right].copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bucket = _AREA_FAMILIES[int(rng.integers(len(_AREA_FAMILIES)))]
    chosen = rng.choice(len(bucket), size=2, replace=False)
    a, b = (bucket[int(i)] for i in chosen)
    alternatives = [shape for shape in _SHAPES if shape not in (a, b)]
    shapes = (a, b, alternatives[int(rng.integers(len(alternatives)))])
    base_palette = tuple(map(int, rng.choice(np.arange(1, 10), size=2, replace=False)))
    changing_palette = bool(rng.integers(0, 2))
    origin = int(rng.integers(0, 4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if not changing_palette or (split == "train" and index == 0):
                palette = base_palette
            elif split == "train" and index == 1:
                palette = base_palette[::-1]
            else:
                palette = tuple(map(int, rng.choice(np.arange(1, 10), size=2, replace=False)))
            for _ in range(100):
                objects = _plan(rng, shapes, index, split, origin)
                pair = _render(rng, objects, palette)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct packed example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(grid, records, signal, mode):
    output = grid.copy()
    if mode == "closure":
        groups = _groups(records, include_open=True)
    else:
        groups = _groups(records)
    if mode in ("grouping", "area"):
        pools = defaultdict(list)
        for group in groups.values():
            for obj in group:
                pools[obj["mask"].size if mode == "area" else 0].append(obj)
        for members in pools.values():
            h = max(o["mask"].shape[0] for o in members)
            w = max(o["mask"].shape[1] for o in members)
            shared = np.zeros((h, w), dtype=bool)
            for obj in members:
                a, b = obj["mask"].shape
                shared[:a, :b] |= obj["mask"]
            for obj in members:
                a, b = obj["mask"].shape
                t, l, y, x = obj["box"]
                output[t + 1:t + y - 1, l + 1:l + x - 1] = np.where(
                    np.rot90(shared[:a, :b], -obj["turn"]), signal, 0)
        return output
    for members in groups.values():
        if mode == "intersection":
            shared = np.logical_and.reduce([o["mask"] for o in members])
        elif mode == "xor":
            shared = np.logical_xor.reduce([o["mask"] for o in members])
        else:
            shared = np.logical_or.reduce([o["mask"] for o in members])
        for obj in members:
            t, l, h, w = obj["box"]
            output[t + 1:t + h - 1, l + 1:l + w - 1] = np.where(
                np.rot90(shared, -obj["turn"]), signal, 0)
    return output


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, origins, taught, singleton_keys = set(), set(), set(), set()
    failures = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == out.shape, "canvas dimensions changed")
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            _, ink, signal, records = _parse(grid)
            groups = _groups(records)
            active = {shape: members for shape, members in groups.items() if len(members) > 1}
            _require(bool(active), "need a nontrivial sharing group")
            _require(np.array_equal(solve(grid), out), "input-only solve disagrees")
            change = grid != out
            _require(np.any(change) and np.all(grid[change] == 0)
                     and np.all(out[change] == signal), "only new picture pixels may be added")
            allowed = np.zeros(grid.shape, dtype=bool)
            for shape, members in groups.items():
                union = np.logical_or.reduce([o["mask"] for o in members])
                if len(members) > 1:
                    _require(np.count_nonzero(~union) >= 2, "union needs visible empty cells")
                    common = np.logical_and.reduce([o["mask"] for o in members])
                    _require(np.any(common), "overlap demonstration missing")
                    for i, obj in enumerate(members):
                        others = np.logical_or.reduce([o["mask"] for j, o in enumerate(members) if j != i])
                        _require(np.any(obj["mask"] & ~others), "each peer must contribute an exclusive pixel")
                for obj in members:
                    t, l, h, w = obj["box"]
                    local_out = np.rot90(out[t + 1:t + h - 1, l + 1:l + w - 1] == signal, obj["turn"])
                    _require(np.array_equal(local_out, union), "registered peers have different resulting pictures")
                    allowed[t + 1:t + h - 1, l + 1:l + w - 1] = True
                    if split == "train":
                        origins.add(obj["turn"])
                        (taught if len(members) > 1 else singleton_keys).add(shape)
            _require(np.array_equal(out[~allowed], grid[~allowed]), "outline, beacon, open picture, or background changed")
            poisoned = False
            for obj in records:
                if not obj["closed"] and obj["mask"].shape in active:
                    union = np.logical_or.reduce([o["mask"] for o in active[obj["mask"].shape]])
                    poisoned |= bool(np.any(obj["mask"] & ~union))
            _require(poisoned, "an open peer must carry an excluded pixel")
            if split == "train":
                for mode in ("closure", "intersection", "xor", "grouping", "area"):
                    if not np.array_equal(_ablation(grid, records, signal, mode), out):
                        failures.add(mode)
                # Opposite beacon corners at the same physical dimensions
                # distinguish raw upper-left overlay from corner registration.
                for members in active.values():
                    raw = defaultdict(list)
                    for obj in members:
                        raw[np.rot90(obj["mask"], -obj["turn"]).shape].append(obj)
                    for peers in raw.values():
                        if len(peers) < 2:
                            continue
                        merged = np.logical_or.reduce([np.rot90(o["mask"], -o["turn"]) for o in peers])
                        if any(not np.array_equal(merged, np.rot90(
                                np.logical_or.reduce([o["mask"] for o in members]), -p["turn"])) for p in peers):
                            failures.add("registration")
            else:
                _require(len(active) == 2, "test must combine two sharing groups")
                _require(set(active) <= taught, "test group shape was not taught")
    _require(origins == {0, 1, 2, 3}, "training must expose all beacon corners")
    _require(len(taught & singleton_keys) >= 2, "training must show two shapes as peers and singletons")
    _require(failures == {"closure", "intersection", "xor", "grouping", "area", "registration"},
             "training failed to refute a specified shortcut")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _edge_case_checks():
    """Small specification fixtures, independent of generation and its plans."""
    grid = np.zeros((14, 21), dtype=int)
    for t, l, h, w in ((1, 1, 5, 5), (1, 10, 5, 5),
                       (8, 1, 5, 5), (9, 9, 3, 11)):
        grid[t, l:l + w] = grid[t + h - 1, l:l + w] = 7
        grid[t:t + h, l] = grid[t:t + h, l + w - 1] = 7
    for r, c in ((0, 0), (0, 15), (13, 6), (12, 8)):
        grid[r, c] = 2
    grid[8, 1:5] = 0  # four consecutive missing cells, including a corner
    grid[10, 3] = grid[10, 13] = 2  # open peer and singleton pictures
    grid[2:5, 11:14] = 2  # the second closed 3x3 picture is full
    expected = grid.copy()
    expected[2:5, 2:5] = 2  # the initially empty first picture receives it
    _require(np.array_equal(solve(grid), expected), "empty/full picture fixture failed")

    sparse = grid.copy()
    sparse[2:5, 2:5] = sparse[2:5, 11:14] = 0
    sparse[2, 2] = sparse[4, 11] = 2  # registered TL and BR, disjoint
    expected = sparse.copy()
    expected[4, 4] = expected[2, 13] = 2
    saved = sparse.copy()
    _require(np.array_equal(solve(sparse), expected), "disjoint rotated picture fixture failed")
    _require(np.array_equal(sparse, saved), "solve mutated its input")
    _require(np.array_equal(solve(sparse.tolist()), expected), "nested list input failed")

    tiny = np.array([[7, 7, 7, 0, 0, 0, 0, 0, 0],
                     [7, 0, 7, 0, 0, 0, 0, 0, 0],
                     [7, 7, 7, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 2, 2, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 7, 7, 7, 0],
                     [0, 0, 0, 0, 0, 7, 2, 7, 0],
                     [0, 0, 0, 0, 0, 7, 7, 7, 0]], dtype=int)
    expected = tiny.copy()
    expected[1, 1] = 2
    _require(np.array_equal(solve(tiny), expected), "one-cell cavity / grid-edge fixture failed")


def self_test():
    """Bounded checks, including variation in 42 independently seeded puzzles."""
    _edge_case_checks()
    shapes, palettes, layouts, peer_counts, gap_counts, origins = set(), set(), set(), set(), set(), set()
    signatures = set()
    boundary = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic puzzle")
        signatures.add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            grid, ink, signal, records = _parse(pair["input"])
            shapes.add(grid.shape)
            palettes.add((ink, signal))
            # Normalize both semantic colors, rather than counting recolorings.
            layouts.add(_key(np.where(grid == ink, 1, np.where(grid == signal, 2, 0))))
            for obj in records:
                gap_counts.add(obj["gap"])
                origins.add(obj["turn"])
                t, l, h, w = obj["box"]
                boundary += int(t == 0 or l == 0 or t + h == grid.shape[0] or l + w == grid.shape[1])
            peer_counts.update(len(m) for m in _groups(records).values())
    _require(len(signatures) == 42, "seeds do not distinguish puzzles")
    _require(len(shapes) >= 45 and len(layouts) >= 200 and len(palettes) >= 25,
             "insufficient geometry, structural, or role-palette variation")
    _require({1, 2, 3, 4} <= peer_counts and {0, 1, 2, 3} <= gap_counts,
             "missing singleton, peer-count, or gap-length cases")
    _require(origins == {0, 1, 2, 3} and boundary > 5, "missing rotations or boundary examples")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (2.0, 1), (4, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "canvas_shapes": len(shapes),
            "normalized_inputs": len(layouts), "palettes": len(palettes),
            "peer_counts": sorted(peer_counts), "gap_lengths": sorted(gap_counts),
            "boundary_windows": boundary}
