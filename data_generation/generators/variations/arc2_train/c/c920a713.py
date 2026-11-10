"""Overlapping rectangular frames serialized as depth-ordered rings."""

import hashlib
import numpy as np

_SOURCE = {
    "3bec3aae5b53650501140eff5365dc55691270819994a66d1f3ca754f0e79b9e",
    "18ee0e688276f0e43d66a193d4a48ee53615b345f5e07ff21c3c0ba6541e8f58",
    "81bbb162408a4cdb64f4991b8cba0f6cb244c4347200220fbf9bc7979b87754d",
    "b847311eea9bfaca7b778358871263a19e0b1f8ad6caf9011c385ac65ac6c317",
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return grid.shape, grid.tobytes()


def _digest(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _counts(seed, rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _perimeter(rect):
    top, bottom, left, right = rect
    cells = {(top, x) for x in range(left, right+1)}
    cells.update((bottom, x) for x in range(left, right+1))
    cells.update((y, left) for y in range(top, bottom+1))
    cells.update((y, right) for y in range(top, bottom+1))
    return cells


def _rings(order):
    size = 2*len(order)-1
    result = np.empty((size, size), dtype=np.int8)
    for depth, color in enumerate(order):
        result[depth:size-depth, depth:size-depth] = color
    return result


def _analyze(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) >= 3, "need background and at least two frames")
    sorted_counts = np.sort(counts)
    _require(sorted_counts[-1] > sorted_counts[-2], "background is not identifiable")
    background = int(values[int(np.argmax(counts))])
    colors = tuple(int(v) for v in values if int(v) != background)
    _require(2 <= len(colors) <= 9, "invalid frame count")
    _require(sum(int(grid[y, x] == background) for y in (0, grid.shape[0]-1)
                 for x in (0, grid.shape[1]-1)) >= 1,
             "background must be visible at a corner")
    rects, perimeters = {}, {}
    for color in colors:
        ys, xs = np.where(grid == color)
        rect = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        top, bottom, left, right = rect
        _require(bottom-top >= 2 and right-left >= 2, "frame is smaller than three by three")
        cells = _perimeter(rect)
        _require(all((int(y), int(x)) in cells for y, x in zip(ys, xs)),
                 "visible color departs from its rectangular perimeter")
        for side in (((top, x) for x in range(left, right+1)),
                     ((bottom, x) for x in range(left, right+1)),
                     ((y, left) for y in range(top, bottom+1)),
                     ((y, right) for y in range(top, bottom+1))):
            _require(sum(int(grid[cell] == color) for cell in side) >= 2,
                     "a frame side lacks recoverable visible evidence")
        rects[color] = rect
        perimeters[color] = cells
    cover = {}
    for color, cells in perimeters.items():
        for cell in cells:
            cover.setdefault(cell, set()).add(color)
    for cell, latent in cover.items():
        _require(int(grid[cell]) in latent, "a latent border has no visible frame")
    for y, x in zip(*np.where(grid != background)):
        _require((int(y), int(x)) in perimeters[int(grid[y, x])],
                 "foreground is not part of its frame")
    above = {color: set() for color in colors}
    contacts = set()
    for cell, latent in cover.items():
        if len(latent) < 2:
            continue
        visible = int(grid[cell])
        for hidden in latent - {visible}:
            above[hidden].add(visible)
        for a in latent:
            for b in latent:
                if a < b:
                    contacts.add((a, b))
    indegree = {color: 0 for color in colors}
    for targets in above.values():
        for upper in targets:
            indegree[upper] += 1
    order = []
    for _ in colors:
        ready = [c for c in colors if c not in order and indegree[c] == 0]
        _require(len(ready) == 1, "occlusion does not give one total order")
        lower = ready[0]
        order.append(lower)
        for upper in above[lower]:
            indegree[upper] -= 1
    reconstructed = np.full(grid.shape, background, dtype=np.int8)
    for color in order:
        for cell in perimeters[color]:
            reconstructed[cell] = color
    _require(np.array_equal(reconstructed, grid),
             "crossing colors conflict with a global stack")
    return {"background": background, "order": tuple(order), "rects": rects,
            "cover": cover, "contacts": len(contacts),
            "direct": sum(map(len, above.values()))}


def _canvas(rng, n):
    minimum = 10 if n <= 5 else (14 if n <= 7 else 19)
    if rng.random() < .20:
        side = int(rng.integers(minimum, minimum+6))
        return side, side
    return int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))


def _rectangle(rng, height, width, scale):
    def extent(limit):
        maximum = min(limit, max(4, int(round(limit*scale))))
        return 3 if rng.random() < .12 else int(rng.integers(3, maximum+1))
    tall, wide = extent(height), extent(width)
    top = int(rng.integers(0, height-tall+1))
    left = int(rng.integers(0, width-wide+1))
    if rng.random() < .13:
        top = 0 if rng.random() < .5 else height-tall
    if rng.random() < .13:
        left = 0 if rng.random() < .5 else width-wide
    return top, top+tall-1, left, left+wide-1


def _layout(rng, n, height, width, scale):
    rects = [_rectangle(rng, height, width, scale)]
    borders = [_perimeter(rects[0])]
    for _ in range(1, n):
        for attempt in range(350):
            rect = _rectangle(rng, height, width, scale)
            border = _perimeter(rect)
            if rect not in rects and border & borders[-1]:
                rects.append(rect)
                borders.append(border)
                break
        else:
            return None
    return rects, borders


def _pair(rng, n, background):
    colors = tuple(int(v) for v in rng.choice(
        np.array([c for c in range(10) if c != background]), n, replace=False))
    for _ in range(180):
        height, width = _canvas(rng, n)
        scale = float(rng.uniform(.42, .88))
        layout = _layout(rng, n, height, width, scale)
        if layout is None:
            continue
        rects, borders = layout
        grid = np.full((height, width), background, dtype=np.int8)
        for color, border in zip(colors, borders):
            for cell in border:
                grid[cell] = color
        try:
            info = _analyze(grid)
        except AssertionError:
            continue
        if info["order"] != colors:
            continue
        if any(info["rects"][color] != rect for color, rect in zip(colors, rects)):
            continue
        return {"input": grid, "output": _rings(info["order"])}
    raise ValueError("could not construct a recoverable frame stack")


def _frame_count(rng):
    return int(rng.choice(np.arange(2, 10), p=(.035, .07, .12, .20,
                                             .245, .17, .115, .045)))


def _nonmonotonic(values):
    return list(values) != sorted(values) and list(values) != sorted(values, reverse=True)


def _area_order(info):
    return [(bottom-top+1)*(right-left+1)
            for color in info["order"]
            for top, bottom, left, right in [info["rects"][color]]]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_background = first_n = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(20):
                n = _frame_count(rng)
                if split == "train" and index == 0 and n < 4:
                    n = int(rng.integers(4, 7))
                if split == "train" and index == 1 and n == first_n:
                    n = 2+(n-1) % 8
                background_options = [c for c in range(10)
                                      if not (split == "train" and index == 1
                                              and c == previous_background)]
                background = int(rng.choice(background_options))
                try:
                    pair = _pair(rng, n, background)
                except ValueError:
                    continue
                if split == "train" and index == 0:
                    info = _analyze(pair["input"])
                    if not (_nonmonotonic(info["order"])
                            and _nonmonotonic(_area_order(info))):
                        continue
                key = _key(pair["input"])
                if key in seen or _digest(pair) in _SOURCE:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_n = n
                    previous_background = background
                break
            else:
                raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_counts, train_backgrounds = set(), set()
    color_order_evidence = area_order_evidence = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer), "invalid grid type")
                _require(all(1 <= side <= 30 for side in grid.shape), "invalid grid size")
                _require(np.all((grid >= 0) & (grid <= 9)), "invalid ARC color")
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(_digest(pair) not in _SOURCE, "source pair copy")
            info = _analyze(pair["input"])
            _require(np.array_equal(pair["output"], _rings(info["order"])),
                     "output does not encode the inferred stack")
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "unchanged input is not a meaningful example")
            if split == "train":
                train_counts.add(len(info["order"]))
                train_backgrounds.add(info["background"])
                color_order_evidence |= _nonmonotonic(info["order"])
                area_order_evidence |= _nonmonotonic(_area_order(info))
    _require(len(train_counts) >= 2, "training must show variable output size")
    _require(len(train_backgrounds) >= 2, "training must show variable background")
    _require(color_order_evidence and area_order_evidence,
             "training must disprove numeric-color and area rank shortcuts")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    _require(not _equal(generate_puzzle(987), generate_puzzle(988)),
             "different seeds produced the same puzzle")
    for seed, train, test in ((0, None, None), (314159, 7, 3),
                              (5000, 2, 1), (5001, 10, 1), (5002, 10, 4),
                              (6000, None, 1), (6001, None, 4),
                              (6002, 2, None), (6003, 10, None)):
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        _require(_equal(a, b), "generation is not deterministic")
        validate_generated_puzzle(a)
        if train is not None:
            _require(len(a["train"]) == train, "requested training count ignored")
        if test is not None:
            _require(len(a["test"]) == test, "requested test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7000, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts were accepted")
    train_lengths, test_lengths = set(), set()
    shapes, frame_counts, backgrounds, contacts, structures = set(), set(), set(), set(), set()
    covered_corners = shared_lines = triple_cells = edge_frames = narrow_frames = 0
    coincident_sides = mixed_intersections = combined_geometry = 0
    small_canvases = large_canvases = zero_frames = 0
    all_wide = all_tall = square_heavy = monotonic_areas = 0
    test_smaller = test_equal = test_larger = novel_tests = reused_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        train_infos = [_analyze(p["input"]) for p in puzzle["train"]]
        test_infos = [_analyze(p["input"]) for p in puzzle["test"]]
        train_colors = set().union(*(set(i["order"]) for i in train_infos))
        max_train = max(len(i["order"]) for i in train_infos)
        for info in test_infos:
            n = len(info["order"])
            test_smaller += n < max_train
            test_equal += n == max_train
            test_larger += n > max_train
            novel_tests += bool(set(info["order"]) - train_colors)
            reused_tests += not bool(set(info["order"]) - train_colors)
        for pair, info in zip(puzzle["train"]+puzzle["test"], train_infos+test_infos):
            grid = pair["input"]
            shapes.add(grid.shape)
            frame_counts.add(len(info["order"]))
            backgrounds.add(info["background"])
            contacts.add(info["contacts"])
            structures.add((grid.shape, tuple(sorted(info["rects"].values()))))
            small_canvases += min(grid.shape) <= 12
            large_canvases += max(grid.shape) >= 26
            zero_frames += info["background"] != 0 and 0 in info["order"]
            triple_cells += any(len(c) >= 3 for c in info["cover"].values())
            rects = info["rects"]
            dimensions = [(b-t+1, r-l+1) for t, b, l, r in rects.values()]
            all_wide += all(w > h for h, w in dimensions)
            all_tall += all(h > w for h, w in dimensions)
            square_heavy += sum(h == w for h, w in dimensions) >= max(2, len(dimensions)//2)
            monotonic_areas += not _nonmonotonic(_area_order(info))
            has_corner = has_edge = has_narrow = False
            for color, (top, bottom, left, right) in rects.items():
                has_corner |= any(int(grid[cell]) != color for cell in
                                  ((top, left), (top, right), (bottom, left), (bottom, right)))
                has_edge |= top == 0 or left == 0 or bottom == grid.shape[0]-1 or right == grid.shape[1]-1
                has_narrow |= bottom-top == 2 or right-left == 2
            covered_corners += has_corner
            edge_frames += has_edge
            narrow_frames += has_narrow
            has_triple = any(len(c) >= 3 for c in info["cover"].values())
            combined_geometry += has_corner and has_edge and has_narrow and has_triple
            rows = [v for t, b, _, _ in rects.values() for v in (t, b)]
            cols = [v for _, _, l, r in rects.values() for v in (l, r)]
            shared_lines += len(rows) != len(set(rows)) or len(cols) != len(set(cols))
            entries = list(rects.items())
            has_coincident = has_mixed = False
            for index, (a, (ta, ba, la, ra)) in enumerate(entries):
                for b, (tb, bb, lb, rb) in entries[index+1:]:
                    has_coincident |= (bool({ta, ba} & {tb, bb}) and min(ra, rb)-max(la, lb) >= 1) or (bool({la, ra} & {lb, rb}) and min(ba, bb)-max(ta, tb) >= 1)
                    cells = _perimeter(rects[a]) & _perimeter(rects[b])
                    has_mixed |= len({int(grid[cell]) for cell in cells}) > 1
            coincident_sides += has_coincident
            mixed_intersections += has_mixed
    _require(len(train_lengths) >= 4 and len(test_lengths) >= 3, "counts lack diversity")
    _require(len(shapes) >= 30 and len(structures) >= 150, "geometry lacks diversity")
    _require(len(frame_counts) >= 7 and max(frame_counts) > 7, "frame counts lack diversity")
    _require(len(backgrounds) >= 7 and len(contacts) >= 10, "roles or contacts lack diversity")
    _require(all((covered_corners, shared_lines, triple_cells, edge_frames, narrow_frames)),
             "reviewed overlap geometry lacks coverage")
    _require(all((coincident_sides, mixed_intersections, combined_geometry,
                  small_canvases, large_canvases, zero_frames,
                  all_wide, all_tall, square_heavy, monotonic_areas)),
             "reviewed geometry, scale, or color combinations lack coverage")
    _require(all((test_smaller, test_equal, test_larger, novel_tests, reused_tests)),
             "train/test regimes lack independent variation")
