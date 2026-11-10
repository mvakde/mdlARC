"""Generate corner-anchored, quarter-turn stencil-copying ARC puzzles."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = frozenset({
    "fe76d4ea8f68be85c842c497820eff7a16c47822fbf24881664fcb3745e63818",
    "7f4539f5c06e1281e14cdbb3db3326486948e8a5090915a0668cc1c2865a2768",
    "dfbdd7238a86c2a00a11c995729f43feb8b89b73d0b7b16210755269fe4c6ab5",
    "0d1481ec43ecbdbc90d6540ab6319a92de1883fb92d5b7daf281d6ba1734d395",
})


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count_arg(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _counts(rng, seed, num_train, num_test):
    _count_arg(num_train, "num_train", 2, 10)
    _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        nt = 1 + seed % 4
        minimum = max(2, nt + 1)
        nr = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        nt = int(num_test)
        nr = int(rng.integers(max(2, nt + 1), 11))
    elif num_test is None:
        nr = int(num_train)
        nt = int(rng.integers(1, min(4, nr - 1) + 1))
    else:
        nr, nt = int(num_train), int(num_test)
    if nr <= nt:
        raise ValueError("num_train must exceed num_test")
    return nr, nt


def _corners(r, c):
    return ((r, c), (r, c + 2), (r + 2, c + 2), (r + 2, c))


def _frame_from_box(br, bc, size, orientation):
    d = size - 1
    return ((br + d, bc + d), (br + d, bc), (br, bc), (br, bc + d))[orientation]


def _footprint(frame, size):
    r, c, o = frame
    mr, mc = _corners(r, c)[o]
    return ((mr - size + 1, mc - size + 1),
            (mr - size + 1, mc), (mr, mc), (mr, mc - size + 1))[o]


def _rotate(base, orientation):
    return np.rot90(base, -(orientation - 2) % 4).copy()


def _unrotate(oriented, orientation):
    return np.rot90(oriented, (orientation - 2) % 4).copy()


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, n)) for n in (1, 2, 3))


def _stencil(rng, size, forced_asymmetric=False, seven=False):
    if size == 2:
        counts = (2, 3, 4)
    elif size == 3:
        counts = (4, 5, 6, 7, 8, 9)
    else:
        counts = tuple(range(6, 14))
    for _ in range(400):
        count = 7 if seven else int(rng.choice(counts))
        chosen = rng.choice(size * size, size=count, replace=False)
        mask = np.zeros((size, size), dtype=bool)
        mask.flat[chosen] = True
        if not (mask[-1].any() and mask[:, -1].any()):
            continue
        if forced_asymmetric and not _asymmetric(mask):
            continue
        return mask
    raise RuntimeError("unable to sample stencil")


def _bbox(frame, size):
    r, c, _ = frame
    sr, sc = _footprint(frame, size)
    return (min(r, sr), max(r + 2, sr + size - 1),
            min(c, sc), max(c + 2, sc + size - 1))


def _boxes_disjoint(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _layout(rng, height, width, size, orientations, compact):
    box = size + 2
    if compact:
        row_span = min(height - box + 1, max(20, size * 4 + 8))
        col_span = min(width - box + 1, max(20, size * 4 + 8))
        row_offset = int(rng.integers(0, height - box - row_span + 2))
        col_offset = int(rng.integers(0, width - box - col_span + 2))
    else:
        row_span, col_span = height - box + 1, width - box + 1
        row_offset = col_offset = 0
    for _ in range(55):
        frames = []
        boxes = []
        for o in orientations:
            choices = rng.permutation(row_span * col_span)
            for index in choices:
                br = row_offset + int(index) // col_span
                bc = col_offset + int(index) % col_span
                r, c = _frame_from_box(br, bc, size, o)
                frame = (r, c, o)
                bbox = _bbox(frame, size)
                if all(_boxes_disjoint(bbox, old) for old in boxes):
                    frames.append(frame)
                    boxes.append(bbox)
                    break
            else:
                break
        if len(frames) == len(orientations):
            return frames
    raise RuntimeError("unable to pack objects")


def _paint(grid, mask, start, color):
    rr, cc = np.nonzero(mask)
    sr, sc = start
    grid[sr + rr, sc + cc] = color


def _recover_frames(grid, marker):
    cells = frozenset(map(tuple, np.argwhere(grid == marker).tolist()))
    _require(len(cells) >= 9 and len(cells) % 3 == 0, "invalid marker count")
    candidates = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            corners = _corners(r, c)
            present = frozenset(p for p in corners if p in cells)
            if len(present) == 3:
                missing = next(o for o, p in enumerate(corners) if p not in cells)
                candidates.append(((r, c, missing), present))
    by_cell = {cell: [] for cell in cells}
    for i, (_, points) in enumerate(candidates):
        for point in points:
            by_cell[point].append(i)
    solutions = []
    def search(remaining, chosen):
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen)
            return
        pivot = min(remaining, key=lambda x: sum(candidates[i][1] <= remaining for i in by_cell[x]))
        for i in by_cell[pivot]:
            if candidates[i][1] <= remaining:
                search(remaining - candidates[i][1], chosen + (i,))
    search(cells, ())
    _require(len(solutions) == 1, "marker frames do not partition uniquely")
    return sorted(candidates[i][0] for i in solutions[0])


def _analyze_with_roles(pair, background, marker, payload):
    source, target = pair["input"], pair["output"]
    frames = _recover_frames(source, marker)
    _require(np.array_equal(source == marker, target == marker), "markers changed")
    _require(set(np.unique(target).tolist()) == {background, marker, payload}, "palette changed")
    input_payload = set(map(tuple, np.argwhere(source == payload).tolist()))
    output_payload = set(map(tuple, np.argwhere(target == payload).tolist()))
    marker_cells = set(map(tuple, np.argwhere(source == marker).tolist()))
    valid = []
    for size in (2, 3, 4):
        footprints = []
        boxes = []
        for frame in frames:
            sr, sc = _footprint(frame, size)
            if sr < 0 or sc < 0 or sr + size > source.shape[0] or sc + size > source.shape[1]:
                break
            cells = {(sr + r, sc + c) for r in range(size) for c in range(size)}
            if cells & marker_cells or any(cells & old for old in footprints):
                break
            box = _bbox(frame, size)
            if any(not _boxes_disjoint(box, old) for old in boxes):
                break
            footprints.append(cells)
            boxes.append(box)
        if len(footprints) != len(frames):
            continue
        frame = frames[0]
        sr, sc = _footprint(frame, size)
        oriented = target[sr:sr + size, sc:sc + size] == payload
        base = _unrotate(oriented, frame[2])
        if not (base[-1].any() and base[:, -1].any()) or int(base.sum()) < 2:
            continue
        expected = set()
        parts = []
        for frame in frames:
            start = _footprint(frame, size)
            rr, cc = np.nonzero(_rotate(base, frame[2]))
            part = {(start[0] + int(r), start[1] + int(c)) for r, c in zip(rr, cc)}
            expected |= part
            parts.append(part)
        if expected != output_payload:
            continue
        for exemplar_index, part in enumerate(parts):
            if not part <= input_payload:
                continue
            extra = input_payload - part
            if len(extra) > 2:
                continue
            box = boxes[exemplar_index]
            if any((r, c) in marker_cells or not (box[0] - 1 <= r <= box[1] + 1 and box[2] - 1 <= c <= box[3] + 1) for r, c in extra):
                continue
            valid.append({"palette": (background, marker, payload), "frames": frames,
                          "size": size, "stencil": base, "cells": int(base.sum()),
                          "exemplar": exemplar_index, "overdraw": len(extra),
                          "boxes": boxes, "height": source.shape[0], "width": source.shape[1]})
    _require(len(valid) == 1, "stencil size or exemplar is ambiguous")
    return valid[0]


def _analyze_pair(pair):
    source = pair["input"]
    values, counts = np.unique(source, return_counts=True)
    _require(len(values) == 3, "input needs three distinct colors")
    background = int(values[np.argmax(counts)])
    _require(int(max(counts)) > int(np.partition(counts, -2)[-2]), "background ambiguous")
    others = [int(x) for x in values if x != background]
    valid = []
    for marker, payload in (others, others[::-1]):
        try:
            valid.append(_analyze_with_roles(pair, background, marker, payload))
        except (AssertionError, ValueError):
            pass
    _require(len(valid) == 1, "color roles are ambiguous")
    return valid[0]


def _input_candidates(source):
    values, counts = np.unique(source, return_counts=True)
    if len(values) != 3:
        return []
    background = int(values[np.argmax(counts)])
    results = []
    for marker in [int(x) for x in values if x != background]:
        payload = next(int(x) for x in values if x != background and x != marker)
        try:
            frames = _recover_frames(source, marker)
        except AssertionError:
            continue
        pixels = set(map(tuple, np.argwhere(source == payload).tolist()))
        for size in (2, 3, 4):
            boxes = [_bbox(f, size) for f in frames]
            if any(not _boxes_disjoint(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i)):
                continue
            for exemplar in frames:
                sr, sc = _footprint(exemplar, size)
                if sr < 0 or sc < 0 or sr + size > source.shape[0] or sc + size > source.shape[1]:
                    continue
                if any(_footprint(f, size)[0] < 0 or _footprint(f, size)[1] < 0 or _footprint(f, size)[0] + size > source.shape[0] or _footprint(f, size)[1] + size > source.shape[1] for f in frames):
                    continue
                footprint = {(sr+r, sc+c) for r in range(size) for c in range(size)}
                if not pixels <= footprint:
                    continue
                oriented = source[sr:sr+size, sc:sc+size] == payload
                base = _unrotate(oriented, exemplar[2])
                if not base[-1].any() or not base[:, -1].any() or int(base.sum()) < 2:
                    continue
                result = source.copy()
                result[result == payload] = background
                for frame in frames:
                    fr, fc = _footprint(frame, size)
                    rr, cc = np.nonzero(_rotate(base, frame[2]))
                    if any(result[fr + int(r), fc + int(c)] == marker for r, c in zip(rr, cc)):
                        break
                    result[fr + rr, fc + cc] = payload
                else:
                    results.append(result)
    return results


def _pair_digest(pair):
    a, b = pair["input"], pair["output"]
    return hashlib.sha256(str(a.shape).encode() + a.astype(np.int8).tobytes() + b.astype(np.int8).tobytes()).hexdigest()


def _signature(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def _make_pair(rng, regime, palette, size, forced_orientations=(), source_like=False, noisy=False, force_asymmetric=False):
    for _ in range(100):
        height = int(rng.integers(22, 31))
        width = int(rng.integers(22, 31))
        count = 8 if source_like else int(rng.integers(3, 9 if regime == "target_test" else 8))
        if count < len(forced_orientations):
            count = len(forced_orientations)
        orientations = list(forced_orientations) + rng.integers(0, 4, size=count - len(forced_orientations)).tolist()
        rng.shuffle(orientations)
        try:
            frames = _layout(rng, height, width, size, orientations, source_like)
        except RuntimeError:
            continue
        base = _stencil(rng, size, force_asymmetric, seven=source_like)
        background, marker, payload = palette
        source = np.full((height, width), background, dtype=np.int8)
        for r, c, o in frames:
            for i, (rr, cc) in enumerate(_corners(r, c)):
                if i != o:
                    source[rr, cc] = marker
        exemplar_index = int(rng.integers(len(frames)))
        exemplar = frames[exemplar_index]
        _paint(source, _rotate(base, exemplar[2]), _footprint(exemplar, size), payload)
        target = source.copy()
        target[target == payload] = background
        for frame in frames:
            _paint(target, _rotate(base, frame[2]), _footprint(frame, size), payload)
        if noisy:
            box = _bbox(exemplar, size)
            options = [(r, c) for r in range(max(0, box[0]-1), min(height, box[1]+2))
                       for c in range(max(0, box[2]-1), min(width, box[3]+2))
                       if source[r, c] == background and target[r, c] == background
                       and all((r < b[0] or r > b[1] or c < b[2] or c > b[3])
                               for j, f in enumerate(frames) if j != exemplar_index
                               for b in [_bbox(f, size)])]
            if not options:
                continue
            selected = rng.choice(len(options), size=min(len(options), int(rng.integers(1, 3))), replace=False)
            for i in np.atleast_1d(selected):
                source[options[int(i)]] = payload
        pair = {"input": source, "output": target}
        try:
            info = _analyze_pair(pair)
            if regime == "target_test":
                answers = _input_candidates(source)
                _require(len(answers) == 1 and np.array_equal(answers[0], target), "test input not uniquely solvable")
            _require(_pair_digest(pair) not in _SOURCE_PAIR_DIGESTS, "source copy")
            return pair
        except AssertionError:
            continue
    raise RuntimeError("could not construct unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    shared_palette = bool(rng.integers(0, 2))
    palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    def colors():
        return palette if shared_palette else tuple(int(x) for x in rng.choice(10, 3, replace=False))
    sizes = [2, 3]
    if train_count >= 3 and rng.random() < .27:
        sizes.append(4)
    sizes.extend(int(rng.choice((2, 3, 4) if 4 in sizes else (2, 3))) for _ in range(train_count - len(sizes)))
    rng.shuffle(sizes)
    # The first two demonstrations jointly cover all missing corners. The
    # 3x3 demonstration makes a quarter-turn visibly different.
    first3 = sizes.index(3)
    other = next(i for i in range(train_count) if i != first3)
    forced = {first3: (0, 1), other: (2, 3)}
    used = set()
    def unique_pair(regime, size, forced_orientations=(), source_like=False, noisy=False, force_asymmetric=False):
        for _ in range(25):
            child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            pair = _make_pair(child, regime, colors(), size, forced_orientations, source_like, noisy, force_asymmetric)
            sig = _signature(pair["input"])
            if sig not in used:
                used.add(sig)
                return pair
        raise RuntimeError("duplicate puzzle input")
    train = [unique_pair("source_train", size, forced.get(i, ()), noisy=(i not in forced and rng.random() < .22), force_asymmetric=(i == first3)) for i, size in enumerate(sizes)]
    test = []
    for _ in range(test_count):
        source_like = bool(rng.random() < .32)
        size = 3 if source_like else int(rng.choice(sizes))
        test.append(unique_pair("target_test", size, source_like=source_like))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "grid type")
                _require(all(1 <= n <= 30 for n in grid.shape) and int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid bounds")
            _require(pair["input"].shape == pair["output"].shape and not np.array_equal(pair["input"], pair["output"]), "dimension or transformation")
            insig = _signature(pair["input"])
            psig = (insig, _signature(pair["output"]))
            _require(insig not in seen_inputs and psig not in seen_pairs, "duplicate puzzle pair")
            seen_inputs.add(insig)
            seen_pairs.add(psig)
            _require(_pair_digest(pair) not in _SOURCE_PAIR_DIGESTS, "exact source copy")
            info = _analyze_pair(pair)
            if split == "test":
                _require(info["overdraw"] == 0, "test overdraw is ambiguous")
                answers = _input_candidates(pair["input"])
                _require(len(answers) == 1 and np.array_equal(answers[0], pair["output"]), "test input has no unique stencil answer")
            information[split].append(info)
    trained_sizes = {x["size"] for x in information["train"]}
    trained_orientations = {f[2] for x in information["train"] for f in x["frames"]}
    _require({2, 3} <= trained_sizes and trained_orientations == {0, 1, 2, 3}, "training coverage")
    _require(any(x["size"] == 3 and _asymmetric(x["stencil"]) and len({f[2] for f in x["frames"]}) > 1 for x in information["train"]), "visible rotation demonstration")
    for x in information["test"]:
        _require(x["size"] in trained_sizes and {f[2] for f in x["frames"]} <= trained_orientations, "test convention not trained")


def self_test() -> None:
    def equal(a, b):
        return all(np.array_equal(x[k], y[k]) for split in ("train", "test") for x, y in zip(a[split], b[split]) for k in ("input", "output")) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))
    _require(equal(generate_puzzle(12345), generate_puzzle(12345)), "automatic determinism")
    _require(equal(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3)), "explicit determinism")
    for n, (nr, nt) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(8000+n, nr, nt)
        _require((nr is None or len(p["train"]) == nr) and (nt is None or len(p["test"]) == nt), "partial or boundary count")
    for n, arguments in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(9000+n, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    sizes, counts, shapes, palettes, stencils, layouts = set(), set(), set(), set(), set(), set()
    empty_anchors = symmetric = solid = noisy = source_like = varied_palettes = 0
    subset_orientations = easier_tests = six_by_six_tests = diagonal_touch = 0
    test_sizes = set()
    test_frame_counts = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        infos = {split: [_analyze_pair(pair) for pair in p[split]] for split in ("train", "test")}
        varied_palettes += len({x["palette"] for split in infos.values() for x in split}) > 1
        for split in ("train", "test"):
            for x in infos[split]:
                sizes.add(x["size"])
                counts.add(len(x["frames"]))
                shapes.add((x["height"], x["width"]))
                palettes.add(x["palette"])
                stencils.add((x["size"], x["stencil"].tobytes()))
                layouts.add((x["height"], x["width"], tuple(x["frames"])))
                empty_anchors += not bool(x["stencil"][0, 0])
                symmetric += not _asymmetric(x["stencil"])
                solid += x["size"] == 2 and x["cells"] == 4
                noisy += x["overdraw"] > 0
                source_like += split == "test" and len(x["frames"]) == 8 and x["size"] == 3 and x["cells"] == 7
                subset_orientations += len({f[2] for f in x["frames"]}) < 4
                if split == "test":
                    test_sizes.add(x["size"])
                    test_frame_counts.add(len(x["frames"]))
                    easier_tests += len(x["frames"]) <= 5
                    six_by_six_tests += (len({b[0] for b in x["boxes"]}) >= 6
                                         and len({b[2] for b in x["boxes"]}) >= 6)
                    diagonal_touch += any(
                        (a[1] + 1 == b[0] or b[1] + 1 == a[0])
                        and (a[3] + 1 == b[2] or b[3] + 1 == a[2])
                        for i, a in enumerate(x["boxes"]) for b in x["boxes"][:i]
                    )
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _require({2, 3} <= sizes and len(counts) >= 5 and len(shapes) >= 25 and len(palettes) >= 25 and len(stencils) >= 25 and len(layouts) >= 100, "structural diversity")
    _require(empty_anchors >= 20 and symmetric >= 20 and solid >= 5 and noisy >= 1 and source_like >= 5 and varied_palettes >= 5, "review coverage diversity")
    _require(subset_orientations >= 50 and easier_tests >= 15
             and six_by_six_tests >= 15 and diagonal_touch >= 3,
             "orientation, layout, or easy-test diversity")
    _require(test_sizes == {2, 3, 4} and len(test_frame_counts) >= 5,
             "test scale or frame-count diversity")
