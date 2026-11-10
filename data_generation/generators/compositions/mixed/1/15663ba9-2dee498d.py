"""Label contour curvature, then stably deduplicate the labeled columns.

Only NumPy and the standard library are used. No files are read by the solver
or generator. Tests retain hashes and counters, not a corpus of sample grids.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


CONVEX = 4
CONCAVE = 2
_INPUT_COLORS = (0, 1, 3, 5, 6, 7, 8, 9)
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 142305657, 344790878, 478332575, 970893749,
              3424771, 263924815, 148830932, 70527619, 110840995,
              128576874, 931034920, 768202411, 223827936, 260833077,
              623669792, 891620357, 53558083, 507934926, 124054441,
              544718466, 809859508, 228988316, 211507207, 846445732,
              458398807, 662012507, 853561866, 909378832, 534994070,
              821606197, 311116661, 101493239, 798841528, 665990911,
              559407286, 996349, 282320635, 715969166, 514985823,
              372246588, 96802941)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "grid must be integer")
    _require(bool(np.all((0 <= array) & (array <= 9))), "illegal color")
    return array.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, point)) for point in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                other = (r + dr, c + dc)
                if other in remaining:
                    remaining.remove(other)
                    cells.add(other)
                    stack.append(other)
        result.append(cells)
    return result


def _trace(cells):
    def neighbors(cell):
        r, c = cell
        return sorted((r + dr, c + dc) for dr, dc in _DIRECTIONS
                      if (r + dr, c + dc) in cells)
    _require(len(cells) >= 8 and all(len(neighbors(p)) == 2 for p in cells),
             "each contour must be a simple orthogonal cycle")
    start = min(cells)
    path, visited = [start], {start}
    previous, current = start, neighbors(start)[0]
    while current != start:
        _require(current not in visited, "cycle revisits a cell")
        path.append(current)
        visited.add(current)
        following = next(p for p in neighbors(current) if p != previous)
        previous, current = current, following
    _require(len(path) == len(cells), "disconnected cycle")
    return path


def _inside_cells(cells):
    """Padded flood fill also handles contours near the canvas boundary."""
    rows, cols = zip(*cells)
    r0, c0 = min(rows) - 1, min(cols) - 1
    h, w = max(rows) - r0 + 2, max(cols) - c0 + 2
    exterior, stack = {(r0, c0)}, [(r0, c0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRECTIONS:
            p = (r + dr, c + dc)
            if (r0 <= p[0] < r0 + h and c0 <= p[1] < c0 + w
                    and p not in cells and p not in exterior):
                exterior.add(p)
                stack.append(p)
    return {(r, c) for r in range(r0, r0 + h)
            for c in range(c0, c0 + w)} - exterior - cells


def _classify(cells):
    path = _trace(cells)
    interior = _inside_cells(cells)
    _require(bool(interior), "contour has no enclosed background cell")
    area = sum(c * path[(i + 1) % len(path)][0]
               - r * path[(i + 1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    _require(area != 0, "zero-area contour")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        cross = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        classes[(r, c)] = 0 if cross == 0 else (1 if cross * area > 0 else 2)
    counts = Counter(classes.values())
    _require(counts[1] - counts[2] == 4, "invalid turning balance")
    return {"cells": cells, "path": path, "classes": classes,
            "interior": interior, "convex": counts[1], "concave": counts[2]}


def _analyze(source):
    _require(min(source.shape) >= 5, "a framed contour needs at least 5 cells")
    background = int(source[0, 0])
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    _require(bool(np.all(border == background)), "outer frame must be background")
    _require(not np.any(np.isin(source, (CONVEX, CONCAVE))),
             "input cannot use the reserved bend labels")
    infos = []
    for cells in _components(source != background):
        colors = {int(source[p]) for p in cells}
        _require(len(colors) == 1, "each contour must start monochrome")
        info = _classify(cells)
        info["color"] = colors.pop()
        infos.append(info)
    _require(bool(infos), "no contours")
    for i, info in enumerate(infos):
        for other in infos[i + 1:]:
            _require(not (info["interior"] & other["cells"]
                          or other["interior"] & info["cells"]), "nested contours")
            _require(all(abs(r - s) > 1 or abs(c - t) > 1
                         for r, c in info["cells"] for s, t in other["cells"]),
                     "contours must have a blank cell of separation")
    return background, infos


def _label(source, infos):
    labeled = source.copy()
    for info in infos:
        for cell, kind in info["classes"].items():
            if kind:
                labeled[cell] = CONVEX if kind == 1 else CONCAVE
    return labeled


def _column_keys(grid):
    return [tuple(map(int, grid[:, c])) for c in range(grid.shape[1])]


def _retained_indices(grid):
    seen, indices = set(), []
    for c, key in enumerate(_column_keys(grid)):
        if key not in seen:
            seen.add(key)
            indices.append(c)
    return indices


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only rule: classify on the original geometry, then compress once."""
    source = _grid(input_grid)
    _, infos = _analyze(source)
    labeled = _label(source, infos)
    return labeled[:, _retained_indices(labeled)].copy()


def _mask_from_vertices(vertices):
    cleaned = []
    for p in vertices:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    if cleaned[-1] == cleaned[0]:
        cleaned.pop()
    cells = []
    for i, (r, c) in enumerate(cleaned):
        nr, nc = cleaned[(i + 1) % len(cleaned)]
        _require((r == nr) != (c == nc), "non-orthogonal polygon edge")
        dr, dc = int(np.sign(nr - r)), int(np.sign(nc - c))
        cells.extend((r + j * dr, c + j * dc)
                     for j in range(abs(nr - r) + abs(nc - c)))
    _require(len(set(cells)) == len(cells), "polygon crosses itself")
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    mask = np.zeros((max(r for r, _ in cells) - r0 + 1,
                     max(c for _, c in cells) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _rectangle(h, w):
    return _mask_from_vertices([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])


def _profile_mask(top, bottom, widths):
    """Boundary of an orthogonal ribbon with independently stepped edges."""
    xs = [0]
    for width in widths:
        xs.append(xs[-1] + int(width))
    vertices = [(int(top[0]), 0)]
    for i in range(1, len(top)):
        vertices.extend(((int(top[i - 1]), xs[i]), (int(top[i]), xs[i])))
    vertices.extend(((int(top[-1]), xs[-1]), (int(bottom[-1]), xs[-1])))
    for i in range(len(bottom) - 1, 0, -1):
        vertices.extend(((int(bottom[i]), xs[i]), (int(bottom[i - 1]), xs[i])))
    vertices.append((int(bottom[0]), 0))
    return _mask_from_vertices(vertices)


def _ribbon(rng, target):
    if target == 0:
        mask = _rectangle(int(rng.integers(3, 9)), int(rng.integers(3, 11)))
    else:
        segments = int(rng.integers((target + 1) // 2 + 1,
                                    min(target + 1, 8) + 1))
        slots = [(edge, i) for edge in range(2) for i in range(1, segments)]
        selected = {slots[int(j)] for j in rng.choice(len(slots), target, replace=False)}
        levels = []
        top_band = int(rng.integers(2, 5))
        bottom_base = top_band + int(rng.integers(2, 5))
        for edge, low in ((0, 0), (1, bottom_base)):
            high = top_band if edge == 0 else bottom_base + int(rng.integers(2, 5))
            values = [int(rng.integers(low, high + 1))]
            for i in range(1, segments):
                choices = [v for v in range(low, high + 1) if v != values[-1]]
                values.append(int(rng.choice(choices)) if (edge, i) in selected else values[-1])
            levels.append(values)
        # At most 27 cells wide, so every orientation fits a one-cell frame.
        widths = [2] * segments
        for _ in range(int(rng.integers(0, 27 - 2 * segments))):
            widths[int(rng.integers(segments))] += 1
        mask = _profile_mask(levels[0], levels[1], widths)
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return np.ascontiguousarray(mask)


def _place_masks(rng, shapes, colors, background):
    max_h = max(s.shape[0] for s in shapes)
    max_w = max(s.shape[1] for s in shapes)
    for _ in range(16):
        h = int(rng.integers(max(max_h + 2, 12 if len(shapes) == 1 else 23), 31))
        w = int(rng.integers(max(max_w + 2, 12 if len(shapes) == 1 else 23), 31))
        result = np.full((h, w), background, dtype=np.int8)
        forbidden = np.zeros((h, w), dtype=bool)
        occupied_interiors = set()
        success = True
        for shape, color in zip(shapes, colors):
            sh, sw = shape.shape
            found = False
            for _ in range(90):
                r, c = int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw))
                # Reject interior placements, but permit overlapping bounding boxes.
                local = forbidden[r:r + sh, c:c + sw]
                if np.any(local & shape):
                    continue
                cells = {(r + int(sr), c + int(sc)) for sr, sc in np.argwhere(shape)}
                if cells & occupied_interiors:
                    continue
                interior = _inside_cells(cells)
                if any(result[p] != background for p in interior):
                    continue
                result[r:r + sh, c:c + sw][shape] = color
                occupied_interiors.update(interior)
                for rr, cc in cells:
                    forbidden[rr - 1:rr + 2, cc - 1:cc + 2] = True
                found = True
                break
            if not found:
                success = False
                break
        if success:
            return result
    return None


def _dependency_witness(rng, background, foreground):
    """A stepped ribbon and two loops split one raw column equivalence class."""
    upper_depth, lower_depth = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    gap = int(rng.integers(2, 5))
    lower_start = upper_depth + gap
    a, b = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    main = _profile_mask([0, upper_depth],
                         [lower_start, lower_start + lower_depth], [a, b])
    rw1, rw2 = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    top_loop, bottom_loop = _rectangle(upper_depth + 1, rw1), _rectangle(lower_depth + 1, rw2)
    separation = int(rng.integers(2, 5))
    x = main.shape[1] + separation
    inner_h, inner_w = main.shape[0], x + max(rw1, rw2)
    margin_top, margin_left = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    h = inner_h + margin_top + int(rng.integers(1, 4))
    w = inner_w + margin_left + int(rng.integers(1, 4))
    result = np.full((h, w), background, dtype=np.int8)
    for shape, y, xx in ((main, 0, 0), (top_loop, 0, x),
                          (bottom_loop, lower_start, x)):
        sh, sw = shape.shape
        result[margin_top + y:margin_top + y + sh,
               margin_left + xx:margin_left + xx + sw][shape] = foreground
    if rng.random() < .5:
        result = np.fliplr(result)
    if rng.random() < .5:
        result = np.flipud(result)
    return result.copy()


def _facts(source, labeled, infos):
    raw, keys = _column_keys(source), _column_keys(labeled)
    keep = _retained_indices(labeled)
    last = [i for i, key in enumerate(keys) if key not in keys[i + 1:]]
    blank = (int(source[0, 0]),) * source.shape[0]
    split = any(raw[i] == raw[j] and keys[i] != keys[j]
                for i in range(len(keys)) for j in range(i + 1, len(keys)))
    separated = any(keys[i] != blank and keys[i] == keys[j]
                    and any(k != keys[i] for k in keys[i + 1:j])
                    for i in range(len(keys)) for j in range(i + 2, len(keys)))
    last_new = max(keep)
    early_repeat = any(keys[i] != blank and keys[i] in keys[:i]
                       for i in range(1, last_new))
    first_foreground = [keys[i] for i in keep if keys[i] != blank]
    last_foreground = [keys[i] for i in last if keys[i] != blank]
    return {"split_raw_equality": split, "separated_duplicate": separated,
            "early_foreground_duplicate": early_repeat,
            "crop_fails": keys[:len(keep)] != [keys[i] for i in keep],
            "last_order_fails": [keys[i] for i in last] != [keys[i] for i in keep],
            "last_foreground_order_fails": first_foreground != last_foreground,
            "components": len(infos), "concaves": max(i["concave"] for i in infos),
            "removed": len(keys) - len(keep)}


def _new_pair(rng, index, test, background, palette, seen):
    for _ in range(40):
        if not test and index == 0:
            source = _dependency_witness(rng, background, int(rng.choice(palette)))
        else:
            target = int(rng.integers(7, 13)) if test else (
                int(rng.integers(1, 7)) if index == 1 else int(rng.integers(0, 7)))
            shapes = [_ribbon(rng, target)]
            count = int(rng.choice([1, 2, 3, 4], p=[.42, .32, .20, .06]))
            for _ in range(count - 1):
                # Auxiliary loops vary independently and need not be rectangles.
                if rng.random() < .25:
                    auxiliary = _profile_mask([0, 1], [4, 5], [2, 2])
                    auxiliary = np.rot90(auxiliary, int(rng.integers(4)))
                else:
                    auxiliary = _rectangle(int(rng.integers(3, 7)), int(rng.integers(3, 7)))
                shapes.append(auxiliary)
            colors = [int(rng.choice(palette)) for _ in shapes]
            source = _place_masks(rng, shapes, colors, background)
            if source is None:
                continue
        if max(source.shape) > 30:
            continue
        _, infos = _analyze(source)
        labeled = _label(source, infos)
        facts = _facts(source, labeled, infos)
        key = (source.shape, source.tobytes())
        if key in seen or facts["removed"] == 0:
            continue
        if not test and index == 0 and not all(facts[k] for k in (
                "split_raw_equality", "separated_duplicate", "crop_fails",
                "early_foreground_duplicate", "last_foreground_order_fails")):
            continue
        if not test and index == 1 and not facts["crop_fails"]:
            continue
        seen.add(key)
        return {"input": source, "output": labeled[:, _retained_indices(labeled)].copy()}
    raise RuntimeError("bounded construction failed to produce a unique witness")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer))
                 and not isinstance(value, (bool, np.bool_)) and lo <= value <= hi,
                 f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer))
             and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(_INPUT_COLORS)]
    backgrounds, train_colors, test_colors = colors[:3], colors[3:6], colors[6:]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            is_test = split == "test"
            background = backgrounds[(index + int(is_test)) % len(backgrounds)]
            palette = test_colors if is_test else train_colors
            puzzle[split].append(_new_pair(rng, index, is_test, background, palette, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _point_inside(path, row, col):
    # Ray casting through cell centers, sampled away from all contour edges.
    crossings = 0
    for (r, c), (nr, nc) in zip(path, path[1:] + path[:1]):
        if c == nc and min(r, nr) < row < max(r, nr) and c > col:
            crossings += 1
    return crossings % 2 == 1


def _audit_labels(source, infos):
    """Check signed turns by a different geometric calculation."""
    result = source.copy()
    for info in infos:
        for cell, kind in info["classes"].items():
            if not kind:
                continue
            r, c = cell
            quadrants = sum(_point_inside(info["path"], r + dr, c + dc)
                            for dr in (-.25, .25) for dc in (-.25, .25))
            _require(quadrants in (1, 3), "bend interior angle is ambiguous")
            expected_kind = 1 if quadrants == 1 else 2
            _require(kind == expected_kind, "signed-turn and interior classifications differ")
            result[cell] = CONVEX if quadrants == 1 else CONCAVE
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_facts, split_colors, complexities = set(), [], {}, {}
    for split in ("train", "test"):
        split_colors[split], complexities[split] = set(), []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "each pair must have exactly input/output")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = (source.shape, source.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _, infos = _analyze(source)
            labeled = _audit_labels(source, infos)
            # Explicit pairwise equality rather than the solver's set-based filter.
            keys = _column_keys(labeled)
            keep = [i for i in range(len(keys)) if all(keys[i] != keys[j] for j in range(i))]
            _require(target.shape == (source.shape[0], len(keep)), "wrong output shape")
            _require(np.array_equal(target, labeled[:, keep]), "wrong labeled first-occurrence output")
            _require(len(set(_column_keys(target))) == target.shape[1], "duplicate output column")
            _require(target.shape[1] < source.shape[1], "no compression")
            _require(not np.array_equal(target, source[:, _retained_indices(source)]),
                     "omitting A does not change the answer")
            _require(not np.array_equal(target, labeled), "omitting B does not change the answer")
            _require({CONVEX, CONCAVE} <= set(map(int, target.flat))
                     or all(i["concave"] == 0 for i in infos), "bend labels lost")
            split_colors[split].update(i["color"] for i in infos)
            complexities[split].append(max(i["concave"] for i in infos))
            if split == "train":
                train_facts.append(_facts(source, labeled, infos))
    _require(any(f["split_raw_equality"] for f in train_facts), "no causal order witness")
    _require(any(f["crop_fails"] and f["separated_duplicate"]
                 and f["early_foreground_duplicate"] and f["last_foreground_order_fails"]
                 for f in train_facts), "insufficient dedup evidence")
    _require(any(f["concaves"] for f in train_facts), "training lacks concave bends")
    _require(not (split_colors["train"] & split_colors["test"]), "test contour palette is not held out")
    _require(min(complexities["test"]) > max(complexities["train"]),
             "test must extend the per-contour bend count")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, one puzzle in memory at a time; no disk writes or traces."""
    hashes, shapes, layouts, turn_patterns = set(), set(), set(), set()
    object_counts, concaves, backgrounds, palettes = set(), set(), set(), set()
    witness_count = pairs = near_matches = adjacent_bends = histogram_collisions = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        local_layouts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                _require(np.array_equal(solve(source), target), "input-only solve mismatch")
                background, infos = _analyze(source)
                labeled = _label(source, infos)
                facts = _facts(source, labeled, infos)
                pairs += 1
                witness_count += facts["split_raw_equality"]
                backgrounds.add(background)
                palettes.add(tuple(map(int, np.unique(source))))
                shapes.add((source.shape, target.shape))
                object_counts.add(len(infos))
                pattern = np.where(source == background, 0, 1).astype(np.uint8)
                signature = (source.shape, hashlib.sha256(pattern.tobytes()).hexdigest())
                layouts.add(signature)
                local_layouts.add(signature)
                columns = _column_keys(target)
                near_matches += any(sum(a != b for a, b in zip(x, y)) == 1
                                    for i, x in enumerate(columns) for y in columns[i + 1:])
                histogram_collisions += len({tuple(sorted(k)) for k in columns}) < len(columns)
                for info in infos:
                    concaves.add(info["concave"])
                    kinds = [info["classes"][p] for p in info["path"]]
                    adjacent_bends += any(kinds[i] and kinds[(i + 1) % len(kinds)]
                                          for i in range(len(kinds)))
                    turns = [k for k in kinds if k]
                    candidates = [tuple(seq[i:] + seq[:i])
                                  for seq in (turns, turns[::-1]) for i in range(len(turns))]
                    turn_patterns.add(min(candidates))
        _require(len(local_layouts) >= 3, "only palette changes within a puzzle")
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic generation")
    _require(len(hashes) == 42 and len(layouts) >= 180, "insufficient structural diversity")
    _require(len(shapes) >= 90 and len(turn_patterns) >= 20, "geometry coverage too narrow")
    _require({1, 2, 3, 4} <= object_counts, "missing component counts")
    _require(set(range(13)) <= concaves, "missing bend complexities")
    _require(len(backgrounds) == 8 and len(palettes) >= 40, "palette roles too narrow")
    _require(witness_count >= 42 and histogram_collisions > 0 and adjacent_bends > 0,
             "missing family-specific evidence")
    # Every legal count, including the largest retained puzzle (14 pairs).
    count_cases = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(12345 + a * 10 + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
            count_cases += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(hashes),
            "input_geometries": len(layouts), "shape_pairs": len(shapes),
            "turn_patterns": len(turn_patterns), "component_counts": sorted(object_counts),
            "concave_counts": sorted(concaves), "background_colors": len(backgrounds),
            "palettes": len(palettes), "causal_witness_pairs": witness_count,
            "near_match_pairs": near_matches, "adjacent_bend_contours": adjacent_bends,
            "histogram_collision_pairs": histogram_collisions,
            "legal_count_cases": count_cases}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
