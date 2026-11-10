"""Generate scattered-square inventory puzzles with rank-wise nested outputs."""

import hashlib
import numpy as np


_SOURCE_INPUT_HASHES = {
    "6cd4afe0cb813e5c969e03a871be2784b756830c3ee7bdb09ee1126e4a4c60e8",
    "4a733a86a91296c49cc729cc9a6c8f8cc1f9f364708992c8dd6c7b9f445021b6",
    "1ce072b623feb319dd2008eff8dfedc72ded9e189fdf133c9ad5d9fd165856d9",
    "3c3274008678e01b1f0ea4fa05a2e771d73420a527b4dc8c2505f54b6f620861",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    nt = _count_arg("num_train", num_train, 2, 10)
    nq = _count_arg("num_test", num_test, 1, 4)
    if nt is not None and nq is not None:
        if nt <= nq:
            raise ValueError("num_train must exceed num_test")
        return nt, nq
    if nt is None and nq is None:
        nq = 1 + seed % 4
        nt = nq + 1 + (seed // 4) % (10 - nq)
    elif nt is None:
        nt = int(rng.integers(nq + 1, 11))
    else:
        nq = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, nq


def _digest(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _components(grid, background):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == background:
                continue
            cells = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not visited[yy, xx] and int(grid[yy, xx]) == color:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            r0 = min(y for y, x in cells)
            r1 = max(y for y, x in cells)
            c0 = min(x for y, x in cells)
            c1 = max(x for y, x in cells)
            ch, cw = r1 - r0 + 1, c1 - c0 + 1
            components.append((color, ch, cw, len(cells) == ch * cw,
                               r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
                               (r0, c0, r1, c1), len(cells)))
    return components


def _analyze_input(grid):
    h, w = grid.shape
    hist = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(hist))
    _require(hist[background] > max(hist[c] for c in range(10) if c != background),
             "background must be dominant")
    components = _components(grid, background)
    squares = [x for x in components if x[1] == x[2] and x[1] >= 2 and x[3]]
    _require(squares, "missing complete squares")
    classes = {}
    boxes = []
    for color, side, _, _, edge, box, _ in squares:
        if side in classes:
            _require(classes[side][0] == color, "ambiguous same-size colors")
            classes[side][1] += 1
        else:
            classes[side] = [color, 1]
        boxes.append((side, color, box, edge))
    _require(len({v[0] for v in classes.values()}) == len(classes),
             "one color cannot describe two sizes")
    payload = {v[0] for v in classes.values()}
    remaining = set(np.unique(grid).tolist()) - payload - {background}
    _require(len(remaining) == 1, "exactly one noise color is required")
    noise = remaining.pop()
    fragments = 0
    for color, ch, cw, solid, edge, _, _ in components:
        if color == noise:
            _require(not (ch == cw and ch >= 2 and solid), "noise forms a square")
        elif ch != cw or not solid:
            side = next(s for s, data in classes.items() if data[0] == color)
            _require(edge and solid and ch != cw and max(ch, cw) == side,
                     "invalid payload fragment")
            fragments += 1
    return {"shape": (h, w), "background": background, "noise": noise,
            "noise_cells": int(hist[noise]), "classes": classes,
            "boxes": boxes, "fragments": fragments,
            "edge_squares": sum(edge for _, _, _, edge in boxes)}


def _output(facts):
    classes = facts["classes"]
    counts = [data[1] for data in classes.values()]
    height = max(classes) + 1
    width = sum(max(side for side, (_, n) in classes.items() if n > rank) + 1
                for rank in range(max(counts)))
    result = np.full((height, width), facts["background"], dtype=np.int8)
    cursor = 0
    for rank in range(max(counts)):
        active = sorted((side for side, (_, n) in classes.items() if n > rank), reverse=True)
        for side in active:
            result[:side, cursor:cursor + side] = classes[side][0]
        cursor += active[0] + 1
    return result


def _reserve(mask, r, c, hh, ww):
    h, w = mask.shape
    mask[max(0, r - 1):min(h, r + hh + 1), max(0, c - 1):min(w, c + ww + 1)] = True


def _place_squares(h, w, bg, classes, rng, want_edge):
    items = [(s, color) for s, color, count in classes for _ in range(count)]
    items.sort(key=lambda x: -x[0])
    for attempt in range(90):
        grid = np.full((h, w), bg, dtype=np.int8)
        reserved = np.zeros((h, w), dtype=bool)
        order = items.copy()
        rng.shuffle(order)
        order.sort(key=lambda x: -x[0])
        if want_edge:
            edge_candidates = [(r, c) for r in range(h - order[0][0] + 1)
                               for c in range(w - order[0][0] + 1)
                               if r == 0 or c == 0 or r + order[0][0] == h or c + order[0][0] == w]
            r, c = edge_candidates[int(rng.integers(len(edge_candidates)))]
            side, color = order.pop(0)
            grid[r:r + side, c:c + side] = color
            _reserve(reserved, r, c, side, side)
        success = True
        for side, color in order:
            positions = [(r, c) for r in range(h - side + 1)
                         for c in range(w - side + 1)
                         if not reserved[r:r + side, c:c + side].any()]
            if not positions:
                success = False
                break
            r, c = positions[int(rng.integers(len(positions)))]
            grid[r:r + side, c:c + side] = color
            _reserve(reserved, r, c, side, side)
        if success:
            return grid, reserved
    raise RuntimeError("square packing failed")


def _add_fragment(grid, reserved, classes, rng):
    h, w = grid.shape
    candidates = []
    for side, color, _ in classes:
        if side < 3:
            continue
        for depth in range(1, side):
            for c in range(1, w - side):
                candidates.extend(((0, c, depth, side, color),
                                   (h - depth, c, depth, side, color)))
            for r in range(1, h - side):
                candidates.extend(((r, 0, side, depth, color),
                                   (r, w - depth, side, depth, color)))
    rng.shuffle(candidates)
    for r, c, hh, ww, color in candidates:
        if not reserved[r:r + hh, c:c + ww].any():
            grid[r:r + hh, c:c + ww] = color
            _reserve(reserved, r, c, hh, ww)
            return True
    return False


def _add_noise(grid, bg, noise, count, rng):
    h, w = grid.shape
    cells = np.argwhere(grid == bg)
    rng.shuffle(cells)
    placed = 0
    for rr, cc in cells:
        r, c = int(rr), int(cc)
        grid[r, c] = noise
        square = False
        for y in (r - 1, r):
            for x in (c - 1, c):
                if 0 <= y < h - 1 and 0 <= x < w - 1 and np.all(grid[y:y + 2, x:x + 2] == noise):
                    square = True
        if square:
            grid[r, c] = bg
            continue
        placed += 1
        if placed == count:
            return
    raise RuntimeError("noise placement failed")


def _sample_palette(rng, parameters, split, index, sides):
    mode = parameters["palette_mode"]
    base = parameters["palette"]
    if mode == "stable":
        bg, noise = base[:2]
        colors = [base[2 + side - 2] for side in sides]
    elif mode == "reuse" and split == "test":
        bg, noise = base[2], base[0]
        available = list(base[3:])
        rng.shuffle(available)
        colors = [base[1], *available[:len(sides) - 1]]
    elif mode == "reuse":
        bg, noise = base[:2]
        available = list(base[3:])
        rng.shuffle(available)
        colors = ([base[2], *available[:len(sides) - 1]] if index == 0
                  else [int(x) for x in rng.choice(np.asarray(base[2:]), size=len(sides), replace=False)])
    else:
        p = [int(x) for x in rng.permutation(10)]
        bg, noise = p[:2]
        colors = p[2:2 + len(sides)]
    return int(bg), int(noise), [int(c) for c in colors]


def _sample_spec(rng, parameters, split, index):
    if split == "train" and index == 0:
        sides = (2, 3, 4)
        counts = (int(rng.integers(3, 5)), 2, 2)
    elif split == "train" and index == 1:
        sides = tuple(sorted(int(x) for x in rng.choice(np.arange(2, 6), size=3, replace=False)))
        leader = int(rng.integers(1, 3))
        c = [int(x) for x in rng.integers(1, 4, size=3)]
        c[leader] = max(c) + int(rng.integers(1, 3))
        counts = tuple(c)
    else:
        nclass = int(rng.choice((2, 3, 4), p=(0.22, 0.56, 0.22)))
        sides = tuple(sorted(int(x) for x in rng.choice(np.arange(2, 6), size=nclass, replace=False)))
        counts = tuple(int(x) for x in rng.integers(1, 6, size=nclass))
    if split == "train":
        h, w = (int(x) for x in rng.integers(22, 31, size=2))
    else:
        h, w = (int(x) for x in rng.integers(19, 31, size=2))
    bg, noise, colors = _sample_palette(rng, parameters, split, index, sides)
    density = float(rng.uniform(0.025, 0.115))
    return {"h": h, "w": w, "bg": bg, "noise": noise,
            "classes": [(s, color, count) for s, color, count in zip(sides, colors, counts)],
            "noise_count": max(8, round(h * w * density)),
            "fragments": int(rng.choice((0, 1, 2), p=(0.5, 0.4, 0.1))),
            "edge": bool(rng.random() < 0.48)}


def _make_pair(rng, parameters, split, index):
    for _ in range(100):
        spec = _sample_spec(rng, parameters, split, index)
        classes = spec["classes"]
        nmax = max(n for _, _, n in classes)
        width = sum(max(s for s, _, n in classes if n > rank) + 1 for rank in range(nmax))
        area = sum(s * s * n for s, _, n in classes)
        if width > 30 or area > spec["h"] * spec["w"] * 0.32:
            continue
        try:
            grid, reserved = _place_squares(spec["h"], spec["w"], spec["bg"], classes, rng, spec["edge"])
            for _ in range(spec["fragments"]):
                if not _add_fragment(grid, reserved, classes, rng):
                    break
            _add_noise(grid, spec["bg"], spec["noise"], spec["noise_count"], rng)
            facts = _analyze_input(grid)
            if tuple(sorted(facts["classes"])) != tuple(s for s, _, _ in classes):
                continue
            if any(facts["classes"][s] != [color, n] for s, color, n in classes):
                continue
            if _digest(grid) in _SOURCE_INPUT_HASHES:
                continue
            output = _output(facts)
            if output.shape[1] <= 30:
                return {"input": grid, "output": output}
        except (RuntimeError, AssertionError):
            continue
    raise RuntimeError("could not generate pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _pair_counts(seed, rng, num_train, num_test)
    parameters = {"palette": [int(x) for x in rng.permutation(10)],
                  "palette_mode": str(rng.choice(("stable", "free", "reuse"), p=(0.24, 0.43, 0.33)))}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(pair_rng, parameters, split, index)
                key = _digest(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate-pair retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    nt, nq = (len(puzzle[s]) if isinstance(puzzle[s], list) else -1 for s in ("train", "test"))
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen = set()
    training = []
    testing = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid dimensions")
                _require(0 <= grid.min() and grid.max() <= 9, "grid colors")
            key = _digest(pair["input"])
            _require(key not in seen and key not in _SOURCE_INPUT_HASHES, "duplicate or source input")
            seen.add(key)
            facts = _analyze_input(pair["input"])
            _require(2 <= len(facts["classes"]) <= 4, "class count")
            _require(all(2 <= side <= 5 for side in facts["classes"]), "square size")
            _require(facts["noise_cells"] >= 8, "missing noise")
            _require(pair["input"].shape != pair["output"].shape or
                     not np.array_equal(pair["input"], pair["output"]), "unchanged pair")
            _require(np.array_equal(pair["output"], _output(facts)), "wrong rank-wise output")
            (training if split == "train" else testing).append(facts)
    _require(training[0]["classes"].keys() == {2, 3, 4}, "first training pair must show three nested classes")
    _require(all(training[0]["classes"][s][1] >= 2 for s in (2, 3, 4)), "first training pair must show repeated nests")
    _require(max(x["classes"][2][1] for x in training[:1]) > training[0]["classes"][3][1],
             "first training pair must show a size-class tail")
    _require(len({tuple(sorted((s, v[1]) for s, v in f["classes"].items())) for f in training}) >= 2,
             "training inventories must vary")
    _require(all(f["shape"][0] >= 19 and f["shape"][1] >= 19 for f in testing), "test canvas too small")


def _same_puzzle(a, b):
    return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output")) and \
           all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 3)):
        validate_generated_puzzle(generate_puzzle(2000 + (nt or 0) * 10 + (nq or 0), nt, nq))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(5, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    a = generate_puzzle(991, 5, 2)
    _require(_same_puzzle(a, generate_puzzle(991, 5, 2)), "nondeterministic")
    _require(not _same_puzzle(a, generate_puzzle(992, 5, 2)), "seed has no effect")
    train_counts, test_counts, shapes, inventories, layouts, roles = (set() for _ in range(6))
    class_counts = set()
    edge = fragments = clustered = high_noise = 0
    no_fragments = square_train = square_test = middle_leaders = 0
    count_one = contacts = boundary_noise = combined = 0
    train_shape_change = train_background_change = stable_palettes = role_shifts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_facts = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                facts = _analyze_input(pair["input"])
                puzzle_facts[split].append(facts)
                shapes.add((split, facts["shape"]))
                profile = tuple((s, v[1]) for s, v in sorted(facts["classes"].items()))
                inventories.add((split, profile))
                class_counts.add(len(profile))
                layouts.add(tuple(sorted((s, box) for s, _, box, _ in facts["boxes"])))
                roles.add((facts["background"], facts["noise"], tuple(v[0] for _, v in sorted(facts["classes"].items()))))
                edge += facts["edge_squares"] > 0
                fragments += facts["fragments"] > 0
                no_fragments += facts["fragments"] == 0
                high_noise += facts["noise_cells"] >= 50
                noise_components = [x for x in _components(pair["input"], facts["background"]) if x[0] == facts["noise"]]
                has_cluster = any(x[-1] > 1 for x in noise_components)
                clustered += has_cluster
                a = pair["input"]
                boundary_noise += bool(np.any(a[0] == facts["noise"]) or np.any(a[-1] == facts["noise"]) or
                                       np.any(a[:, 0] == facts["noise"]) or np.any(a[:, -1] == facts["noise"]))
                adjacent = False
                for _, _, (r0, c0, r1, c1), _ in facts["boxes"]:
                    adjacent |= (r0 > 0 and np.any(a[r0 - 1, c0:c1 + 1] == facts["noise"]))
                    adjacent |= (r1 + 1 < a.shape[0] and np.any(a[r1 + 1, c0:c1 + 1] == facts["noise"]))
                    adjacent |= (c0 > 0 and np.any(a[r0:r1 + 1, c0 - 1] == facts["noise"]))
                    adjacent |= (c1 + 1 < a.shape[1] and np.any(a[r0:r1 + 1, c1 + 1] == facts["noise"]))
                contacts += adjacent
                combined += bool(facts["edge_squares"] and facts["fragments"] and has_cluster and adjacent)
                if a.shape[0] == a.shape[1]:
                    if split == "train":
                        square_train += 1
                    else:
                        square_test += 1
                middle_leaders += len(profile) == 3 and profile[1][1] > max(profile[0][1], profile[2][1])
                count_one += any(n == 1 for _, n in profile)
        train_facts = puzzle_facts["train"]
        test_facts = puzzle_facts["test"]
        train_shape_change += len({f["shape"] for f in train_facts}) > 1
        train_background_change += len({f["background"] for f in train_facts}) > 1
        all_facts = train_facts + test_facts
        color_by_side = {}
        consistent = True
        for f in all_facts:
            for side, (color, _) in f["classes"].items():
                if side in color_by_side and color_by_side[side] != color:
                    consistent = False
                color_by_side[side] = color
        stable_palettes += (len({f["background"] for f in all_facts}) == 1 and
                            len({f["noise"] for f in all_facts}) == 1 and consistent)
        train_payloads = {color for f in train_facts for color, _ in f["classes"].values()}
        role_shifts += any(f["background"] in train_payloads and
                           f["noise"] in {t["background"] for t in train_facts} and
                           any(color in {t["noise"] for t in train_facts}
                               for color, _ in f["classes"].values()) for f in test_facts)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "pair count diversity")
    _require(len(shapes) >= 25 and len(inventories) >= 50 and len(layouts) >= 100, "structural diversity")
    _require(len(roles) >= 30 and class_counts == {2, 3, 4}, "role or class diversity")
    _require(edge >= 15 and fragments >= 15 and clustered >= 15 and high_noise >= 15,
             "review feature coverage")
    _require(no_fragments >= 15 and square_train >= 3 and square_test >= 3,
             "optional fragments or square-canvas coverage")
    _require(middle_leaders >= 5 and count_one >= 15, "rank-tail coverage")
    _require(contacts >= 15 and boundary_noise >= 15 and combined >= 10,
             "noisy-neighborhood combinations")
    _require(train_shape_change >= 35 and train_background_change >= 5,
             "within-puzzle shape or palette variation")
    _require(stable_palettes >= 3 and role_shifts >= 3,
             "stable and reassigned palettes must both occur")
