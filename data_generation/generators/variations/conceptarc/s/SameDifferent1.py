"""Solved ARC puzzles: retain every translated copy of a repeated object shape."""

import hashlib
import numpy as np


SOURCE_INPUTS = frozenset({
    "a96c61ec979b97119f8e48529d645c5d255d4447e1e54d75785e28eab67df738",
    "663e0cf6226fad75ceadac7f406d1e171bb97dea3ee9880666ab47369e496d46",
    "20320eb24e1c38513a2af8443b50d5e79d6fa703746076f7a4768f2cb448be05",
    "4186545b7ffcd3f75ba980cd7e9fb2d04ae6d48cc75c9f98183998eead72431b",
    "60957226db7098feea3cea20db8dca6d63ea81d2828b3db0024943e035a499ce",
    "75c778f32a451d76326aa546f04fbfb6ee361de5fddcadb17db82f242c2effc4",
})
FAMILIES = ("solid", "frame", "c", "bar", "hole", "stroke")


def signature(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


def normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    cells = {(r - top, c - left) for r, c in cells}
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    return tuple("".join("1" if (r, c) in cells else "0" for c in range(width))
                 for r in range(height))


def cells_of(pattern):
    return {(r, c) for r, row in enumerate(pattern)
            for c, value in enumerate(row) if value == "1"}


def connected(cells):
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == cells


def transform(pattern, turns=0, mirror=False):
    cells = cells_of(pattern)
    if mirror:
        cells = {(r, -c) for r, c in cells}
    for _ in range(turns):
        cells = {(c, -r) for r, c in cells}
    return normalize(cells)


def make_motif(rng, family):
    if family == "solid":
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif family == "frame":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 7))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    elif family == "c":
        h, w = int(rng.integers(3, 8)), int(rng.integers(2, 7))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c == 0}
    elif family == "bar":
        length, thick = int(rng.integers(3, 9)), int(rng.integers(1, 3))
        cells = {(r, c) for r in range(length) for c in range(thick)}
    elif family == "hole":
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        cells = {(r, c) for r in range(h) for c in range(w)}
        interior = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
        for idx in rng.choice(len(interior), size=int(rng.integers(1, min(4, len(interior))+1)), replace=False):
            cells.remove(interior[int(idx)])
        if not connected(cells):
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h-1) or c in (0, w-1)}
    else:
        h, w = int(rng.integers(2, 6)), int(rng.integers(3, 7))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        goal = int(rng.integers(5, min(h*w, 13)+1))
        while len(cells) < goal:
            frontier = {(nr, nc) for r, c in cells
                        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells}
            if not frontier:
                break
            choices = sorted(frontier)
            cells.add(choices[int(rng.integers(len(choices)))])
    pattern = normalize(cells)
    return transform(pattern, int(rng.integers(4)), bool(rng.integers(2)))


def one_cell_decoy(pattern, rng):
    h, w = len(pattern), len(pattern[0])
    original = cells_of(pattern)
    options = []
    for r in range(h):
        for c in range(w):
            changed = original ^ {(r, c)}
            if changed and connected(changed) and normalize(changed) != pattern:
                candidate = normalize(changed)
                if (len(candidate), len(candidate[0])) == (h, w):
                    options.append(candidate)
    return options[int(rng.integers(len(options)))] if options else None


def lookalike(pattern, rng, mode):
    if mode == "one_cell":
        return one_cell_decoy(pattern, rng)
    candidates = sorted({transform(pattern, turns, mirrored)
                         for turns in range(4) for mirrored in (False, True)} - {pattern})
    return candidates[int(rng.integers(len(candidates)))] if candidates else None


def objects(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    result = []
    for row in range(h):
        for col in range(w):
            color = int(grid[row, col])
            if color == 0 or visited[row, col]:
                continue
            visited[row, col] = True
            todo = [(row, col)]
            cells = set()
            while todo:
                r, c = todo.pop()
                cells.add((r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= nr < h and 0 <= nc < w and not visited[nr, nc]
                            and int(grid[nr, nc]) == color):
                        visited[nr, nc] = True
                        todo.append((nr, nc))
            result.append((color, cells, normalize(cells)))
    return result


def solve(grid):
    found = objects(grid)
    counts = {}
    for _, _, shape in found:
        counts[shape] = counts.get(shape, 0) + 1
    answer = np.zeros_like(grid)
    for color, cells, shape in found:
        if counts[shape] > 1:
            for r, c in cells:
                answer[r, c] = color
    return answer


def pair_counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError("Illegal pair count")
    if num_train is None:
        if num_test is None:
            num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def colors_for_pair(rng, copies, n_distractors, role):
    palette_size = int(rng.integers(2, 7))
    if role == "cross":
        palette_size = max(4, palette_size)
    palette = [int(x) for x in rng.choice(np.arange(1, 10), size=palette_size, replace=False)]
    if role == "same" or (role == "free" and rng.random() < 0.3):
        keep = [palette[0]] * copies
    else:
        keep = [palette[0], palette[1]]
        keep.extend(int(rng.choice(palette)) for _ in range(copies-2))
        rng.shuffle(keep)
    erased = []
    for i in range(n_distractors):
        if i == 0 and (role == "cross" or rng.random() < 0.55):
            erased.append(int(rng.choice(keep)))
        else:
            erased.append(int(rng.choice(palette)))
    return keep, erased


def place_shelf(rng, items, dense):
    order = list(map(int, rng.permutation(len(items))))
    gap = 1 if dense else int(rng.integers(2, 4))
    choices = list(map(int, rng.permutation(np.arange(2, min(4, len(items))+1))))
    for columns in choices:
        rows = [order[i:i+columns] for i in range(0, len(order), columns)]
        widths = [sum(len(items[i][0][0]) for i in row) + gap*(len(row)-1) for row in rows]
        heights = [max(len(items[i][0]) for i in row) for row in rows]
        top, left = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        bottom, right = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        h, w = top + sum(heights) + gap*(len(rows)-1) + bottom, left + max(widths) + right
        h, w = max(8, h), max(8, w)
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=np.int8)
        output = np.zeros_like(grid)
        y = top
        for row, row_height in zip(rows, heights):
            x = left
            for i in row:
                pattern, color, retain = items[i]
                for pr, pc in cells_of(pattern):
                    grid[y+pr, x+pc] = color
                    if retain:
                        output[y+pr, x+pc] = color
                x += len(pattern[0]) + gap
            y += row_height + gap
        return grid, output
    raise RuntimeError("Objects do not fit ARC bounds")


def place_scattered(rng, items):
    total = sum(len(p)*len(p[0]) for p, _, _ in items)
    side = max(12, min(25, int(np.sqrt(total*3.6))+3))
    for extra in (0, 3, 6, 9):
        h = min(30, side + extra + int(rng.integers(-2, 3)))
        w = min(30, side + extra + int(rng.integers(-2, 3)))
        h, w = max(h, 8), max(w, 8)
        grid, output = np.zeros((h, w), np.int8), np.zeros((h, w), np.int8)
        boxes = []
        success = True
        for i in rng.permutation(len(items)):
            pattern, color, retain = items[int(i)]
            ph, pw = len(pattern), len(pattern[0])
            if ph > h or pw > w:
                success = False
                break
            for _ in range(300):
                r = int(rng.integers(h-ph+1))
                c = int(rng.integers(w-pw+1))
                if any(r < br+bh+1 and br < r+ph+1 and c < bc+bw+1 and bc < c+pw+1
                       for br, bc, bh, bw in boxes):
                    continue
                boxes.append((r, c, ph, pw))
                for pr, pc in cells_of(pattern):
                    grid[r+pr, c+pc] = color
                    if retain:
                        output[r+pr, c+pc] = color
                break
            else:
                success = False
                break
        if success:
            return grid, output
    return place_shelf(rng, items, False)


def generate_pair(rng, target, role, regime):
    copies = int(rng.choice((2, 3, 4, 5), p=(.38, .34, .20, .08)))
    if role in ("cross", "same"):
        distractor_count = int(rng.integers(1, 4))
    else:
        distractor_count = int(rng.choice((0, 1, 2, 3), p=(.18, .38, .30, .14)))
    near_mode = ("one_cell" if role == "cross" else
                 str(rng.choice(("one_cell", "transform", "random"), p=(.43, .25, .32))))
    decoys = []
    for i in range(distractor_count):
        proposal = lookalike(target, rng, near_mode) if i == 0 and near_mode != "random" else None
        if proposal is None or proposal in decoys or proposal == target:
            for _ in range(80):
                proposal = make_motif(rng, str(rng.choice(FAMILIES)))
                if proposal != target and proposal not in decoys:
                    break
            else:
                raise RuntimeError("Cannot construct distinct singleton")
        decoys.append(proposal)
    keep_colors, erase_colors = colors_for_pair(rng, copies, distractor_count, role)
    items = [(target, c, True) for c in keep_colors]
    items.extend((shape, c, False) for shape, c in zip(decoys, erase_colors))
    dense = bool(rng.random() < (.72 if regime == "source_train" else .63))
    if dense:
        grid, answer = place_shelf(rng, items, True)
    else:
        grid, answer = place_scattered(rng, items)
    if bool(rng.integers(2)):
        grid, answer = grid.T.copy(), answer.T.copy()
    if bool(rng.integers(2)):
        grid, answer = grid[:, ::-1].copy(), answer[:, ::-1].copy()
    if bool(rng.integers(2)):
        grid, answer = grid[::-1].copy(), answer[::-1].copy()
    return grid, answer


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = pair_counts(rng, num_train, num_test)
    focus = str(rng.choice(FAMILIES))
    roles = ["cross", "same"] + ["free"]*(n_train-2)
    rng.shuffle(roles)
    pair_seeds = rng.integers(0, 2**63-1, size=n_train+n_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_targets = []
    training_families = []
    for section, count, offset in (("train", n_train, 0), ("test", n_test, n_train)):
        for index in range(count):
            prng = np.random.default_rng(int(pair_seeds[offset+index]))
            for attempt in range(30):
                if section == "train":
                    family = focus if prng.random() < .35 else str(prng.choice(FAMILIES))
                    target = make_motif(prng, family)
                    if index == n_train-1 and all(len(p) == len(p[0]) for p in training_targets):
                        family = str(prng.choice(("c", "bar", "stroke")))
                        target = make_motif(prng, family)
                        if len(target) == len(target[0]):
                            continue
                    if target in training_targets:
                        continue
                    role = roles[index]
                else:
                    mode = str(prng.choice(("reuse", "family", "novel"), p=(.24, .35, .41)))
                    if mode == "reuse":
                        target = training_targets[int(prng.integers(len(training_targets)))]
                    else:
                        if mode == "family":
                            family = str(prng.choice(training_families))
                        else:
                            available = [f for f in FAMILIES if f not in training_families]
                            family = str(prng.choice(available if available else FAMILIES))
                        target = make_motif(prng, family)
                    role = "free"
                grid, output = generate_pair(prng, target, role, "source_train" if section == "train" else "target_test")
                fingerprint = (grid.shape, grid.tobytes())
                if fingerprint not in seen and signature(grid) not in SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("Cannot construct distinct pair")
            seen.add(fingerprint)
            puzzle[section].append({"input": grid, "output": output})
            if section == "train":
                training_targets.append(target)
                training_families.append(family)
    validate_generated_puzzle(puzzle)
    return puzzle


def inspect_pair(pair):
    grid, answer = pair["input"], pair["output"]
    found = objects(grid)
    groups = {}
    for color, cells, shape in found:
        groups.setdefault(shape, []).append((color, cells))
    repeated = {shape: members for shape, members in groups.items() if len(members) > 1}
    assert len(repeated) == 1
    shape, copies = next(iter(repeated.items()))
    assert 2 <= len(copies) <= 5
    assert 2 <= len(found) <= 8
    assert len(groups) - 1 <= 3
    expected = np.zeros_like(grid)
    for color, cells in copies:
        for r, c in cells:
            expected[r, c] = color
    assert np.array_equal(answer, expected)
    return shape, copies, [member for s, members in groups.items() if s != shape for member in members]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[x], list) for x in ("train", "test"))
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen = set()
    cross_evidence = False
    same_evidence = False
    train_shapes = []
    changed_train = 0
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, answer = pair["input"], pair["output"]
            for image in (grid, answer):
                assert isinstance(image, np.ndarray) and image.ndim == 2
                assert np.issubdtype(image.dtype, np.integer)
                assert 1 <= image.shape[0] <= 30 and 1 <= image.shape[1] <= 30
                assert np.all((image >= 0) & (image <= 9))
            assert grid.shape == answer.shape
            assert np.any(grid == 0)
            assert signature(grid) not in SOURCE_INPUTS
            fingerprint = (grid.shape, grid.tobytes())
            assert fingerprint not in seen
            seen.add(fingerprint)
            shape, copies, decoys = inspect_pair(pair)
            if section == "train":
                train_shapes.append(shape)
                changed_train += bool(decoys)
                colors = {c for c, _ in copies}
                cross_evidence |= len(colors) >= 2 and bool(colors & {c for c, _ in decoys})
                same_evidence |= len(colors) == 1
    assert changed_train >= 2
    assert cross_evidence and same_evidence
    assert len(set(train_shapes)) >= 2
    assert any(len(shape) != len(shape[0]) for shape in train_shapes)


def same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, shapes, palettes = set(), set(), set(), set(), set()
    layouts, copy_counts, decoy_counts = set(), set(), set()
    compact = same_copy = cross_copy = unchanged_train = unchanged_test = 0
    dense = sparse = near_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                shape, copies, decoys = inspect_pair(pair)
                sizes.add(grid.shape)
                shapes.add(shape)
                palettes.add(frozenset(int(c) for c in np.unique(grid) if c))
                layouts.add(tuple(sorted((min(r for r, _ in cells), min(c for _, c in cells))
                                         for _, cells, _ in objects(grid))))
                copy_counts.add(len(copies))
                decoy_counts.add(len(decoys))
                compact += int(max(grid.shape) <= 12)
                same_copy += int(len({c for c, _ in copies}) == 1)
                cross_copy += int(len({c for c, _ in copies}) > 1)
                density = np.count_nonzero(grid) / grid.size
                dense += int(density >= .25)
                sparse += int(density <= .15)
                if section == "train":
                    unchanged_train += int(not decoys)
                else:
                    unchanged_test += int(not decoys)
                    near_test += int(any(len(s) == len(shape) and len(s[0]) == len(shape[0])
                                         and abs(sum(row.count("1") for row in s) -
                                                 sum(row.count("1") for row in shape)) == 1
                                         for _, _, s in objects(grid) if s != shape))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 25 and len(shapes) >= 30 and len(palettes) >= 25
    assert len(layouts) >= 80
    assert copy_counts >= {2, 3, 4} and decoy_counts >= {0, 1, 2, 3}
    assert compact >= 5 and same_copy >= 15 and cross_copy >= 30
    assert unchanged_train >= 1 and unchanged_test >= 1 and near_test >= 5
    assert dense >= 10 and sparse >= 10
    for n_train, n_test in ((2, 1), (10, 1), (10, 4),
                            (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(923, n_train, n_test)
        assert same_puzzle(p, generate_puzzle(923, n_train, n_test))
        if n_train is not None:
            assert len(p["train"]) == n_train
        if n_test is not None:
            assert len(p["test"]) == n_test
    for n_train, n_test in ((1, None), (11, None), (None, 0), (None, 5),
                            (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
