"""Generate seed-directed reflections of an irregular reference motif."""

from __future__ import annotations

import numpy as np


_SOURCE_INPUTS = {
    "000000000000/000010000000/001010000000/001111100000/"
    "000100000000/000000000000/000000000000/000000000000/"
    "000002000000/000000000000/000000000000/000000000000",
    "00000000000/00000000000/00000000000/00000000000/"
    "01100000000/00100000060/01111000000/00000000000/"
    "00000000000/00000000000/00000000000/00000000000",
    "0000000000000/0001100000000/0010000000020/0001100000000/"
    "0000100000000/0000000000000/0000000000000/0000000000000/"
    "0000000000000/0080000000000/0000000000000/0000000000000/"
    "0000000000000/0000000000000",
    "11000000000/01110000000/01000003000/01110000000/"
    "00000000000/00000000000/00000000000/00000000000/"
    "00000000000/00000000000/80000000000",
    "0000000000000000/0001000000030000/0011000000000000/"
    "1101000000000000/0010000000000000/0000000000000000/"
    "0000000000000000/0000000000000000/0000000000000000/"
    "0600000000000000/0000000000000000/0000000000000000/"
    "0000000000000000/0000000000000000",
}


def _grid_text(grid: np.ndarray) -> str:
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, train: int | None,
                    test: int | None) -> tuple[int, int]:
    train = _count(train, "num_train", 2, 10)
    test = _count(test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(mask: np.ndarray) -> int:
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            for point in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    frontier.append(point)
    return count


def _grow_connected(rng: np.random.Generator, height: int, width: int,
                    target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[int(rng.integers(height)), int(rng.integers(width))] = True
    while int(mask.sum()) < target:
        frontier = set()
        for row, col in np.argwhere(mask):
            for nr, nc in ((row - 1, col), (row + 1, col),
                           (row, col - 1), (row, col + 1)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        options = sorted(frontier)
        if not options:
            break
        row, col = options[int(rng.integers(len(options)))]
        mask[row, col] = True
    return mask


def _sample_motif(rng: np.random.Generator, regime: str
                  ) -> tuple[np.ndarray, str]:
    styles = ("branch", "line", "dense", "disconnected")
    for _ in range(300):
        style = styles[int(rng.integers(len(styles)))]
        if regime == "target_test" and rng.random() < 0.25:
            height, width = (int(rng.integers(5, 9)),
                             int(rng.integers(5, 9)))
        else:
            height, width = (int(rng.integers(3, 9)),
                             int(rng.integers(3, 9)))
        if style == "line":
            mask = np.zeros((height, width), dtype=bool)
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            mask[row, :] = True
            mask[:, col] = True
            candidates = []
            for rr, cc in np.argwhere(mask):
                for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                               (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        candidates.append((nr, nc))
            rng.shuffle(candidates)
            for nr, nc in candidates[:int(rng.integers(1, 4))]:
                mask[nr, nc] = True
        elif style == "disconnected":
            mask = rng.random((height, width)) < rng.uniform(0.22, 0.39)
        else:
            density = rng.uniform(0.35, 0.56) if style == "branch" else rng.uniform(0.55, 0.73)
            target = max(6, int(round(height * width * density)))
            mask = _grow_connected(rng, height, width, target)

        occupied = np.argwhere(mask)
        if len(occupied) < 5:
            continue
        low = occupied.min(axis=0)
        high = occupied.max(axis=0)
        mask = mask[low[0]:high[0] + 1, low[1]:high[1] + 1]
        mh, mw = mask.shape
        if mh < 3 or mw < 3 or int(mask.sum()) == mh * mw:
            continue
        if not np.any(mask.sum(axis=1) == 1) or not np.any(mask.sum(axis=0) == 1):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        parts = _components(mask)
        if style == "disconnected" and not 2 <= parts <= 6:
            continue
        if style != "disconnected" and parts != 1:
            continue
        return mask, style
    raise ValueError("could not sample a suitable motif")


def _choose_anchor(rng: np.random.Generator, mask: np.ndarray,
                   orientation: str) -> tuple[int, int]:
    occupied = [tuple(map(int, point)) for point in np.argwhere(mask)]
    if orientation == "H":
        candidates = [(r, c) for r, c in occupied if int(mask[r].sum()) == 1]
        interior = [(r, c) for r, c in candidates if 0 < r < mask.shape[0] - 1]
    else:
        candidates = [(r, c) for r, c in occupied if int(mask[:, c].sum()) == 1]
        interior = [(r, c) for r, c in candidates if 0 < c < mask.shape[1] - 1]
    if interior and rng.random() < 0.72:
        candidates = interior
    return candidates[int(rng.integers(len(candidates)))]


def _orientation_plan(rng: np.random.Generator, num_train: int,
                      num_test: int, consistent: bool,
                      expanded: bool) -> tuple[list[list[tuple[str, int]]],
                                               list[list[tuple[str, int]]]]:
    hside = int(rng.choice([-1, 1]))
    vside = int(rng.choice([-1, 1]))
    train = [[("H", hside)], [("H", hside if consistent else -hside),
                              ("V", vside)]]
    if num_train >= 3:
        train.append([("V", vside if consistent else -vside)])
    if expanded:
        doubled = "H" if rng.random() < 0.5 else "V"
        train.append([("H", hside), ("V", vside), (doubled, -hside if doubled == "H" else -vside)])
    while len(train) < num_train:
        choice = int(rng.integers(4 if expanded else 3))
        if choice == 0:
            orientations = ["H"]
        elif choice == 1:
            orientations = ["V"]
        elif choice == 2:
            orientations = ["H", "V"]
        else:
            orientations = ["H", "V", "H" if rng.random() < 0.5 else "V"]
        spec = []
        for orientation in orientations:
            allowed = [hside if orientation == "H" else vside]
            if not consistent:
                allowed.append(-allowed[0])
            used = {side for name, side in spec if name == orientation}
            remaining = [side for side in allowed if side not in used]
            if not remaining:
                break
            spec.append((orientation, remaining[int(rng.integers(len(remaining)))]))
        if len(spec) == len(orientations):
            train.append(spec)
    rng.shuffle(train)

    test = []
    for _ in range(num_test):
        if expanded and rng.random() < 0.45:
            doubled = "H" if rng.random() < 0.5 else "V"
            spec = [("H", hside), ("V", vside),
                    (doubled, -hside if doubled == "H" else -vside)]
        elif rng.random() < 0.8:
            spec = [("H", hside if consistent else int(rng.choice([-1, 1]))),
                    ("V", vside if consistent else int(rng.choice([-1, 1])))]
        else:
            axis = "H" if rng.random() < 0.5 else "V"
            spec = [(axis, (hside if axis == "H" else vside)
                     if consistent else int(rng.choice([-1, 1])))]
        test.append(spec)
    return train, test


def _make_pair(rng: np.random.Generator, regime: str, spec: list[tuple[str, int]],
               fixed_palette: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        mask, _style = _sample_motif(rng, regime)
        height, width = mask.shape
        occupied = [tuple(map(int, point)) for point in np.argwhere(mask)]
        dual = [(r, c) for r, c in occupied
                if int(mask[r].sum()) == 1 and int(mask[:, c].sum()) == 1]
        shared = dual[int(rng.integers(len(dual)))] if dual and rng.random() < 0.24 else None
        copies = []
        for orientation, side in spec:
            anchor = shared if shared is not None else _choose_anchor(rng, mask, orientation)
            gap = int(rng.integers(1, 8))
            if orientation == "H":
                constant = (2 * width + gap - 1) if side == 1 else -gap - 1
                reflected = {(r, constant - c) for r, c in occupied}
                seed = (anchor[0], constant - anchor[1])
            else:
                constant = (2 * height + gap - 1) if side == 1 else -gap - 1
                reflected = {(constant - r, c) for r, c in occupied}
                seed = (constant - anchor[0], anchor[1])
            copies.append((orientation, side, gap, seed, reflected))
        all_groups = [set(occupied)] + [part[4] for part in copies]
        if any(all_groups[i] & all_groups[j]
               for i in range(len(all_groups)) for j in range(i + 1, len(all_groups))):
            continue
        all_points = set().union(*all_groups)
        min_row = min(r for r, _ in all_points)
        max_row = max(r for r, _ in all_points)
        min_col = min(c for _, c in all_points)
        max_col = max(c for _, c in all_points)
        foot_h, foot_w = max_row - min_row + 1, max_col - min_col + 1
        if foot_h > 30 or foot_w > 30:
            continue
        # Padding is sampled independently of train/test, and zero reaches the edge.
        top = int(rng.integers(0, min(5, 31 - foot_h)))
        bottom = int(rng.integers(0, min(5, 31 - foot_h - top)))
        left = int(rng.integers(0, min(5, 31 - foot_w)))
        right = int(rng.integers(0, min(5, 31 - foot_w - left)))
        rows, cols = foot_h + top + bottom, foot_w + left + right
        if rows < 7 or cols < 7:
            continue
        row_shift, col_shift = top - min_row, left - min_col
        if fixed_palette is None:
            background, template = map(int, rng.choice(10, size=2, replace=False))
        else:
            background, template = fixed_palette
        accents = [int(c) for c in rng.permutation(10)
                   if int(c) not in (background, template)][:len(copies)]
        inp = np.full((rows, cols), background, dtype=np.int8)
        for row, col in occupied:
            inp[row + row_shift, col + col_shift] = template
        out = inp.copy()
        for color, (_orientation, _side, _gap, seed, reflected) in zip(accents, copies):
            sr, sc = seed
            inp[sr + row_shift, sc + col_shift] = color
            for row, col in reflected:
                out[row + row_shift, col + col_shift] = color
        # The seed is present in both grids, and every other painted cell is new.
        for color, (_orientation, _side, _gap, seed, _reflected) in zip(accents, copies):
            sr, sc = seed
            out[sr + row_shift, sc + col_shift] = color
        if np.count_nonzero(inp == background) <= int(mask.sum()):
            continue
        return inp, out
    raise ValueError("could not place a valid reflection construction")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Palette stability and side consistency are puzzle conventions.
    fixed_palette = tuple(map(int, rng.choice(10, size=2, replace=False))) if rng.random() < 0.55 else None
    consistent = rng.random() < 0.18
    expanded = num_train >= 4 and not consistent and rng.random() < 0.62
    train_specs, test_specs = _orientation_plan(rng, num_train, num_test,
                                                consistent, expanded)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for spec in specs:
            for _ in range(100):
                inp, out = _make_pair(rng, "source_train" if split == "train" else "target_test",
                                      spec, fixed_palette)
                signature = (inp.shape, inp.tobytes())
                if signature not in seen and _grid_text(inp) not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError("could not sample distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.shape == out.shape and inp.ndim == 2
    assert all(1 <= value <= 30 for value in inp.shape)
    for grid in (inp, out):
        assert np.issubdtype(grid.dtype, np.integer)
        assert bool(np.all((0 <= grid) & (grid <= 9)))
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    groups = [(int(color), int(count)) for color, count in zip(colors, counts)
              if int(color) != background]
    templates = [color for color, count in groups if count >= 5]
    assert len(templates) == 1
    template = templates[0]
    seed_colors = [color for color, count in groups if color != template]
    assert 1 <= len(seed_colors) <= 4
    assert all(count == 1 for color, count in groups if color != template)
    motif = [tuple(map(int, point)) for point in np.argwhere(inp == template)]
    assert len(motif) >= 5
    expected = inp.copy()
    copies = []
    occupied = set(motif)
    for color in seed_colors:
        seed = tuple(map(int, np.argwhere(inp == color)[0]))
        row_match = [point for point in motif if point[0] == seed[0]]
        col_match = [point for point in motif if point[1] == seed[1]]
        horizontal = len(row_match) == 1 and len(col_match) == 0
        vertical = len(col_match) == 1 and len(row_match) == 0
        assert horizontal ^ vertical
        if horizontal:
            anchor = row_match[0]
            constant = anchor[1] + seed[1]
            reflected = {(row, constant - col) for row, col in motif}
            orientation = "H"
            side = 1 if seed[1] > anchor[1] else -1
        else:
            anchor = col_match[0]
            constant = anchor[0] + seed[0]
            reflected = {(constant - row, col) for row, col in motif}
            orientation = "V"
            side = 1 if seed[0] > anchor[0] else -1
        assert seed in reflected and len(reflected) == len(motif)
        assert not reflected & occupied
        assert all(0 <= row < inp.shape[0] and 0 <= col < inp.shape[1]
                   for row, col in reflected)
        assert all(inp[row, col] == background or (row, col) == seed
                   for row, col in reflected)
        occupied.update(reflected)
        for row, col in reflected:
            expected[row, col] = color
        copies.append((orientation, side, anchor, seed, reflected))
    assert np.array_equal(out, expected)
    assert np.any(inp != out)
    assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
    return {"background": background, "template": template,
            "seed_colors": set(seed_colors), "motif": motif, "copies": copies,
            "components": _components(inp == template)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_orientations = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _inspect_pair(pair["input"], pair["output"])
            signature = (pair["input"].shape, pair["input"].tobytes())
            assert signature not in seen
            seen.add(signature)
            if split == "train":
                train_orientations.update(copy[0] for copy in info["copies"])
    assert train_orientations == {"H", "V"}


def self_test() -> None:
    styles = set()
    sizes = set()
    sides = set()
    interior_h = interior_v = dual = 0
    edge = near = far = mixed_gaps = 0
    connected = disconnected = multis = unseen_test_color = 0
    counts = set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_colors = set()
        pair_infos = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _inspect_pair(inp, out)
                pair_infos.append((split, info))
                sizes.add(inp.shape)
                signatures.add((inp.shape, inp.tobytes()))
                connected += info["components"] == 1
                disconnected += info["components"] > 1
                multis += len(info["copies"]) >= 3
                if split == "train":
                    train_colors.update(info["seed_colors"])
                else:
                    unseen_test_color += bool(info["seed_colors"] - train_colors)
                occupied = np.argwhere(out != info["background"])
                edge += bool(np.any(occupied[:, 0] == 0) or
                             np.any(occupied[:, 0] == out.shape[0] - 1) or
                             np.any(occupied[:, 1] == 0) or
                             np.any(occupied[:, 1] == out.shape[1] - 1))
                motif = info["motif"]
                minr, maxr = min(r for r, _ in motif), max(r for r, _ in motif)
                minc, maxc = min(c for _, c in motif), max(c for _, c in motif)
                gaps = []
                for orientation, side, anchor, seed_point, reflected in info["copies"]:
                    sides.add((orientation, side))
                    if orientation == "H":
                        interior_h += minr < anchor[0] < maxr
                        gap = (min(c for _, c in reflected) - maxc - 1) if side == 1 else (
                            minc - max(c for _, c in reflected) - 1)
                    else:
                        interior_v += minc < anchor[1] < maxc
                        gap = (min(r for r, _ in reflected) - maxr - 1) if side == 1 else (
                            minr - max(r for r, _ in reflected) - 1)
                    gaps.append(gap)
                    near += gap <= 2
                    far += gap >= 5
                mixed_gaps += bool(gaps and min(gaps) <= 2 and max(gaps) >= 5)
                dual += any(a[0] == b[0] and a[1] == b[1]
                            for i, (_, _, a, _, _) in enumerate(info["copies"])
                            for _, _, b, _, _ in info["copies"][i + 1:])
        assert {copy[0] for split, info in pair_infos if split == "train"
                for copy in info["copies"]} == {"H", "V"}
    assert len(counts) >= 10 and len(sizes) >= 50 and len(signatures) >= 150
    assert sides == {("H", -1), ("H", 1), ("V", -1), ("V", 1)}
    assert interior_h > 20 and interior_v > 20 and dual > 0
    assert edge > 20 and near > 20 and far > 20 and mixed_gaps > 5
    assert connected > 25 and disconnected > 15 and multis > 5
    assert unseen_test_color > 5
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 4), (2, None)):
        validate_generated_puzzle(generate_puzzle(1000 + (train or 0) * 10 + (test or 0),
                                                  train, test))
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
