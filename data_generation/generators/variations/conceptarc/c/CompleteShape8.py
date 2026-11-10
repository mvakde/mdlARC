"""Translated stencil completion puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    "0000000555/0004000555/0000000555/0004440000/0000000000/0000000000/0000660000/0000000000/0000006000/0000000000",
    "0000000505/0000000050/0700000555/0070000000/0000006000/0000000000/0000600000/0000000000/0000000000/0000000000",
    "0000000505/0000000500/0700000555/0000000000/0070000000/0000000000/0000000000/0000808000/0000800000/0000000000",
    "0000000050/0000000505/0030000050/0000000000/0030000000/0000000808/0000000000/0909000000/0000000000/0000000000",
    "0000000500/0000005050/0000050005/0000005050/0002000500/0000000000/0000000000/0000000500/0002000000/0000000000/0000000500",
}


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_key(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _bounds(cells):
    rows = [p[0] for p in cells]
    columns = [p[1] for p in cells]
    return min(rows), max(rows), min(columns), max(columns)


def _connected(cells):
    cells = set(cells)
    seen = {next(iter(cells))}
    queue = list(seen)
    for r, c in queue:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p in cells and p not in seen:
                    seen.add(p)
                    queue.append(p)
    return len(seen) == len(cells)


def _count_request(seed, num_train, num_test, rng):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if num_train is None and num_test is None:
        ntrain = 2 + int(seed) % 9
        ntest = 1 + (int(seed) // 9) % min(4, ntrain - 1)
    elif num_train is None:
        ntest = int(num_test)
        ntrain = int(rng.integers(max(2, ntest + 1), 11))
    elif num_test is None:
        ntrain = int(num_train)
        ntest = int(rng.integers(1, min(4, ntrain - 1) + 1))
    else:
        ntrain, ntest = int(num_train), int(num_test)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    return ntrain, ntest


def _shape(rng, regime):
    if regime == "source_train":
        sizes = ((3, 3), (2, 3), (3, 2), (3, 4), (4, 3))
        weights = (5, 1, 1, 1, 1)
    else:
        sizes = ((3, 3), (3, 4), (4, 3), (4, 4), (5, 5), (3, 5), (5, 3))
        weights = (2, 2, 2, 1, 3, 1, 1)
    index = int(rng.choice(len(sizes), p=np.array(weights) / sum(weights)))
    h, w = sizes[index]
    area = h * w
    low = 8 if regime == "target_test" and area >= 20 else 4
    high = min(area, 17)
    for _ in range(80):
        count = int(rng.integers(low, high + 1))
        selected = rng.choice(area, size=count, replace=False)
        cells = tuple(sorted((int(i) // w, int(i) % w) for i in selected))
        if _bounds(cells) == (0, h - 1, 0, w - 1):
            return cells, h, w
    return tuple((r, c) for r in range(h) for c in range(w)), h, w


def _place(rng, shape, h, w, canvas_h, canvas_w, ncopy):
    origins, footprints = [], []
    for _ in range(ncopy + 1):
        options = []
        for r in range(canvas_h - h + 1):
            for c in range(canvas_w - w + 1):
                cells = {(r + dr, c + dc) for dr, dc in shape}
                if any(cells.intersection(other) for other in footprints):
                    continue
                distance = min((max(abs(a - b), abs(x - y))
                                for other in footprints for a, x in cells
                                for b, y in other), default=99)
                if distance >= 2 or (distance == 1 and rng.random() < .16):
                    options.append(((r, c), cells))
        if not options:
            return None
        origin, cells = options[int(rng.integers(len(options)))]
        origins.append(origin)
        footprints.append(cells)
    return origins, footprints


def _clue_indices(rng, shape, count):
    if count == 2 and rng.random() < .75:
        unique_pairs = []
        for i in range(len(shape)):
            for j in range(i + 1, len(shape)):
                delta = (shape[j][0] - shape[i][0],
                         shape[j][1] - shape[i][1])
                if sum((b[0] - a[0], b[1] - a[1]) == delta
                       for a in shape for b in shape) == 1:
                    unique_pairs.append((i, j))
        if unique_pairs:
            return unique_pairs[int(rng.integers(len(unique_pairs)))]
    return rng.choice(len(shape), size=count, replace=False)


def _pair(rng, regime, background, same_color, require_multi=False):
    for _ in range(200):
        shape, h, w = _shape(rng, regime)
        ncopy = int(rng.choice((1, 2, 3), p=(.15, .65, .20) if regime == "source_train"
                               else (.20, .45, .35)))
        if require_multi:
            ncopy = max(ncopy, 2)
        side = 2 if regime == "source_train" else 3
        canvas_h = int(rng.integers(max(8, h + side * ncopy + 1), 19))
        canvas_w = int(rng.integers(max(8, w + side * ncopy + 1), 19))
        if rng.random() < .5:
            canvas_h, canvas_w = canvas_w, canvas_h
        placement = _place(rng, shape, h, w, canvas_h, canvas_w, ncopy)
        if placement is None:
            continue
        origins, footprints = placement
        palette = [c for c in range(10) if c != background]
        reference_color = int(rng.choice(palette))
        others = [c for c in palette if c != reference_color]
        rng.shuffle(others)
        colors = list(others[:ncopy])
        if same_color:
            colors[int(rng.integers(ncopy))] = reference_color
        elif rng.random() < .14 and ncopy >= 2:
            colors[-1] = colors[0]
        inp = np.full((canvas_h, canvas_w), background, dtype=np.int8)
        out = inp.copy()
        for r, c in footprints[0]:
            inp[r, c] = reference_color
        for i in range(ncopy):
            count = len(shape)
            if regime == "target_test" and max(h, w) >= 5 and rng.random() < .38:
                clues = 2
            elif regime == "target_test" and rng.random() < .60:
                clues = int(rng.integers(2, max(3, count // 2 + 1)))
            else:
                clues = int(rng.integers(2, count - 1))
            selected = _clue_indices(rng, shape, clues)
            origin = origins[i + 1]
            for j in selected:
                dr, dc = shape[int(j)]
                inp[origin[0] + dr, origin[1] + dc] = colors[i]
            for r, c in footprints[i + 1]:
                out[r, c] = colors[i]
        if np.array_equal(inp, out) or _source_key(inp) in _SOURCE_INPUTS:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Could not construct pair")


def _reference_candidates(inp, background):
    height, width = inp.shape
    found = set()
    for h in range(2, min(5, height) + 1):
        for w in range(2, min(5, width) + 1):
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    box = inp[r:r + h, c:c + w]
                    colors = set(int(x) for x in box.flat) - {background}
                    if len(colors) != 1:
                        continue
                    color = next(iter(colors))
                    shape = tuple(sorted((int(a), int(b)) for a, b in np.argwhere(box == color)))
                    if not 4 <= len(shape) <= 17 or _bounds(shape) != (0, h - 1, 0, w - 1):
                        continue
                    cells = frozenset((r + dr, c + dc) for dr, dc in shape)
                    if cells not in found:
                        found.add(cells)
                        yield shape, (r, c), color, cells


def _solutions(inp, background, limit=2, with_interpretations=False):
    """Enumerate reference/partition hypotheses from the input alone."""
    nonback = {(int(r), int(c)) for r, c in np.argwhere(inp != background)}
    height, width = inp.shape
    answers = set()
    interpretations = set()
    for shape, _, _, ref_cells in _reference_candidates(inp, background):
        remaining = nonback - ref_cells
        if len(remaining) < 2:
            continue
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        options = {}
        for clue in remaining:
            color = int(inp[clue])
            for dr, dc in shape:
                row, col = clue[0] - dr, clue[1] - dc
                if not (0 <= row <= height - sh and 0 <= col <= width - sw):
                    continue
                key = (row, col, color)
                if key in options:
                    continue
                full = frozenset((row + a, col + b) for a, b in shape)
                if full.intersection(ref_cells):
                    continue
                captured = frozenset(p for p in full if p in remaining and int(inp[p]) == color)
                if not 2 <= len(captured) <= len(shape) - 2:
                    continue
                if any(p in nonback and p not in captured for p in full):
                    continue
                options[key] = (full, captured, color)
        by_clue = {clue: [] for clue in remaining}
        for option in options.values():
            for clue in option[1]:
                by_clue[clue].append(option)
        if any(not choices for choices in by_clue.values()):
            continue

        def search(left, occupied, chosen):
            if len(answers) >= limit or len(interpretations) >= limit:
                return
            if not left:
                if not 1 <= len(chosen) <= 3:
                    return
                result = np.full(inp.shape, background, dtype=np.int8)
                for full, _, color in chosen:
                    for rr, cc in full:
                        result[rr, cc] = color
                answers.add(_key(result))
                interpretations.add((ref_cells, tuple(sorted(
                    (tuple(sorted(full)), color) for full, _, color in chosen))))
                return
            if len(chosen) == 3:
                return
            clue = min(left, key=lambda p: len(by_clue[p]))
            for option in by_clue[clue]:
                full, captured, _ = option
                if captured <= left and not full.intersection(occupied):
                    search(left - captured, occupied | full, chosen + [option])

        search(remaining, ref_cells, [])
        if len(answers) >= limit or len(interpretations) >= limit:
            break
    return (answers, interpretations) if with_interpretations else answers


def _background(inp):
    values, counts = np.unique(inp, return_counts=True)
    return int(values[np.argmax(counts)])


def _pair_valid(pair):
    inp, out = pair["input"], pair["output"]
    bg = _background(inp)
    assert bg == _background(out)
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    answers, interpretations = _solutions(inp, bg, with_interpretations=True)
    assert answers == {_key(out)} and len(interpretations) == 1


def _profile(pair):
    inp, out = pair["input"], pair["output"]
    bg = _background(inp)
    reference = {tuple(int(v) for v in cell) for cell in np.argwhere(
        (inp != bg) & (out == bg))}
    assert reference
    r0, r1, c0, c1 = _bounds(reference)
    shape = tuple(sorted((r - r0, c - c0) for r, c in reference))
    copies, remainder = divmod(int(np.count_nonzero(out != bg)), len(shape))
    assert remainder == 0 and 1 <= copies <= 3
    answers, interpretations = _solutions(inp, bg, with_interpretations=True)
    assert answers == {_key(out)} and len(interpretations) == 1
    ref_hypothesis, placed = next(iter(interpretations))
    assert ref_hypothesis == reference and len(placed) == copies
    clue_counts, copy_colors, disconnected_fragments = [], [], []
    boxes = [(r0, r1, c0, c1)]
    for full, color in placed:
        clues = {cell for cell in full if int(inp[cell]) == color}
        clue_counts.append(len(clues))
        copy_colors.append(color)
        disconnected_fragments.append(not _connected(clues))
        a, b, x, y = _bounds(full)
        boxes.append((a, b, x, y))
    overlap = any(max(a[0], b[0]) <= min(a[1], b[1]) and
                  max(a[2], b[2]) <= min(a[3], b[3])
                  for i, a in enumerate(boxes) for b in boxes[i + 1:])
    return {
        "canvas": inp.shape,
        "box": (r1 - r0 + 1, c1 - c0 + 1),
        "occupancy": len(shape),
        "copies": copies,
        "background": bg,
        "reference_color": int(inp[next(iter(reference))]),
        "reference_origin": (r0, c0),
        "disconnected_reference": not _connected(reference),
        "same_color": int(inp[next(iter(reference))]) in set(int(v) for v in out.flat),
        "clues": tuple(clue_counts),
        "copy_colors": tuple(copy_colors),
        "disconnected_fragment": any(disconnected_fragments),
        "overlapping_boxes": overlap,
        "interior_reference": (r0 > 0 and c0 > 0 and
                               r1 < inp.shape[0] - 1 and c1 < inp.shape[1] - 1),
    }


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _count_request(seed, num_train, num_test, rng)
    background = int(rng.choice(10, p=(.65, .04, .04, .04, .04,
                                              .04, .04, .04, .04, .03)))
    vary_background = bool(rng.random() < .50)
    train_same = bool(rng.integers(0, 2))
    forced_train_index = int(rng.integers(ntrain)) if train_same else -1
    forced_test_index = int(rng.integers(ntest)) if not train_same else -1
    puzzle = {"train": [], "test": []}
    seen = set()
    for group, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if vary_background and ((group == "train" and index == ntrain - 1)
                                    or (group == "test" and rng.random() < .55)
                                    or (group == "train" and index > 0
                                        and rng.random() < .30)):
                pair_background = int(rng.choice(
                    [color for color in range(10) if color != background]))
            else:
                pair_background = background
            for _ in range(800):
                same = (group == "train" and
                        (index == forced_train_index or rng.random() < .12)
                        or group == "test" and
                        (index == forced_test_index or rng.random() < .28))
                pair = _pair(rng, "source_train" if group == "train" else "target_test",
                             pair_background, same,
                             require_multi=(group == "train" and index == 0))
                key = _key(pair["input"])
                if key in seen:
                    continue
                try:
                    _pair_valid(pair)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[group].append(pair)
                break
            else:
                raise RuntimeError("Could not construct unambiguous puzzle")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_backgrounds, test_backgrounds = set(), set()
    train_copy_counts = []
    for group in ("train", "test"):
        for pair in puzzle[group]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            ikey, okey = _key(pair["input"]), _key(pair["output"])
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            assert _source_key(pair["input"]) not in _SOURCE_INPUTS
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            if group == "train":
                train_backgrounds.add(_background(pair["input"]))
            else:
                test_backgrounds.add(_background(pair["input"]))
            profile = _profile(pair)
            if group == "train":
                train_copy_counts.append(profile["copies"])
            assert 4 <= profile["occupancy"] <= 17
            if group == "train":
                assert profile["box"] in ((3, 3), (2, 3), (3, 2),
                                           (3, 4), (4, 3))
            else:
                assert profile["box"] in ((3, 3), (3, 4), (4, 3), (4, 4),
                                           (5, 5), (3, 5), (5, 3))
    assert len(train_backgrounds) > 1 or test_backgrounds <= train_backgrounds
    assert max(train_copy_counts) >= 2


def _same_puzzles(a, b):
    return all(_key(p[x]) == _key(q[x]) for group in ("train", "test")
               for p, q in zip(a[group], b[group]) for x in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert _same_puzzles(generate_puzzle(17), generate_puzzle(17))
    profiles = [(_profile(pair), group, seed)
                for seed, puzzle in enumerate(puzzles)
                for group in ("train", "test") for pair in puzzle[group]]
    assert len({p["canvas"] for p, _, _ in profiles}) >= 30
    assert len({p["box"] for p, _, _ in profiles}) >= 6
    assert len({p["occupancy"] for p, _, _ in profiles}) >= 10
    assert {p["copies"] for p, _, _ in profiles} == {1, 2, 3}
    assert len({p["background"] for p, _, _ in profiles}) >= 3
    assert sum(len({p["background"] for p, group, seed2 in profiles
                    if group == "train" and seed2 == seed}) > 1
               for seed in range(42)) >= 10
    assert sum(p["disconnected_reference"] for p, _, _ in profiles) >= 10
    assert sum(p["same_color"] for p, group, _ in profiles if group == "test") >= 5
    assert sum(p["same_color"] for p, group, _ in profiles if group == "train") >= 5
    assert sum(p["disconnected_fragment"] for p, _, _ in profiles) >= 20
    assert sum(p["overlapping_boxes"] for p, _, _ in profiles) >= 2
    assert sum(p["interior_reference"] for p, _, _ in profiles) >= 20
    assert sum(2 in p["clues"] for p, _, _ in profiles) >= 20
    assert sum(any(n >= p["occupancy"] - 3 for n in p["clues"])
               for p, _, _ in profiles) >= 20
    assert sum(len(set(p["copy_colors"])) < p["copies"]
               for p, _, _ in profiles) >= 2
    assert sum(max(p["box"]) >= 5 and 2 in p["clues"]
               for p, group, _ in profiles if group == "test") >= 3
    assert sum(max(p["box"]) >= 5 and p["copies"] == 1
               for p, group, _ in profiles if group == "test") >= 1
    assert sum(max(p["box"]) >= 5 and p["copies"] >= 2
               for p, group, _ in profiles if group == "test") >= 3
    unseen_test_colors = 0
    same_first_test = 0
    for seed in range(42):
        train = [p for p, group, s in profiles if s == seed and group == "train"]
        test = [p for p, group, s in profiles if s == seed and group == "test"]
        unseen_test_colors += bool(
            set(c for p in test for c in p["copy_colors"]) -
            set(c for p in train for c in p["copy_colors"]))
        same_first_test += (any(p["same_color"] for p in test)
                            and not any(p["same_color"] for p in train))
    assert unseen_test_colors >= 5 and same_first_test >= 3
    assert sum(len({p["reference_origin"] for p, _, seed2 in profiles
                    if seed2 == seed}) > 1 for seed in range(42)) >= 35
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, None, 1), (105, None, 4), (106, 2, None),
                         (107, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(500, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
