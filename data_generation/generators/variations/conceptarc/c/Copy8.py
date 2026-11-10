"""Puzzles in which one translated copy of a figure is partly erased."""

import hashlib
import numpy as np


# An exact source input match would also be an exact source-pair copy.
_SOURCE_INPUT_DIGESTS = {
    "ffb18693e233d2fac9a0fb3675dae1273862ab3f46709ca8369fb613774320d9",
    "3724540279f1d913153e4d67471ba927d18d53b0a4e0f2351ffc0dfd7ec12d91",
    "548724eea7ecb74f6684a07b8d783a3532b49fddd04c88fca73659b9e840ef6e",
    "9f60fdfed316ea6337f9f3ce5af577c5040212feb3376fdd2983b024107ff36f",
    "fad13064d2dffb3024c4ca415567ac3032b4edabf45a315d8da5c4bf3d3e747e",
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, bool) or
            not isinstance(value, (int, np.integer)) or
            not low <= value <= high
        ):
            raise ValueError("invalid pair count")
    if num_test is None:
        top = 4 if num_train is None else min(4, num_train - 1)
        num_test = int(rng.integers(1, top + 1))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _motif(rng, family):
    """Use semantic colors 1 for the body and 2 for the accent."""
    w = int(rng.integers(5, 10))
    h = int(rng.integers(4, 8))
    if family == "cap":
        depth = int(rng.integers(2, 5))
        m = np.zeros((h + depth, w), dtype=np.int8)
        m[depth:, :] = 1
        tip = int(rng.integers(1, w - 1))
        for y in range(depth):
            radius = y + int(rng.integers(0, 2))
            m[y, max(0, tip - radius):min(w, tip + radius + 1)] = 2
    elif family == "v":
        depth = int(rng.integers(2, 5))
        m = np.zeros((h + depth, w), dtype=np.int8)
        m[0, :] = m[h - 1, :] = 1
        m[:h, 0] = m[:h, -1] = 1
        if h >= 5 and rng.integers(2):
            m[h // 2, :] = 1
        for y in range(depth):
            m[h + y, min(w // 2, y)] = 2
            m[h + y, max(w // 2, w - 1 - y)] = 2
    elif family == "side":
        m = np.zeros((h, w + 3), dtype=np.int8)
        m[:, :w] = 1
        m[0, w:] = m[-1, w:] = 2
        m[:, -1] = 2
        if rng.integers(2):
            m[h // 2, w] = 2
    elif family == "edge":
        h = int(rng.integers(6, 10))
        m = np.zeros((h, w + 2), dtype=np.int8)
        for y in range(h - 1):
            m[y, 1] = m[y, min(w + 1, 2 + y)] = 1
        m[h - 2, 1:] = 1
        m[:, 0] = m[-1, :] = 2
    elif family == "arms":
        m = np.zeros((h + 3, w + 2), dtype=np.int8)
        m[:h, 1:w + 1] = 1
        side = int(rng.integers(0, 2))
        x = 0 if side == 0 else w + 1
        m[h - 1:h + 3, x] = 2
        reach = int(rng.integers(2, 5))
        if side == 0:
            m[h + 2, :reach] = 2
        else:
            m[h + 2, -reach:] = 2
    elif family == "mono":
        depth = int(rng.integers(2, 5))
        m = np.zeros((h + depth + 1, w), dtype=np.int8)
        m[:h, :] = 1
        m[h:h + depth, int(rng.integers(1, w - 1))] = 1
        m[-1, :] = 1
    else:
        raise ValueError("unknown motif family")
    return m


def _erase(rng, motif, style):
    ygrid, xgrid = np.indices(motif.shape)
    if style == "accent":
        erased = motif == 2
    elif style == "mono":
        # Preserve the full slab, as in the source monochrome test. Its
        # boundaries identify the offset; the attached stem is restored.
        row_counts = np.count_nonzero(motif, axis=1)
        slab_end = next(i for i, count in enumerate(row_counts)
                        if count < row_counts[0])
        erased = (motif != 0) & (ygrid >= slab_end)
    elif style in ("body", "strokes"):
        color = 2 if style == "strokes" else 1
        ys, xs = np.nonzero(motif == color)
        if not len(ys):
            return None
        k = int(rng.integers(len(ys)))
        radius = int(rng.integers(2, 5))
        erased = (motif == color) & (
            abs(ygrid - ys[k]) + abs(xgrid - xs[k]) <= radius
        )
    elif style == "mixed":
        if not np.any(motif == 2):
            return None
        erased = np.zeros(motif.shape, dtype=bool)
        for color in (1, 2):
            ys, xs = np.nonzero(motif == color)
            k = int(rng.integers(len(ys)))
            radius = int(rng.integers(1, 3))
            erased |= (motif == color) & (
                abs(ygrid - ys[k]) + abs(xgrid - xs[k]) <= radius
            )
    else:
        raise ValueError("unknown erasure style")
    if not np.any(erased) or np.count_nonzero((motif != 0) & ~erased) < 6:
        return None
    return erased


def _make_pair(rng, family, style, axis, reverse, stagger, gap):
    if axis == "vertical" and stagger and gap == 0:
        # A cut through two joined, staggered figures can create a second
        # plausible translated pair with a different answer.
        return None
    motif = _motif(rng, family)
    erased = _erase(rng, motif, style)
    if erased is None:
        return None
    h, w = motif.shape
    shift = int(rng.integers(-3, 4)) if stagger and axis == "horizontal" else 0
    side_shift = int(rng.integers(-3, 4)) if stagger and axis == "vertical" else 0
    if stagger and shift == side_shift == 0:
        shift = 1 if axis == "horizontal" else 0
        side_shift = 1 if axis == "vertical" else 0
    if axis == "horizontal":
        origins = ((max(0, -shift), 0), (max(0, shift), w + gap))
    else:
        origins = ((0, max(0, -side_shift)), (h + gap, max(0, side_shift)))
    top, left, bottom, right = (int(rng.integers(0, 4)) for _ in range(4))
    height = top + max(y + h for y, x in origins) + bottom
    width = left + max(x + w for y, x in origins) + right
    if height > 30 or width > 30:
        return None
    background = 0 if rng.integers(5) else int(rng.integers(1, 10))
    choices = [c for c in range(10) if c != background]
    body, accent = (int(c) for c in rng.choice(choices, size=2, replace=False))
    local = np.where(motif == 1, body, np.where(motif == 2, accent, background))
    out = np.full((height, width), background, dtype=np.int8)
    inp = out.copy()
    for i, (y, x) in enumerate(origins):
        y += top
        x += left
        out[y:y + h, x:x + w] = local
        inp[y:y + h, x:x + w] = local
        if i != int(reverse):
            inp[y:y + h, x:x + w][erased] = background
    if np.count_nonzero(out == background) <= out.size // 2:
        return None
    return {"input": inp, "output": out}


def _cells(grid, background):
    return {
        (int(y), int(x)): int(grid[y, x])
        for y, x in np.argwhere(grid != background)
    }


def _decompose(out, background):
    """Recover equal translated figures with a spatial cut, even when joined."""
    cells = _cells(out, background)
    n = len(cells)
    if n < 12 or n % 2:
        return []
    decompositions = []
    for axis, index in (("horizontal", 1), ("vertical", 0)):
        coordinates = sorted(point[index] for point in cells)
        if coordinates[n // 2 - 1] == coordinates[n // 2]:
            continue
        cut = coordinates[n // 2 - 1]
        first = {p: c for p, c in cells.items() if p[index] <= cut}
        second = {p: c for p, c in cells.items() if p[index] > cut}
        if len(first) != len(second):
            continue
        dy = min(y for y, x in second) - min(y for y, x in first)
        dx = min(x for y, x in second) - min(x for y, x in first)
        if {(y + dy, x + dx): c for (y, x), c in first.items()} == second:
            decompositions.append((axis, first, second))
    return decompositions


def _offsets(inp, source, visible, axis, direction):
    if not visible:
        return set()
    by_color = {}
    for point, color in source.items():
        by_color.setdefault(color, []).append(point)
    reference = min(visible, key=lambda p: len(by_color.get(visible[p], [])))
    candidates = set()
    for point in by_color.get(visible[reference], []):
        dy, dx = reference[0] - point[0], reference[1] - point[1]
        if axis == "horizontal" and dx * direction <= 0:
            continue
        if axis == "vertical" and dy * direction <= 0:
            continue
        translated = {(y + dy, x + dx): c for (y, x), c in source.items()}
        if any(y < 0 or x < 0 or y >= inp.shape[0] or x >= inp.shape[1]
               for y, x in translated):
            continue
        if any(p in source for p in translated):
            continue
        if all(translated.get(p) == c for p, c in visible.items()):
            candidates.add((dy, dx))
    return candidates


def _input_solution(inp):
    """Predict from pixels alone by finding the smallest complete template."""
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    points = _cells(inp, background)
    outputs = {}
    for index in (0, 1):
        coordinates = [p[index] for p in points]
        for cut in range(min(coordinates), max(coordinates)):
            low = {p: c for p, c in points.items() if p[index] <= cut}
            high = {p: c for p, c in points.items() if p[index] > cut}
            for source, target in ((low, high), (high, low)):
                if len(source) <= len(target) or len(target) < 2:
                    continue
                by_color = {}
                for p, c in source.items():
                    by_color.setdefault(c, []).append(p)
                rare = min(target, key=lambda p: len(by_color.get(target[p], [])))
                area = (max(y for y, x in source) - min(y for y, x in source) + 1) * (
                    max(x for y, x in source) - min(x for y, x in source) + 1)
                score = (len(source), area)
                for point in by_color.get(target[rare], []):
                    dy, dx = rare[0] - point[0], rare[1] - point[1]
                    if (index == 0 and dy == 0) or (index == 1 and dx == 0):
                        continue
                    moved = {(y + dy, x + dx): c for (y, x), c in source.items()}
                    if any(y < 0 or x < 0 or y >= inp.shape[0] or x >= inp.shape[1]
                           for y, x in moved):
                        continue
                    if any(p in source for p in moved):
                        continue
                    if not all(moved.get(p) == c for p, c in target.items()):
                        continue
                    if not all((p[index] > cut) == (rare[index] > cut) for p in moved):
                        continue
                    answer = inp.copy()
                    for (y, x), c in moved.items():
                        answer[y, x] = c
                    if np.array_equal(answer, inp) or not _decompose(answer, background):
                        continue
                    key = answer.tobytes()
                    if key not in outputs or score < outputs[key][1]:
                        outputs[key] = (answer, score)
    if not outputs:
        return None
    best = min(score for answer, score in outputs.values())
    winners = [answer for answer, score in outputs.values() if score == best]
    return winners[0] if len(winners) == 1 else None


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors, counts = np.unique(out, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert np.count_nonzero(out == background) > out.size // 2
    assert np.all((inp == background) | (inp == out))
    assert np.any(inp != out)
    digest = hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest()
    assert digest not in _SOURCE_INPUT_DIGESTS
    valid = []
    for axis, first, second in _decompose(out, background):
        for source, target, direction in ((first, second, 1), (second, first, -1)):
            if any(inp[p] != c for p, c in source.items()):
                continue
            visible = {p: c for p, c in target.items() if inp[p] != background}
            if not visible or len(visible) == len(target):
                continue
            offset = (
                min(y for y, x in target) - min(y for y, x in source),
                min(x for y, x in target) - min(x for y, x in source),
            )
            if _offsets(inp, source, visible, axis, direction) == {offset}:
                valid.append((axis, source, target, visible, direction))
    assert len(valid) == 1, "ambiguous or malformed duplicate completion"
    axis, source, target, visible, direction = valid[0]
    missing = set(target) - set(visible)
    source_colors = set(source.values())
    missing_colors = {target[p] for p in missing}
    visible_colors = set(visible.values())
    sy0, sy1 = min(y for y, x in source), max(y for y, x in source)
    sx0, sx1 = min(x for y, x in source), max(x for y, x in source)
    ty0, ty1 = min(y for y, x in target), max(y for y, x in target)
    tx0, tx1 = min(x for y, x in target), max(x for y, x in target)
    if axis == "horizontal":
        gap = tx0 - sx1 - 1 if direction > 0 else sx0 - tx1 - 1
        is_staggered = ty0 != sy0
    else:
        gap = ty0 - sy1 - 1 if direction > 0 else sy0 - ty1 - 1
        is_staggered = tx0 != sx0
    return {
        "axis": axis, "direction": direction, "gap": gap,
        "stagger": is_staggered, "background": background,
        "source_colors": source_colors, "missing_colors": missing_colors,
        "visible_colors": visible_colors,
        "partial_color": bool(missing_colors & visible_colors),
        "mixed": len(missing_colors) > 1, "mono": len(source_colors) == 1,
        "height": inp.shape[0], "width": inp.shape[1],
        "edge": sy0 == 0 or sx0 == 0 or ty0 == 0 or tx0 == 0,
        "touching": gap == 0,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = (pair["input"].shape, pair["input"].tobytes())
            assert key not in seen
            seen.add(key)
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    assert any(not a["partial_color"] and not a["mono"] for a in train)
    assert any(a["partial_color"] for a in train)
    if any(a["direction"] < 0 for a in analyses["test"]):
        assert any(a["direction"] < 0 for a in train)
    if any(a["stagger"] for a in analyses["test"]):
        assert any(a["stagger"] for a in train)
    for pair in puzzle["test"]:
        assert np.array_equal(_input_solution(pair["input"]), pair["output"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # These puzzle-wide choices set the challenge mix, not hidden solving rules.
    reverse_challenge = bool(rng.integers(2))
    stagger_challenge = bool(rng.integers(2))
    train, test, seen = [], [], set()
    families = ("cap", "v", "side", "edge", "arms", "mono")
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                family = ("cap", "v", "side", "arms")[int(rng.integers(4))]
                style = "accent"
                axis, reverse, stagger = "horizontal", False, False
            elif split == "train" and index == 1:
                family = ("v", "edge", "cap", "side", "arms")[int(rng.integers(5))]
                style = "body"
                axis, reverse, stagger = "horizontal", reverse_challenge, stagger_challenge
            else:
                family = families[int(rng.integers(len(families)))]
                style = "mono" if family == "mono" else (
                    "accent", "body", "mixed", "strokes"
                )[int(rng.integers(4))]
                axis = "vertical" if rng.integers(4) == 0 else "horizontal"
                reverse = reverse_challenge and bool(rng.integers(2))
                stagger = stagger_challenge and bool(rng.integers(2))
            if split == "test" and index == 0:
                axis = "vertical" if rng.integers(2) else axis
                reverse, stagger = reverse_challenge, stagger_challenge
            for attempt in range(160):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                gap = int(pair_rng.integers(0, 5))
                pair = _make_pair(pair_rng, family, style, axis, reverse, stagger, gap)
                if pair is None:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                try:
                    _analyze(pair)
                except AssertionError:
                    continue
                if split == "test" and not np.array_equal(
                    _input_solution(pair["input"]), pair["output"]
                ):
                    continue
                seen.add(key)
                break
            else:
                raise RuntimeError("could not construct an unambiguous pair")
            (train if split == "train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(a, b):
    return all(
        len(a[k]) == len(b[k]) and all(
            np.array_equal(pa[g], pb[g])
            for pa, pb in zip(a[k], b[k])
            for g in ("input", "output")
        )
        for k in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, puzzles = set(), set(), set()
    palettes, shapes, gaps, placements, motifs = set(), set(), set(), set(), set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                features.append(a)
                palettes.add(tuple(sorted(int(c) for c in np.unique(pair["output"]))))
                shapes.add(pair["input"].shape)
                gaps.add(a["gap"])
                placements.add((a["axis"], a["direction"], a["stagger"]))
                bg = a["background"]
                ys, xs = np.nonzero(pair["output"] != bg)
                crop = pair["output"][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                motifs.add((crop.shape, crop.tobytes()))
                signature.append((pair["input"].shape, pair["input"].tobytes(),
                                  pair["output"].tobytes()))
        puzzles.add(tuple(signature))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) == 42 and len(palettes) >= 20 and len(shapes) >= 25
    assert len(motifs) >= 90 and len(placements) >= 5
    assert {0, 1, 2} <= gaps
    assert any(a["mixed"] for a in features)
    assert any(a["mono"] for a in features)
    assert any(a["background"] != 0 for a in features)
    assert any(a["edge"] for a in features)
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1),
                          (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(131, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(131, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
