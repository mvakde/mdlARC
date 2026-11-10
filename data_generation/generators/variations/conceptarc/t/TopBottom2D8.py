"""Puzzles where the color on the lowest occupied row recolors the foreground."""

import numpy as np


SOURCE_INPUTS = {
    ("0000000000", "0033330000", "0030030000", "0030030000",
     "0033330000", "0000000000", "0000022220", "0000020020",
     "0000020020", "0000022220"),
    ("0000000000", "0444001110", "0040000100", "0040000100",
     "0000333000", "0000030000", "0000030000", "0555000000",
     "0050000000", "0050000000"),
    ("0006000000", "0060600000", "0006000600", "0000006060",
     "0100000600", "1010000000", "0100000000", "0000000800",
     "0000008080", "0000000800"),
    ("0400000000", "0400000000", "0400000002", "0400000002",
     "0000600002", "0000600002", "0000600800", "0000600800",
     "0000000800", "0000000800"),
    ("0022222220", "0020000020", "0022222220", "0044444440",
     "0040000040", "0044444440", "0088888880", "0080000080",
     "0088888880", "0000000000"),
    ("0000200000", "0000020000", "0030002000", "0003000200",
     "0000300000", "0000030000", "0000003000", "0400000300",
     "0040000030", "0004000000"),
}
FAMILIES = ("rect", "t", "x", "bar", "outline", "diagonal", "zigzag", "sparse")


def _motif(rng, kind):
    if kind == "rect":
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        a = np.zeros((h, w), bool)
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = True
    elif kind == "t":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        a = np.zeros((h, w), bool)
        a[0, :] = True
        a[1:, w // 2] = True
    elif kind == "x":
        n = int(rng.choice((3, 5)))
        a = np.eye(n, dtype=bool) | np.fliplr(np.eye(n, dtype=bool))
        if rng.integers(2):
            a[n // 2, n // 2] = False
    elif kind == "bar":
        a = np.ones((int(rng.integers(2, 9)), 1), bool)
    elif kind == "outline":
        h, w = int(rng.integers(3, 5)), int(rng.integers(5, 12))
        a = np.zeros((h, w), bool)
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = True
    elif kind == "diagonal":
        n = int(rng.integers(3, 8))
        a = np.eye(n, dtype=bool)
        if rng.integers(2):
            a = np.fliplr(a)
    elif kind == "zigzag":
        n = int(rng.integers(4, 8))
        a = np.zeros((n, 3), bool)
        for r in range(n):
            a[r, (r + r // 2) % 3] = True
    elif kind == "sparse":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        a = np.zeros((h, w), bool)
        a[0, int(rng.integers(w))] = True
        a[-1, int(rng.integers(w))] = True
        for _ in range(int(rng.integers(2, 6))):
            a[int(rng.integers(h)), int(rng.integers(w))] = True
    else:
        raise ValueError("unknown motif")
    if kind in ("t", "bar", "outline", "zigzag", "sparse"):
        a = np.rot90(a, int(rng.integers(4)))
    if rng.integers(2):
        a = np.fliplr(a)
    rr, cc = np.where(a)
    return np.array(
        a[rr.min():rr.max() + 1, cc.min():cc.max() + 1],
        dtype=bool, copy=True,
    )


def _dimensions(rng, reversal):
    if reversal:
        return int(rng.integers(11, 20)), int(rng.integers(10, 20))
    draw = rng.random()
    if draw < 0.18:
        return 10, 10
    if draw < 0.30:
        return int(rng.integers(8, 11)), int(rng.integers(8, 11))
    if draw < 0.92:
        return int(rng.integers(11, 22)), int(rng.integers(10, 22))
    return int(rng.integers(22, 28)), int(rng.integers(18, 28))


def _first_masks(rng, relation):
    if relation == "equal":
        if rng.integers(2):
            a = np.ones((int(rng.integers(4, 8)), 1), bool)
            b = a.T.copy()
        else:
            a, b = np.zeros((3, 5), bool), np.zeros((4, 4), bool)
            for mask in (a, b):
                mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return [a, b]
    for _ in range(200):
        kind = str(rng.choice(FAMILIES[:6]))
        a = _motif(rng, kind)
        b = _motif(rng, kind if rng.random() < 0.55
                   else str(rng.choice(FAMILIES[:6])))
        if abs(int(a.sum()) - int(b.sum())) >= 2:
            masks = sorted((a, b), key=lambda m: int(m.sum()))
            return masks if relation == "fewer" else masks[::-1]
    raise RuntimeError("first-pair motif selection")


def _masks(rng, n, regime, profile, relation):
    if profile == "first":
        return _first_masks(rng, relation)
    if profile == "reversal":
        length = int(rng.integers(5, 8))
        wide = int(rng.integers(length + 1, 9))
        return [np.ones((length, 1), bool), np.ones((1, wide), bool)]
    weights = ([13, 13, 12, 15, 10, 12, 5, 5] if regime == "train"
               else [10, 10, 10, 13, 11, 13, 9, 8])
    probabilities = np.array(weights) / sum(weights)
    mixed = rng.random() < (0.60 if regime == "train" else 0.78)
    common = str(rng.choice(FAMILIES, p=probabilities))
    kinds = [str(rng.choice(FAMILIES, p=probabilities)) if mixed else common
             for _ in range(n)]
    if mixed and n > 1 and len(set(kinds)) == 1:
        kinds[-1] = FAMILIES[(FAMILIES.index(kinds[0]) +
                              int(rng.integers(1, 8))) % 8]
    return [_motif(rng, kind) for kind in kinds]


def _place(rng, masks, colors, h, w, reversal):
    if any(m.shape[0] > h or m.shape[1] > w for m in masks):
        return None
    for _ in range(240):
        ids = np.zeros((h, w), np.int8)
        mh, mw = masks[0].shape
        bottom = h - 1 - int(rng.integers(min(4, h - mh + 1)))
        top = bottom - mh + 1
        col = int(rng.integers(w - mw + 1))
        ids[top:top + mh, col:col + mw][masks[0]] = 1
        good = True
        for i, mask in enumerate(masks[1:], 1):
            mh, mw = mask.shape
            lo = top + 1 if reversal and i == 1 else 0
            hi = bottom - mh
            if hi < lo:
                good = False
                break
            for _ in range(85):
                r = int(rng.integers(lo, hi + 1))
                c = int(rng.integers(w - mw + 1))
                patch = ids[r:r + mh, c:c + mw]
                if not np.any(patch[mask]):
                    patch[mask] = i + 1
                    break
            else:
                good = False
                break
        if good:
            result = np.zeros((h, w), np.int8)
            for i, color in enumerate(colors, 1):
                result[ids == i] = color
            if np.count_nonzero(result) * 2 < result.size:
                return result
    return None


def _colors(rng, n, target_rank=None, new_color=None):
    if n == 2 and target_rank is not None:
        ordered = sorted(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
        return ordered if target_rank == "min" else ordered[::-1]
    k = int(rng.integers(2, min(4, n) + 1))
    pool = [c for c in range(1, 10) if c != new_color]
    picked = list(map(int, rng.choice(pool, k - int(new_color is not None),
                                      replace=False)))
    if new_color is not None:
        picked.append(int(new_color))
    target = int(rng.choice(picked))
    assignments = [target] + [c for c in picked if c != target]
    while len(assignments) < n:
        assignments.append(int(rng.choice(picked)))
    return [target] + list(map(int, rng.permutation(assignments[1:])))


def _bottom_color(inp):
    bottom = int(np.flatnonzero(np.any(inp != 0, axis=1))[-1])
    colors = set(map(int, inp[bottom][inp[bottom] != 0]))
    assert len(colors) == 1
    return next(iter(colors))


def _solve(inp):
    target = _bottom_color(inp)
    return np.where(inp == 0, 0, target).astype(np.int8)


def _pair(rng, regime, profile, relation=None, rank=None, distinct_target=None,
          new_color=None):
    for _ in range(160):
        h, w = _dimensions(rng, profile == "reversal")
        if profile in ("first", "reversal"):
            n = 2
        else:
            n = int(rng.integers(2, min(6, max(3, h * w // 40)) + 1))
        masks = _masks(rng, n, regime, profile, relation)
        colors = _colors(rng, n, target_rank=rank, new_color=new_color)
        if distinct_target is not None and colors[0] == distinct_target:
            continue
        inp = _place(rng, masks, colors, h, w, profile == "reversal")
        if inp is not None and len(np.unique(inp)) >= 3:
            return inp, _solve(inp)
    raise RuntimeError("could not construct pair")


def _count(value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("count must be an integer")
    if not lo <= value <= hi:
        raise ValueError("count out of range")
    return int(value)


def _source_input(inp):
    return inp.shape == (10, 10) and tuple(
        "".join(str(int(v)) for v in row) for row in inp) in SOURCE_INPUTS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, 2, 10)
    num_test = _count(num_test, 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("train count must exceed test count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    # Background 0 and the bottom-row rule are the only shared conventions.
    first_rank = "min" if rng.integers(2) else "max"
    second_rank = "max" if first_rank == "min" else "min"
    puzzle = {"train": [], "test": []}
    seen, train_colors = set(), set()
    for regime, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            profile = ("first" if regime == "train" and i == 0 else
                       "reversal" if regime == "train" and i == 1 else "general")
            relation = ("fewer", "equal", "more")[seed % 3] if profile == "first" else None
            rank = first_rank if profile == "first" else second_rank if profile == "reversal" else None
            novel = sorted(set(range(1, 10)) - train_colors)
            new_color = (int(rng.choice(novel)) if regime == "test" and i == 0
                         and novel and rng.random() < 0.6 else None)
            first_target = (_bottom_color(puzzle["train"][0]["input"])
                            if profile == "reversal" else None)
            for _ in range(30):
                local = np.random.default_rng(int(rng.integers(0, 2**63)))
                inp, out = _pair(local, regime, profile, relation, rank,
                                 first_target, new_color)
                key = (inp.shape, inp.tobytes())
                if key not in seen and not _source_input(inp):
                    break
            else:
                raise RuntimeError("duplicate input")
            seen.add(key)
            puzzle[regime].append({"input": inp, "output": out})
            if regime == "train":
                train_colors.update(int(v) for v in np.unique(inp) if v)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, targets = set(), []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            assert np.count_nonzero(inp) * 2 < inp.size
            assert 2 <= len(set(map(int, np.unique(inp))) - {0}) <= 4
            target = _bottom_color(inp)
            assert np.array_equal(out, np.where(inp == 0, 0, target))
            assert not np.array_equal(inp, out)
            assert not _source_input(inp)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            if regime == "train":
                targets.append(target)
    assert targets[0] != targets[1] and len(set(targets)) >= 2
    second = puzzle["train"][1]["input"]
    winner = targets[1]
    loser = next(c for c in np.unique(second) if c not in (0, winner))
    wr = np.flatnonzero(np.any(second == winner, axis=1))
    lr = np.flatnonzero(np.any(second == loser, axis=1))
    assert wr[0] < lr[0] and wr[-1] > lr[-1]
    assert np.count_nonzero(second == winner) < np.count_nonzero(second == loser)


def _fingerprint(puzzle):
    return tuple((name, p["input"].shape, p["input"].tobytes(),
                  p["output"].tobytes())
                 for name in ("train", "test") for p in puzzle[name])


def _observed_family(inp, color):
    rr, cc = np.where(inp == color)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    mask = inp[r0:r1, c0:c1] == color
    h, w = mask.shape
    if (h == 1 or w == 1) and np.all(mask):
        return "horizontal_bar" if h == 1 else "vertical_bar"
    if h >= 3 and w >= 3:
        border = np.zeros((h, w), bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        if np.array_equal(mask, border):
            return "border"
    if h == w and h >= 3:
        eye = np.eye(h, dtype=bool)
        if np.array_equal(mask, eye) or np.array_equal(mask, np.fliplr(eye)):
            return "diagonal"
        if h in (3, 5):
            x = eye | np.fliplr(eye)
            hollow = x.copy()
            hollow[h // 2, h // 2] = False
            if np.array_equal(mask, x) or np.array_equal(mask, hollow):
                return "x"
    for turn in range(4):
        rotated = np.rot90(mask, turn)
        rh, rw = rotated.shape
        if rh >= 3 and rw >= 3:
            t = np.zeros((rh, rw), bool)
            t[0, :] = True
            t[1:, rw // 2] = True
            if np.array_equal(rotated, t):
                return "t"
    return None


def _distant_same_color(inp):
    for color in np.unique(inp):
        if color == 0:
            continue
        rr, cc = np.where(inp == color)
        if np.any(np.diff(np.unique(rr)) >= 3) or np.any(np.diff(np.unique(cc)) >= 3):
            return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    dimensions, shapes, palettes, counts = set(), set(), set(), set()
    relations, first_ranks = set(), set()
    families = {"train": set(), "test": set()}
    four_colors = compact = small = large = 0
    mixed = distant = novel = reversals = close = edge = interior = 0
    outline_widths, outline_offsets = set(), set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_colors = set(int(c) for pair in puzzle["train"]
                           for c in np.unique(pair["input"]) if c)
        local_dimensions, local_palettes = set(), set()
        first = puzzle["train"][0]["input"]
        target = _bottom_color(first)
        other = next(c for c in np.unique(first) if c not in (0, target))
        relations.add(int(np.sign(np.count_nonzero(first == target) -
                                  np.count_nonzero(first == other))))
        first_ranks.add(int(target < other))
        for name in ("train", "test"):
            for pair in puzzle[name]:
                inp, out = pair["input"], pair["output"]
                dimensions.add(inp.shape)
                local_dimensions.add(inp.shape)
                shapes.add((inp.shape, (inp != 0).tobytes()))
                colors = set(map(int, np.unique(inp))) - {0}
                palettes.add(tuple(sorted(colors)))
                local_palettes.add(tuple(sorted(colors)))
                four_colors += len(colors) >= 4
                compact += inp.shape == (10, 10)
                small += min(inp.shape) < 10
                large += max(inp.shape) >= 22
                touches = bool(np.any(inp[0]) or np.any(inp[-1]) or
                               np.any(inp[:, 0]) or np.any(inp[:, -1]))
                edge += touches
                interior += not touches
                distant += _distant_same_color(inp)
                if name == "test":
                    novel += bool(colors - train_colors)
                target = _bottom_color(inp)
                rows = {c: np.flatnonzero(np.any(inp == c, axis=1)) for c in colors}
                reversals += any(rows[target][0] < rows[c][0] and
                                 rows[target][-1] > rows[c][-1]
                                 for c in colors if c != target)
                close += rows[target][-1] - max(
                    rows[c][-1] for c in colors if c != target) <= 2
                observed = [_observed_family(inp, c) for c in colors]
                known = set(f for f in observed if f is not None)
                families[name].update(known)
                mixed += len(known) >= 2
                borders = []
                for c, family in zip(colors, observed):
                    if family == "border":
                        rr, cc = np.where(inp == c)
                        borders.append((int(cc.min()), int(cc.max() - cc.min() + 1)))
                if len(borders) >= 2:
                    outline_widths.add(tuple(sorted(w for _, w in borders)))
                    outline_offsets.add(abs(borders[0][0] - borders[1][0]))
                assert np.array_equal(out, _solve(inp))
        assert len(local_dimensions) >= 2 and len(local_palettes) >= 2
    assert relations == {-1, 0, 1} and first_ranks == {0, 1}
    assert len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3
    assert len(dimensions) >= 35 and len(shapes) >= 150 and len(palettes) >= 20
    assert four_colors > 0 and compact > 0 and small > 0 and large > 0
    assert mixed >= 10 and distant >= 10 and novel >= 10 and reversals >= 42
    assert close >= 10
    assert edge > 0 and interior > 0
    assert len(outline_widths) >= 3 and len(outline_offsets) >= 3
    required_families = {"border", "t", "x", "diagonal",
                         "horizontal_bar", "vertical_bar"}
    assert required_families <= families["train"]
    assert required_families <= families["test"]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, tr, te)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(1000 + i, tr, te))
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(7, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
