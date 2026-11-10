"""Variations of central-object extraction within an enclosing boundary."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "238262080fc62e12795389b5e0da2d0138658932dee1cae6528a00a3bfb171ab",
    "d8e09fd3357f3e2d7d8795fc8ec2daa1395d977c1f7af75e2e07cf385b2db5ce",
    "0dc3228388664b4bb2d3b3cafb11835e416a3f459d333cd610759379bf110ddf",
    "3237467e234f4cc70c65a4037e6424aae93c64aab131a4d5e0be6959a6d2951e",
    "c49170eceb6000b08767d1871414d085122f3b12a764024aa5198b7bffc0e138",
    "9bed860b103b1352846ac97b3870578ee4484d1b678da4a13a96603aa8b0a3c6",
}


def _ring(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def _diamond(side: int) -> np.ndarray:
    yy, xx = np.indices((side, side))
    return (np.abs(2 * yy - side + 1) + np.abs(2 * xx - side + 1)
            == 2 * (side // 2))


def _paint(grid: np.ndarray, y: int, x: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    region = grid[y:y + h, x:x + w]
    region[mask] = color


def _choose_counts(rng: np.random.Generator, num_train: int | None,
                   num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _roles(rng: np.random.Generator, palette: tuple[int, ...]) -> tuple[int, int]:
    outer = int(rng.choice(palette))
    target = outer if rng.random() < 0.25 else int(rng.choice(
        [c for c in palette if c != outer]))
    return outer, target


def _decoys(rng: np.random.Generator, grid: np.ndarray, background: int,
            palette: tuple[int, ...], target_color: int,
            outer: tuple[int, int, int, int], target: tuple[int, int, int, int],
            count: int, diamond_outer: bool = False) -> None:
    oy, ox, oh, ow = outer
    ty, tx, th, tw = target
    cy, cx = oy + (oh - 1) / 2, ox + (ow - 1) / 2
    for _ in range(count):
        unused = [c for c in palette if c not in set(int(v) for v in np.unique(grid))]
        color = (target_color if rng.random() < 0.30
                 else int(rng.choice(unused if unused and rng.random() < 0.7
                                     else palette)))
        clearance = 3 if color == target_color else 1
        for _attempt in range(90):
            h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
            if oh - h <= 2 or ow - w <= 2:
                break
            y = int(rng.integers(oy + 1, oy + oh - h))
            x = int(rng.integers(ox + 1, ox + ow - w))
            if (y <= ty + th + clearance - 1 and y + h >= ty - clearance
                    and x <= tx + tw + clearance - 1 and x + w >= tx - clearance):
                continue
            if diamond_outer:
                corners = ((y, x), (y + h - 1, x), (y, x + w - 1),
                           (y + h - 1, x + w - 1))
                if any(abs(yy - cy) + abs(xx - cx) >= (oh - 1) / 2 - 1
                       for yy, xx in corners):
                    continue
            region = grid[y:y + h, x:x + w]
            if np.any(region != background):
                continue
            mask = (_ring(h, w) if h >= 4 and w >= 4 and rng.random() < 0.4
                    else np.ones((h, w), bool))
            _paint(grid, y, x, mask, color)
            break


def _rectangular_pair(rng: np.random.Generator, background: int,
                      palette: tuple[int, ...], kind: str,
                      contamination: bool, density: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "solid":
        th = tw = int(rng.integers(2, 6))
    else:
        th, tw = int(rng.integers(4, 9)), int(rng.integers(4, 9))
    gt, gb, gl, gr = (int(rng.integers(4, 8)) for _ in range(4))
    oh, ow = th + gt + gb, tw + gl + gr
    mt, mb, ml, mr = (int(rng.integers(0, 4)) for _ in range(4))
    grid = np.full((oh + mt + mb, ow + ml + mr), background, np.int8)
    oy, ox, ty, tx = mt, ml, mt + gt, ml + gl
    outer_color, target_color = _roles(rng, palette)
    _paint(grid, oy, ox, _ring(oh, ow), outer_color)
    if kind == "hollow" and rng.random() < 0.25:
        middle_color = int(rng.choice([c for c in palette if c != target_color]))
        _paint(grid, ty - 2, tx - 2, _ring(th + 4, tw + 4), middle_color)
    if kind == "hollow" and contamination:
        # A neighboring object continues outside the crop, while the target
        # outline occludes its crossing and its tip remains visible in the hole.
        foreign = int(rng.choice([c for c in palette if c != target_color]))
        width = min(tw - 2, int(rng.integers(2, 5)))
        intruder = _ring(4, width) if width >= 3 else np.ones((4, width), bool)
        if rng.random() < 0.5:
            _paint(grid, ty - 2, tx + 1, intruder, foreign)
        else:
            _paint(grid, ty + th - 2, tx + 1, intruder, foreign)
    target_mask = np.ones((th, tw), bool) if kind == "solid" else _ring(th, tw)
    _paint(grid, ty, tx, target_mask, target_color)
    _decoys(rng, grid, background, palette, target_color,
            (oy, ox, oh, ow), (ty, tx, th, tw), density)
    output = np.full((th, tw), background, np.int8)
    output[target_mask] = target_color
    return grid, output


def _diamond_pair(rng: np.random.Generator, background: int,
                  palette: tuple[int, ...], density: int) -> tuple[np.ndarray, np.ndarray]:
    target_side = int(rng.integers(5, 9))
    outer_side = target_side + int(rng.integers(14, 17))
    mt, mb, ml, mr = (int(rng.integers(0, 4)) for _ in range(4))
    grid = np.full((outer_side + mt + mb, outer_side + ml + mr), background, np.int8)
    oy, ox = mt, ml
    shift_y, shift_x = (int(rng.integers(-1, 2)) for _ in range(2))
    ty = oy + (outer_side - target_side) // 2 + shift_y
    tx = ox + (outer_side - target_side) // 2 + shift_x
    outer_color, target_color = _roles(rng, palette)
    _paint(grid, oy, ox, _diamond(outer_side), outer_color)
    if rng.random() < 0.75:
        middle_side = target_side + 8
        middle_color = (outer_color if outer_color != target_color and rng.random() < 0.6
                        else int(rng.choice([c for c in palette if c != target_color])))
        _paint(grid, ty - (middle_side - target_side) // 2,
               tx - (middle_side - target_side) // 2,
               _diamond(middle_side), middle_color)
    _paint(grid, ty, tx, _diamond(target_side), target_color)
    _decoys(rng, grid, background, palette, target_color,
            (oy, ox, outer_side, outer_side),
            (ty, tx, target_side, target_side), min(density, 3), True)
    _paint(grid, oy, ox, _diamond(outer_side), outer_color)
    output = np.full((target_side, target_side), background, np.int8)
    output[_diamond(target_side)] = target_color
    return grid, output


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    background = int(rng.choice(np.arange(10)))
    palette_size = int(rng.integers(4, 9))
    palette = tuple(int(c) for c in rng.choice(
        [c for c in range(10) if c != background], palette_size, replace=False))
    train_kinds = ["solid", "hollow"]
    train_kinds += [str(rng.choice(("solid", "hollow", "diamond"),
                                   p=(0.33, 0.35, 0.32)))
                    for _ in range(train_count - 2)]
    if train_count >= 3 and "diamond" not in train_kinds and rng.random() < 0.55:
        train_kinds[int(rng.integers(2, train_count))] = "diamond"
    rng.shuffle(train_kinds)
    test_kinds = [str(rng.choice(("solid", "hollow", "diamond"),
                                 p=(0.33, 0.32, 0.35))) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, kinds in (("train", train_kinds), ("test", test_kinds)):
        first_hollow = True
        for kind in kinds:
            for _ in range(12):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                density = int(child_rng.choice((0, 1, 2, 3, 4, 5),
                                               p=(0.14, 0.20, 0.23, 0.20, 0.15, 0.08)))
                if kind == "diamond":
                    inp, out = _diamond_pair(child_rng, background, palette, density)
                else:
                    contaminate = (kind == "hollow" and
                                   (first_hollow if regime == "train"
                                    else child_rng.random() < 0.4))
                    inp, out = _rectangular_pair(child_rng, background, palette,
                                                 kind, contaminate, density)
                key = (inp.shape, inp.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append({"input": inp, "output": out})
                    if kind == "hollow":
                        first_hollow = False
                    break
            else:
                raise RuntimeError("Could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_hash(inp: np.ndarray, out: np.ndarray) -> str:
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes()
                          + bytes(out.shape) + out.astype(np.int8).tobytes()).hexdigest()


def _pair_features(inp: np.ndarray, out: np.ndarray, background: int) -> tuple[str, bool]:
    occupied = np.argwhere(inp != background)
    assert len(occupied) > 0
    top, left = np.min(occupied, axis=0)
    bottom, right = np.max(occupied, axis=0)
    oh, ow = int(bottom - top + 1), int(right - left + 1)
    outer = inp[top:bottom + 1, left:right + 1]
    color = int(outer[0, 0])
    if color != background and np.all(outer[_ring(oh, ow)] == color):
        kind_outer = "rectangle"
    else:
        assert oh == ow
        mask = _diamond(oh)
        color = int(outer[0, oh // 2])
        assert color != background and np.all(outer[mask] == color)
        kind_outer = "diamond"
    assert 0 <= top <= 3 and 0 <= left <= 3
    assert 0 <= inp.shape[0] - bottom - 1 <= 3
    assert 0 <= inp.shape[1] - right - 1 <= 3

    colors = set(int(c) for c in np.unique(out)) - {background}
    assert len(colors) == 1
    target_color = colors.pop()
    th, tw = out.shape
    target_mask = out == target_color
    assert np.any(target_mask[0]) and np.any(target_mask[-1])
    assert np.any(target_mask[:, 0]) and np.any(target_mask[:, -1])
    if np.all(target_mask):
        kind = "solid"
        assert kind_outer == "rectangle" and th == tw and 2 <= th <= 5
    elif np.array_equal(target_mask, _ring(th, tw)):
        kind = "hollow"
        assert kind_outer == "rectangle" and 4 <= th <= 8 and 4 <= tw <= 8
    else:
        kind = "diamond"
        assert kind_outer == "diamond" and th == tw and 5 <= th <= 8
        assert np.array_equal(target_mask, _diamond(th))
    assert np.all(out[~target_mask] == background)

    # Locate the answer's outline in visible cells, without construction metadata.
    candidates = []
    for y in range(int(top + 1), int(bottom - th + 2)):
        for x in range(int(left + 1), int(right - tw + 2)):
            region = inp[y:y + th, x:x + tw]
            if np.all(region[target_mask] == target_color):
                distance = max(abs((2 * y + th - 1) - (top + bottom)),
                               abs((2 * x + tw - 1) - (left + right)))
                if distance <= 4:
                    candidates.append((y, x))
    assert len(candidates) == 1
    y, x = candidates[0]
    crop = inp[y:y + th, x:x + tw]
    assert np.all(crop[target_mask] == target_color)
    assert not np.any(crop[~target_mask] == target_color)
    contamination = bool(np.any(crop[~target_mask] != background))
    if kind != "hollow":
        assert not contamination
    if kind == "diamond":
        assert oh == ow and 14 <= oh - th <= 16
    else:
        assert 4 <= y - top <= 7 and 4 <= bottom - (y + th - 1) <= 7
        assert 4 <= x - left <= 7 and 4 <= right - (x + tw - 1) <= 7
    assert _source_hash(inp, out) not in _SOURCE_HASHES
    return kind, contamination


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    backgrounds = []
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.shape, out.tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert inp.shape != out.shape or not np.array_equal(inp, out)
            background = int(np.bincount(inp.ravel().astype(int), minlength=10).argmax())
            backgrounds.append(background)
            features[regime].append(_pair_features(inp, out, background))
    assert len(set(backgrounds)) == 1
    train_kinds = [kind for kind, _ in features["train"]]
    assert "solid" in train_kinds and "hollow" in train_kinds
    assert any(kind == "hollow" and dirty for kind, dirty in features["train"])
    assert all(kind in ("solid", "hollow", "diamond") for kind, _ in features["test"])


def _same_puzzle(a: dict, b: dict) -> bool:
    return (len(a["train"]) == len(b["train"])
            and len(a["test"]) == len(b["test"])
            and all(np.array_equal(pa[field], pb[field])
                    for regime in ("train", "test")
                    for pa, pb in zip(a[regime], b[regime])
                    for field in ("input", "output")))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        if seed < 6:
            assert _same_puzzle(puzzle, generate_puzzle(seed))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    fingerprints = {tuple((pair["input"].shape, pair["input"].tobytes(),
                           pair["output"].shape, pair["output"].tobytes())
                          for regime in ("train", "test") for pair in p[regime])
                    for p in puzzles}
    assert len(fingerprints) >= 40
    kinds, backgrounds, dimensions, hollow_sizes, diamond_parities = set(), set(), set(), set(), set()
    dirty = clean = edge = diamond_train = diamond_test = no_diamond_test = 0
    first_hollow = nonfinal_diamond = source_shift = same_role = color_rich = 0
    asymmetric = offset = simple = target_decoy = 0
    for p in puzzles:
        bk = int(np.bincount(p["train"][0]["input"].ravel().astype(int)).argmax())
        backgrounds.add(bk)
        train_shapes = [_pair_features(q["input"], q["output"], bk)[0] for q in p["train"]]
        test_shapes = [_pair_features(q["input"], q["output"], bk)[0] for q in p["test"]]
        diamond_train += "diamond" in train_shapes
        diamond_test += "diamond" in test_shapes
        no_diamond_test += "diamond" not in test_shapes
        source_shift += "diamond" not in train_shapes and "diamond" in test_shapes
        first_hollow += train_shapes[0] == "hollow"
        nonfinal_diamond += any(kind == "diamond" for kind in test_shapes[:-1])
        for q in p["train"] + p["test"]:
            inp, out = q["input"], q["output"]
            kind, contaminated = _pair_features(inp, out, bk)
            kinds.add(kind)
            dimensions.add((inp.shape, out.shape))
            dirty += contaminated
            clean += not contaminated
            if kind == "hollow":
                hollow_sizes.add(out.shape)
            if kind == "diamond":
                diamond_parities.add(out.shape[0] % 2)
            color_rich += len(np.unique(inp)) >= 6
            occupied = np.argwhere(inp != bk)
            edge += np.any(np.min(occupied, axis=0) == 0)
            top, left = np.min(occupied, axis=0)
            bottom, right = np.max(occupied, axis=0)
            oh, ow = bottom - top + 1, right - left + 1
            asymmetric += kind != "diamond" and oh != ow
            target_color = int(next(iter(set(np.unique(out)) - {bk})))
            boundary_color = int(inp[top, left] if kind != "diamond"
                                 else inp[top, left + oh // 2])
            same_role += boundary_color == target_color
            mask = out == target_color
            matches = []
            for y in range(top + 1, bottom - out.shape[0] + 2):
                for x in range(left + 1, right - out.shape[1] + 2):
                    region = inp[y:y + out.shape[0], x:x + out.shape[1]]
                    if np.all(region[mask] == target_color):
                        delta = max(abs(2*y+out.shape[0]-1-top-bottom),
                                    abs(2*x+out.shape[1]-1-left-right))
                        if delta <= 4:
                            matches.append((y, x))
            assert len(matches) == 1
            y, x = matches[0]
            offset += (2*y+out.shape[0]-1 != top+bottom
                       or 2*x+out.shape[1]-1 != left+right)
            residual = inp != bk
            outer_mask = _diamond(oh) if kind == "diamond" else _ring(oh, ow)
            residual[top:bottom+1, left:right+1][outer_mask] = False
            residual[y:y+out.shape[0], x:x+out.shape[1]][mask] = False
            simple += not np.any(residual)
            target_decoy += np.any(residual & (inp == target_color))
    assert kinds == {"solid", "hollow", "diamond"}
    assert len(backgrounds) >= 4 and len(dimensions) >= 60
    assert dirty >= 30 and clean >= 80 and edge >= 30
    assert diamond_train >= 8 and diamond_test >= 8 and no_diamond_test >= 8
    assert (4, 4) in hollow_sizes and (4, 6) in hollow_sizes
    assert diamond_parities == {0, 1}
    assert first_hollow >= 6 and nonfinal_diamond >= 1 and source_shift >= 1
    assert same_role >= 20 and color_rich >= 10 and asymmetric >= 40
    assert offset >= 40 and simple >= 3 and target_decoy >= 10
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        p = generate_puzzle(812, train, test)
        validate_generated_puzzle(p)
        assert _same_puzzle(p, generate_puzzle(812, train, test))
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pair counts accepted")
