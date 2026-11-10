"""Canvas parity selects a stamp corner before binary-mask substitution.

Only NumPy and the standard library are required. No files, parent generators,
sample labels, or generation parameters are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


SELF_TEST_SEEDS = (
    0, 874407984, 878050662, 428590807, 127656788, 344088483,
    597646198, 136908489, 953561790, 703094126, 571314953, 88702248,
    651617374, 93021046, 876620971, 805818595, 215824516, 563099530,
    482546340, 391174740, 51620699, 69871696, 978481898, 577302664,
    246368240, 236647292, 509562122, 214954757, 555602432, 136091180,
    104511208, 623112327, 196385773, 138248881, 987529886, 438303940,
    797772550, 197564674, 632798455, 802795823, 459339658, 966532066,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lower, upper=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(f"invalid {name}: {value!r}")
    return int(value)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=False)


def _bbox(binary):
    rr, cc = np.nonzero(binary)
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _crop(grid, box):
    r0, r1, c0, c1 = box
    return grid[r0:r1, c0:c1].copy()


def _separated(a, b):
    # Half-open boxes: strict inequality leaves at least one blank line.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _touches_sides(binary):
    return all(np.any(edge) for edge in
               (binary[0], binary[-1], binary[:, 0], binary[:, -1]))


def _components(binary, diagonal=True):
    remaining = set(map(tuple, np.argwhere(binary)))
    count = 0
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return count


def _transforms(array):
    return [view for k in range(4)
            for view in (np.rot90(array, k), np.fliplr(np.rot90(array, k)))]


def _asymmetric(array):
    return all(not np.array_equal(array, t) for t in _transforms(array)[1:])


def _read_input(input_grid):
    """Recover both tight encodings by color roles, without mask connectivity."""
    grid = _grid(input_grid)
    if grid.shape[0] != grid.shape[1]:
        raise ValueError("the canvas must be square")
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if counts.max() * 2 <= grid.size:
        raise ValueError("background must be a strict majority")
    candidates = []
    for signal in map(int, colors):
        if signal == background or len(colors) < 4:
            continue
        mask_box = _bbox(grid == signal)
        rest = (grid != signal) & (grid != background)
        if len(np.unique(grid[rest])) < 2:
            continue
        stamp_box = _bbox(rest)
        if not _separated(mask_box, stamp_box):
            continue
        mask_crop = _crop(grid, mask_box)
        stamp = _crop(grid, stamp_box)
        if not np.all((mask_crop == signal) | (mask_crop == background)):
            continue
        if np.any(stamp == signal):
            continue
        if (not all(1 <= d <= 15 for d in mask_crop.shape)
                or stamp.shape[0] < 2 or stamp.shape[1] < 3):
            continue
        if _components(stamp != background) != 1:
            continue
        candidates.append({"background": background, "signal": signal,
                           "mask": mask_crop == signal, "stamp": stamp,
                           "mask_box": mask_box, "stamp_box": stamp_box,
                           "side": grid.shape[0]})
    if len(candidates) != 1:
        raise ValueError("expected one separated monochrome-mask/multicolor-stamp reading")
    return candidates[0]


def _patch(info):
    stamp = info["stamp"]
    return (stamp[:2, :2] if info["side"] % 2 else stamp[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select using whole-canvas parity; substitute in the untransformed mask."""
    info = _read_input(input_grid)
    mask, patch = info["mask"], _patch(info)
    rr, cc = np.indices((2 * mask.shape[0], 2 * mask.shape[1]))
    return np.where(mask[rr // 2, cc // 2], patch[rr % 2, cc % 2],
                    info["background"]).astype(np.int8)


def _expand(mask, patch, background):
    # Generation uses literal block writes; solve uses fine/coarse coordinates.
    out = np.full((2 * mask.shape[0], 2 * mask.shape[1]), background, dtype=np.int8)
    for r, c in np.argwhere(mask):
        out[2*r:2*r+2, 2*c:2*c+2] = patch
    return out


def _key(array):
    return array.shape, array.tobytes()


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def _mask(rng, shape, family, witness=False):
    h, w = shape
    rr, cc = np.indices(shape)
    if witness:
        mask = np.zeros(shape, dtype=bool)
        if family == "islands":
            mask[0, :2] = True
            mask[-1, 0] = mask[-1, -1] = True
        else:
            mask[0, 0] = True
            mask[1, 1:] = True
            mask[1:, -1] = True
        _require(_asymmetric(mask), "witness mask must fix its orientation")
        return mask
    if family in ("path", "diagonal"):
        mask = np.zeros(shape, dtype=bool)
        r = c = 0
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            steps = []
            if r < h - 1:
                steps.append((1, 0))
            if c < w - 1:
                steps.append((0, 1))
            if family == "diagonal" and r < h - 1 and c < w - 1:
                steps += [(1, 1)] * 3
            dr, dc = steps[int(rng.integers(len(steps)))]
            r, c = r + dr, c + dc
            mask[r, c] = True
    elif family == "ring":
        mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
    elif family == "ladder":
        mask = (rr % int(rng.integers(2, 4)) == 0) | (rr == h-1) | (cc == 0)
    elif family == "dense":
        mask = rng.random(shape) < float(rng.uniform(.65, .92))
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    elif family == "islands":
        mask = (rr % 2 == 0) & (cc % 2 == 0)
        mask[0, 0] = mask[-1, -1] = True
    elif family == "sparse":
        mask = rng.random(shape) < float(rng.uniform(.1, .35))
        mask[0, 0] = mask[-1, -1] = True
    elif family == "full":
        mask = np.ones(shape, dtype=bool)
    else:
        raise ValueError("unknown mask family")
    if rng.random() < .5:
        mask = np.flipud(mask)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _stamp(rng, shape, background, payload, witness=False):
    h, w = shape
    rr, cc = np.indices(shape)
    k = len(payload)
    mode = int(rng.integers(6))
    for _ in range(64):
        if mode == 0:
            labels = rr // int(rng.integers(1, 3))
        elif mode == 1:
            labels = cc // int(rng.integers(1, 3))
        elif mode == 2:
            labels = rr + int(rng.integers(1, k + 1)) * cc
        elif mode == 3:
            labels = np.minimum.reduce([rr, cc, h-1-rr, w-1-cc])
        elif mode == 4:
            tile = rng.integers(k, size=(int(rng.integers(1, min(4, h)+1)),
                                        int(rng.integers(1, min(4, w)+1))))
            labels = tile[rr % tile.shape[0], cc % tile.shape[1]]
        else:
            labels = rng.integers(k, size=shape)
        stamp = payload[(labels + int(rng.integers(k))) % k].astype(np.int8)
        if rng.random() < .65:
            stamp[rng.random(shape) < float(rng.uniform(.03, .18))] = background
        if witness:
            # Two equally colorful, asymmetric candidates with literal holes.
            values = np.array([background, *map(int, payload[:3])], dtype=np.int8)
            left = rng.permutation(values).reshape(2, 2)
            right = rng.permutation(values).reshape(2, 2)
            if any(np.array_equal(right, view) for view in _transforms(left)):
                # An adjacent transposition is outside the square's D4 group.
                right = left.ravel()[[0, 1, 3, 2]].reshape(2, 2)
            stamp[:2, :2], stamp[:2, -2:] = left, right
        elif w >= 4:
            if rng.random() < .18:
                # Either candidate can be a solid block, independent of parity.
                col = 0 if rng.random() < .5 else w - 2
                stamp[:2, col:col+2] = rng.choice(payload)
            if rng.random() < .12:
                stamp[:2, -2:] = stamp[:2, :2]
        foreground = stamp != background
        bottom_decoys_clear = (not witness or all(
            not np.array_equal(bottom, view)
            for bottom in (stamp[-2:, :2], stamp[-2:, -2:])
            for top in (left, right) for view in _transforms(top)))
        if (bottom_decoys_clear and len(np.unique(stamp[foreground])) >= 2 and _touches_sides(foreground)
                and _components(foreground) == 1
                and np.any(stamp[:2, :2] != background)
                and np.any(stamp[:2, -2:] != background)):
            return stamp
    # Dense fallback keeps both crop candidates visible and distinct.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[-1, -1] = payload[1]
    if witness:
        stamp[:2, :2] = [[background, payload[0]], [payload[1], payload[2]]]
        stamp[:2, -2:] = [[background, payload[0]], [payload[2], payload[1]]]
    return stamp


def _place(rng, mask, stamp, background, signal, *, parity=None,
           lower=4, upper=30, exclude=(), rim=None):
    """Place rectangles in either order along either axis, on a square."""
    mh, mw = mask.shape
    sh, sw = stamp.shape
    rim = int(rng.integers(2)) if rim is None else rim
    foreground = int(mask.sum() + np.count_nonzero(stamp != background))
    choices = []
    for horizontal in (False, True):
        for gap in range(1, 5):
            need = max(max(mh, sh), mw+sw+gap) if horizontal else max(mh+sh+gap, max(mw, sw))
            for side in range(max(lower, need + 2*rim), upper + 1):
                if (side not in exclude and side*side > 2*foreground
                        and (parity is None or side % 2 == parity)):
                    choices.append((side, horizontal, gap))
    if not choices:
        raise ValueError("rectangles do not fit the requested canvas range")
    side, horizontal, gap = choices[int(rng.integers(len(choices)))]
    reverse = bool(rng.integers(2))
    if horizontal:
        start = int(rng.integers(rim, side - mw - sw - gap - rim + 1))
        mc, sc = (start+sw+gap, start) if reverse else (start, start+mw+gap)
        mr = int(rng.integers(rim, side - mh - rim + 1))
        sr = int(rng.integers(rim, side - sh - rim + 1))
    else:
        start = int(rng.integers(rim, side - mh - sh - gap - rim + 1))
        mr, sr = (start+sh+gap, start) if reverse else (start, start+mh+gap)
        mc = int(rng.integers(rim, side - mw - rim + 1))
        sc = int(rng.integers(rim, side - sw - rim + 1))
    grid = np.full((side, side), background, dtype=np.int8)
    grid[mr:mr+mh, mc:mc+mw] = np.where(mask, signal, background)
    grid[sr:sr+sh, sc:sc+sw] = stamp
    return grid


def _colors(rng, stable):
    palette = stable if stable is not None else rng.permutation(10)
    k = int(rng.integers(3, 7))
    return int(palette[0]), int(palette[1]), np.array(palette[2:2+k], dtype=np.int8)


def _shape(rng, lo, hi, aspect=None):
    options = [(h, w) for h in range(lo, hi+1) for w in range(lo, hi+1)
               if aspect is None or (h < w if aspect == "wide" else h > w)]
    return options[int(rng.integers(len(options)))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["canvas_growth", "mask_growth", "stamp_growth", "aspect"]))
    aspect = str(rng.choice(["wide", "tall"])) if regime == "aspect" else None
    islands = bool(rng.integers(2))
    stable = rng.permutation(10) if rng.random() < .25 else None
    background, signal, payload = _colors(rng, stable)
    mask = _mask(rng, _shape(rng, 3, 6, aspect),
                 "islands" if islands else "diagonal", witness=True)
    stamp = _stamp(rng, (int(rng.integers(3, 6)), int(rng.integers(4, 8))),
                   background, payload, witness=True)
    first = _place(rng, mask, stamp, background, signal, lower=10, upper=23, rim=1)
    # This changes only whole-canvas parity: all source objects stay identical.
    second = np.pad(first, ((0, 1), (0, 1)), constant_values=background)
    patch1 = stamp[:2, :2] if first.shape[0] % 2 else stamp[:2, -2:]
    patch2 = stamp[:2, :2] if second.shape[0] % 2 else stamp[:2, -2:]
    puzzle = {"train": [{"input": first, "output": _expand(mask, patch1, background)},
                        {"input": second, "output": _expand(mask, patch2, background)}],
              "test": []}
    seen = {_key(first), _key(second)}
    base = first.shape[0]
    family_pool = ["path", "diagonal", "ring", "ladder", "dense", "full"]
    if islands:
        family_pool += ["islands", "sparse"]
    first_test_parity = int(rng.integers(2))
    for split, count in (("train", num_train-2), ("test", num_test)):
        for index in range(count):
            for _ in range(64):
                bg, sig, colors = _colors(rng, stable)
                if split == "test" and regime == "mask_growth":
                    mshape = _shape(rng, 7, 15)
                elif split == "test" and regime == "aspect":
                    mshape = _shape(rng, 2, 8, "tall" if aspect == "wide" else "wide")
                else:
                    mshape = _shape(rng, 1, 6, aspect)
                if split == "test" and regime == "stamp_growth":
                    sshape = int(rng.integers(6, 11)), int(rng.integers(8, 13))
                else:
                    sshape = int(rng.integers(2, 6)), int(rng.integers(3, 8))
                m = _mask(rng, mshape, str(rng.choice(family_pool)))
                if not islands and _components(m) != 1:
                    continue
                s = _stamp(rng, sshape, bg, colors)
                if split == "train":
                    parity = (base + index) % 2 if index < 2 else None
                    excluded = {p["input"].shape[0] for p in puzzle["train"]} if index < 2 else ()
                    lower, upper = base+2 if index < 2 else 4, 26
                else:
                    parity = (first_test_parity + index) % 2
                    excluded = ()
                    lower = (max(p["input"].shape[0] for p in puzzle["train"]) + 1
                             if regime == "canvas_growth" else 4)
                    upper = 30
                try:
                    grid = _place(rng, m, s, bg, sig, parity=parity,
                                  lower=lower, upper=upper, exclude=excluded)
                    info = _read_input(grid)
                except ValueError:
                    continue
                key = _key(grid)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({"input": grid,
                                      "output": _expand(m, _patch(info), bg)})
                break
            else:
                raise RuntimeError("bounded sampling could not construct a unique readable input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual_twins(train):
    for i, a in enumerate(train):
        ga = np.asarray(a["input"])
        for b in train[i+1:]:
            gb = np.asarray(b["input"])
            small, big = (ga, gb) if ga.shape[0] < gb.shape[0] else (gb, ga)
            if big.shape[0] != small.shape[0] + 1:
                continue
            bg = _read_input(small)["background"]
            if (np.array_equal(big[:-1, :-1], small) and np.all(big[-1] == bg)
                    and np.all(big[:, -1] == bg)
                    and not np.array_equal(a["output"], b["output"])):
                return True
    return False


def validate_generated_puzzle(puzzle):
    """Semantic, schema, and demonstration guards for generated puzzles."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "split keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            info = _read_input(grid)
            mask, patch, bg = info["mask"], _patch(info), info["background"]
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            _require(out.shape == (2*mask.shape[0], 2*mask.shape[1]), "output extent")
            _require(info["signal"] not in out, "mask signal leaked into output")
            # Every inactive block and every patch hole is checked literally.
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    expected = patch if mask[r, c] else np.full((2, 2), bg, dtype=np.int8)
                    _require(np.array_equal(out[2*r:2*r+2, 2*c:2*c+2], expected),
                             "incorrect active/inactive block or altered patch")
            _require(np.array_equal(solve(grid), out), "input-only solve disagreement")
            if split == "train":
                train_info.append(info)
    _require(_counterfactual_twins(puzzle["train"]), "missing parity-only counterfactuals")
    for parity in (0, 1):
        witnesses = [i for i in train_info if i["side"] % 2 == parity
                     and not np.array_equal(i["stamp"][:2, :2], i["stamp"][:2, -2:])]
        _require(bool(witnesses), "missing contrasting parity branch")
    _require(any(np.any(~i["mask"]) and i["mask"].sum() >= 2
                 and _asymmetric(i["mask"]) and _asymmetric(_patch(i))
                 and np.any(_patch(i) == i["background"]) for i in train_info),
             "need mask, orientation, and literal-hole witnesses")
    if n >= 4:
        _require(all(len({i["side"] for i in train_info if i["side"] % 2 == p}) >= 2
                     for p in (0, 1)), "need multiple sizes for each parity")
    if any(_components(_read_input(p["input"])["mask"]) > 1 for p in puzzle["test"]):
        _require(any(_components(i["mask"]) > 1 for i in train_info), "unintroduced islands")


def self_test():
    """Bounded streaming suite: 42 seeds, all count settings, causal checks.

    Returns measurements; writes no files and retains no grid corpus.
    """
    fingerprints, mask_shapes, stamp_shapes, sides, backgrounds = set(), set(), set(), set(), set()
    mask_hashes, patch_hashes, layouts, test_parities = set(), set(), set(), set()
    counts = {"pairs": 0, "islands": 0, "diagonal_only": 0, "full_masks": 0,
              "patch_holes": 0, "equal_corners": 0, "parity_counterfactuals": 0,
              "fixed_left_errors": 0, "fixed_right_errors": 0, "crop_only_errors": 0}
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed nondeterminism")
        fingerprints.add(_digest(puzzle))
        counts["parity_counterfactuals"] += int(_counterfactual_twins(puzzle["train"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                mask, patch = info["mask"], _patch(info)
                mb, sb = info["mask_box"], info["stamp_box"]
                layout = ("left" if mb[3] < sb[2] else "right") if (mb[3] < sb[2] or sb[3] < mb[2]) else ("above" if mb[1] < sb[0] else "below")
                layouts.add(layout)
                sides.add(info["side"])
                backgrounds.add(info["background"])
                mask_shapes.add(mask.shape)
                stamp_shapes.add(info["stamp"].shape)
                mask_hashes.add(hashlib.sha256(bytes(mask.shape) + mask.tobytes()).hexdigest())
                role_map = {info["background"]: 0}
                labels = bytes(role_map.setdefault(int(v), len(role_map)) for v in patch.flat)
                patch_hashes.add(labels)
                counts["pairs"] += 1
                counts["islands"] += int(_components(mask) > 1)
                counts["diagonal_only"] += int(_components(mask) == 1 and _components(mask, False) > 1)
                counts["full_masks"] += int(mask.all())
                counts["patch_holes"] += int(np.any(patch == info["background"]))
                counts["equal_corners"] += int(np.array_equal(info["stamp"][:2, :2], info["stamp"][:2, -2:]))
                if split == "train":
                    counts["fixed_left_errors"] += int(not np.array_equal(
                        _expand(mask, info["stamp"][:2, :2], info["background"]), pair["output"]))
                    counts["fixed_right_errors"] += int(not np.array_equal(
                        _expand(mask, info["stamp"][:2, -2:], info["background"]), pair["output"]))
                    counts["crop_only_errors"] += int(not np.array_equal(patch, pair["output"]))
                if split == "test":
                    test_parities.add(info["side"] % 2)
        # Relabeling all ten colors must preserve the geometric operation.
        pair = puzzle["test"][0]
        permutation = np.random.default_rng(seed + 1).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "literal color IDs affect the rule")
    _require(len(fingerprints) == 42, "42-seed puzzle diversity")
    _require(counts["parity_counterfactuals"] == 42, "causal parity evidence")
    _require(len(mask_shapes) >= 25 and len(stamp_shapes) >= 20 and len(mask_hashes) >= 75,
             "geometric diversity collapsed")
    _require(len(sides) >= 12 and len(backgrounds) == 10 and len(patch_hashes) >= 10,
             "size, role, or crop diversity collapsed")
    _require(layouts == {"left", "right", "above", "below"} and test_parities == {0, 1},
             "layout/parity diversity collapsed")
    _require(all(counts[k] > 0 for k in ("islands", "diagonal_only", "full_masks", "equal_corners")),
             "structural subfamilies absent")
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(123456, a, b))
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    # A validator must reject a wrong label, not merely trust the generator.
    damaged = generate_puzzle(0, 2, 1)
    out = damaged["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"valid": True, "seeds": len(SELF_TEST_SEEDS), "distinct_puzzles": len(fingerprints),
            "legal_count_combinations": legal_calls, "mask_shapes": len(mask_shapes),
            "stamp_shapes": len(stamp_shapes), "mask_patterns": len(mask_hashes),
            "color_normalized_patches": len(patch_hashes), "canvas_sides": sorted(sides),
            "background_colors": sorted(backgrounds), "layouts": sorted(layouts),
            "test_parities": sorted(test_parities), **counts}
