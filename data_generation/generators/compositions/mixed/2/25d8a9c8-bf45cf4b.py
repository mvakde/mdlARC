"""Whole-row equality controls exact, two-scale stamp substitution.

The framed rectangle is data, including its background-valued rows. The other
rectangle is the stamp. No random state or demonstrations are used by solve.
Only standard-library modules and NumPy are used; generation writes no files.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 191494586, 402648957, 662157425, 254525666, 75504858,
              918921676, 976363275, 339501883, 61022099, 28839913, 995540801,
              267821878, 719458703, 140640340, 147633040, 625389010, 859582377,
              586330913, 901060857, 869897983, 432422405, 637677476, 646891098,
              923440234, 795108405, 935376802, 575751252, 216802868, 435637646,
              776656249, 164353670, 438743779, 703862474, 522483712, 442597432,
              685818130, 806973544, 857332739, 958679880, 26635234, 174066135)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or array.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in array.shape)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return array


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _box(binary):
    r, c = np.nonzero(binary)
    if not len(r):
        raise ValueError("missing visible encoding")
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def _crop(grid, box):
    r0, r1, c0, c1 = box
    return grid[r0:r1, c0:c1].copy()


def _separated(a, b):
    # Exclusive ends: equality leaves no blank line, hence strict inequalities.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read_input(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    rim = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if not np.all(rim == bg):
        raise ValueError("the outer rim must identify one background color")
    readings = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        box = _box(grid == color)
        r0, r1, c0, c1 = box
        h, w = r1 - r0, c1 - c0
        if min(h, w) < 3:
            continue
        perimeter = np.ones((h, w), dtype=bool)
        perimeter[1:-1, 1:-1] = False
        if not np.array_equal(_crop(grid, box) == color, perimeter):
            continue
        outside = grid != bg
        outside[r0:r1, c0:c1] = False
        if not np.any(outside):
            continue
        sb = _box(outside)
        if not _separated(box, sb):
            continue
        mask_data = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1].copy()
        stamp = _crop(grid, sb)
        if len(set(map(int, np.unique(stamp))) - {bg}) < 2:
            continue
        if (mask_data.shape[0] * stamp.shape[0] > 30
                or mask_data.shape[1] * stamp.shape[1] > 30):
            continue
        readings.append({"background": bg, "frame": color, "mask": mask_data,
                         "stamp": stamp, "mask_box": box, "stamp_box": sb})
    if len(readings) != 1:
        raise ValueError("input must have exactly one framed-mask/separate-stamp reading")
    return readings[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read visible encodings, classify COMPLETE rows, and expand their cells."""
    info = _read_input(input_grid)
    mask, stamp, bg = info["mask"], info["stamp"], info["background"]
    active = np.all(mask == mask[:, :1], axis=1)
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(active[rr // sh], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def _expand(active, stamp, bg):
    """Generator/validator block writes, distinct from solve's pixel indexing."""
    mh, mw = active.shape
    sh, sw = stamp.shape
    output = np.full((mh * sh, mw * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(active):
        output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = stamp
    return output


def _row_activity(mask):
    return np.repeat(np.all(mask == mask[:, :1], axis=1)[:, None], mask.shape[1], axis=1)


def _transforms(array):
    return [f(np.rot90(array, k)) for k in range(4)
            for f in (lambda a: a, np.fliplr)]


def _asymmetric(array):
    return all(not np.array_equal(array, a) for a in _transforms(array)[1:])


def _mixed_row(rng, width, palette):
    a, b = map(int, rng.choice(palette, 2, replace=False))
    style = int(rng.integers(5))
    if style == 0:
        row = np.full(width, a, dtype=np.int8)
        row[int(rng.integers(width))] = b
    elif style == 1:
        row = np.array([a if i % 2 else b for i in range(width)], dtype=np.int8)
    elif style == 2:
        cut = int(rng.integers(1, width))
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
    elif style == 3:
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, width), dtype=np.int8)
        row[0], row[-1] = a, b
    return row


def _make_mask(rng, shape, palette, bg, witness=0):
    h, w = shape
    non_bg = [int(x) for x in palette if int(x) != bg]
    a, b = map(int, rng.choice(non_bg, 2, replace=False))
    if witness == 1:
        # Every column is indispensable: deleting column j makes its negative
        # row uniform. Both colors also occur in positive rows.
        mask = np.full((w + 2, w), a, dtype=np.int8)
        mask[1] = b
        for j in range(w):
            mask[j + 2, j] = b
        for _ in range(32):
            order = rng.permutation(h)
            states = np.array([True, True] + [False] * w)[order]
            if not np.array_equal(states, states[::-1]):
                return mask[order].copy()
        return mask
    family = int(rng.integers(6))
    if family == 0:
        states = np.arange(h) % 2 == int(rng.integers(2))
    elif family == 1:
        start, stop = sorted(map(int, rng.choice(h + 1, 2, replace=False)))
        states = (np.arange(h) >= start) & (np.arange(h) < stop)
    elif family == 2:
        states = np.ones(h, dtype=bool)
    elif family == 3:
        states = np.zeros(h, dtype=bool)
    else:
        states = rng.random(h) < float(rng.choice([.25, .5, .75]))
    mask = np.empty(shape, dtype=np.int8)
    for i, positive in enumerate(states):
        mask[i] = int(rng.choice(palette)) if positive else _mixed_row(rng, w, palette)
    if witness == 2:
        mask[0] = bg  # A uniform background row is data, not an empty margin.
        mask[1] = a
        mask[2] = _mixed_row(rng, w, np.array([a, b], dtype=np.int8))
        rng.shuffle(mask, axis=0)
    return mask


def _make_stamp(rng, shape, palette, bg, witness=False):
    h, w = shape
    rr, cc = np.indices(shape)
    family = 5 if witness else int(rng.integers(6))
    hole_kind = 5 if witness else int(rng.integers(6))
    for _ in range(128):
        k = len(palette)
        if family == 0:
            labels = rr if rng.random() < .5 else cc
        elif family == 1:
            labels = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
        elif family == 2:
            labels = rr + int(rng.integers(1, k + 1)) * cc
        elif family == 3:
            tile = rng.integers(k, size=(int(rng.integers(1, min(3, h) + 1)),
                                         int(rng.integers(1, min(3, w) + 1))))
            labels = tile[rr % tile.shape[0], cc % tile.shape[1]]
        elif family == 4:
            labels = np.abs(2 * rr - h + 1) + np.abs(2 * cc - w + 1)
        else:
            labels = rng.integers(k, size=shape)
        stamp = palette[(labels + int(rng.integers(k))) % k].astype(np.int8)
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        boundary = [(r, c) for r in range(h) for c in range(w)
                    if r in (0, h - 1) or c in (0, w - 1)]
        if interior and hole_kind in (1, 2, 3, 5):
            if hole_kind == 3:
                r0 = int(rng.integers(1, h - 1))
                c0 = int(rng.integers(1, w - 1))
                stamp[r0:int(rng.integers(r0 + 1, h)),
                      c0:int(rng.integers(c0 + 1, w))] = bg
            else:
                count = 1 if hole_kind in (1, 5) else int(rng.integers(1, len(interior) + 1))
                for j in rng.choice(len(interior), count, replace=False):
                    stamp[interior[int(j)]] = bg
        if hole_kind in (4, 5):
            stamp[boundary[int(rng.integers(len(boundary)))]] = bg
        fg = stamp != bg
        if not all(np.any(side) for side in (fg[0], fg[-1], fg[:, 0], fg[:, -1])):
            continue
        present = set(map(int, np.unique(stamp))) - {bg}
        if len(present) < 2:
            # Holes can erase the only contrasting ring/band. A small visible
            # accent preserves the multicolor role without advertising a row.
            sites = np.flatnonzero(fg)
            contrasting = [c for c in palette if int(c) not in present]
            stamp.flat[int(rng.choice(sites))] = int(rng.choice(contrasting))
        if witness and not _asymmetric(stamp):
            continue
        return stamp
    raise RuntimeError("bounded stamp construction failed")


def _layout(rng, mask, stamp, bg, frame):
    panel = np.full((mask.shape[0] + 2, mask.shape[1] + 2), frame, dtype=np.int8)
    panel[1:-1, 1:-1] = mask
    ph, pw = panel.shape
    sh, sw = stamp.shape
    horizontal, reverse = map(bool, rng.integers(2, size=2))
    gap = int(rng.integers(1, 5))
    base_h = max(ph, sh) + 2 if horizontal else ph + sh + gap + 2
    base_w = pw + sw + gap + 2 if horizontal else max(pw, sw) + 2
    h = base_h + int(rng.integers(min(30 - base_h, 6) + 1))
    w = base_w + int(rng.integers(min(30 - base_w, 6) + 1))
    if horizontal:
        start = int(rng.integers(1, w - pw - sw - gap))
        pc, sc = (start + sw + gap, start) if reverse else (start, start + pw + gap)
        pr, sr = int(rng.integers(1, h - ph)), int(rng.integers(1, h - sh))
    else:
        start = int(rng.integers(1, h - ph - sh - gap))
        pr, sr = (start + sh + gap, start) if reverse else (start, start + ph + gap)
        pc, sc = int(rng.integers(1, w - pw)), int(rng.integers(1, w - sw))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[pr:pr + ph, pc:pc + pw] = panel
    grid[sr:sr + sh, sc:sc + sw] = stamp
    return grid


def _shapes(rng, mode, max_width, split, index):
    if split == "train" and index == 0:
        mask = max_width + 2, max_width
        if mode == "stamp_growth":
            return mask, (3, 3)
        return mask, (int(rng.integers(3, min(5, 30 // mask[0]) + 1)),
                      int(rng.integers(3, min(5, 30 // mask[1]) + 1)))
    if split == "test" and mode == "stamp_growth":
        stamp = tuple(map(int, rng.integers(4, 8, size=2)))
        mask = (int(rng.integers(2, min(8, 30 // stamp[0]) + 1)),
                int(rng.integers(2, min(max_width, 30 // stamp[1]) + 1)))
        return mask, stamp
    if split == "test" and mode == "mask_growth":
        h = int(rng.integers(max_width + 3, 11))
    else:
        upper = max_width + 2 if mode == "mask_growth" else 8
        h = int(rng.integers(3 if split == "train" and index == 1 else 2, upper + 1))
    w = max_width - 1 if split == "train" and index == 1 else int(rng.integers(2, max_width + 1))
    upper_stamp = 3 if mode == "stamp_growth" else 8
    stamp = (int(rng.integers(2, min(upper_stamp, 30 // h) + 1)),
             int(rng.integers(2, min(upper_stamp, 30 // w) + 1)))
    if split == "train" and index == 1 and mode == "stamp_growth" and stamp == (3, 3):
        stamp = (2, 3)
    return (h, w), stamp


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["mask_growth", "stamp_growth", "recombination"]))
    max_width = int(rng.integers(3, 9 if mode == "recombination" else 6))
    stable = bool(rng.random() < .25)
    base_palette = rng.permutation(10)
    puzzle = {"train": [], "test": []}
    used, train_masks, test_masks = set(), set(), set()
    first_stamp_shape = None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            for _ in range(64):
                colors = base_palette if stable else rng.permutation(10)
                bg, frame = map(int, colors[:2])
                mask_palette = np.array([bg, *colors[2:int(rng.integers(4, 8))]], dtype=np.int8)
                payload = np.array(colors[2:int(rng.integers(4, 8))], dtype=np.int8)
                mshape, sshape = _shapes(rng, mode, max_width, split, index)
                if split == "train" and index == 1 and sshape == first_stamp_shape:
                    continue
                witness = index + 1 if split == "train" and index < 2 else 0
                mask = _make_mask(rng, mshape, mask_palette, bg, witness)
                token = mask.shape, tuple(map(bool, np.all(mask == mask[:, :1], axis=1)))
                if split == "test" and token in train_masks | test_masks:
                    continue
                stamp = _make_stamp(rng, sshape, payload, bg, witness == 1)
                grid = _layout(rng, mask, stamp, bg, frame)
                try:
                    info = _read_input(grid)
                except ValueError:
                    continue
                if not np.array_equal(info["mask"], mask) or not np.array_equal(info["stamp"], stamp):
                    continue
                key = _key(grid)
                if key in used:
                    continue
                used.add(key)
                (train_masks if split == "train" else test_masks).add(token)
                output = _expand(_row_activity(mask), stamp, bg)
                puzzle[split].append({"input": grid, "output": output})
                if split == "train" and index == 0:
                    first_stamp_shape = stamp.shape
                break
            else:
                raise RuntimeError("bounded unique-pair construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _singleton_columns(mask):
    columns = set()
    if mask.shape[1] < 3:
        return columns
    for row in mask:
        colors, counts = np.unique(row, return_counts=True)
        if len(colors) == 2 and sorted(map(int, counts)) == [1, len(row) - 1]:
            exceptional = colors[np.argmin(counts)]
            columns.add(int(np.flatnonzero(row == exceptional)[0]))
    return columns


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            info = _read_input(grid)
            mask, stamp, bg = info["mask"], info["stamp"], info["background"]
            active = _row_activity(mask)
            _require(np.array_equal(output, _expand(active, stamp, bg)), "incorrect block output")
            _require(info["frame"] not in output, "frame leaked to output")
            _require(set(map(int, np.unique(output))) <= set(map(int, np.unique(stamp))) | {bg},
                     "new output colors")
            _require(not np.array_equal(grid, output), "identity pair")
            info["active"] = active
            info["token"] = mask.shape, tuple(map(bool, active[:, 0]))
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    widest = max(i["mask"].shape[1] for i in train + test)
    _require(any(i["mask"].shape[1] == widest and
                 _singleton_columns(i["mask"]) == set(range(widest)) for i in train),
             "training must expose an exception in EVERY column of the widest mask")
    _require(any(np.any(np.all(i["mask"] == i["background"], axis=1)) for i in train),
             "missing uniform-background-row demonstration")
    _require(any(not np.array_equal(i["active"][:, 0], i["active"][::-1, 0]) for i in train),
             "missing vertical orientation evidence")
    _require(any(_asymmetric(i["stamp"]) for i in train), "missing unchanged stamp orientation evidence")
    _require(any(np.any(i["stamp"][1:-1, 1:-1] == i["background"]) for i in train),
             "missing internal stamp hole evidence")
    _require(any(any(np.any(s == i["background"]) for s in
                         (i["stamp"][0], i["stamp"][-1], i["stamp"][:, 0], i["stamp"][:, -1]))
                 for i in train), "missing tight-crop edge-gap evidence")
    _require(len({i["mask"].shape for i in train}) >= 2, "training mask sizes must vary")
    _require(len({i["stamp"].shape for i in train}) >= 2, "training stamp sizes must vary")
    _require(any(set(map(int, np.unique(i["mask"][i["active"][:, 0]]))) &
                 set(map(int, np.unique(i["mask"][~i["active"][:, 0]]))) for i in train),
             "no color shared by positive and negative rows")
    _require(any(not np.array_equal(_expand(i["mask"] != i["background"], i["stamp"], i["background"]),
                                   _expand(i["active"], i["stamp"], i["background"])) for i in train),
             "omitting parent A never changes a demonstration")
    _require(any(i["stamp"].size > 1 and np.any(i["active"]) for i in train),
             "omitting parent B never changes a demonstration")
    train_tokens = {i["token"] for i in train}
    test_tokens = [i["token"] for i in test]
    _require(not train_tokens.intersection(test_tokens) and len(set(test_tokens)) == len(test_tokens),
             "tests must have held-out, distinct coarse arrangements")
    return None


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _pattern(array):
    roles = {}
    return array.shape, tuple(roles.setdefault(int(c), len(roles)) for c in array.flat)


def self_test():
    """Stream 42 seeds and all legal counts; retain only bounded feature sets."""
    fingerprints, mask_patterns, stamp_patterns, shapes = set(), set(), set(), set()
    backgrounds, frame_colors, layouts, activities = set(), set(), set(), set()
    extreme = {False: 0, True: 0}
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _read_input(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                mask, stamp = info["mask"], info["stamp"]
                mask_patterns.add(_pattern(mask))
                stamp_patterns.add(_pattern(stamp))
                shapes.add(pair["output"].shape)
                backgrounds.add(info["background"])
                frame_colors.add(info["frame"])
                states = tuple(map(bool, np.all(mask == mask[:, :1], axis=1)))
                activities.add(states)
                extreme[True] += all(states)
                extreme[False] += not any(states)
                mb, sb = info["mask_box"], info["stamp_box"]
                layouts.add("left" if mb[3] < sb[2] else "right" if sb[3] < mb[2]
                            else "above" if mb[1] < sb[0] else "below")
    _require(len(fingerprints) == 42, "different seeds repeated puzzles")
    _require(len(mask_patterns) >= 100 and len(stamp_patterns) >= 100, "weak structural variation")
    _require(len(shapes) >= 50 and len(activities) >= 40, "weak geometric/row variation")
    _require(len(backgrounds) == len(frame_colors) == 10, "palette roles do not vary enough")
    _require(len(layouts) == 4 and min(extreme.values()) > 0, "missing layouts or extreme row masks")
    original = generate_puzzle(0)
    _require(_fingerprint(original) == _fingerprint(generate_puzzle(0)), "nondeterministic seed")
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "wrong counts")
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    pair = original["train"][0]
    info = _read_input(pair["input"])
    permutation = np.array([7, 2, 9, 0, 6, 1, 4, 8, 5, 3], dtype=np.int8)
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
             "solve uses literal colors")
    # Local causal checks: one equality change must toggle exactly one stamp band.
    witness = next(p for p in original["train"]
                   if _singleton_columns(_read_input(p["input"])["mask"]))
    info = _read_input(witness["input"])
    mask, stamp, bg = info["mask"], info["stamp"], info["background"]
    positive = int(np.flatnonzero(np.all(mask == mask[:, :1], axis=1))[0])
    changed = witness["input"].copy()
    mb = info["mask_box"]
    alternate = next(c for c in range(10) if c not in (info["frame"], int(mask[positive, 0])))
    changed[mb[0] + 1 + positive, mb[2] + 1] = alternate
    expected = witness["output"].copy()
    expected[positive * stamp.shape[0]:(positive + 1) * stamp.shape[0]] = bg
    _require(np.array_equal(solve(changed), expected), "equality change did not control expansion")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    output = damaged["test"][0]["output"]
    output[0, 0] = (int(output[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": pairs, "legal_count_combinations": legal,
            "distinct_puzzles": len(fingerprints), "mask_structures": len(mask_patterns),
            "stamp_structures": len(stamp_patterns), "output_shapes": len(shapes),
            "row_activity_patterns": len(activities), "background_colors": len(backgrounds),
            "frame_colors": len(frame_colors), "layouts": sorted(layouts),
            "all_positive_pairs": extreme[True], "all_negative_pairs": extreme[False]}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
