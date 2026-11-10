"""Generate ARC puzzles in which a detached glyph chooses a target's paint."""

import hashlib
import numpy as np


_ORIGINAL_GLYPHS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
)
# SHA-256 of the six original 14x14 input/output byte pairs. These protect
# exact source examples without excluding their dimensions or palette.
_SOURCE_PAIR_DIGESTS = frozenset((
    "015441178238fde0b8b4bcfb753b85141a193787126c95a3a289284df3d88cb9",
    "a14db2313e7f44750a2469d227853c0ac7174de799c11ce9444d2b5594bafccf",
    "aacca784fefbaa82f6405fa6dbef7325bfe69b4332e322b85d53a0c51e587a0e",
    "39e67d9ca31a2b3d5479fb4c924c156bef7c76bf054f285c3875df977c4eb491",
    "f9f5a3a0a8241c9fd3dba5e7a0dd3cd911d3e5e36e4ec84145c43551eaf3c661",
    "505943cda6b920dc3f58267573dbb6e0ae225f0a13042e1b1661d768f1072cd6",
))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _signature(mask):
    return (tuple(mask.shape), np.asarray(mask, dtype=bool).tobytes())


def _crop(mask):
    points = np.argwhere(mask)
    assert len(points)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0)
    return mask[top:bottom + 1, left:right + 1].copy(), (int(top), int(bottom), int(left), int(right))


def _new_glyph(rng, used):
    dimensions = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
    for _ in range(100):
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        mask = rng.random((h, w)) < float(rng.uniform(0.39, 0.66))
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not 5 <= int(mask.sum()) <= 12:
            continue
        if not _eight_connected(mask):
            continue
        sig = _signature(mask)
        if sig not in used:
            return mask
    # A deterministic, still distinct fallback for exceptionally unlucky draws.
    for h, w in dimensions:
        for k in range(1, h * w - 1):
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            if 5 <= int(mask.sum()) <= 12 and _signature(mask) not in used:
                return mask
    raise RuntimeError("could not construct a distinct glyph")


def _vocabulary(rng, train_count):
    n = int(rng.integers(2, min(4, train_count) + 1))
    if n == 3 and rng.random() < 0.28:
        return [g.copy() for g in _ORIGINAL_GLYPHS]
    glyphs = []
    used = set()
    for _ in range(n):
        glyph = _new_glyph(rng, used)
        glyphs.append(glyph)
        used.add(_signature(glyph))
    return glyphs


def _training_schedule(rng, count, classes):
    schedule = list(range(classes))
    while len(schedule) < count:
        if rng.random() < 0.5:
            schedule.extend(int(i) for i in rng.permutation(classes))
        else:
            schedule.append(int(rng.integers(classes)))
    schedule = schedule[:count]
    return [schedule[int(i)] for i in rng.permutation(count)]


def _test_schedule(rng, count, classes):
    schedule = []
    while len(schedule) < count:
        schedule.extend(int(i) for i in rng.permutation(classes))
    return schedule[:count]


def _dimensions(rng, regime):
    # Both splits can use every size family; tests favor less common larger shapes.
    weights = (18, 16, 18, 18, 13, 17) if regime == "source_train" else (9, 20, 20, 17, 23, 11)
    mode = int(rng.choice(6, p=np.array(weights) / sum(weights)))
    if mode == 0:
        return int(rng.integers(3, 7)), int(rng.integers(6, 11))
    if mode == 1:
        return int(rng.integers(10, 23)), int(rng.integers(5, 10))
    if mode == 2:
        return int(rng.integers(3, 10)), int(rng.integers(11, 23))
    if mode == 3:
        side = int(rng.integers(7, 18))
        return max(3, min(22, side + int(rng.integers(-2, 3)))), max(5, min(22, side + int(rng.integers(-2, 3))))
    if mode == 4:
        return int(rng.integers(14, 23)), int(rng.integers(14, 23))
    return int(rng.integers(5, 21)), int(rng.integers(6, 21))


def _target_mask(rng, h, w, style):
    mask = np.zeros((h, w), dtype=bool)
    if style == 0:  # branched diagonal strokes
        col = int(rng.integers(w))
        for r in range(h):
            mask[r, col] = True
            col = max(0, min(w - 1, col + int(rng.integers(-1, 2))))
        for _ in range(max(2, (h + w) // 3)):
            starts = np.argwhere(mask)
            r, c = (int(x) for x in starts[int(rng.integers(len(starts)))])
            length = int(rng.integers(2, max(3, min(h, w) + 1)))
            dr, dc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            for j in range(length):
                rr, cc = r + j * dr, c + j * dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True
    elif style == 1:  # frame, with independently placed interior strokes
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < 0.5:
                mask[int(rng.integers(h)), int(rng.integers(w)):int(rng.integers(w))] = True
            else:
                c = int(rng.integers(w))
                a, b = sorted((int(rng.integers(h)), int(rng.integers(h))))
                mask[a:b + 1, c] = True
        if h > 4 and w > 5:
            holes = rng.random((h - 2, w - 2)) < 0.16
            mask[1:-1, 1:-1] |= holes
    elif style == 2:  # repeated bars and crossbars
        step_r = int(rng.integers(2, 5))
        step_c = int(rng.integers(2, 5))
        mask[int(rng.integers(step_r))::step_r, :] = True
        mask[:, int(rng.integers(step_c))::step_c] = True
        mask &= rng.random((h, w)) < float(rng.uniform(0.89, 1.0))
    elif style == 3:  # sparse repeated motifs and isolated fragments
        mask = rng.random((h, w)) < float(rng.uniform(0.21, 0.4))
        for r in range(0, h, int(rng.integers(2, 5))):
            c = int(rng.integers(w))
            mask[r, c:min(w, c + int(rng.integers(2, 5)))] = True
    elif style == 4:  # solid masks are a valid special case
        mask[:] = True
    elif style == 5:  # irregular outline
        left = int(rng.integers(0, max(1, w // 4)))
        right = int(rng.integers(max(left + 1, 3 * w // 4), w))
        for r in range(h):
            l = max(0, min(w - 2, left + int(rng.integers(-1, 2))))
            rr = max(l + 1, min(w - 1, right + int(rng.integers(-1, 2))))
            mask[r, l] = mask[r, rr] = True
            if r in (0, h - 1) or rng.random() < 0.17:
                mask[r, l:rr + 1] = True
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(h))
            mask[r, min(w - 1, left):max(left + 1, right)] = True
    else:  # connected-ish blob and internal holes
        center = (h // 2, w // 2)
        mask[center] = True
        for end in ((0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                    (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)):
            r, c = center
            while (r, c) != end:
                old_r, old_c = r, c
                if r != end[0] and (c == end[1] or rng.random() < 0.65):
                    r += 1 if end[0] > r else -1
                if c != end[1] and (r == end[0] or rng.random() < 0.65):
                    c += 1 if end[1] > c else -1
                if (r, c) == (old_r, old_c):
                    if r != end[0]:
                        r += 1 if end[0] > r else -1
                    else:
                        c += 1 if end[1] > c else -1
                mask[r, c] = True
        for _ in range(2):
            frontier = np.zeros((h, w), dtype=bool)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    src_r = slice(max(0, -dr), min(h, h - dr))
                    src_c = slice(max(0, -dc), min(w, w - dc))
                    dst_r = slice(max(0, dr), min(h, h + dr))
                    dst_c = slice(max(0, dc), min(w, w + dc))
                    frontier[dst_r, dst_c] |= mask[src_r, src_c]
            mask |= frontier & (rng.random((h, w)) < 0.22)
    # Ensure that the chosen dimensions are the actual tight target bounds.
    for border in (0, h - 1):
        if not mask[border].any():
            points = np.argwhere(mask)
            r, c = (int(x) for x in points[int(rng.integers(len(points)))])
            if style == 3:
                mask[border, c] = True
            else:
                mask[min(r, border):max(r, border) + 1, c] = True
    for border in (0, w - 1):
        if not mask[:, border].any():
            points = np.argwhere(mask)
            r, c = (int(x) for x in points[int(rng.integers(len(points)))])
            if style == 3:
                mask[r, border] = True
            else:
                mask[r, min(c, border):max(c, border) + 1] = True
    return mask


def _palette(rng, paints, previous=None):
    available = [c for c in range(10) if c not in paints]
    for _ in range(30):
        colors = [int(x) for x in rng.choice(available, size=3, replace=False)]
        if previous is None or all(colors[i] != previous[i] for i in range(3)):
            return colors
    # There are at least six colors left, so this always has three alternatives.
    return [c for c in available if c not in previous][:3]


def _place(rng, target, glyph, colors, paint):
    th, tw = target.shape
    gh, gw = glyph.shape
    for _ in range(30):
        direction = int(rng.integers(4))
        gap = int(rng.integers(1, 5))
        vertical = direction < 2
        core_h = th + gh + gap if vertical else max(th, gh)
        core_w = max(tw, gw) if vertical else tw + gw + gap
        if core_h > 30 or core_w > 30:
            continue
        extra_h = int(rng.integers(0, min(30 - core_h, 8) + 1))
        extra_w = int(rng.integers(0, min(30 - core_w, 8) + 1))
        height, width = core_h + extra_h, core_w + extra_w
        if 2 * (int(target.sum()) + int(glyph.sum())) >= height * width:
            continue
        if vertical:
            block_r = int(rng.integers(extra_h + 1))
            tr = block_r if direction == 0 else block_r + gh + gap
            gr = block_r + th + gap if direction == 0 else block_r
            tc = int(rng.integers(width - tw + 1))
            gc = int(rng.integers(width - gw + 1))
        else:
            block_c = int(rng.integers(extra_w + 1))
            tc = block_c if direction == 2 else block_c + gw + gap
            gc = block_c + tw + gap if direction == 2 else block_c
            tr = int(rng.integers(height - th + 1))
            gr = int(rng.integers(height - gh + 1))
        bg, ink, marker = colors
        inp = np.full((height, width), bg, dtype=np.int8)
        out = np.full((height, width), bg, dtype=np.int8)
        inp[tr:tr + th, tc:tc + tw][target] = ink
        out[tr:tr + th, tc:tc + tw][target] = paint
        inp[gr:gr + gh, gc:gc + gw][glyph] = marker
        return {"input": inp, "output": out}
    return None


def _source_copy(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape != (14, 14) or out.shape != (14, 14):
        return False
    digest = hashlib.sha256(inp.astype(np.int8).tobytes() + b"|" + out.astype(np.int8).tobytes()).hexdigest()
    return digest in _SOURCE_PAIR_DIGESTS


def _grid_signature(grid):
    return (tuple(grid.shape), grid.tobytes())


def _make_pair(rng, regime, glyph, paint, paints, previous_palette, seen, reuse=None, dimensions=None):
    for attempt in range(100):
        if reuse is None:
            h, w = dimensions if dimensions is not None else _dimensions(rng, regime)
            style = int(rng.integers(7))
            if style == 4 and h * w > 210:
                continue
            target = _target_mask(rng, h, w, style)
        else:
            target = reuse.copy()
        if int(target.sum()) < max(12, int(glyph.sum() * 1.7) + 1):
            continue
        if target.size <= 2 * glyph.size:
            continue
        colors = _palette(rng, paints, previous_palette)
        pair = _place(rng, target, glyph, colors, paint)
        if pair is None or _source_copy(pair):
            continue
        sig = _grid_signature(pair["input"])
        if sig in seen:
            continue
        seen.add(sig)
        return pair, target, colors
    raise RuntimeError("could not construct a unique valid pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(rng, seed, num_train, num_test)
    glyphs = _vocabulary(rng, ntrain)
    paints = [int(x) for x in rng.choice(np.arange(1, 10), size=len(glyphs), replace=False)]
    train_classes = _training_schedule(rng, ntrain, len(glyphs))
    test_classes = _test_schedule(rng, ntest, len(glyphs))
    repeat_index = None
    if rng.random() < (0.52 if ntrain >= 3 else 0.2):
        eligible = [i for i in range(1, ntrain) if any(train_classes[j] != train_classes[i] for j in range(i))]
        if eligible:
            repeat_index = int(rng.choice(eligible))
    train, test = [], []
    train_masks = []
    seen = set()
    previous_palette = None
    for i, cls in enumerate(train_classes):
        reuse = None
        if i == repeat_index:
            candidates = [j for j in range(i)
                          if train_classes[j] != cls
                          and int(train_masks[j].sum()) >= max(12, int(glyphs[cls].sum() * 1.7) + 1)
                          and train_masks[j].size > 2 * glyphs[cls].size]
            if candidates:
                reuse = train_masks[int(rng.choice(candidates))]
        try:
            pair, mask, previous_palette = _make_pair(rng, "source_train", glyphs[cls], paints[cls], paints, previous_palette, seen, reuse)
        except RuntimeError:
            if reuse is None:
                raise
            pair, mask, previous_palette = _make_pair(rng, "source_train", glyphs[cls], paints[cls], paints, previous_palette, seen)
        train.append(pair)
        train_masks.append(mask)
    train_configs = {(info["glyph"], info["target"]) for info in map(_analyze_pair, train)}
    for cls in test_classes:
        same_size = None
        if rng.random() < 0.3:
            eligible_sizes = [mask.shape for mask in train_masks
                              if mask.size > 2 * glyphs[cls].size]
            if eligible_sizes:
                same_size = eligible_sizes[int(rng.integers(len(eligible_sizes)))]
        for _ in range(20):
            pair, _, colors = _make_pair(rng, "target_test", glyphs[cls], paints[cls], paints, previous_palette, seen, dimensions=same_size)
            info = _analyze_pair(pair)
            if (info["glyph"], info["target"]) not in train_configs:
                previous_palette = colors
                break
        else:
            raise RuntimeError("could not construct a new test configuration")
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    in_colors = set(int(c) for c in np.unique(inp))
    out_colors = set(int(c) for c in np.unique(out))
    assert len(in_colors) == 3 and len(out_colors) == 2
    common = in_colors & out_colors
    assert len(common) == 1
    bg = next(iter(common))
    paint = next(iter(out_colors - common))
    roles = list(in_colors - common)
    roles.sort(key=lambda c: int(np.count_nonzero(inp == c)), reverse=True)
    target_color, glyph_color = roles
    target = inp == target_color
    marker = inp == glyph_color
    assert int(target.sum()) >= max(12, int(marker.sum() * 1.7) + 1)
    glyph, (gr0, gr1, gc0, gc1) = _crop(marker)
    target_crop, (tr0, tr1, tc0, tc1) = _crop(target)
    assert 3 <= glyph.shape[0] <= 5 and 3 <= glyph.shape[1] <= 5
    assert 5 <= int(glyph.sum()) <= 12
    assert _eight_connected(glyph)
    assert target_crop.size > 2 * glyph.size
    assert (tr1 + 1 < gr0 or gr1 + 1 < tr0 or tc1 + 1 < gc0 or gc1 + 1 < tc0)
    assert int(np.count_nonzero(inp == bg)) > inp.size // 2
    assert np.array_equal(out, np.where(target, paint, bg))
    assert not np.array_equal(inp, out)
    layout = ("above" if tr1 < gr0 else "below" if gr1 < tr0
              else "left" if tc1 < gc0 else "right")
    return {"glyph": _signature(glyph), "paint": paint, "background": bg,
            "target_color": target_color, "glyph_color": glyph_color,
            "target": _signature(target_crop), "height": target_crop.shape[0],
            "width": target_crop.shape[1], "target_cells": int(target.sum()),
            "glyph_cells": int(marker.sum()), "layout": layout,
            "edge": tr0 == 0 or tc0 == 0 or tr1 == inp.shape[0] - 1 or tc1 == inp.shape[1] - 1 or gr0 == 0 or gc0 == 0 or gr1 == inp.shape[0] - 1 or gc1 == inp.shape[1] - 1}


def _predict_from_demonstrations(train_pairs, test_input):
    mapping = {}
    for pair in train_pairs:
        info = _analyze_pair(pair)
        glyph = info["glyph"]
        if glyph in mapping:
            assert mapping[glyph] == info["paint"]
        mapping[glyph] = info["paint"]
    colors, counts = np.unique(test_input, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    others = [int(c) for c in colors if c != bg]
    assert len(others) == 2
    others.sort(key=lambda c: int(np.count_nonzero(test_input == c)), reverse=True)
    target = test_input == others[0]
    glyph = _signature(_crop(test_input == others[1])[0])
    assert glyph in mapping
    return np.where(target, mapping[glyph], bg)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_sig = _grid_signature(pair["input"])
            pair_sig = (input_sig, _grid_signature(pair["output"]))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            assert not _source_copy(pair)
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            infos[split].append(_analyze_pair(pair))
    mapping = {}
    for info in infos["train"]:
        if info["glyph"] in mapping:
            assert mapping[info["glyph"]] == info["paint"]
        mapping[info["glyph"]] = info["paint"]
    assert len(mapping) >= 2 and len(set(mapping.values())) == len(mapping)
    all_infos = infos["train"] + infos["test"]
    paints = set(mapping.values())
    for info in all_infos:
        assert info["glyph"] in mapping and mapping[info["glyph"]] == info["paint"]
        assert not ({info["background"], info["target_color"], info["glyph_color"]} & paints)
    # Every test configuration must differ from its demonstrations.
    train_configs = {(i["glyph"], i["target"], i["height"], i["width"]) for i in infos["train"]}
    for info, pair in zip(infos["test"], puzzle["test"]):
        assert (info["glyph"], info["target"], info["height"], info["width"]) not in train_configs
        assert np.array_equal(_predict_from_demonstrations(puzzle["train"], pair["input"]), pair["output"])


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        p = generate_puzzle(seed, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for seed, kwargs in ((1101, {"num_test": 1}), (1102, {"num_test": 4}),
                         (1103, {"num_train": 2}), (1104, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(seed, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": 1.5}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    target_shapes, glyph_shapes, target_masks, layouts = set(), set(), set(), set()
    roles, vocabularies, fingerprints = set(), set(), set()
    vocabulary_counts = set()
    edge_count = solid_count = fragmented_count = repeated_count = 0
    connected_count = sparse_count = canvas_30_count = 0
    zero_bg = zero_target = zero_glyph = target_8 = 0
    repeated_puzzles = varied_target_for_glyph = 0
    multiple_test_classes = test_not_least_frequent = 0
    new_test_size = reused_test_size = combined_axes = 0
    test_classes, test_heights = set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        infos = [_analyze_pair(x) for s in ("train", "test") for x in p[s]]
        train_infos = infos[:len(p["train"])]
        test_infos = infos[len(p["train"]):]
        fingerprints.add(tuple(_grid_signature(x["input"]) for s in ("train", "test") for x in p[s]))
        vocabularies.add(tuple(sorted(i["glyph"] for i in train_infos)))
        vocabulary_counts.add(len({i["glyph"] for i in train_infos}))
        repeated_count += len(train_infos) - len({i["target"] for i in train_infos})
        repeated_puzzles += any(a["target"] == b["target"] and a["paint"] != b["paint"]
                                for j, a in enumerate(train_infos) for b in train_infos[j + 1:])
        varied_target_for_glyph += any(a["glyph"] == b["glyph"] and a["target"] != b["target"]
                                       for j, a in enumerate(train_infos) for b in train_infos[j + 1:])
        train_sizes = {(i["height"], i["width"]) for i in train_infos}
        new_test_size += any((i["height"], i["width"]) not in train_sizes for i in test_infos)
        reused_test_size += any((i["height"], i["width"]) in train_sizes for i in test_infos)
        multiple_test_classes += len({i["glyph"] for i in test_infos}) > 1
        frequencies = {g: sum(i["glyph"] == g for i in train_infos)
                       for g in {i["glyph"] for i in train_infos}}
        test_not_least_frequent += any(frequencies[i["glyph"]] > min(frequencies.values())
                                       for i in test_infos)
        combined_axes += len({i["background"] for i in infos}) > 1 and any(
            i["edge"] and not _four_connected(np.frombuffer(i["target"][1], dtype=bool).reshape(i["target"][0]))
            for i in infos)
        for i, pair in zip(infos, p["train"] + p["test"]):
            target_shapes.add((i["height"], i["width"]))
            glyph_shapes.add(i["glyph"][0])
            target_masks.add(i["target"])
            roles.add((i["background"], i["target_color"], i["glyph_color"]))
            layouts.add(i["layout"])
            edge_count += i["edge"]
            solid_count += i["target_cells"] == i["height"] * i["width"]
            sparse_count += i["target_cells"] * 5 < 2 * i["height"] * i["width"]
            canvas_30_count += 30 in pair["input"].shape
            zero_bg += i["background"] == 0
            zero_target += i["target_color"] == 0
            zero_glyph += i["glyph_color"] == 0
            target_8 += i["target_color"] == 8
            crop = np.frombuffer(i["target"][1], dtype=bool).reshape(i["target"][0])
            fragmented = not _four_connected(crop)
            fragmented_count += fragmented
            connected_count += not fragmented
        for i in test_infos:
            test_classes.add(i["glyph"])
            test_heights.add(i["height"])
        assert len({i["background"] for i in infos}) >= 2
        assert len({i["target_color"] for i in infos}) >= 2
        assert len({i["glyph_color"] for i in infos}) >= 2
    assert len(fingerprints) == 42
    assert len(target_shapes) >= 70 and len(target_masks) >= 100
    assert len(glyph_shapes) >= 4 and len(vocabularies) >= 15
    assert vocabulary_counts == {2, 3, 4}
    assert len(roles) >= 40 and edge_count >= 15
    assert solid_count >= 3 and fragmented_count >= 20 and repeated_count >= 4
    assert connected_count >= 15 and sparse_count >= 15 and canvas_30_count >= 3
    assert min(zero_bg, zero_target, zero_glyph, target_8) >= 3
    assert layouts == {"above", "below", "left", "right"}
    assert repeated_puzzles >= 4 and varied_target_for_glyph >= 8
    assert multiple_test_classes >= 5 and test_not_least_frequent >= 5
    assert new_test_size >= 5 and reused_test_size >= 4 and combined_axes >= 10
    assert len(test_classes) >= 10 and len(test_heights) >= 8


def _connected(mask, diagonal):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(x) for x in points[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        offsets = ((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1))
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _four_connected(mask):
    return _connected(mask, False)


def _eight_connected(mask):
    return _connected(mask, True)
