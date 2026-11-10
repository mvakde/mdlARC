"""Measure empty rectangles in logical cells; paint them in original pixels.

No files, parent modules, or generation state are used by solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


BACKGROUND = 0
HIGHLIGHT = 6
BLOCKERS = (1, 2, 3, 4, 5, 7, 8, 9)
TARGET_SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6))
HOST_SEEDS = (0, 192015362, 103930447, 695144072, 3111565, 507397271,
              434884587, 206899655, 388347571, 444149161, 686658249,
              164246794, 711271647, 318175823, 566954515, 923865765,
              832167147, 360593501, 756411842, 799554887, 235196533,
              293850813, 886546217, 586684556, 263593082, 664873076,
              205401418, 755007945, 101054715, 245985612, 284021883,
              210829971, 631585667, 813588585, 255465342, 90582167,
              943334004, 451513863, 394408703, 338239255, 679600966,
              760930067)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _collapse(grid):
    """Independent maximal runs of identical complete row/column vectors."""
    rs = np.r_[0, 1 + np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1))]
    cs = np.r_[0, 1 + np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0))]
    rows = np.diff(np.r_[rs, grid.shape[0]])
    cols = np.diff(np.r_[cs, grid.shape[1]])
    return grid[np.ix_(rs, cs)].copy(), rows, cols


def _runs(vector):
    padded = np.r_[False, vector, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return zip(edges[::2], edges[1::2])


def _largest(mask, rows=None, cols=None, minimum=2):
    """Return area and ALL winning (top,left,bottom,right) half-open boxes.

    Positive band weights also implement deliberate ablations. For a fixed
    row span only full horizontal true runs can maximize area, so subruns
    need not be enumerated. Vertical spans are all enumerated.
    """
    h, w = mask.shape
    rp = np.r_[0, np.cumsum(np.ones(h, dtype=int) if rows is None else rows)]
    cp = np.r_[0, np.cumsum(np.ones(w, dtype=int) if cols is None else cols)]
    best, winners = 0, []
    for top in range(h):
        common = np.ones(w, dtype=bool)
        for bottom in range(top + 1, h + 1):
            common &= mask[bottom - 1]
            height = int(rp[bottom] - rp[top])
            if height < minimum:
                continue
            for left, right in _runs(common):
                width = int(cp[right] - cp[left])
                if width < minimum:
                    continue
                area = height * width
                box = (top, int(left), bottom, int(right))
                if area > best:
                    best, winners = area, [box]
                elif area == best:
                    winners.append(box)
    return best, winners


def _winner(mask):
    _, boxes = _largest(mask)
    if len(boxes) != 1:
        raise ValueError("Expected a unique largest logical background rectangle")
    return boxes[0]


def _box_mask(shape, box):
    mask = np.zeros(shape, dtype=bool)
    t, l, b, r = box
    mask[t:b, l:r] = True
    return mask


def _expand(grid, rows, cols):
    return np.repeat(np.repeat(grid, rows, axis=0), cols, axis=1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    if np.any(grid == HIGHLIGHT):
        raise ValueError("Color 6 is reserved for output highlighting")
    compact, rows, cols = _collapse(grid)
    box = _winner(compact == BACKGROUND)
    selected = _expand(_box_mask(compact.shape, box), rows, cols)
    output = grid.copy()
    output[selected] = HIGHLIGHT
    return output


def _position(rng, length, size):
    free = length - size
    if free == 0:
        return 0
    if rng.random() < .45:
        return int(rng.choice((0, free)))
    return int(rng.integers(free + 1))


def _paint(mask, box):
    t, l, b, r = box
    mask[t:b, l:r] = True


def _attached(mask, box):
    t, l, b, r = box
    return bool((t > 0 and np.any(mask[t - 1, l:r]))
                or (b < mask.shape[0] and np.any(mask[b, l:r]))
                or (l > 0 and np.any(mask[t:b, l - 1]))
                or (r < mask.shape[1] and np.any(mask[t:b, r])))


def _attach(rng, mask, target):
    """Try a protrusion; selection is checked again on the completed mask."""
    h, w = mask.shape
    t, l, b, r = target
    choices = []
    for y in range(t, b):
        if l:
            choices.append((y, l - 1))
        if r < w:
            choices.append((y, r))
    for x in range(l, r):
        if t:
            choices.append((t - 1, x))
        if b < h:
            choices.append((b, x))
    rng.shuffle(choices)
    for y, x in choices:
        if mask[y, x]:
            continue
        mask[y, x] = True
        if _largest(mask)[1] == [target]:
            return True
        mask[y, x] = False
    return False


def _sample_mask(rng, target_shape, profile):
    th, tw = target_shape
    area = th * tw
    rival = None
    if profile == "rival":
        candidates = [(a, b) for a in range(2, 6) for b in range(2, 6)
                      if .60 * area <= a * b < area]
        if not candidates:
            return None
        rh, rw = candidates[int(rng.integers(len(candidates)))]
        h = th + rh + int(rng.integers(1, 4))
        w = tw + rw + int(rng.integers(1, 4))
    elif profile == "line":
        # A long horizontal logical line; transpose/reflection later remove
        # any orientation or location convention.
        h = int(rng.integers(th + 3, th + 7))
        w = int(rng.integers(max(tw + 2, area + 1), max(tw + 2, area + 1) + 3))
    else:
        h = int(rng.integers(th + 2, min(14, th + 8) + 1))
        w = int(rng.integers(tw + 2, min(14, tw + 8) + 1))
    if max(h, w) > 18:
        return None
    t, l = _position(rng, h, th), _position(rng, w, tw)
    target = (t, l, t + th, l + tw)
    mask = _box_mask((h, w), target)
    if profile == "rival":
        # Disjoint on BOTH axes makes either omitted axis measurable in the
        # compulsory comparison demonstration. The optional cases are freer.
        positions = [(y, x) for y in range(h - rh + 1) for x in range(w - rw + 1)
                     if (y + rh < t or y > t + th)
                     and (x + rw < l or x > l + tw)]
        if not positions:
            return None
        y, x = positions[int(rng.integers(len(positions)))]
        rival = (y, x, y + rh, x + rw)
        _paint(mask, rival)
    elif profile == "line":
        length = int(rng.integers(area + 1, w + 1))
        ys = [y for y in range(h) if y < t - 1 or y > t + th]
        if not ys:
            return None
        y = int(rng.choice(ys))
        x = int(rng.integers(w - length + 1))
        mask[y, x:x + length] = True
    if profile in ("line", "overlap") or rng.random() < .65:
        if not _attach(rng, mask, target):
            return None
        if rng.random() < .4:
            _attach(rng, mask, target)
    # Irregular background components, incidental competitors and extensions
    # are accepted only if the exact finished-grid winner remains unique.
    mask |= rng.random(mask.shape) < float(rng.uniform(.015, .13))
    if _largest(mask)[1] != [target]:
        return None
    if profile == "line" and _largest(mask, minimum=1)[0] <= area:
        return None
    return mask, target, rival


def _bands(rng, length):
    mode = int(rng.integers(5))
    if mode == 0:
        return np.full(length, int(rng.integers(1, min(3, 30 // length) + 1)), dtype=int)
    bands = np.ones(length, dtype=int)
    budget = int(rng.integers(1, 30 - length + 1))
    if mode == 1:
        bands[int(rng.integers(length))] += min(budget, 11)
    else:
        for _ in range(budget):
            bands[int(rng.integers(length))] += 1
    return bands


def _ablation_flags(compact, rows, cols, box):
    mask = compact == BACKGROUND
    return {
        "no_collapse": box not in _largest(mask, rows, cols)[1],
        "row_only": box not in _largest(mask, cols=cols)[1],
        "column_only": box not in _largest(mask, rows=rows)[1],
        "allow_lines": box not in _largest(mask, minimum=1)[1],
        "fill_all": np.count_nonzero(mask) > (box[2] - box[0]) * (box[3] - box[1]),
    }


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _make_pair(rng, shape, profile, palette):
    for _ in range(240):
        sampled = _sample_mask(rng, shape, profile)
        if sampled is None:
            continue
        mask, target, rival = sampled
        latent = rng.choice(palette, size=mask.shape).astype(np.int8)
        latent[mask] = BACKGROUND
        # Sometimes expose a row boundary at exactly one blocker cell. This
        # makes hidden local seams observable through the complete vector.
        if rng.random() < .65 and len(palette) >= 2:
            t, l, b, r = target
            y = int(rng.integers(t, b - 1))
            possible = np.flatnonzero(latent[y] != BACKGROUND)
            if len(possible):
                latent[y + 1] = latent[y]
                x = int(rng.choice(possible))
                latent[y + 1, x] = int(rng.choice([c for c in palette if c != latent[y, x]]))
        recovered, _, _ = _collapse(latent)
        if recovered.shape != latent.shape or _largest(latent == BACKGROUND)[1] != [target]:
            continue
        if profile == "line":
            if (_largest(latent == BACKGROUND, minimum=1)[0] <= shape[0] * shape[1]
                    or not _attached(latent == BACKGROUND, target)):
                continue
        rows, cols = _bands(rng, latent.shape[0]), _bands(rng, latent.shape[1])
        if rival is not None:
            rows = np.ones(latent.shape[0], dtype=int)
            cols = np.ones(latent.shape[1], dtype=int)
            a, c, b, d = rival
            rows[a:b] = int(rng.integers(3, 5))
            cols[c:d] = int(rng.integers(3, 5))
        if rows.sum() > 30 or cols.sum() > 30:
            continue
        if max(rows.max(), cols.max()) == 1:
            continue
        flags = _ablation_flags(latent, rows, cols, target)
        if profile == "rival" and not all(flags[k] for k in ("no_collapse", "row_only", "column_only")):
            continue
        changed = _box_mask(latent.shape, target)
        compact_output = latent.copy()
        compact_output[changed] = HIGHLIGHT
        inp = _expand(latent, rows, cols)
        out = _expand(compact_output, rows, cols)
        # Jointly transpose or reflect input/output. The rule is unchanged.
        if rng.random() < .5:
            inp, out = inp.T, out.T
        if rng.random() < .5:
            inp, out = inp[::-1], out[::-1]
        if rng.random() < .5:
            inp, out = inp[:, ::-1], out[:, ::-1]
        return {"input": inp.copy(), "output": out.copy()}
    raise RuntimeError(f"Bounded construction failed for {shape}/{profile}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    palette = rng.choice(BLOCKERS, size=int(rng.integers(2, 9)), replace=False)
    # A shape and its transpose form one held-out geometry class. Tests use
    # that class; training never does, even after transposition augmentation.
    held = TARGET_SHAPES[int(rng.integers(len(TARGET_SHAPES)))]
    train_shapes = [s for s in TARGET_SHAPES if sorted(s) != sorted(held)]
    compare_shapes = [s for s in train_shapes if 6 <= s[0] * s[1] <= 16]
    comparison = compare_shapes[int(rng.integers(len(compare_shapes)))]
    line_shapes = [s for s in train_shapes if 4 <= s[0] * s[1] <= 9
                   and sorted(s) != sorted(comparison)]
    plans = [("train", comparison, "rival"),
             ("train", line_shapes[int(rng.integers(len(line_shapes)))], "line")]
    for _ in range(int(num_train) - 2):
        shape = train_shapes[int(rng.integers(len(train_shapes)))]
        options = ["plain", "overlap"]
        if shape != (2, 2):
            options.append("rival")
        if shape[0] * shape[1] <= 12:
            options.append("line")
        plans.append(("train", shape, str(rng.choice(options))))
    for _ in range(int(num_test)):
        options = ["plain", "overlap"]
        if held != (2, 2):
            options.append("rival")
        if held[0] * held[1] <= 12:
            options.append("line")
        plans.append(("test", held, str(rng.choice(options))))
    puzzle, used = {"train": [], "test": []}, set()
    for split, shape, profile in plans:
        for _ in range(12):
            active = palette if len(puzzle["train"]) == 0 else rng.choice(
                palette, size=int(rng.integers(2, len(palette) + 1)), replace=False)
            pair = _make_pair(rng, shape, profile, active)
            signature = _signature(pair["input"])
            if signature not in used:
                used.add(signature)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Could not produce distinct inputs")
    # Shuffle demonstrations so their evidence requirements do not impose a
    # visible pair-index geometry convention.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    compact, rows, cols = _collapse(inp)
    box = _winner(compact == BACKGROUND)
    flags = _ablation_flags(compact, rows, cols, box)
    t, l, b, r = box
    selected = _box_mask(compact.shape, box)
    reachable = selected.copy()
    for _ in range(compact.size):
        grown = reachable.copy()
        grown[1:] |= reachable[:-1]
        grown[:-1] |= reachable[1:]
        grown[:, 1:] |= reachable[:, :-1]
        grown[:, :-1] |= reachable[:, 1:]
        grown &= compact == BACKGROUND
        if np.array_equal(grown, reachable):
            break
        reachable = grown
    flags["connected_fill"] = np.count_nonzero(reachable) > np.count_nonzero(selected)
    flags["single_witness"] = bool(
        np.any(np.count_nonzero(compact[1:] != compact[:-1], axis=1) == 1)
        or np.any(np.count_nonzero(compact[:, 1:] != compact[:, :-1], axis=0) == 1))
    return compact, rows, cols, box, flags


def validate_generated_puzzle(puzzle):
    """Validate the semantic contract and the demonstration/shift contract."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and HIGHLIGHT not in inp
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            compact, rows, cols, box, flags = _features(inp)
            assert np.array_equal(_expand(compact, rows, cols), inp)
            assert np.array_equal(_collapse(compact)[0], compact)
            selected = _expand(_box_mask(compact.shape, box), rows, cols)
            assert np.array_equal(inp != out, selected)
            assert np.all(inp[selected] == BACKGROUND)
            assert np.all(out[selected] == HIGHLIGHT)
            assert np.array_equal(inp[~selected], out[~selected])
            assert np.array_equal(solve(inp), out)
            t, l, bottom, right = box
            infos[split].append((tuple(sorted((bottom - t, right - l))), flags))
    train_flags = [f for _, f in infos["train"]]
    for name in ("no_collapse", "row_only", "column_only", "allow_lines", "fill_all", "connected_fill"):
        assert any(f[name] for f in train_flags), f"Training lacks a {name} counterexample"
    trained_shapes = {shape for shape, _ in infos["train"]}
    assert len(trained_shapes) >= 2
    assert all(shape not in trained_shapes for shape, _ in infos["test"])


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 supplied seeds, bounded memory, counterexamples and metamorphic tests."""
    # Two literal witnesses check the intended selection independently of the
    # sampling recipe. They also pin expansion/projection and logical 2-Dness.
    literal = ("1212121212", "1000121212", "2000212121", "1000121212",
               "2121212121", "1212100002", "2121200001", "1212121212")
    compact = np.array([[int(c) for c in row] for row in literal], dtype=np.int8)
    rows = np.array([1, 1, 1, 1, 1, 3, 3, 1])
    cols = np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 1])
    assert _winner(compact == 0) == (1, 1, 4, 4)
    assert _largest(compact == 0, rows, cols) == (72, [(5, 5, 7, 9)])
    expected = compact.copy()
    expected[1:4, 1:4] = 6
    assert np.array_equal(solve(_expand(compact, rows, cols)), _expand(expected, rows, cols))
    line = ("101212121212", "100121212121", "200212121212",
            "121212121212", "000000012121")
    compact = np.array([[int(c) for c in row] for row in line], dtype=np.int8)
    expected = compact.copy()
    expected[1:3, 1:3] = 6
    assert np.array_equal(solve(compact), expected)
    assert np.array_equal(solve(_expand(compact, [1, 2, 1, 1, 4], [1] * 12)),
                          _expand(expected, [1, 2, 1, 1, 4], [1] * 12))
    signatures, shapes, compact_shapes, targets, profiles, palettes = (set() for _ in range(6))
    counts = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                compact, rows, cols, box, flags = _features(inp)
                shapes.add(inp.shape)
                compact_shapes.add(compact.shape)
                targets.add((box[2] - box[0], box[3] - box[1]))
                profiles.add((tuple(rows), tuple(cols)))
                palettes.add(tuple(np.unique(inp)))
                counts.update({k: int(v) for k, v in flags.items()})
                # Geometry equivariance checks output projection, including
                # runs at boundaries, without relying on generation metadata.
                assert np.array_equal(solve(inp.T), out.T)
                assert np.array_equal(solve(inp[::-1, ::-1]), out[::-1, ::-1])
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(compact_shapes) >= 25
    assert len(targets) >= 12 and len(profiles) >= 100 and len(palettes) >= 25
    assert counts["single_witness"] > 10
    first = generate_puzzle(0, 2, 1)
    assert _digest(first) == _digest(generate_puzzle(0, 2, 1))
    # A validator must also accept the nested-list serialization.
    validate_generated_puzzle({s: [{k: v.tolist() for k, v in p.items()} for p in first[s]]
                               for s in ("train", "test")})
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(192015362, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Exhaustive small masks check area, minimum dimensions and tie reporting.
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected, winners = 0, set()
        for t in range(2):
            for b in range(t + 2, 4):
                for l in range(2):
                    for r in range(l + 2, 4):
                        if np.all(mask[t:b, l:r]):
                            area = (b - t) * (r - l)
                            if area > expected:
                                expected, winners = area, {(t, l, b, r)}
                            elif area == expected:
                                winners.add((t, l, b, r))
        area, found = _largest(mask)
        assert area == expected and set(found) == winners
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "compact_shapes": len(compact_shapes), "target_shapes": len(targets),
            "band_profiles": len(profiles), "palettes": len(palettes),
            "counterexamples": dict(counts)}
