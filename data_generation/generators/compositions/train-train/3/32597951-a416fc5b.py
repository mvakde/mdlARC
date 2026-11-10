"""Camouflaged bow wave: recover a window, then complete its two forward wings.

Only standard-library modules and NumPy are used.  The decoder reads every role
from the input; generation metadata and training outputs are never solver inputs.
"""
from __future__ import annotations

import hashlib
import numpy as np


# Clockwise, starting at north.  These are travel directions, not source places.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
TEXTURES = ("random", "clustered", "horizontal", "vertical", "checker",
            "diagonal_down", "diagonal_up", "tile")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _decode(input_grid):
    """Identify a unique marker box and the two occurrences of the primary."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    census = dict(zip(map(int, colors), map(int, counts)))
    if len(census) != 6:
        raise ValueError("a window uses six distinct input colors")
    primaries = [c for c, n in census.items() if n == 2]
    tabs = [c for c, n in census.items() if n == 1]
    if len(primaries) != 1 or len(tabs) != 2:
        raise ValueError("expected two primary pixels and two singleton color tabs")
    primary = primaries[0]
    primary_points = [tuple(map(int, p)) for p in np.argwhere(a == primary)]
    candidates = []
    for marker in set(census) - {primary, *tabs}:
        if census[marker] < 6:
            continue
        points = np.argwhere(a == marker)
        top, left = map(int, points.min(axis=0))
        bottom, right = map(int, points.max(axis=0))
        h, w = bottom - top + 1, right - left + 1
        if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
            continue
        pivot = ((top + bottom) // 2, (left + right) // 2)
        if pivot not in primary_points:
            continue
        source = next(p for p in primary_points if p != pivot)
        if top <= source[0] <= bottom and left <= source[1] <= right:
            continue
        delta = (pivot[0] - source[0], pivot[1] - source[1])
        dr, dc = tuple(int(np.sign(x)) for x in delta)
        k = max(map(abs, delta))
        if (dr, dc) not in DIRECTIONS or delta != (k * dr, k * dc):
            continue
        exit_distance = min(n for n, step in ((h // 2, dr), (w // 2, dc)) if step)
        if k < exit_distance + 2:
            continue
        left_tab = (source[0] - dc, source[1] + dr)
        right_tab = (source[0] + dc, source[1] - dr)
        if not all(0 <= r < a.shape[0] and 0 <= c < a.shape[1]
                   for r, c in (left_tab, right_tab)):
            continue
        left_color, right_color = int(a[left_tab]), int(a[right_tab])
        if {left_color, right_color} != set(tabs):
            continue
        patch = a[top:bottom + 1, left:right + 1]
        remainder = set(map(int, np.unique(patch))) - {marker, primary}
        if len(remainder) != 1 or np.count_nonzero(patch == primary) != 1:
            continue
        complement = remainder.pop()
        if complement in tabs:
            continue
        ambient = set(census) - {primary, marker, left_color, right_color, complement}
        if len(ambient) != 1:
            continue
        marked_base = ambient.pop()
        outside = np.ones(a.shape, dtype=bool)
        outside[top:bottom + 1, left:right + 1] = False
        if any(np.count_nonzero((a == color) & outside) < 8
               for color in (marked_base, complement)):
            continue
        if min(census[marked_base], census[complement]) < 10:
            continue
        candidates.append(dict(box=(top, bottom, left, right), pivot=pivot,
                               source=source, direction=(dr, dc), distance=k,
                               primary=primary, marker=marker, complement=complement,
                               marked_base=marked_base, left_color=left_color,
                               right_color=right_color, left_tab=left_tab,
                               right_tab=right_tab))
    if len(candidates) != 1:
        raise ValueError("the camouflaged window must be uniquely identifiable")
    return a, candidates[0]


def _geometry(shape, info):
    rr, cc = np.indices(shape)
    r = rr - info["pivot"][0]
    c = cc - info["pivot"][1]
    dr, dc = info["direction"]
    forward = dr * r + dc * c
    # Positive means left in screen coordinates (rows increase downwards).
    side = dr * c - dc * r
    t, b, l, q = info["box"]
    inside = (rr >= t) & (rr <= b) & (cc >= l) & (cc <= q)
    return inside, forward, side


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _decode(input_grid)
    inside, forward, side = _geometry(a.shape, info)
    eligible = inside & (a == info["complement"]) & (forward > 0)
    out = a.copy()
    out[eligible & (side > 0)] = info["left_color"]
    out[eligible & (side < 0)] = info["right_color"]
    return out


def _field(rng, shape, kind):
    h, w = shape
    r, c = np.indices(shape)
    phase = int(rng.integers(0, 12))
    if kind == "random":
        return rng.random(shape) < rng.uniform(0.22, 0.82)
    if kind == "clustered":
        bh, bw = map(int, rng.integers(2, 5, size=2))
        coarse = rng.random(((h + bh - 1) // bh, (w + bw - 1) // bw)) < rng.uniform(.3, .7)
        field = np.repeat(np.repeat(coarse, bh, 0), bw, 1)[:h, :w]
        return field ^ (rng.random(shape) < .09)
    if kind in ("horizontal", "vertical", "diagonal_down", "diagonal_up"):
        period = int(rng.integers(2, 7))
        run = int(rng.integers(1, period))
        coord = {"horizontal": r, "vertical": c,
                 "diagonal_down": r - c, "diagonal_up": r + c}[kind]
        return (coord + phase) % period < run
    if kind == "checker":
        bh, bw = map(int, rng.integers(1, 4, size=2))
        return (r // bh + c // bw + phase) % 2 == 0
    if kind == "tile":
        bh, bw = map(int, rng.integers(2, 6, size=2))
        tile = rng.random((bh, bw)) < rng.uniform(.3, .7)
        return tile[(r + phase) % bh, (c + phase) % bw]
    raise ValueError("unknown texture")


def _witnesses(a, info):
    inside, forward, side = _geometry(a.shape, info)
    holes = inside & (a == info["complement"])
    markers = inside & (a == info["marker"])
    witnesses = {
        "left": holes & (forward > 0) & (side > 0),
        "right": holes & (forward > 0) & (side < 0),
        "rear": holes & (forward < 0) & (side != 0),
        "axis": holes & (forward > 0) & (side == 0),
        "lateral_tie": holes & (forward == 0) & (side != 0),
        "mask": markers & (forward > 0) & (side != 0),
        "clipping": (~inside) & (a == info["complement"]) & (forward > 0) & (side != 0),
    }
    dr, dc = info["direction"]
    if dr and dc:
        rr, cc = np.indices(a.shape)
        cr, cp = info["pivot"]
        witnesses["diagonal_half_plane"] = (holes & (forward > 0) & (side != 0)
            & ((dr * (rr - cr) < 0) | (dc * (cc - cp) < 0)))
    return witnesses


def _make_pair(rng, direction, palette, *, elongated=False, cornerless=False,
               forbidden_shape=None, texture=None):
    marked_base, complement, marker, primary, left_color, right_color = palette
    dr, dc = direction
    for attempt in range(400):
        if elongated:
            h = int(rng.choice((5, 7, 9)))
            w = int(rng.choice((15, 17, 19, 21, 23)))
            if rng.random() < .5:
                h, w = w, h
        else:
            h, w = map(int, rng.choice((5, 7, 9, 11, 13), size=2))
        if (h, w) == forbidden_shape:
            continue
        ah, aw = h // 2, w // 2
        distance = min(n for n, step in ((ah, dr), (aw, dc)) if step) + int(rng.integers(2, 6))
        source = (-distance * dr, -distance * dc)
        lt = (source[0] - dc, source[1] + dr)
        rt = (source[0] + dc, source[1] - dr)
        rmin = min(-ah, source[0], lt[0], rt[0])
        rmax = max(ah, source[0], lt[0], rt[0])
        cmin = min(-aw, source[1], lt[1], rt[1])
        cmax = max(aw, source[1], lt[1], rt[1])
        base_h, base_w = rmax - rmin + 1, cmax - cmin + 1
        if max(base_h, base_w) > 30:
            continue
        extra_h = int(rng.integers(0, min(7, 30 - base_h) + 1))
        extra_w = int(rng.integers(0, min(7, 30 - base_w) + 1))
        before_h = int(rng.integers(0, extra_h + 1))
        before_w = int(rng.integers(0, extra_w + 1))
        cr, cc = before_h - rmin, before_w - cmin
        shape = base_h + extra_h, base_w + extra_w
        box = cr - ah, cr + ah, cc - aw, cc + aw
        info = dict(box=box, pivot=(cr, cc), direction=direction,
                    marker=marker, complement=complement)
        kind = texture if texture is not None else str(rng.choice(TEXTURES))
        # A fallback avoids pathological rejection for a texture/shape pairing;
        # this changes sampling only, never the transformation.
        if attempt >= 300:
            kind = "random"
        field = _field(rng, shape, kind)
        t, b, l, q = box
        patch = field[t:b + 1, l:q + 1]
        if cornerless:
            # Only used with an irregular field.  The corners are still ordinary
            # field pixels, so the rectangle must be recovered from side extrema.
            patch[0, 0] = patch[0, -1] = patch[-1, 0] = patch[-1, -1] = False
        marker_mask = patch.copy()
        marker_mask[ah, aw] = False  # the visible pivot replaces this one pixel
        if (marker_mask.sum() < 6 or not marker_mask[0].any()
                or not marker_mask[-1].any() or not marker_mask[:, 0].any()
                or not marker_mask[:, -1].any()):
            continue
        a = np.where(field, marked_base, complement).astype(np.int8)
        a[t:b + 1, l:q + 1][patch] = marker
        a[cr, cc] = primary
        for (r, c), color in ((source, primary), (lt, left_color), (rt, right_color)):
            a[cr + r, cc + c] = color
        if not all(m.any() for m in _witnesses(a, info).values()):
            continue
        try:
            _, decoded = _decode(a)
        except ValueError:
            continue
        assert decoded["box"] == box and decoded["direction"] == direction
        return {"input": a, "output": solve(a)}
    raise RuntimeError("could not construct a witnessed camouflaged window")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = tuple(map(int, rng.choice(10, size=6, replace=False)))
    marker, primary = base[2:4]
    mixed_palette = bool(rng.integers(0, 2))
    mode = int(seed) % 3
    cardinal = [0, 2, 4, 6]
    diagonal = [1, 3, 5, 7]
    heldout = int(rng.choice(cardinal if mode == 0 else diagonal)) if mode < 2 else None
    first = int(rng.choice([x for x in cardinal if x != heldout]))
    second = int(rng.choice([x for x in diagonal if x != heldout]))
    train_dirs = [first, second] + [int(rng.choice([x for x in range(8) if x != heldout]))
                                   for _ in range(nt - 2)]
    test_dirs = [heldout if heldout is not None else int(rng.choice(train_dirs))]
    test_dirs += [int(rng.integers(0, 8)) for _ in range(ne - 1)]
    pairs, seen = [], set()
    first_shape = None
    for index, direction_index in enumerate(train_dirs + test_dirs):
        palette = base
        if mixed_palette:
            f, g, left, right = map(int, rng.choice([x for x in range(10)
                                                   if x not in (marker, primary)],
                                                  size=4, replace=False))
            palette = f, g, marker, primary, left, right
        for _ in range(30):
            pair = _make_pair(
                rng, DIRECTIONS[direction_index], palette,
                elongated=(index >= nt and mode == 2) or
                          (index >= 2 and mode != 2 and rng.random() < .45),
                cornerless=index == 0,
                forbidden_shape=first_shape if index == 1 else None,
                texture="random" if index == 0 else
                        str(rng.choice(TEXTURES[2:])) if index == 1 else None)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("failed to construct distinct inputs")
        seen.add(key)
        pairs.append(pair)
        if index == 0:
            _, info = _decode(pair["input"])
            t, b, l, q = info["box"]
            first_shape = b - t + 1, q - l + 1
    train = [pairs[int(i)] for i in rng.permutation(nt)]
    puzzle = {"train": train, "test": pairs[nt:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check outputs, observable roles, ablation witnesses, and teaching set."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, conventions, training_shapes, training_directions = set(), set(), set(), set()
    test_profiles = []
    cornerless_cardinal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            _, info = _decode(a)
            conventions.add((info["marker"], info["primary"]))
            t, b, l, q = info["box"]
            cr, cc = info["pivot"]
            dr, dc = info["direction"]
            # Deliberately check each changed and unchanged cell with scalar
            # arithmetic, rather than validating outputs by calling solve again.
            expected = a.copy()
            for r in range(t, b + 1):
                for c in range(l, q + 1):
                    dot = dr * (r - cr) + dc * (c - cc)
                    cross = dr * (c - cc) - dc * (r - cr)
                    if int(a[r, c]) == info["complement"] and dot > 0 and cross:
                        expected[r, c] = info["left_color"] if cross > 0 else info["right_color"]
            assert np.array_equal(out, expected), "incorrect selective wing completion"
            witness = _witnesses(a, info)
            assert all(mask.any() for mask in witness.values()), "missing conceptual witness"
            assert np.all(out[witness["left"]] == info["left_color"])
            assert np.all(out[witness["right"]] == info["right_color"])
            for name in ("rear", "axis", "lateral_tie", "mask", "clipping"):
                assert np.array_equal(out[witness[name]], a[witness[name]])
            if split == "train":
                training_shapes.add((b - t + 1, q - l + 1))
                training_directions.add(DIRECTIONS.index(info["direction"]))
                if dr == 0 or dc == 0:
                    cornerless_cardinal |= all(int(a[r, c]) == info["complement"]
                                               for r in (t, b) for c in (l, q))
            else:
                test_profiles.append((DIRECTIONS.index(info["direction"]), b - t + 1, q - l + 1))
    assert len(conventions) == 1, "marker and primary identities are shared within a puzzle"
    assert len(training_shapes) >= 2
    assert any(i % 2 == 0 for i in training_directions)
    assert any(i % 2 == 1 for i in training_directions)
    assert cornerless_cardinal, "show bounding-box completion beyond the marker hull"
    assert any(di not in training_directions or
               max(h, w) > max(max(shape) for shape in training_shapes)
               for di, h, w in test_profiles), "missing direction or extent shift"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed geometry/ablation/metamorphic and API checks."""
    shapes, boxes, directions, palettes, distances, identities = (set() for _ in range(6))
    aspect = set()
    densities, changed_counts = [], []
    edges, test_shifts, fingerprints = set(), set(), set()
    empty_rows = empty_columns = disconnected = 0
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        train_dirs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                original_input = a.copy()
                _, info = _decode(a)
                inside, forward, side = _geometry(a.shape, info)
                t, b, l, q = info["box"]
                h, w = b - t + 1, q - l + 1
                di = DIRECTIONS.index(info["direction"])
                if split == "train":
                    train_dirs.add(di)
                elif seed % 3 < 2:
                    if pair is puzzle["test"][0]:
                        assert di not in train_dirs
                        test_shifts.add("heldout_cardinal" if di % 2 == 0 else "heldout_diagonal")
                else:
                    assert max(h, w) >= 15 and min(h, w) <= 9
                    test_shifts.add("elongation")
                shapes.add(a.shape)
                boxes.add((h, w))
                directions.add(di)
                identities.add((info["marker"], info["primary"]))
                palettes.add(tuple(info[k] for k in ("marked_base", "complement", "left_color", "right_color")))
                distances.add(info["distance"])
                aspect.add("square" if h == w else "tall" if h > w else "wide")
                patch = a[t:b + 1, l:q + 1] == info["marker"]
                densities.append(float(patch.mean()))
                changed_counts.append(int(np.count_nonzero(a != out)))
                empty_rows += int(np.any(~patch[1:-1].any(axis=1)))
                empty_columns += int(np.any(~patch[:, 1:-1].any(axis=0)))
                # Four-neighbor marker components, measured from returned pixels.
                remaining = set(map(tuple, np.argwhere(patch)))
                components = 0
                while remaining:
                    components += 1
                    stack = [remaining.pop()]
                    while stack:
                        r, c = stack.pop()
                        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if p in remaining:
                                remaining.remove(p)
                                stack.append(p)
                disconnected += components > 1
                for edge, contact in (("top", t == 0), ("bottom", b == a.shape[0] - 1),
                                      ("left", l == 0), ("right", q == a.shape[1] - 1)):
                    if contact:
                        edges.add(edge)
                # The witnesses force a different output for a concrete
                # simplification: erase a wing, fill rear/axis/tie/markers, or
                # omit clipping.  Swapping the tabs must also change the result.
                for mask in _witnesses(a, info).values():
                    assert mask.any()
                swapped = a.copy()
                swapped[info["left_tab"]], swapped[info["right_tab"]] = (
                    a[info["right_tab"]], a[info["left_tab"]])
                assert not np.array_equal(solve(swapped), out)
                assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(out))
                assert np.array_equal(solve(a.tolist()), out)
                assert np.array_equal(a, original_input), "solve mutated the input"
                pair_count += 1
    assert len(fingerprints) == 42
    assert len(shapes) >= 60 and len(boxes) >= 35
    assert directions == set(range(8)) and len(identities) >= 20
    assert len(palettes) >= 60 and len(distances) >= 8
    assert aspect == {"square", "tall", "wide"}
    assert edges == {"top", "bottom", "left", "right"}
    assert test_shifts == {"heldout_cardinal", "heldout_diagonal", "elongation"}
    assert min(densities) < .30 and max(densities) > .70
    assert min(changed_counts) <= 8 and max(changed_counts) >= 45
    assert min(empty_rows, empty_columns, disconnected) > 0
    # A valid family member outside the sampler's size and source-gap ranges.
    rr, cc = np.indices((30, 30))
    large = ((rr + cc) % 2).astype(np.int8)
    block = large[0:29, 5:10]
    block[block == 0] = 6
    large[14, 7] = large[14, 20] = 8
    large[15, 20], large[13, 20] = 4, 5
    result = solve(large)
    assert result[0, 5] == 5 and result[28, 5] == 4
    assert result[14, 5] == 1 and result[14, 9] == 1
    assert np.array_equal(solve(np.rot90(large)), np.rot90(result))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42000 + nt * 10 + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pair_count, canvas_shapes=len(shapes),
                window_shapes=len(boxes), directions=len(directions),
                primary_marker_palettes=len(identities), other_palettes=len(palettes),
                source_distances=len(distances), marker_density=[min(densities), max(densities)],
                completion_counts=[min(changed_counts), max(changed_counts)],
                disconnected_marker_examples=disconnected,
                empty_internal_marker_rows=empty_rows, empty_internal_marker_columns=empty_columns,
                boundary_contacts=sorted(edges), shifts=sorted(test_shifts))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
