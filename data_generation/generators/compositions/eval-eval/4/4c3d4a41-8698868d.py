"""Self-adjusting cavity lock: topology determines the teeth of a comb.

The input-only reader and solver do not access seeds, examples, or files.
Generation uses component opening depths; prediction floods each clipped image.
All diagnostics are bounded and in memory. No execution traces are stored.
"""

from __future__ import annotations

import hashlib
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(cells)
    return result


def _enclosed(mask, diagonal=False):
    h, w = mask.shape
    return [cells for cells in _components(mask, diagonal)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)]


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer arrays in 0..9 of size 1..30")
    return a.astype(np.int8, copy=True)


def _minimal_shift(mask, target, *, diagonal=False, area=False, erase=False):
    """Search actual clipped images, treating their boundaries as exterior.

Optional variants exist only to test specific competing rules. The public
solver always uses four-connectivity, enclosed component count and first stop.
"""
    for d in range(mask.shape[0] - 1):
        residual = mask[d:]
        cavities = _enclosed(residual, diagonal)
        value = sum(map(len, cavities)) if area else len(cavities)
        if erase:
            value = len(_components(residual))
        if (value <= target if area else value == target):
            return d
    raise ValueError("target cavity count cannot be reached with two pin rows left")


def _read_canonical(a, bg):
    """Read one orientation: clipping roof above, aligned count cards below."""
    objects = _components(a != bg)
    assert 3 <= len(objects) <= 5
    boxes = [_box(cells) for cells in objects]
    # The housing, including all attached pins, is taller/wider and has a
    # strictly larger bounding-box area than any one count card.
    main = max(range(len(boxes)), key=lambda i: (boxes[i][1] - boxes[i][0]) *
               (boxes[i][3] - boxes[i][2]))
    t, stop, left, right_stop = boxes[main]
    b, right = stop - 1, right_stop - 1
    s = int(a[t, left])
    assert s != bg and np.all(_perimeter(a[t:stop, left:right_stop]) == s)
    assert b - t >= 9
    occupied = a[t + 1, left + 1:right] != bg
    indices = np.flatnonzero(occupied) + left + 1
    assert len(indices)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    assert len(runs) == len(objects) - 1 and 2 <= len(runs) <= 4
    assert runs[0][0] >= left + 2 and runs[-1][-1] <= right - 2
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[t, left:right_stop] = expected[b, left:right_stop] = s
    expected[t:stop, left] = expected[t:stop, right] = s
    rods, used_cards = [], set()
    for run in runs:
        c, end = int(run[0]), int(run[-1]) + 1
        width = end - c
        assert 5 <= width <= 8
        column = a[t + 1:b, c:end]
        rows = np.flatnonzero(np.any(column != bg, axis=1))
        assert len(rows) and np.array_equal(rows, np.arange(len(rows)))
        length = len(rows)
        assert 7 <= length <= 16 and b - t - 1 - length >= 1
        pin = column[:length].copy()
        assert np.all(_perimeter(pin) != bg) and s not in pin
        assert all(len(np.unique(row[row != bg])) == 1 for row in pin)
        assert len(_components(pin != bg)) == 1
        mask = pin == bg
        initial = len(_components(mask))
        assert initial <= 5
        matches = [(i, box) for i, box in enumerate(boxes) if i != main
                   and box[2:] == (c, end) and box[0] > b + 1]
        assert len(matches) == 1
        i, (r0, r1, c0, c1) = matches[0]
        assert i not in used_cards and 4 <= r1 - r0 <= 6
        used_cards.add(i)
        card = a[r0:r1, c0:c1]
        assert np.all(_perimeter(card) == s)
        assert set(map(int, np.unique(card))) <= {bg, s}
        dots = card == bg
        assert not np.any(dots[:-1] & dots[1:])
        assert not np.any(dots[:, :-1] & dots[:, 1:])
        assert len(_components(~dots)) == 1
        target = int(dots.sum())
        assert 0 <= target <= initial
        shift = _minimal_shift(mask, target)
        expected[t + 1:t + 1 + length, c:end] = pin
        expected[r0:r1, c0:c1] = card
        rods.append(dict(col=c, end=end, pin=pin, mask=mask, target=target,
                         initial=initial, shift=shift, card=(r0, r1, c0, c1),
                         gap=b - t - 1 - length))
    assert len(used_cards) == len(objects) - 1
    assert np.array_equal(a, expected), "stray cells or malformed lock"
    return dict(background=bg, structure=s, top=t, base=b, left=left,
                right=right, rods=rods)


def _read(input_grid):
    grid = _array(input_grid)
    border = _perimeter(grid)
    if not np.all(border == border[0]):
        raise ValueError("the exterior border must identify one background")
    bg = int(border[0])
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation).copy()
        try:
            f = _read_canonical(a, bg)
        except (AssertionError, ValueError, IndexError):
            continue
        candidates.append((a, f, rotation))
    if len(candidates) != 1:
        raise ValueError("input must have one lock direction and aligned set of cards")
    return candidates[0]


def _prediction(a, f, shifts=None):
    result = a.copy()
    for i, rod in enumerate(f["rods"]):
        d = rod["shift"] if shifts is None else shifts[i]
        pin, c, end = rod["pin"], rod["col"], rod["end"]
        t, b = f["top"], f["base"]
        result[t + 1:b, c:end] = f["background"]
        survivor = pin[d:]
        result[t + 1:t + 1 + len(survivor), c:end] = survivor
        result[t + 1 + len(survivor):b, c:end] = f["structure"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the direction, count targets, and first qualifying pin positions."""
    a, f, rotation = _read(input_grid)
    return np.rot90(_prediction(a, f), rotation).copy()


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _hole_mask(rng, length, width, count):
    mask = np.zeros((length, width), dtype=bool)
    if count == 0:
        return mask
    if rng.random() < 0.55:
        # Recursive one-cell walls give big cavities, bars and tied opening
        # depths without a small artificial area budget.
        boxes = [(1, length - 1, 1, width - 1)]
        while len(boxes) < count:
            cuts = []
            for i, (r0, r1, c0, c1) in enumerate(boxes):
                cuts.extend((i, 0, p) for p in range(r0 + 1, r1 - 1))
                cuts.extend((i, 1, p) for p in range(c0 + 1, c1 - 1))
            if not cuts:
                break
            i, axis, p = _choice(rng, cuts)
            r0, r1, c0, c1 = boxes.pop(i)
            boxes.extend(((r0, p, c0, c1), (p + 1, r1, c0, c1)) if axis == 0
                         else ((r0, r1, c0, p), (r0, r1, p + 1, c1)))
        if len(boxes) == count:
            for r0, r1, c0, c1 in boxes:
                if rng.random() < 0.6:
                    r0 = int(rng.integers(r0, r1))
                    r1 = int(rng.integers(r0 + 1, r1 + 1))
                    c0 = int(rng.integers(c0, c1))
                    c1 = int(rng.integers(c0 + 1, c1 + 1))
                mask[r0:r1, c0:c1] = True
            return mask
    # Independent seed cavities; a finite constructive fallback supports all
    # five counts at the smallest admitted cross-section.
    for attempt in range(8):
        mask[:] = False
        for flat in rng.permutation((length - 2) * (width - 2)):
            r, c = divmod(int(flat), width - 2)
            r, c = r + 1, c + 1
            if any(mask[r + dr, c + dc] for dr, dc in _FOUR):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
            if int(mask.sum()) == count:
                break
        if int(mask.sum()) == count:
            break
    else:
        mask[:] = False
        for r, c in ((1, 1), (1, width - 2), (3, 1), (3, width - 2), (5, 1))[:count]:
            mask[r, c] = True
    target_area = int(rng.integers(count, (length - 2) * (width - 2) + 1))
    # Two passes over a finite interior; growth preserves both cavity count
    # and one connected pin. This is geometry generation, never a saved trace.
    for _ in range(2):
        for flat in rng.permutation((length - 2) * (width - 2)):
            if int(mask.sum()) >= target_area:
                break
            r, c = divmod(int(flat), width - 2)
            r, c = r + 1, c + 1
            if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _FOUR):
                continue
            mask[r, c] = True
            if len(_components(mask)) != count or len(_components(~mask)) != 1:
                mask[r, c] = False
    return mask


def _dots(rng, height, width, count):
    for _ in range(12):
        mask = np.zeros((height, width), dtype=bool)
        for flat in rng.permutation((height - 2) * (width - 2)):
            if int(mask.sum()) == count:
                break
            r, c = divmod(int(flat), width - 2)
            r, c = r + 1, c + 1
            if any(mask[r + dr, c + dc] for dr, dc in _FOUR):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
        if int(mask.sum()) == count:
            return mask
    # Five singleton dots in a 3x3 interior, all separated by foreground.
    mask[:] = False
    cells = [(r, c) for r in (1, 3) for c in (1, 3)] + [(2, 2)]
    assert height >= 5 or count <= 3
    if height == 4:
        cells = [(1, 1), (1, 3), (2, 2)]
    for i in rng.permutation(len(cells))[:count]:
        mask[cells[int(i)]] = True
    assert int(mask.sum()) == count and len(_components(~mask)) == 1
    return mask


def _opening_shifts(mask):
    """Constructor path: components open when their first row reaches roof."""
    openings = [min(r for r, c in cells) for cells in _components(mask)]
    return {k: min(d for d in range(mask.shape[0] - 1)
                   if sum(row > d for row in openings) == k)
            for k in {sum(row > d for row in openings)
                      for d in range(mask.shape[0] - 1)}}


def _make_pair(rng, n, length_range, count_cap, rotation, witness=0, count_floor=0):
    # Allocate a hard width budget, including real lane spacing and side gaps.
    widths, gaps, insets = [5] * n, [1] * (n - 1), [2, 2]
    budget = 28 - sum(widths) - sum(gaps) - sum(insets)
    slots = [(widths, i, 3) for i in range(n)] + [(gaps, i, 2) for i in range(n - 1)]
    slots += [(insets, i, 1) for i in range(2)]
    for index in rng.permutation(len(slots)):
        values, i, cap = slots[int(index)]
        extra = int(rng.integers(min(cap, budget) + 1))
        values[i] += extra
        budget -= extra
    length_max = int(rng.integers(length_range[0], length_range[1] + 1))
    lengths = rng.integers(length_range[0], length_max + 1, n).tolist()
    lengths[int(rng.integers(n))] = length_max
    interior = length_max + int(rng.integers(1, 4))
    base, housing_width = interior + 1, sum(widths) + sum(gaps) + sum(insets)
    bg, s = map(int, rng.choice(10, 2, replace=False))
    available = [c for c in range(10) if c not in (bg, s)]
    pool_size = int(rng.integers(2 if witness else 1, 7))
    pool = list(map(int, rng.choice(available, pool_size, replace=False)))
    rods, cards = [], []
    col = insets[0]
    for i, (length, width) in enumerate(zip(lengths, widths)):
        if witness == 1 and i == 0:
            mask = np.zeros((length, width), dtype=bool)
            start = int(rng.integers(2, length - 3))
            mask[start:length - 2, 1:width - 1] = True
            target = 0
        elif witness == 1 and i == 1:
            mask = np.zeros((length, width), dtype=bool)
            start = int(rng.integers(1, length - 4))
            mask[start, 1] = True
            mask[start + 1:start + 4, 2] = True
            if rng.random() < 0.5:
                mask = np.fliplr(mask).copy()
            target = 1
        elif witness == 2 and i < 2:
            # Equal initial and target counts, different opening depths.
            mask = np.zeros((length, width), dtype=bool)
            start = i + 1
            mask[start, 1:3] = True
            mask[length - 2, width - 2] = True
            target = 1
        else:
            count = int(rng.integers(count_floor if i == 0 else 0, count_cap + 1))
            mask = _hole_mask(rng, length, width, count)
            options = _opening_shifts(mask)
            target = _choice(rng, sorted(options))
            if witness == 1:
                target = count  # the third lane is a stationary witness
        options = _opening_shifts(mask)
        d = options[target]
        row_colors = []
        while len(row_colors) < length:
            row_colors.extend([_choice(rng, pool)] * int(rng.integers(1, 4)))
        row_colors = row_colors[:length]
        if witness and i == 1:
            row_colors[-2:] = pool[:2]
        pin = np.repeat(np.array(row_colors, dtype=np.int8)[:, None], width, axis=1)
        pin[mask] = bg
        # Cards have isolated clue cells; their spatial pattern carries no code.
        heights = [h for h in range(4, 7) if (h - 2) * (width - 2) >= 2 * target - 1
                   and (h >= 5 or target <= 3)]
        card_height = _choice(rng, heights)
        gap_cap = min(3, 28 - base - 1 - card_height)
        assert gap_cap >= 1
        card_top = base + 1 + int(rng.integers(1, gap_cap + 1))
        card = np.full((card_height, width), s, dtype=np.int8)
        card[_dots(rng, card_height, width, target)] = bg
        rods.append((col, pin, d))
        cards.append((card_top, col, card))
        if i < n - 1:
            col += width + gaps[i]
    height = max(r + len(card) for r, c, card in cards)
    a = np.full((height, housing_width), bg, dtype=np.int8)
    a[0] = a[base] = s
    a[:base + 1, 0] = a[:base + 1, -1] = s
    out = a.copy()
    for c, pin, d in rods:
        a[1:1 + len(pin), c:c + pin.shape[1]] = pin
        # Constructor writes input pixel destinations, rather than invoking
        # either the solver or its clipped-image cavity search.
        out[1:base, c:c + pin.shape[1]] = s
        for r in range(d, len(pin)):
            out[1 + r - d, c:c + pin.shape[1]] = pin[r]
    for r, c, card in cards:
        a[r:r + len(card), c:c + card.shape[1]] = card
        out[r:r + len(card), c:c + card.shape[1]] = card
    vertical_room, horizontal_room = 30 - height, 30 - housing_width
    top = int(rng.integers(1, min(2, vertical_room - 1) + 1))
    bottom = int(rng.integers(1, min(2, vertical_room - top) + 1))
    left = int(rng.integers(1, min(2, horizontal_room - 1) + 1))
    right = int(rng.integers(1, min(2, horizontal_room - left) + 1))
    pads = ((top, bottom), (left, right))
    a, out = np.pad(a, pads, constant_values=bg), np.pad(out, pads, constant_values=bg)
    if rng.random() < 0.5:
        a, out = np.fliplr(a), np.fliplr(out)
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("counts must be integers: 2..10 train and 1..4 test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fresh(pair, seen):
    """At most fourteen pairs; ninety role-color assignments suffice."""
    if _key(pair["input"]) not in seen:
        return pair
    colors = list(map(int, np.unique(pair["input"])))
    order = colors[:2] + [c for c in range(10) if c not in colors[:2]]
    for first in range(10):
        for second in range(10):
            if first == second:
                continue
            target = [first, second] + [c for c in range(10) if c not in (first, second)]
            lut = np.empty(10, dtype=np.int8)
            lut[order] = target
            candidate = {role: lut[a] for role, a in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("fourteen-pair uniqueness bound exceeded")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    schedule = _choice(rng, ("lanes", "depth", "counts"))
    direction = int(rng.integers(4))
    mixed_directions = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = i + 1 if split == "train" and i < 2 else 0
            n = 3 if witness == 1 else int(rng.integers(2, 4))
            lengths, cap = (7, 12), 5
            if schedule == "lanes" and split == "test":
                n = 4
            elif schedule == "depth":
                lengths = (7, 9) if split == "train" else (13, 16)
            elif schedule == "counts":
                cap = 2 if split == "train" else 5
            rotation = (direction + i) % 4 if mixed_directions else direction
            floor = 4 if schedule == "counts" and split == "test" else 0
            pair = _make_pair(rng, n, lengths, cap, rotation, witness, floor)
            pair = _fresh(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _features(f):
    rods = f["rods"]
    report = dict(lanes=len(rods), widths=tuple(r["pin"].shape[1] for r in rods),
                  lengths=tuple(len(r["pin"]) for r in rods),
                  targets=tuple(r["target"] for r in rods),
                  initial=tuple(r["initial"] for r in rods),
                  shifts=tuple(r["shift"] for r in rods),
                  teeth=tuple(r["shift"] + r["gap"] for r in rods))
    report["stationary"] = any(r["shift"] == 0 for r in rods)
    report["moving"] = any(r["shift"] > 0 for r in rods)
    report["unequal_shifts"] = len(set(report["shifts"])) > 1
    report["open_bay"] = any(r["shift"] > 0 and np.any(r["mask"][r["shift"]]) for r in rods)
    report["multicell"] = any(len(cells) > 1 for r in rods for cells in _components(r["mask"]))
    report["area_inversion"] = any(a["initial"] < b["initial"]
        and int(a["mask"].sum()) > int(b["mask"].sum()) for a in rods for b in rods)
    report["band_order"] = any(len(np.unique(r["pin"][r["shift"]:][
        r["pin"][r["shift"]:] != f["background"]])) > 1 for r in rods)
    report["geometry_witness"] = any(a["initial"] == b["initial"] and a["target"] == b["target"]
        and a["shift"] != b["shift"] for i, a in enumerate(rods) for b in rods[i + 1:])
    report["ignore_cards_differs"] = any(_minimal_shift(r["mask"], 0) != r["shift"] for r in rods)
    for name, options in (("eight_differs", {"diagonal": True}),
                          ("area_differs", {"area": True}),
                          ("erase_differs", {"erase": True})):
        report[name] = False
        for rod in rods:
            try:
                d = _minimal_shift(rod["mask"], rod["target"], **options)
            except ValueError:
                continue
            else:
                report[name] |= d != rod["shift"]
    return report


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a, f, rotation = _read(inp)
            prediction = _prediction(a, f)
            assert np.array_equal(np.rot90(prediction, rotation), out), "wrong lock response"
            assert not np.array_equal(inp, out)
            for rod in f["rods"]:
                d, pin = rod["shift"], rod["pin"]
                survivor = prediction[f["top"] + 1:f["top"] + 1 + len(pin) - d,
                                      rod["col"]:rod["end"]]
                assert np.array_equal(survivor, pin[d:]), "ordered pixels were changed"
                assert len(_enclosed(survivor == f["background"])) == rod["target"]
                assert all(len(_enclosed(rod["mask"][earlier:])) != rod["target"]
                           for earlier in range(d)), "pin did not stop at first position"
                r0, r1, c0, c1 = rod["card"]
                assert np.array_equal(prediction[r0:r1, c0:c1], a[r0:r1, c0:c1])
                assert np.all(prediction[f["top"] + 1 + len(pin) - d:f["base"],
                                         rod["col"]:rod["end"]] == f["structure"])
            if split == "train":
                training.append(_features(f))
    for flag in ("stationary", "moving", "unequal_shifts", "multicell", "area_inversion", "open_bay",
                 "band_order", "geometry_witness", "eight_differs", "area_differs", "erase_differs",
                 "ignore_cards_differs"):
        assert any(row[flag] for row in training), "missing training witness: " + flag


def self_test():
    """Stream 42 seeds; no growing sample archive, trace, or per-step log."""
    aggregate = {name: set() for name in ("shapes", "widths", "lengths", "targets", "initial",
                                         "shifts", "teeth", "lanes", "directions", "palettes", "masks")}
    signatures, pairs = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, f, rotation = _read(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                digest.update(pair["input"].tobytes())
                digest.update(bytes(pair["input"].shape))
                features = _features(f)
                aggregate["shapes"].add(pair["input"].shape)
                aggregate["lanes"].add(len(f["rods"]))
                aggregate["directions"].add(rotation)
                aggregate["palettes"].add(len(np.unique(a)))
                for name in ("widths", "lengths", "targets", "initial", "shifts", "teeth"):
                    aggregate[name].update(features[name])
                for rod in f["rods"]:
                    aggregate["masks"].add((rod["mask"].shape, rod["mask"].tobytes()))
                pairs += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert aggregate["lanes"] == {2, 3, 4} and aggregate["directions"] == {0, 1, 2, 3}
    assert len(aggregate["shapes"]) >= 25 and len(aggregate["masks"]) >= 100
    assert aggregate["widths"] == {5, 6, 7, 8}
    assert aggregate["initial"] == set(range(6)) and aggregate["targets"] == set(range(6))
    assert min(aggregate["shifts"]) == 0 and max(aggregate["shifts"]) >= 8
    assert max(aggregate["lengths"]) == 16 and len(aggregate["palettes"]) >= 4
    # Simultaneous openings can skip counts; zero is still the first event
    # opening the deepest cavity, and an initially solid pin stays still.
    tied = np.zeros((9, 6), dtype=bool)
    tied[2, 1] = tied[2, 4] = True
    tied[5:7, 2:4] = True
    assert _opening_shifts(tied) == {0: 5, 1: 2, 3: 0}
    assert _minimal_shift(tied, 1) == 2 and _minimal_shift(tied, 0) == 5
    try:
        _minimal_shift(tied, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("unreachable tied-opening target accepted")
    assert _minimal_shift(np.zeros((7, 5), dtype=bool), 0) == 0
    # Input-only metamorphic checks: rotation, arbitrary role recoloring, and
    # relocation of clue dots leave the physical rule covariant/invariant.
    puzzle = generate_puzzle(127, 2, 1)
    for pair in puzzle["train"]:
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(pair["input"], rotation)),
                                  np.rot90(pair["output"], rotation))
        lut = np.array([7, 3, 9, 1, 0, 5, 8, 2, 6, 4], dtype=np.int8)
        assert np.array_equal(solve(lut[pair["input"]]), lut[pair["output"]])
        a, f, rot = _read(pair["input"])
        expected = np.rot90(pair["output"], -rot).copy()
        for rod in f["rods"]:
            r0, r1, c0, c1 = rod["card"]
            card = a[r0:r1, c0:c1]
            a[r0:r1, c0:c1] = np.fliplr(card).copy()
            expected[r0:r1, c0:c1] = a[r0:r1, c0:c1]
        assert np.array_equal(solve(np.rot90(a, rot)), np.rot90(expected, rot))
    again = generate_puzzle(127, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in puzzle
               for p, q in zip(puzzle[s], again[s]) for k in p)
    # All legal sizes are intentionally cheap; the host repeats these checks.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42 + nt, nt, ne)
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "variation": {name: len(values) for name, values in aggregate.items()},
            "independent_solver": False}
