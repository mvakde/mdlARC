"""Counter-calibrated screw ribbons: unwind and repair a repeated binary motif.

Only standard-library modules and NumPy are used. Generation constructs labels
from clean templates; solve parses and infers a consensus from the input alone.
"""
from __future__ import annotations

import json
from math import gcd
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be a 1..30 rectangle of integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _decode(value):
    """Find the unique padded rectangular frame, not a learned literal color."""
    g = _grid(value)
    candidates = []
    for frame in np.unique(g):
        coords = np.argwhere(g == frame)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        box = g[r0:r1, c0:c1]
        if min(box.shape) < 7 or max(box.shape) < 12:
            continue
        if not (np.all(box[[0, -1], :] == frame)
                and np.all(box[:, [0, -1]] == frame)):
            continue
        ring = np.concatenate((box[1, 1:-1], box[-2, 1:-1],
                               box[2:-2, 1], box[2:-2, -2]))
        if len(np.unique(ring)) != 1 or ring[0] == frame:
            continue
        bg = int(ring[0])
        payload = box[2:-2, 2:-2]
        colors = set(map(int, np.unique(payload)))
        if len(colors) != 2 or int(frame) not in colors or bg in colors:
            continue
        ink = next(iter(colors - {int(frame)}))
        outside = np.ones(g.shape, bool)
        outside[r0:r1, c0:c1] = False
        exterior = set(map(int, np.unique(g[outside])))
        if len(exterior) != 2 or bg not in exterior:
            continue
        counter_ink = next(iter(exterior - {bg}))
        if len({int(frame), bg, ink, counter_ink}) != 4:
            continue
        counter = outside & (g == counter_ink)
        groups = _components(counter)
        if not 1 <= len(groups) <= 12 or len(_components(counter, True)) != len(groups):
            continue
        vertical = payload.shape[0] > payload.shape[1]
        bits = (payload == ink).T.copy() if vertical else (payload == ink).copy()
        h, length = bits.shape
        if not (3 <= h < length and length >= 8):
            continue
        # Every counter lies strictly on one side of the frame, with a gap.
        q = np.argwhere(counter)
        side = ("top" if np.all(q[:, 0] < r0) else
                "bottom" if np.all(q[:, 0] >= r1) else
                "left" if np.all(q[:, 1] < c0) else
                "right" if np.all(q[:, 1] >= c1) else None)
        if side is None:
            continue
        gap = (r0 - 1 - int(q[:, 0].max()) if side == "top" else
               int(q[:, 0].min()) - r1 if side == "bottom" else
               c0 - 1 - int(q[:, 1].max()) if side == "left" else
               int(q[:, 1].min()) - c1)
        if gap < 1:
            continue
        candidates.append({"grid": g, "frame": int(frame), "bg": bg,
                           "ink": ink, "counter_ink": counter_ink,
                           "counter": counter, "groups": groups,
                           "count": len(groups), "vertical": vertical,
                           "bits": bits, "rectangle": (r0, r1, c0, c1),
                           "side": side, "gap": gap})
    if len(candidates) != 1:
        raise ValueError("input must have one padded frame and one separated object collection")
    return candidates[0]


def _candidates(bits, pitch, per_slice=False):
    """All strict-majority repairs with >=3 samples per phase coordinate."""
    h, length = bits.shape
    results = []
    limit = max(1, bits.size // 8)
    for period in range(1, length // 3 + 1):
        aligned = np.stack([np.roll(bits[:, j], -pitch * (j if per_slice else j // period))
                            for j in range(length)], axis=1)
        tile = np.empty((h, period), bool)
        errors = 0
        for phase in range(period):
            votes = aligned[:, phase::period]
            ones, total = votes.sum(axis=1), votes.shape[1]
            if np.any(2 * ones == total):
                break
            tile[:, phase] = 2 * ones > total
            errors += int(np.minimum(ones, total - ones).sum())
        else:
            if errors <= limit and tile.any() and not tile.all():
                flat = tile[:, np.arange(length) % period]
                results.append((period, errors, tile, flat))
    return results


def _recover(bits, count):
    candidates = _candidates(bits, count)
    if not candidates:
        raise ValueError("no supported screw-periodic repair")
    flat = candidates[0][3]
    if any(not np.array_equal(flat, item[3]) for item in candidates[1:]):
        raise ValueError("different supported periods imply different straight ribbons")
    return candidates[0][0], candidates[0][2], flat.copy(), candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the repaired, straightened payload; neither frame nor counters."""
    d = _decode(input_grid)
    _, _, flat, _ = _recover(d["bits"], d["count"])
    out = np.where(flat, d["ink"], d["frame"]).astype(np.int8)
    return out.T.copy() if d["vertical"] else out


def _twist(flat, period, pitch):
    return np.stack([np.roll(flat[:, j], pitch * (j // period))
                     for j in range(flat.shape[1])], axis=1)


def _alias_free(h, length, period, pitch):
    # A transverse shift has this order on the circular cross-section. A
    # (order+1)-fold supertile advances by the same pitch as a single tile,
    # giving an exact competing parse if it still has three observations.
    # With zero net pitch, such supertiles flatten to the same output.
    order = h // gcd(h, pitch)
    return order == 1 or period * (order + 1) > length // 3


def _wrong_can_match(bits, pitch, correct, per_slice=False):
    return any(np.array_equal(item[3], correct)
               for item in _candidates(bits, pitch, per_slice))


def _sample_ribbon(rng, h, length, period, pitch, witness=False, area=None):
    # A finite random search plus a deterministic asymmetric template fallback.
    for attempt in range(257):
        if attempt < 256:
            density = float(rng.uniform(.22, .78))
            tile = rng.random((h, period)) < density
        else:
            tile = np.zeros((h, period), bool)
            tile[0, :] = True
            tile[1, -1] = True
        if not tile.any() or tile.all():
            continue
        if witness and tile[0].any() and tile[-1].any():
            # A blank edge row teaches that the frame fixes output extent.
            continue
        flat = tile[:, np.arange(length) % period]
        twisted = _twist(flat, period, pitch)
        noisy = twisted.copy()
        if attempt == 256:
            noisy[0, 0] ^= True
        else:
            errors = int(rng.integers(1, min(5, h * period, max(1, flat.size // 16)) + 1))
            # Distinct unrolled template coordinates: at most one flip each.
            groups = rng.choice(h * period, errors, replace=False)
            for index, raw in enumerate(groups):
                r, phase = divmod(int(raw), period)
                copies = list(range((length - 1 - phase) // period + 1))
                k = 0 if index == 0 else int(rng.choice(copies))
                noisy[(r + pitch * k) % h, phase + k * period] ^= True
        try:
            recovered_period, _, recovered, _ = _recover(noisy, pitch)
        except ValueError:
            continue
        if recovered_period != period or not np.array_equal(recovered, flat):
            continue
        if witness:
            # Reject incidental symmetries that hide any taught mechanism.
            if any(_wrong_can_match(noisy, wrong, flat)
                   for wrong in (0, -pitch, area)):
                continue
            if _wrong_can_match(noisy, pitch, flat, per_slice=True):
                continue
            if (np.array_equal(tile, tile[::-1]) or np.array_equal(tile, tile[:, ::-1])
                    or np.array_equal(tile, ~tile)):
                continue
            # The same helix also has a distinct valid interpretation at
            # twice the pitch: its tile is two successive original copies.
            # The outside count must choose between these clean answers.
            doubled_tile = np.concatenate((tile, np.roll(tile, pitch, axis=0)), axis=1)
            doubled_flat = doubled_tile[:, np.arange(length) % (2 * period)]
            try:
                alt_period, _, alt_flat, _ = _recover(noisy, 2 * pitch)
            except ValueError:
                continue
            if (alt_period != 2 * period or not np.array_equal(alt_flat, doubled_flat)
                    or np.array_equal(alt_flat, flat)):
                continue
        return noisy, flat
    raise RuntimeError("bounded ribbon sampling exhausted")


def _partition(length, pieces, rng=None):
    """Positive zones separated by at least one entirely blank line."""
    available = length - pieces + 1
    if available < pieces:
        return None
    sizes = np.ones(pieces, int)
    if rng is None:
        sizes[:] = available // pieces
        sizes[:available % pieces] += 1
    else:
        for _ in range(available - pieces):
            sizes[int(rng.integers(pieces))] += 1
    result, start = [], 0
    for size in sizes:
        result.append((start, start + int(size)))
        start += int(size) + 1
    return result


def _counter_options(h, w, count, area):
    options = []
    for nr in range(1, min(count, (h + 1) // 2) + 1):
        for nc in range(1, min(count, (w + 1) // 2) + 1):
            if not count <= nr * nc <= count + 3:
                continue
            rows, cols = _partition(h, nr), _partition(w, nc)
            capacities = sorted(((b - a) * (d - c) for a, b in rows for c, d in cols), reverse=True)
            if area is None or sum(capacities[:count]) >= area:
                options.append((nr, nc))
    return options


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _make_counter(h, w, count, rng, area=None):
    options = _counter_options(h, w, count, area)
    if not options:
        raise ValueError("counter cannot fit")
    for attempt in range(17):
        nr, nc = options[int(rng.integers(len(options)))]
        rows = _partition(h, nr, rng if attempt < 16 else None)
        cols = _partition(w, nc, rng if attempt < 16 else None)
        boxes = [(a, b, c, d) for a, b in rows for c, d in cols]
        rng.shuffle(boxes)
        if area is not None:
            boxes.sort(key=lambda box: (box[1] - box[0]) * (box[3] - box[2]), reverse=True)
        boxes = boxes[:count]
        capacities = [(b - a) * (d - c) for a, b, c, d in boxes]
        if area is None or sum(capacities) >= area:
            break
    sizes = [1] * count
    if area is not None:
        for _ in range(area - count):
            i = int(rng.choice([i for i in range(count) if sizes[i] < capacities[i]]))
            sizes[i] += 1
    mask = np.zeros((h, w), bool)
    singleton_collection = area is None and rng.random() < .18
    for i, (a, b, c, d) in enumerate(boxes):
        bh, bw = b - a, d - c
        if area is not None:
            obj = _grow(bh, bw, sizes[i], rng)
        elif singleton_collection:
            obj = _grow(bh, bw, 1, rng)
        else:
            oh, ow = int(rng.integers(1, bh + 1)), int(rng.integers(1, bw + 1))
            mode = int(rng.integers(5))
            if mode == 0:
                obj = _grow(oh, ow, int(rng.integers(1, oh * ow + 1)), rng)
            elif mode == 1:
                obj = np.ones((1, ow), bool)
            elif mode == 2:
                obj = np.ones((oh, 1), bool)
            else:
                obj = np.ones((oh, ow), bool)
                if mode == 4 and oh >= 3 and ow >= 3:
                    obj[1:-1, 1:-1] = False
            a += int(rng.integers(bh - obj.shape[0] + 1))
            c += int(rng.integers(bw - obj.shape[1] + 1))
        mask[a:a + obj.shape[0], c:c + obj.shape[1]] = obj
    _require(len(_components(mask)) == count == len(_components(mask, True)), "counter objects merged")
    return mask


def _place(rng, noisy, flat, period, count, palette, vertical, area):
    frame, bg, ink, counter_ink = palette
    h, length = noisy.shape
    fh, fw = h + 4, length + 4
    ribbon = np.full((fh, fw), frame, np.int8)
    ribbon[1:-1, 1:-1] = bg
    ribbon[2:-2, 2:-2] = np.where(noisy, ink, frame)
    options = []
    for side in ("top", "bottom", "left", "right"):
        for gap in (1, 2):
            for thickness in range(1, 7):
                bh, bw = (thickness, fw) if side in ("top", "bottom") else (fh, thickness)
                total_h = fh + gap + bh if side in ("top", "bottom") else fh
                total_w = fw if side in ("top", "bottom") else fw + gap + bw
                if max(total_h, total_w) <= 30 and _counter_options(bh, bw, count, area):
                    options.append((side, gap, bh, bw, total_h, total_w))
    if not options:
        raise ValueError("no exterior counter layout fits")
    side, gap, bh, bw, th, tw = options[int(rng.integers(len(options)))]
    counters = _make_counter(bh, bw, count, rng, area)
    # Independent asymmetric exterior margins; no fixed input alignment.
    extra_h = int(rng.integers(min(4, 30 - th) + 1))
    extra_w = int(rng.integers(min(4, 30 - tw) + 1))
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    g = np.full((th + extra_h, tw + extra_w), bg, np.int8)
    rr = top + (bh + gap if side == "top" else 0)
    rc = left + (bw + gap if side == "left" else 0)
    cr = top + (fh + gap if side == "bottom" else 0)
    cc = left + (fw + gap if side == "right" else 0)
    g[rr:rr + fh, rc:rc + fw] = ribbon
    region = g[cr:cr + bh, cc:cc + bw]
    region[counters] = counter_ink
    out = np.where(flat, ink, frame).astype(np.int8)
    return {"input": g.T.copy() if vertical else g,
            "output": out.T.copy() if vertical else out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    g = np.asarray(grid, np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    palettes = [tuple(colors[(i + j) % 10] for j in range(4))
                for i in range(int(rng.integers(3, 6)))]
    counts = list(map(int, rng.choice([1, 2, 3], 2, replace=False)))
    periods = list(map(int, rng.permutation([2, 3])))
    area = int(rng.integers(12, 19))
    base_axis = bool(rng.integers(2))
    mixed = rng.random() >= .25
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            witness = split == "train" and index < 2
            for attempt in range(32):
                n = counts[index] if witness else int(rng.integers(1, 7))
                if witness:
                    p = periods[index]
                    choices = [h for h in range(4, 8) if (2 * n) % h and area % h != n % h]
                    h = int(rng.choice(choices))
                    lengths = [length for length in range(max(12, 6 * p), 21)
                               if (length % p == 0) == (index == 0)]
                    length = int(rng.choice(lengths))
                elif split == "train":
                    p = int(rng.integers(2, 6))
                    length = int(rng.integers(max(10, 4 * p), 21))
                    h = int(rng.integers(3, min(8, length - 1) + 1))
                else:
                    # The shift is longer axial extent, including unfamiliar
                    # partial tails, while every decoding convention is taught.
                    length = int(rng.integers(21, 25))
                    p = int(rng.integers(2, length // 4 + 1))
                    h = int(rng.integers(3, 10))
                if not witness and not _alias_free(h, length, p, n):
                    compatible = [q for q in range(2, length // 4 + 1)
                                  if _alias_free(h, length, q, n)]
                    p = int(rng.choice(compatible))
                vertical = (base_axis ^ bool(index) if witness and mixed else
                            bool(rng.integers(2)) if mixed else base_axis)
                palette = palettes[index] if witness else palettes[int(rng.integers(len(palettes)))]
                noisy, flat = _sample_ribbon(rng, h, length, p, n, witness, area if witness else None)
                pair = _place(rng, noisy, flat, p, n, palette, vertical, area if witness else None)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    d = _decode(inp)
    period, tile, flat, candidates = _recover(d["bits"], d["count"])
    normalized_out = out.T if d["vertical"] else out
    expected = np.where(flat, d["ink"], d["frame"])
    _require(np.array_equal(normalized_out, expected), "wrong straightened output")
    clean_twisted = _twist(flat, period, d["count"])
    flips = int(np.count_nonzero(clean_twisted != d["bits"]))
    _require(1 <= flips <= max(1, flat.size // 8), "missing or excessive binary corruption")
    _require(d["bits"].shape[1] // period >= 4, "insufficient generated repetition evidence")
    _require(any(flat[:, phase].any() for phase in range(period)), "empty motif")
    first_copy = d["bits"][:, np.arange(flat.shape[1]) % period]
    try:
        alt_period, _, doubled_flat, _ = _recover(d["bits"], 2 * d["count"])
        doubled_changes = (not np.array_equal(doubled_flat, flat)
                           and alt_period == 2 * period)
    except ValueError:
        doubled_changes, doubled_flat = False, None
    d.update({"period": period, "tile": tile, "flat": flat, "flips": flips,
              "supported_periods": tuple(item[0] for item in candidates),
              "area": int(d["counter"].sum()), "sizes": tuple(map(len, d["groups"])),
              "first_copy_wrong": not np.array_equal(first_copy, flat),
              "blank_edge": not tile[0].any() or not tile[-1].any(),
              "double_count_changes": doubled_changes, "doubled_flat": doubled_flat,
              "zero_pitch_wrong": not _wrong_can_match(d["bits"], 0, flat),
              "reverse_pitch_wrong": not _wrong_can_match(d["bits"], -d["count"], flat),
              "area_pitch_wrong": not _wrong_can_match(d["bits"], int(d["counter"].sum()), flat),
              "per_slice_wrong": not _wrong_can_match(d["bits"], d["count"], flat, True)})
    return d


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _analyze(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(d)
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
    train = records["train"]
    _require(len({d["count"] for d in train}) >= 2, "training needs distinct counts")
    _require(len({d["period"] for d in train}) >= 2, "training needs distinct motif periods")
    _require(any(a["area"] == b["area"] and a["count"] != b["count"]
                 for a in train for b in train), "component count is confounded with ink area")
    for feature in ("first_copy_wrong", "blank_edge", "double_count_changes", "zero_pitch_wrong", "reverse_pitch_wrong",
                    "area_pitch_wrong", "per_slice_wrong"):
        _require(sum(d[feature] for d in train) >= 2, "missing teaching evidence: " + feature)
    _require(any(d["bits"].shape[1] % d["period"] == 0 for d in train), "missing complete-cycle witness")
    _require(any(d["bits"].shape[1] % d["period"] for d in train), "missing partial-cycle witness")
    for role in ("frame", "bg", "ink", "counter_ink"):
        _require(len({d[role] for d in train}) >= 2, "literal color role fixed: " + role)
    _require(min(d["bits"].shape[1] for d in records["test"])
             > max(d["bits"].shape[1] for d in train), "missing longer-ribbon test shift")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> dict:
    """Bounded 42-seed structural/semantic checks and every legal count pair."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_analyze(pair) for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    signatures = {_signature(puzzle) for puzzle in puzzles}
    shapes = {d["grid"].shape for d in rows}
    normalized = set()
    for d in rows:
        mapping = {}
        normalized.add(tuple(mapping.setdefault(int(c), len(mapping)) for c in d["grid"].ravel()))
    structures = {_key(d["tile"]) for d in rows}
    counts = {d["count"] for d in rows}
    periods = {d["period"] for d in rows}
    thicknesses = {d["bits"].shape[0] for d in rows}
    sides = {d["side"] for d in rows}
    _require(len(signatures) == 42, "puzzle diversity collapsed")
    _require(len(shapes) >= 40 and len(normalized) >= 180 and len(structures) >= 130, "geometric diversity collapsed")
    _require(counts == set(range(1, 7)) and periods == set(range(2, 7)), "count/period coverage collapsed")
    _require(thicknesses == set(range(3, 10)), "cross-section range collapsed")
    _require(sides == {"top", "bottom", "left", "right"}, "counter placement collapsed")
    _require({d["vertical"] for d in rows} == {False, True}, "missing axis")
    _require(any(d["count"] >= d["bits"].shape[0] for d in rows), "missing modular wrap case")
    _require(any(d["count"] % d["bits"].shape[0] == 0 for d in rows), "missing no-net-shift case")
    _require(any(max(d["sizes"]) == 1 for d in rows) and any(max(d["sizes"]) >= 10 for d in rows), "counter size variation collapsed")
    for n in range(1, 7):
        _require({d["vertical"] for d in rows if d["count"] == n} == {False, True}, "count/axis coupling")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(9091, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "ignored explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    _require(_signature(generate_puzzle(0)) == _signature(puzzles[0]), "nondeterminism")
    # Actual input intervention: merge all counter objects into one object.
    # Preserve a frame and background gap, and check the resulting query is
    # interpreted using its changed visible count rather than stored metadata.
    changed_count_checks = 0
    for d in rows:
        if d["count"] <= 1:
            continue
        g = d["grid"].copy()
        g[d["counter"]] = d["bg"]
        point = tuple(map(int, np.argwhere(d["counter"])[0]))
        g[point] = d["counter_ink"]
        try:
            alternative = solve(g)
            correct = np.where(d["flat"], d["ink"], d["frame"])
            if d["vertical"]:
                correct = correct.T
            changed_count_checks += not np.array_equal(alternative, correct)
        except ValueError:
            # A changed pitch can invalidate all sparse periodic explanations.
            changed_count_checks += 1
    _require(changed_count_checks >= 80, "object count has no causal effect")
    # Valid counterfactuals, not merely rejected inputs: the identical payload
    # gets a different uniquely supported answer with twice as many objects.
    valid_counterfactuals = 0
    for serial, d in enumerate(rows):
        if not d["double_count_changes"] or 2 * d["count"] > 6:
            continue
        pair = _place(np.random.default_rng(6060 + serial), d["bits"], d["doubled_flat"],
                      2 * d["period"], 2 * d["count"],
                      tuple(d[role] for role in ("frame", "bg", "ink", "counter_ink")),
                      d["vertical"], None)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "valid counterfactual solve failed")
        original = np.where(d["flat"], d["ink"], d["frame"])
        if d["vertical"]:
            original = original.T
        _require(not np.array_equal(original, pair["output"]), "counterfactual answer did not change")
        valid_counterfactuals += 1
    _require(valid_counterfactuals >= 84, "missing valid count-selected alternative answers")
    # Recolor the whole grid: semantic roles must commute with a bijection.
    mapping = np.array([4, 7, 1, 9, 0, 6, 3, 8, 5, 2], np.int8)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "literal color dependence")
        _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "axis transposition failed")
        _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "nested-list input failed")
    # Reject an altered label, not just schema violations.
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("validator accepted an altered output")
    # Regression for the exact supertile alias found during verification.
    tile = np.zeros((9, 2), bool)
    tile[0, 0] = tile[1, 1] = True
    ambiguous = _twist(tile[:, np.arange(24) % 2], 2, 6)
    ambiguous[0, 0] ^= True
    cs = _candidates(ambiguous, 6)
    _require({2, 8}.issubset({item[0] for item in cs}), "alias regression lacks competing parses")
    try:
        _recover(ambiguous, 6)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting period repairs were silently accepted")
    tied = np.zeros((4, 8), bool)
    tied[0, :4:2] = True
    tied[1, :] = True
    _require(2 not in {item[0] for item in _candidates(tied, 0)}, "tied phase got a majority")
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_inputs": len(normalized),
            "binary_templates": len(structures), "counts": sorted(counts),
            "periods": sorted(periods), "thicknesses": sorted(thicknesses),
            "counter_sides": sorted(sides), "counter_interventions": changed_count_checks,
            "valid_count_counterfactuals": valid_counterfactuals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
