"""Registered torn weave: registration-coupled periodic error/erasure repair.

Standard library and NumPy only. No parent imports, cached examples or hidden
palette/period conventions are used by the input-only solver.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _components(cells, diagonal=True):
    todo = set(cells)
    result = []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _repair_band(observations):
    """Unknown=-1. Enumerate all strict-majority, low-error explanations."""
    height, length = observations.shape
    colors = set(map(int, np.unique(observations))) - {-1}
    if len(colors) != 2:
        raise ValueError("band must expose two stitch colors")
    c0, c1 = sorted(colors)
    observed = int(np.count_nonzero(observations >= 0))
    limit = max(1, observed // 12)
    candidates = []
    for period in range(1, length // 3 + 1):
        template = np.empty((height, period), dtype=np.int8)
        errors, supported = 0, True
        for row in range(height):
            for phase in range(period):
                values = observations[row, phase::period]
                n0, n1 = int(np.sum(values == c0)), int(np.sum(values == c1))
                if n0 + n1 < 3 or n0 == n1:
                    supported = False
                    break
                template[row, phase] = c0 if n0 > n1 else c1
                errors += min(n0, n1)
            if not supported:
                break
        if not supported or errors > limit:
            continue
        if any(len(set(map(int, row))) != 2 for row in template):
            continue
        clean = template[:, np.arange(length) % period]
        candidates.append((period, clean, errors))
    if not candidates:
        raise ValueError("no supported vector period")
    if any(not np.array_equal(candidates[0][1], x[1]) for x in candidates[1:]):
        raise ValueError("competing periodic completions")
    period, clean, errors = candidates[0]
    return clean, period, errors, tuple(x[0] for x in candidates)


def _decode_registered(anchor, mover):
    shifted = {(r - 1, c): v for (r, c), v in mover.items()}
    if set(anchor) & set(shifted):
        raise ValueError("registered observations collide")
    merged = dict(anchor)
    merged.update(shifted)
    top, bottom, left, right = _bbox(merged)
    height, length = bottom - top + 1, right - left + 1
    if not (4 <= height <= 10 and 12 <= length <= 26):
        raise ValueError("fabric extent outside family")
    observations = np.full((height, length), -1, dtype=np.int8)
    owners = np.full((height, length), -1, dtype=np.int8)
    for owner, piece in enumerate((anchor, shifted)):
        for (r, c), value in piece.items():
            observations[r - top, c - left] = value
            owners[r - top, c - left] = owner
    row_pairs = []
    for row in observations:
        colors = frozenset(map(int, row)) - {-1}
        if len(colors) != 2:
            raise ValueError("every fabric row must expose both stitch colors")
        row_pairs.append(colors)
    bands = []
    start = 0
    while start < height:
        end = start + 1
        while end < height and row_pairs[end] == row_pairs[start]:
            end += 1
        clean, period, errors, supported = _repair_band(observations[start:end])
        bands.append({"start": start, "end": end, "colors": row_pairs[start],
                      "clean": clean, "period": period, "errors": errors,
                      "supported": supported})
        start = end
    signatures = [b["colors"] for b in bands]
    if not 1 <= len(bands) <= 4 or len(set(signatures)) != len(bands):
        raise ValueError("invalid color bands")
    if len(bands) > 1 and len(set.intersection(*(set(s) for s in signatures))) != 1:
        raise ValueError("bands must share one common stitch color")
    clean = np.concatenate([b["clean"] for b in bands], axis=0)
    return {"observations": observations, "owners": owners, "clean": clean,
            "bands": bands, "bounds": (top, bottom, left, right),
            "anchor": anchor, "shifted": shifted}


def _interpret(input_grid):
    g = _grid(input_grid)
    background = int(g[0, 0])
    if not (np.all(g[0] == background) and np.all(g[-1] == background)
            and np.all(g[:, 0] == background) and np.all(g[:, -1] == background)):
        raise ValueError("input requires a uniform blank outer margin")
    parts = _components({tuple(map(int, q)) for q in np.argwhere(g != background)})
    if len(parts) != 2:
        raise ValueError("expected two eight-connected swatches")
    candidates = []
    for marker in sorted(set(map(int, np.unique(g))) - {background}):
        bars = _components({tuple(map(int, q)) for q in np.argwhere(g == marker)}, False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]) or not 2 <= len(bars[0]) <= 5:
            continue
        for axis in (0, 1):
            oriented = [{(q[axis], q[1 - axis]) for q in bar} for bar in bars]
            if any(len({r for r, c in bar}) != 1
                   or max(c for r, c in bar) - min(c for r, c in bar) + 1 != len(bar)
                   for bar in oriented):
                continue
            records = []
            for bar, obar in zip(bars, oriented):
                own = [part for part in parts if bar <= part]
                if len(own) != 1:
                    break
                r0, _, c0, _ = _bbox(obar)
                data = {(q[axis] - r0, q[1 - axis] - c0): int(g[q])
                        for q in own[0] if int(g[q]) != marker}
                negative = any((-1, c) in data for c in range(len(bar)))
                positive = any((1, c) in data for c in range(len(bar)))
                if negative == positive:
                    break
                records.append((negative, data, own[0]))
            if len(records) != 2 or records[0][0] == records[1][0] or records[0][2] == records[1][2]:
                continue
            records.sort(key=lambda x: not x[0])
            try:
                decoded = _decode_registered(records[0][1], records[1][1])
            except ValueError:
                continue
            decoded.update(background=background, marker=marker, axis=axis,
                           bar_length=len(bars[0]), parts=[x[2] for x in records])
            candidates.append(decoded)
    if len(candidates) != 1:
        raise ValueError("input must have exactly one valid registration and repair")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the completed fabric entirely from the input's visible evidence."""
    parsed = _interpret(input_grid)
    return parsed["clean"].T.copy() if parsed["axis"] else parsed["clean"].copy()


def _local_wrong_witness(parsed):
    obs, owners, clean = parsed["observations"], parsed["owners"], parsed["clean"]
    for band in parsed["bands"]:
        p = band["period"]
        for row in range(band["start"], band["end"]):
            for phase in range(p):
                expected = clean[row, phase]
                for owner in (0, 1):
                    mask = owners[row, phase::p] == owner
                    values = obs[row, phase::p][mask]
                    if len(values) and np.sum(values != expected) > np.sum(values == expected):
                        return True
    return False


def _extent_width(piece):
    _, _, left, right = _bbox(piece)
    return right - left + 1


def _pair_info(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    p = _interpret(inp)
    expected = p["clean"].T if p["axis"] else p["clean"]
    assert np.array_equal(out, expected), "incorrect registration, repair or crop"
    obs, clean = p["observations"], p["clean"]
    holes = obs < 0
    flips = (obs >= 0) & (obs != clean)
    assert np.any(holes) and np.any(flips), "both erasures and swaps must affect the answer"
    assert _local_wrong_witness(p), "missing cross-swatch correction witness"
    widths = [_extent_width(p[k]) for k in ("anchor", "shifted")]
    assert clean.shape[1] > max(widths), "edge-alignment shortcut must have wrong extent"
    assert all(b["errors"] >= 1 for b in p["bands"])
    observed_union = np.where(holes, p["background"], obs)
    assert not np.array_equal(observed_union, clean), "omitting periodic repair must change answer"
    top, bottom, left, right = p["bounds"]
    guides = {(r, c) for r in (-1, 0) for c in range(p["bar_length"])}
    assert guides <= set(p["anchor"]) | set(p["shifted"]), "guides must be covered transparently"
    guide_erase = clean.copy()
    for r, c in guides:
        guide_erase[r - top, c - left] = p["background"]
    assert not np.array_equal(guide_erase, clean)
    # A wrong adjacent-seam order or coincident bars gives a different support
    # pattern or crop even when no valid fabric decoding is possible there.
    for seam in (0, 1):
        wrong = dict(p["anchor"])
        mover_original = {(r + 1, c): v for (r, c), v in p["shifted"].items()}
        wrong.update({(r + seam, c): v for (r, c), v in mover_original.items()})
        assert _bbox(wrong) != p["bounds"], "wrong seam not exposed by material extents"
    common = (set.intersection(*(set(b["colors"]) for b in p["bands"]))
              if len(p["bands"]) > 1 else set())
    accent_hole = bool(common and np.any(holes & (clean != next(iter(common)))))
    return {"axis": p["axis"], "bands": len(p["bands"]),
            "thicknesses": tuple(b["end"] - b["start"] for b in p["bands"]),
            "length": clean.shape[1], "height": clean.shape[0],
            "periods": tuple(b["period"] for b in p["bands"]),
            "bar_length": p["bar_length"], "holes": int(np.sum(holes)),
            "flips": int(np.sum(flips)), "accent_hole": accent_hole,
            "partial": any(clean.shape[1] % b["period"] for b in p["bands"]),
            "palette": (p["background"], p["marker"], tuple(sorted(set(map(int, clean.ravel()))))),
            "shape": inp.shape, "output_shape": out.shape}


def _cut(rng, height, length, offset, right_cut, split, bar_start, bar_length):
    style = int(rng.integers(4))
    if style == 0:
        boundary = rng.integers(0, height - 1, size=length)
    elif style == 1:
        lo, hi = rng.integers(0, height - 1, size=2)
        boundary = np.rint(np.linspace(lo, hi, length)).astype(int)
    elif style == 2:
        wave = int(rng.integers(3, 10))
        phase = int(rng.integers(wave))
        x = (np.arange(length) + phase) % wave
        boundary = np.rint((height - 2) * np.minimum(x, wave - x) / (wave / 2)).astype(int)
    else:
        boundary = np.empty(length, dtype=int)
        cursor = 0
        while cursor < length:
            span = int(rng.integers(1, 6))
            boundary[cursor:cursor + span] = int(rng.integers(height - 1))
            cursor += span
    boundary[bar_start:bar_start + bar_length] = split - 1
    owner = np.empty((height, length), dtype=np.int8)
    for col in range(length):
        if col < offset:
            owner[:, col] = 0
        elif col >= length - right_cut:
            owner[:, col] = 1
        else:
            owner[:, col] = np.arange(height) > boundary[col]
    return owner


def _stage(rng, pieces, background, marker):
    groups = []
    for payload, guide in pieces:
        t, b, l, r = _bbox(set(payload) | guide)
        groups.append(({(y - t, x - l): v for (y, x), v in payload.items()},
                       {(y - t, x - l) for y, x in guide}, b - t + 1, r - l + 1))
    if rng.integers(2):
        groups.reverse()
    a, b = groups
    modes = ["vertical"]
    if a[3] + b[3] + 1 <= 28:
        modes.extend(("horizontal", "diagonal"))
    mode = str(rng.choice(modes))
    gap = int(rng.integers(1, 5))
    if mode == "vertical":
        dr = a[2] + gap
        dc = int(rng.integers(-min(4, b[3] - 1), min(4, a[3] - 1) + 1))
    else:
        gap = min(gap, 28 - a[3] - b[3])
        dc = a[3] + gap
        dr = (a[2] + int(rng.integers(1, 5)) if mode == "diagonal"
              else int(rng.integers(-b[2] + 1, a[2])))
    payload = dict(a[0])
    payload.update({(r + dr, c + dc): v for (r, c), v in b[0].items()})
    guides = a[1] | {(r + dr, c + dc) for r, c in b[1]}
    t, bot, l, right = _bbox(set(payload) | guides)
    h, w = bot - t + 1, right - l + 1
    if h > 28 or w > 28:
        raise ValueError("staging too large")
    pads = [int(rng.integers(1, 4)) for _ in range(4)]
    pads[0] = min(pads[0], 29 - h)
    pads[1] = min(pads[1], 30 - h - pads[0])
    pads[2] = min(pads[2], 29 - w)
    pads[3] = min(pads[3], 30 - w - pads[2])
    grid = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), background, dtype=np.int8)
    for (r, c), v in payload.items():
        grid[r - t + pads[0], c - l + pads[2]] = v
    for r, c in guides:
        grid[r - t + pads[0], c - l + pads[2]] = marker
    return grid


def _make_pair(rng, profile, palette, used):
    for _attempt in range(240):
        count, length = profile["count"], profile["length"]
        height = int(rng.integers(max(4, count), 11))
        thicknesses = np.ones(count, dtype=int)
        for _ in range(height - count):
            thicknesses[int(rng.integers(count))] += 1
        available = list(range(2, min(profile["period_cap"], length // 4) + 1))
        periods = [int(rng.choice(available)) for _ in range(count)]
        if count > 1:
            periods[1] = int(rng.choice([p for p in available if p != periods[0]]))
        background, marker, common = map(int, palette[:3])
        accents = list(map(int, rng.permutation(palette[3:3 + count])))
        clean = np.empty((height, length), dtype=np.int8)
        row_periods, row_bands = [], []
        start = 0
        for index, (cross, period, accent) in enumerate(zip(thicknesses, periods, accents)):
            bits = rng.integers(0, 2, size=(int(cross), period), dtype=np.int8)
            for row in bits:
                if np.all(row == row[0]):
                    row[int(rng.integers(period))] ^= 1
            template = np.where(bits, accent, common).astype(np.int8)
            clean[start:start + cross] = template[:, np.arange(length) % period]
            row_periods.extend([period] * int(cross))
            row_bands.extend([index] * int(cross))
            start += int(cross)
        offset, right_cut = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        bar_length = profile["bar_length"]
        split = int(rng.integers(1, height))
        bar_start = int(rng.integers(offset, length - right_cut - bar_length + 1))
        owner = _cut(rng, height, length, offset, right_cut, split, bar_start, bar_length)
        observations = clean.copy()
        # Protect side strips, end rows and guide backing; delete sparse interior
        # observations, counting true phases solely in the construction path.
        eligible = [(r, c) for r in range(1, height - 1)
                    for c in range(offset, length - right_cut)
                    if not (r in (split - 1, split) and bar_start <= c < bar_start + bar_length)]
        target_holes = int(rng.integers(1, max(2, height * length // 9)))
        holes = 0
        for i in rng.permutation(len(eligible)):
            r, c = eligible[int(i)]
            period = row_periods[r]
            phase = c % period
            if np.count_nonzero(observations[r, phase::period] >= 0) <= 3:
                continue
            observations[r, c] = -1
            holes += 1
            if holes == target_holes:
                break
        if not holes:
            continue
        # On the top row the positive-backed swatch supplies just a short right
        # strip. Flip a phase for which it supplies exactly one observation.
        first_period = row_periods[0]
        forced = [c for c in range(length - right_cut, length)
                  if np.sum((owner[0, c % first_period::first_period] == 1)
                            & (observations[0, c % first_period::first_period] >= 0)) == 1]
        if not forced:
            continue
        forced_col = int(rng.choice(forced))
        flips = {(0, forced_col)}
        used_groups = {(0, forced_col % first_period)}
        for index in range(count):
            rows = [r for r in range(height) if row_bands[r] == index]
            observed = int(np.count_nonzero(observations[rows] >= 0))
            desired = int(rng.integers(1, min(3, max(1, observed // 18)) + 1))
            current = 1 if index == 0 else 0
            options = [(r, c) for r in rows for c in range(length) if observations[r, c] >= 0]
            for i in rng.permutation(len(options)):
                if current >= desired:
                    break
                r, c = options[int(i)]
                group = (r, c % row_periods[r])
                if group in used_groups:
                    continue
                flips.add((r, c))
                used_groups.add(group)
                current += 1
        for r, c in sorted(flips):
            accent = accents[row_bands[r]]
            observations[r, c] = accent if clean[r, c] == common else common
        payloads = [{}, {}]
        for r, c in np.argwhere(observations >= 0):
            r, c = int(r), int(c)
            which = int(owner[r, c])
            payloads[which][(r + which, c)] = int(observations[r, c])
        guide = {(split, c) for c in range(bar_start, bar_start + bar_length)}
        if any(set(data) & guide or len(_components(set(data) | guide)) != 1 for data in payloads):
            continue
        # Check the intended complete pattern before staging, independently of
        # solve's marker interpretation, using only registered observations.
        a = {(r - split, c - bar_start): v for (r, c), v in payloads[0].items()}
        b = {(r - split, c - bar_start): v for (r, c), v in payloads[1].items()}
        try:
            decoded = _decode_registered(a, b)
            if not np.array_equal(decoded["clean"], clean):
                continue
            if tuple(x["period"] for x in decoded["bands"]) != tuple(periods):
                continue
            grid = _stage(rng, [(data, guide) for data in payloads], background, marker)
        except ValueError:
            continue
        k = profile["rotation"]
        pair = {"input": np.rot90(grid, k).copy(), "output": np.rot90(clean, k).copy()}
        if _key(pair["input"]) in used:
            continue
        try:
            _pair_info(pair)
        except (AssertionError, ValueError):
            continue
        return pair
    raise RuntimeError("bounded swatch construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    changing_palette = bool(rng.integers(2))
    mixed_axis = bool(rng.random() < .6)
    rotation = int(rng.integers(4))
    regime = str(rng.choice(("longer", "more_bands", "bar_interpolation")))
    if regime == "more_bands":
        first_counts = list(map(int, rng.permutation((1, 2))))
    else:
        first = int(rng.integers(1, 4))
        second = int(rng.choice([c for c in (2, 3) if c != first]))
        first_counts = [first, second]
    first_bars = list(map(int, rng.permutation((2, 4))))
    puzzle, used = {"train": [], "test": []}, set()
    for split_name, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            training = split_name == "train"
            count = (first_counts[index] if training and index < 2 else
                     int(rng.integers(1, 3) if training else rng.integers(3, 5))
                     if regime == "more_bands" else int(rng.integers(1, 5)))
            length = int(rng.integers(14, 20) if training else rng.integers(22, 27)) if regime == "longer" else int(rng.integers(14, 27))
            bar_length = ((first_bars[index] if index < 2 else int(rng.choice((2, 4))))
                          if training else 3) if regime == "bar_interpolation" else int(rng.integers(2, 6))
            k = rotation
            if mixed_axis:
                k = ((rotation + index + 2 * int(rng.integers(2))) % 4
                     if training and index < 2 else int(rng.integers(4)))
            profile = {"count": count, "length": length, "bar_length": bar_length,
                       "rotation": k, "period_cap": 3 if regime == "longer" else 5}
            colors = rng.permutation(10) if changing_palette else palette
            pair = _make_pair(rng, profile, colors, used)
            used.add(_key(pair["input"]))
            puzzle[split_name].append(pair)
    # Guarantee a training hole whose intended value is an accent, to refute
    # treating unknown material as common-color stitches. Replace only the
    # multi-band teaching example, retaining all requested sampling policies.
    if not any(_pair_info(p)["accent_hole"] for p in puzzle["train"]):
        index = next(i for i, p in enumerate(puzzle["train"]) if _pair_info(p)["bands"] > 1)
        old = puzzle["train"][index]
        info = _pair_info(old)
        parsed = _interpret(old["input"])
        role_colors = [parsed["background"], parsed["marker"]]
        shared = next(iter(set.intersection(*(set(b["colors"]) for b in parsed["bands"]))))
        role_colors += [shared] + [next(iter(set(b["colors"]) - {shared})) for b in parsed["bands"]]
        role_colors += [c for c in range(10) if c not in role_colors]
        # Transposition alone is enough: its parsed axis is a visible convention.
        repair_profile = {"count": info["bands"], "length": info["length"],
                          "bar_length": info["bar_length"], "rotation": info["axis"],
                          "period_cap": 3 if regime == "longer" else 5}
        for _ in range(32):
            replacement = _make_pair(rng, repair_profile, np.array(role_colors), used)
            used.add(_key(replacement["input"]))
            if _pair_info(replacement)["accent_hole"]:
                puzzle["train"][index] = replacement
                break
        else:
            raise RuntimeError("could not expose accent-valued erasure")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            infos[split].append(_pair_info(pair))
    train, test = infos["train"], infos["test"]
    assert len({i["bands"] for i in train}) >= 2
    assert any(len(set(i["periods"])) > 1 for i in train)
    assert any(i["accent_hole"] for i in train)
    assert {i["axis"] for i in test} <= {i["axis"] for i in train}
    longer = min(i["length"] for i in test) > max(i["length"] for i in train)
    more_bands = min(i["bands"] for i in test) > max(i["bands"] for i in train)
    interpolation = ({i["bar_length"] for i in train} == {2, 4}
                     and {i["bar_length"] for i in test} == {3})
    assert longer or more_bands or interpolation, "no meaningful train/test shift"


def self_test():
    """Bounded 42-seed family survey, interface checks and semantic mutations."""
    fingerprints, shapes, structures, palettes = set(), set(), set(), set()
    bands, thicknesses, lengths, periods, bars, axes, hole_counts, flip_counts = (set() for _ in range(8))
    partial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        encoded = json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                              for s in ("train", "test")}, separators=(",", ":"))
        fingerprints.add(encoded)
        if seed in (0, 21, 41):
            repeated = generate_puzzle(seed)
            for s in puzzle:
                for a, b in zip(puzzle[s], repeated[s]):
                    assert all(np.array_equal(a[k], b[k]) for k in a)
        for s in puzzle:
            for pair in puzzle[s]:
                info = _pair_info(pair)
                bands.add(info["bands"])
                thicknesses.update(info["thicknesses"])
                lengths.add(info["length"])
                periods.update(info["periods"])
                bars.add(info["bar_length"])
                axes.add(info["axis"])
                hole_counts.add(info["holes"])
                flip_counts.add(info["flips"])
                partial += info["partial"]
                palettes.add(info["palette"])
                shapes.add((info["shape"], info["output_shape"]))
                roles = {}
                structures.add((pair["input"].shape, tuple(roles.setdefault(int(v), len(roles)) for v in pair["input"].ravel())))
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated input"
                # Reject actual wrong output labels: unfilled material, wrong
                # stitches, shortened domain, and unconditional guide erasure.
                parsed = _interpret(pair["input"])
                clean, obs = parsed["clean"], parsed["observations"]
                mutants = [np.where(obs < 0, parsed["background"], obs), clean[:, :-1]]
                guide_erased = clean.copy()
                top, _, left, _ = parsed["bounds"]
                for r in (-1, 0):
                    for c in range(parsed["bar_length"]):
                        guide_erased[r - top, c - left] = parsed["background"]
                mutants.append(guide_erased)
                changed = clean.copy()
                changed[0, 0] = next(c for c in parsed["bands"][0]["colors"] if c != changed[0, 0])
                mutants.append(changed)
                for wrong in mutants:
                    wrong = wrong.T.copy() if parsed["axis"] else wrong
                    try:
                        _pair_info({"input": pair["input"], "output": wrong})
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("semantic mutant accepted")
    assert len(fingerprints) == 42
    assert bands == {1, 2, 3, 4} and axes == {0, 1} and bars == {2, 3, 4, 5}
    assert periods == {2, 3, 4, 5}
    assert len(lengths) >= 10 and len(thicknesses) >= 6
    assert len(shapes) >= 80 and len(structures) >= 180 and len(palettes) >= 35
    assert len(hole_counts) >= 8 and len(flip_counts) >= 5 and partial >= 50
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(321, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")


if __name__ == "__main__":
    self_test()
    print("registered torn weave self-test passed")
