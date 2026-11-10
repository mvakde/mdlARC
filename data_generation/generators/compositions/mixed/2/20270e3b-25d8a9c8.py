"""Register two colored fragments, then classify complete assembled rows.

Only NumPy and the standard library are used. Generation constructs colored
pieces directly; it never imports or calls either parent generator.
"""
from __future__ import annotations

import hashlib
import json
from itertools import product

import numpy as np


HOST_SEEDS = (0,121283038,395249768,431996432,946929671,880256326,
              824067370,289437579,489443221,691080974,940637029,548484409,
              667580139,661266547,996820104,53727730,254380096,641331656,
              132957751,485869322,447371532,575978581,309360917,905164372,
              882975209,702006510,788430674,687896908,873368621,481433616,
              202520529,215298129,275523211,616261402,33565995,90465726,
              112706328,488338785,404085984,340687370,60911796,437427727)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < lo or
            (hi is not None and value > hi)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or g.dtype.kind not in "iu" or
            not all(1 <= n <= 30 for n in g.shape) or
            np.any(g < 0) or np.any(g > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _digest(puzzle):
    data = {s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                for p in puzzle[s]] for s in ("train", "test")}
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def _components(cells, diagonal=True):
    remaining = set(cells)
    parts = []
    directions = [(r, c) for r, c in product((-1, 0, 1), repeat=2)
                  if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    part.add(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _shift(data, dr, dc=0):
    return {(r + dr, c + dc): color for (r, c), color in data.items()}


def _crop(data, background, axis=0):
    if not data:
        raise ValueError("empty assembly")
    t, b, l, r = _bbox(data)
    result = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
    for (y, x), color in data.items():
        result[y - t, x - l] = color
    return result.T.copy() if axis else result


def _parse(input_grid):
    """Recover roles, ownership and bar normal solely from the input pixels.

    Normal/tangent coordinates are (row,col) for horizontal bars and (col,row)
    for vertical bars. A negatively backed bar is the anchor in either axis.
    """
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg) and
            np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("family inputs have a uniform background border")
    positions = {tuple(map(int, q)) for q in np.argwhere(g != bg)}
    parts = _components(positions)
    if len(parts) != 2:
        raise ValueError("expected two eight-connected fragments")
    choices = []
    for marker in sorted(set(map(int, np.unique(g))) - {bg}):
        guides = {tuple(map(int, q)) for q in np.argwhere(g == marker)}
        bars = _components(guides, diagonal=False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]):
            continue
        length = len(bars[0])
        for axis in (0, 1):
            records = []
            for part in parts:
                owned = [bar for bar in bars if bar <= part]
                if len(owned) != 1:
                    break
                bar = {(p[axis], p[1 - axis]) for p in owned[0]}
                n0, n1, t0, t1 = _bbox(bar)
                if n0 != n1 or t1 - t0 + 1 != length:
                    break
                data = {(p[axis] - n0, p[1 - axis] - t0): int(g[p])
                        for p in part - guides}
                if not data:
                    break
                negative = any((-1, t) in data for t in range(length))
                positive = any((1, t) in data for t in range(length))
                if negative == positive:
                    break
                records.append((negative, data, part, (n0, t0)))
            if len(records) != 2 or records[0][0] == records[1][0]:
                continue
            records.sort(key=lambda item: not item[0])
            a, b = records[0][1], _shift(records[1][1], -1)
            if any(a[q] != b[q] for q in a.keys() & b.keys()):
                continue
            merged = a | b
            assembled = _crop(merged, bg, axis)
            if max(assembled.shape) > 30:
                continue
            choices.append({"background": bg, "marker": marker, "axis": axis,
                            "length": length, "anchor": a,
                            "mover": records[1][1], "parts": parts,
                            "origins": (records[0][3], records[1][3]),
                            "assembled": assembled})
    if len(choices) != 1:
        raise ValueError(f"expected unique bar roles and axis, got {len(choices)}")
    return choices[0]


def _mask(g, predicate="equal", color=None):
    if predicate == "equal":
        return np.all(g == g[:, :1], axis=1)
    if predicate == "ends":
        return g[:, 0] == g[:, -1]
    if predicate == "first_two":
        return g[:, 0] == g[:, min(1, g.shape[1] - 1)]
    if predicate == "last_two":
        return g[:, -1] == g[:, max(0, g.shape[1] - 2)]
    if predicate == "occupied":
        return np.all(g != color, axis=1)
    if predicate == "ignore_background":
        return np.array([len(set(map(int, row)) - {color}) <= 1 for row in g])
    if predicate == "color_match":
        return np.all(g == color, axis=1)
    raise ValueError(predicate)


def _label(assembled, positive, background, predicate="equal", color=None):
    rows = np.where(_mask(assembled, predicate, color), positive, background)
    return np.repeat(rows[:, None], assembled.shape[1], axis=1).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No RNG, seed, training examples, file access, or generation state."""
    p = _parse(input_grid)
    return _label(p["assembled"], p["marker"], p["background"])


def _model(p, name):
    """Deliberately wrong rules for finite demonstration/necessity checks."""
    a, b = p["anchor"], _shift(p["mover"], -1)
    bg, positive = p["background"], p["marker"]
    if name.startswith("color_"):
        return _label(p["assembled"], positive, bg, "color_match", int(name[6:]))
    if name in ("ends", "first_two", "last_two", "occupied", "ignore_background"):
        return _label(p["assembled"], positive, bg, name, bg)
    if name == "aligned_guides":
        b = p["mover"]
    elif name == "reverse_seam":
        b = _shift(p["mover"], 1)
    elif name == "no_translation":
        # Preserve the input displacement between the two bar origins.
        an, at = p["origins"][0]
        bn, bt = p["origins"][1]
        b = _shift(p["mover"], bn - an, bt - at)
    elif name == "anchor_only":
        b = {}
    elif name == "mover_only":
        a = {}
    elif name == "xor":
        shared = a.keys() & b.keys()
        a = {q: c for q, c in a.items() if q not in shared}
        b = {q: c for q, c in b.items() if q not in shared}
    elif name in ("opaque_anchor", "opaque_mover"):
        front, back = (a, b) if name == "opaque_anchor" else (b, a)
        t, bot, l, r = _bbox(front)
        a = front
        b = {q: c for q, c in back.items()
             if not (t <= q[0] <= bot and l <= q[1] <= r)}
    merged = a | b
    if name in ("erase_guides", "paint_guides"):
        guides = {(n, t) for n in (-1, 0) for t in range(p["length"])}
        if name == "erase_guides":
            merged = {q: c for q, c in merged.items() if q not in guides}
        else:
            merged.update({q: positive for q in guides})
    if not merged:
        return None
    assembled = _crop(merged, bg, p["axis"])
    return _label(assembled, positive, bg)


_MODELS = ("aligned_guides", "reverse_seam", "no_translation", "anchor_only",
           "mover_only", "xor", "opaque_anchor", "opaque_mover", "erase_guides",
           "paint_guides", "ends", "first_two", "last_two", "occupied",
           "ignore_background") + tuple(f"color_{c}" for c in range(10))


def _survivors(train):
    candidates = set(_MODELS)
    for pair in train:
        p = _parse(pair["input"])
        candidates = {m for m in candidates
                      if np.array_equal(_model(p, m), pair["output"])}
    return candidates


def _completion_rows(parsed):
    """Positive rows that need both pieces in the same final crop.

    Comparing separately cropped fragment outputs could confuse a crop-size
    change with an equality change. Here missing pixels remain background in
    the exact assembled frame, so a witness requires actual row completion.
    """
    a, b = parsed["anchor"], _shift(parsed["mover"], -1)
    top, bottom, left, right = _bbox(a | b)
    shape = (bottom - top + 1, right - left + 1)
    frames = []
    for data in (a, b):
        frame = np.full(shape, parsed["background"], dtype=np.int8)
        for (r, c), color in data.items():
            frame[r - top, c - left] = color
        frames.append(frame.T.copy() if parsed["axis"] else frame)
    assembled = parsed["assembled"]
    return (_mask(assembled) & (assembled[:, 0] != parsed["background"])
            & ~_mask(frames[0]) & ~_mask(frames[1]))


def _palette(rng):
    colors = rng.permutation(10)
    return int(colors[0]), int(colors[1]), tuple(map(int, colors[2:2 + int(rng.integers(2, 6))]))


def _target(rng, h, w, palette, teaching, axis, base, start, length,
            category="mixed", overlap=False):
    bg, _, colors = palette
    a, b = colors[:2]
    styles = ["uniform", "uniform", "first", "middle", "last", "halves", "hole"]
    if teaching:
        # Positives with different literal colors refute a color-match rule.
        if axis == 0:
            positives = [base - 1, base]
        else:
            inside = int(rng.integers(start, start + length))
            outside = [r for r in range(h) if not start <= r < start + length]
            positives = [inside, int(rng.choice(outside))]
        row_styles = {}
        row_styles[positives[0]], row_styles[positives[1]] = "uniform_a", "uniform_b"
        remaining = [r for r in range(h) if r not in positives]
        rng.shuffle(remaining)
        for r, style in zip(remaining, styles[2:]):
            row_styles[r] = style
        for r in remaining[5:]:
            row_styles[r] = str(rng.choice(styles))
    else:
        row_styles = {}
        for r in range(h):
            if category == "all_positive":
                style = "uniform"
            elif category == "all_negative":
                style = str(rng.choice(styles[2:] + ["alternating", "random", "runs", "cavity"]))
            else:
                style = str(rng.choice(styles + ["alternating", "random", "runs", "cavity"]))
            row_styles[r] = style
    target = np.empty((h, w), dtype=np.int8)
    for r, style in row_styles.items():
        x, y = map(int, rng.choice(colors, 2, replace=False))
        row = np.full(w, x, dtype=np.int8)
        if style == "uniform_a":
            row[:] = a
        elif style == "uniform_b":
            row[:] = b
        elif style in ("first", "middle", "last", "hole"):
            index = (0 if style == "first" else w - 1 if style == "last"
                     else int(rng.integers(1, w - 1)))
            row[index] = bg if style == "hole" else y
        elif style in ("halves", "runs"):
            cut = base if axis == 1 and style == "halves" else int(rng.integers(1, w))
            row[cut:] = y
        elif style == "alternating":
            row[1::2] = y
        elif style == "cavity":
            left = int(rng.integers(1, w - 1))
            right = int(rng.integers(left + 1, w))
            row[left:right] = bg
        elif style == "random":
            row = np.asarray(rng.choice(colors, w), dtype=np.int8)
            if np.all(row == row[0]):
                row[-1] = y if int(row[0]) != y else x
        target[r] = row
    # Empty internal rows are legal whole-row-equality positives. Long vertical
    # guide bars keep each staged fragment connected across the missing strip.
    if not teaching and axis == 1 and length >= 3 and category != "all_negative" and rng.random() < .3:
        possible = [r for r in range(start + 1, start + length - 1) if 0 < r < h - 1]
        if possible:
            target[int(rng.choice(possible))] = bg
    return target


def _pieces(rng, target, axis, base, start, length):
    normal, tangent = target.shape if axis == 0 else target.shape[::-1]
    cuts = np.full(tangent, base, dtype=int)
    outside = [t for t in range(tangent) if not start <= t < start + length]
    # A wandering seam makes an irregular complementary partition. It is not
    # the solution rule: only the two straight guide bars control registration.
    for t in outside:
        cuts[t] = np.clip(base + int(rng.integers(-2, 3)), 1, normal - 1)
    if outside:
        cuts[outside[0]] = max(1, base - 1)
        if len(outside) > 1:
            cuts[outside[-1]] = min(normal - 1, base + 1)
    return cuts


def _split_target(rng, target, bg, axis, base, start, length, overlap):
    cuts = _pieces(rng, target, axis, base, start, length)
    a, b = {}, {}
    for r, c in np.argwhere(target != bg):
        r, c = int(r), int(c)
        n, t = (r, c) if axis == 0 else (c, r)
        q = n - base, t - start
        if n < cuts[t]:
            a[q] = int(target[r, c])
        else:
            b[(q[0] + 1, q[1])] = int(target[r, c])
    if overlap:
        # Copy compatible boundary pixels only outside the guide span. Neither
        # a conflicting-color overwrite nor an extra mating convention is used.
        possible = []
        for q, color in sorted(a.items()):
            n, t = q
            if 0 <= t < length:
                continue
            if (n + 2, t) in b:
                possible.append((q, color))
        if possible:
            rng.shuffle(possible)
            for (n, t), color in possible[:int(rng.integers(1, min(5, len(possible)) + 1))]:
                b[(n + 1, t)] = color
    return a, b


def _stage(rng, a, b, length, axis, palette, staging):
    bg, marker, _ = palette
    guide = {(0, t): marker for t in range(length)}
    groups = []
    for data in (a, b):
        cells = data | guide
        world = {(n, t) if axis == 0 else (t, n): color for (n, t), color in cells.items()}
        y0, y1, x0, x1 = _bbox(world)
        groups.append((_shift(world, -y0, -x0), y1 - y0 + 1, x1 - x0 + 1))
    if rng.integers(2):
        groups.reverse()
    (first, ah, aw), (second, bh, bw) = groups
    gap = int(rng.integers(1, 5))
    if staging == "horizontal":
        dy, dx = int(rng.integers(-bh + 1, ah)), aw + gap
    elif staging == "vertical":
        dy, dx = ah + gap, int(rng.integers(-bw + 1, aw))
    else:
        dy, dx = ah + gap, aw + gap
    cells = first | _shift(second, dy, dx)
    top, bottom, left, right = _bbox(cells)
    h, w = bottom - top + 1, right - left + 1
    if h + 2 > 30 or w + 2 > 30:
        return None
    pads = [1, 1, 1, 1]
    for i, extent in enumerate((h, h, w, w)):
        available = 30 - extent - pads[(i // 2) * 2] - pads[(i // 2) * 2 + 1]
        pads[i] += int(rng.integers(0, min(3, available) + 1))
    g = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), bg, dtype=np.int8)
    for (r, c), color in cells.items():
        g[r - top + pads[0], c - left + pads[2]] = color
    return g


def _make_pair(rng, profile, used, excluded_masks=()):
    for _ in range(180):
        h = int(rng.integers(7, 11)) if profile["teaching"] else int(rng.integers(4, 12))
        w = int(rng.integers(*profile["widths"]))
        axis = profile["axis"]
        normal, tangent = (h, w) if axis == 0 else (w, h)
        length = profile["length"] or int(rng.integers(1, min(6, tangent - 1) + 1))
        if length >= tangent:
            continue
        start = int(rng.integers(tangent - length + 1))
        base = int(rng.integers(2, normal - 1))
        palette = profile["palette"]
        target = _target(rng, h, w, palette, profile["teaching"], axis, base,
                         start, length, profile["category"], profile["overlap"])
        mask = tuple(map(bool, _mask(target)))
        if mask in excluded_masks:
            continue
        a, b = _split_target(rng, target, palette[0], axis, base, start, length,
                             profile["overlap"])
        guide = {(0, t) for t in range(length)}
        if (not a or not b or a.keys() & guide or b.keys() & guide or
                len(_components(a.keys() | guide)) != 1 or
                len(_components(b.keys() | guide)) != 1):
            continue
        g = _stage(rng, a, b, length, axis, palette, profile["staging"])
        if g is None or _key(g) in used:
            continue
        try:
            parsed = _parse(g)
        except ValueError:
            continue
        if not np.array_equal(parsed["assembled"], target):
            continue
        out = _label(target, palette[1], palette[0])
        return {"input": g, "output": out}
    raise RuntimeError("bounded fragment construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    axis_mode = str(rng.choice(("vertical", "vertical", "horizontal", "mixed")))
    regime = str(rng.choice(("bar_length", "staging", "width", "row_mask")))
    stable_palette = bool(rng.integers(2))
    common_palette = _palette(rng)
    overlaps = axis_mode != "horizontal" and bool(rng.integers(2))
    profiles = []
    for i in range(num_train + num_test):
        training = i < num_train
        local = i if training else i - num_train
        axis = 0 if axis_mode == "horizontal" else 1
        if axis_mode == "mixed":
            axis = 1 - local % 2
        length = (1 if local % 2 == 0 else 3) if training and regime == "bar_length" else 2 if regime == "bar_length" else 0
        staging = str(rng.choice(("horizontal", "vertical", "diagonal")))
        if regime == "staging":
            staging = str(rng.choice(("horizontal", "vertical"))) if training else "diagonal"
        widths = (5, 9) if regime == "width" and training else (9, 13) if regime == "width" else (5, 13)
        category = "mixed" if training and local < 2 else str(rng.choice(
            ("mixed", "all_positive", "all_negative"), p=(.78, .11, .11)))
        profiles.append({"axis": axis, "length": length, "staging": staging,
                         "widths": widths, "teaching": training and local < 2,
                         "category": category, "overlap": overlaps and axis == 1,
                         "palette": common_palette if stable_palette else _palette(rng)})
    puzzle, used = {"train": [], "test": []}, set()
    test_masks = set()
    for i, profile in enumerate(profiles):
        split = "train" if i < num_train else "test"
        excluded = test_masks.copy() if split == "test" else set()
        if split == "test" and regime == "row_mask":
            excluded.update(tuple(map(bool, _mask(_parse(p["input"])["assembled"])))
                            for p in puzzle["train"])
        pair = _make_pair(rng, profile, used, excluded)
        used.add(_key(pair["input"]))
        puzzle[split].append(pair)
        if split == "test":
            test_masks.add(tuple(map(bool, _mask(_parse(pair["input"])["assembled"]))))
    # Binary labeling can conceal an assembly difference. Repair only a missing
    # demonstration, requiring wrong surviving models to agree on all tests.
    for attempt in range(48):
        survivors = _survivors(puzzle["train"])
        harmful = {m for m in survivors if any(not np.array_equal(
            _model(_parse(p["input"]), m), p["output"]) for p in puzzle["test"])}
        if not harmful:
            break
        index = attempt % num_train
        profile = dict(profiles[index])
        profile["teaching"] = True
        # Preserve the held-out mask arrangement even when a demonstration is
        # replaced to expose an assembly convention.
        excluded = test_masks if regime == "row_mask" else ()
        replacement = _make_pair(rng, profile, used, excluded)
        # Replacements preserve all mandatory first-example contrasts. Reject a
        # replacement if it loses an already exposed distinction needed by tests.
        old = puzzle["train"][index]
        puzzle["train"][index] = replacement
        after = _survivors(puzzle["train"])
        if any(m in after for m in set(_MODELS) - survivors) or len(after & harmful) >= len(harmful):
            puzzle["train"][index] = old
        else:
            used.add(_key(replacement["input"]))
    else:
        raise RuntimeError("bounded teaching repair exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_positions(g):
    result = set()
    for row in g:
        colors, counts = np.unique(row, return_counts=True)
        if len(colors) == 2 and sorted(map(int, counts)) == [1, len(row) - 1]:
            value = colors[int(np.argmin(counts))]
            pos = int(np.flatnonzero(row == value)[0])
            result.add("first" if pos == 0 else "last" if pos == len(row) - 1 else "middle")
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, contrasts, train_axes, test_axes = set(), set(), set(), set()
    train_classes, positive_payloads, tests = set(), set(), set()
    omission_witness = completion_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            p = _parse(g)
            bg, marker = p["background"], p["marker"]
            a, b = p["anchor"], _shift(p["mover"], -1)
            shared = a.keys() & b.keys()
            _require(all(a[q] == b[q] for q in shared), "conflicting payload union")
            assembled = p["assembled"]
            _require(np.count_nonzero(assembled != bg) == len(a) + len(b) - len(shared), "lost union pixels")
            occupied = assembled != bg
            _require(occupied[0].any() and occupied[-1].any() and
                     occupied[:, 0].any() and occupied[:, -1].any(), "crop not tight")
            flags = tuple(map(bool, _mask(assembled)))
            expected = _label(assembled, marker, bg)
            _require(np.array_equal(expected, out), "incorrect row labels")
            _require(set(map(int, np.unique(out))) <= {bg, marker}, "payload colors leaked")
            _require(all(np.all(row == row[0]) for row in out), "output rows must be solid")
            if split == "train":
                train_axes.add(p["axis"])
                train_classes.update(flags)
                contrasts.update(_exception_positions(assembled))
                positive_payloads.update(int(row[0]) for row, flag in zip(assembled, flags) if flag and int(row[0]) != bg)
                omission_witness |= not np.array_equal(_model(p, "anchor_only"), out) and not np.array_equal(_model(p, "mover_only"), out)
                completion_witness |= bool(np.any(_completion_rows(p)))
            else:
                test_axes.add(p["axis"])
                _require(flags not in tests, "duplicate test row mask")
                tests.add(flags)
    _require(train_classes == {False, True}, "training must teach both row classes")
    _require(len(positive_payloads) >= 2, "training must refute fixed payload-color matching")
    _require(contrasts >= {"first", "middle", "last"}, "missing equality contrasts")
    _require(omission_witness, "training must need both registered fragments")
    _require(completion_witness, "training must show row equality created by union")
    _require(test_axes <= train_axes, "test bar axis is not demonstrated")
    survivors = _survivors(puzzle["train"])
    for pair in puzzle["test"]:
        p = _parse(pair["input"])
        _require(all(np.array_equal(_model(p, m), pair["output"]) for m in survivors),
                 "tests require an untaught alternative distinction")


def self_test():
    """Bounded, in-memory survey; no generated corpus or trace is saved."""
    shapes, masks, palettes, axes, lengths, fingerprints = set(), set(), set(), set(), set(), set()
    overlap_count = blank_rows = mixed_full_rows = 0
    mutant_witnesses = set()
    for i, seed in enumerate(HOST_SEEDS):
        nt, nq = ((2, 1), (4, 1), (7, 3), (10, 4))[i % 4]
        puzzle = generate_puzzle(seed, nt, nq)
        fingerprints.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                assembled, bg = p["assembled"], p["background"]
                shapes.add(assembled.shape)
                axes.add(p["axis"])
                lengths.add(p["length"])
                palettes.add((bg, p["marker"], tuple(map(int, np.unique(assembled)))))
                masks.add(tuple(map(bool, _mask(assembled))))
                overlap_count += bool(p["anchor"].keys() & _shift(p["mover"], -1).keys())
                blank_rows += sum(np.all(row == bg) for row in assembled)
                mixed_full_rows += sum(np.all(row != bg) and len(set(map(int, row))) > 1 for row in assembled)
                _require(np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"]), "solver mismatch")
                for m in _MODELS:
                    if m not in mutant_witnesses and not np.array_equal(_model(p, m), pair["output"]):
                        mutant_witnesses.add(m)
        if i < 2:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed, nt, nq)), "nondeterministic puzzle")
    _require(len(fingerprints) == 42, "duplicate seed puzzles")
    _require(len(shapes) >= 25 and len(masks) >= 50 and len(palettes) >= 40, "insufficient structural diversity")
    _require(axes == {0, 1} and lengths >= {1, 2, 3, 4, 5, 6}, "missing bar variation")
    _require(overlap_count > 10 and blank_rows > 0 and mixed_full_rows > 50, "missing union/equality cases")
    _require(mutant_witnesses == set(_MODELS), "no witness against a semantic mutant")
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(0, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count request ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The validator must reject corrupted answers, including a single wrong cell.
    p = generate_puzzle(0, 2, 1)
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic mutants)")
