"""Mending beams: periodic discrepancies are the obstacles, not visible ink.

Only standard-library modules and NumPy are used. Nothing is read or written by
generation, solving, validation, or self_test. See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)), f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "illegal grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _recognize(value):
    """Find the unique frame with ports; all four colors have geometric roles."""
    grid = _grid(value)
    choices = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation)
        h, width = a.shape
        if h % 2 != 1 or not 19 <= h <= 29 or not 9 <= width <= 30:
            continue
        frame = int(a[-1, 0])
        if not (np.all(a[-1] == frame) and np.all(a[:, 0] == frame)
                and np.all(a[:, -1] == frame)):
            continue
        top = a[0, 1:-1]
        ports = np.flatnonzero(top != frame)
        if not 1 <= len(ports) <= 4 or np.any(np.diff(ports) < 2):
            continue
        guide_values = np.unique(top[ports])
        if len(guide_values) != 1:
            continue
        guide = int(guide_values[0])
        spacers = a[1:-1:2, 1:-1]
        padding_values = np.unique(spacers[1:])
        if len(padding_values) != 1:
            continue
        padding = int(padding_values[0])
        first = np.full(width - 2, padding, dtype=np.int8)
        first[ports] = guide
        if not np.array_equal(spacers[0], first):
            continue
        data = a[2:-2:2, 1:-1]
        other = [int(c) for c in np.unique(data) if c != frame]
        if len(other) != 1 or not np.any(data == frame):
            continue
        ink = other[0]
        if len({frame, padding, ink, guide}) != 4:
            continue
        if set(map(int, np.unique(a))) != {frame, padding, ink, guide}:
            continue
        choices.append({"rotation": rotation, "a": a.copy(), "data": data.copy(),
                        "entries": set(map(int, ports)),
                        "roles": (frame, padding, ink, guide)})
    _require(len(choices) == 1, "input has no unique port/frame interpretation")
    return choices[0]


def _recover(data, frame, ink):
    """Accept only an unambiguous sparse, strict-majority periodic explanation.

    Enumerating three-copy competitors is deliberately broader than the
    constructor, which supplies at least four copies and at most one flipped
    observation per true phase/column. Neither fact is needed by this solver.
    """
    n, width = data.shape
    candidates = []
    allowance = max(1, data.size // 8)
    for period in range(1, n // 3 + 1):
        template = np.empty((period, width), dtype=np.int8)
        valid = True
        for phase in range(period):
            samples = data[phase::period]
            votes = np.count_nonzero(samples == ink, axis=0)
            if np.any(2 * votes == len(samples)):
                valid = False
                break
            template[phase] = np.where(2 * votes > len(samples), ink, frame)
        if not valid or len(np.unique(template)) != 2:
            continue
        ideal = template[np.arange(n) % period]
        errors = int(np.count_nonzero(data != ideal))
        if 1 <= errors <= allowance:
            candidates.append((period, ideal))
    _require(bool(candidates), "no supported sparse periodic repair")
    ideal = candidates[0][1]
    _require(all(np.array_equal(ideal, item[1]) for item in candidates),
             "competing periodic repairs")
    return ideal.copy(), candidates[0][0]


def _runs(row):
    runs = []
    c = 0
    while c < len(row):
        if not row[c]:
            c += 1
            continue
        start = c
        while c + 1 < len(row) and row[c + 1]:
            c += 1
        runs.append((start, c))
        c += 1
    return runs


def _route(defects, entries, policy="normal"):
    """Finite layer propagation on an immutable discrepancy map.

    This is a logical mask, not a saved execution trace. Only the returned
    paint/repair masks and scalar diagnostics are needed by the caller.
    """
    n, width = defects.shape
    paint = np.zeros((n + 1, width), dtype=bool)
    repaired = np.zeros_like(defects)
    active = set(entries)
    incoming = []
    stats = {"reached_runs": 0, "inert_runs": 0, "simultaneous": 0,
             "multi_hit": 0, "shared_flanks": 0, "clipped": 0,
             "singleton": 0, "extinguished": 0}
    reached_lengths = []
    inert_lengths = []
    for r in range(n):
        incoming.append(active.copy())
        paint[r, list(active)] = True
        runs = _runs(defects[r])
        hits = [(a, b) for a, b in runs if any(a <= c <= b for c in active)]
        inert_lengths.extend(b - a + 1 for a, b in runs if (a, b) not in hits)
        stats["simultaneous"] += len(hits) > 1
        following = {c for c in active if not defects[r, c]}
        if policy == "one_run":
            hits = hits[:1]
        elif policy == "ignore_singletons":
            hits = [(a, b) for a, b in hits if a != b]
        elif policy == "pixelwise":
            hits = [(c, c) for c in sorted(active) if defects[r, c]]
        if policy == "straight":
            for a, b in hits:
                repaired[r, a:b + 1] = True
            following = active.copy()
        else:
            for a, b in hits:
                repaired[r, a:b + 1] = True
                paint[r, max(0, a - 1):min(width, b + 2)] = True
                stats["multi_hit"] += sum(a <= c <= b for c in active) > 1
                stats["clipped"] += a == 0 or b == width - 1
                stats["singleton"] += a == b
                stats["reached_runs"] += 1
                reached_lengths.append(b - a + 1)
                for c in (a - 1, b + 1):
                    if 0 <= c < width:
                        stats["shared_flanks"] += c in following
                        following.add(c)
        if active and not following:
            stats["extinguished"] += 1
        active = following
        paint[r + 1, list(active)] = True
    stats["inert_runs"] = len(inert_lengths)
    stats["reached_lengths"] = tuple(reached_lengths)
    stats["inert_lengths"] = tuple(inert_lengths)
    return paint, repaired, incoming, stats


def _answer(parsed, ideal, policy="normal"):
    a = parsed["a"]
    data = parsed["data"]
    frame, padding, ink, guide = parsed["roles"]
    defects = data != ideal
    routing_ideal = ideal
    routing_defects = defects
    route_policy = policy
    if policy == "raw_ink":
        # Give this wrong navigator the true repair values as an extra advantage:
        # even oracle values cannot rescue navigating around visible ink.
        routing_defects = data == ink
        route_policy = "normal"
    elif policy == "copy_first":
        routing_ideal = np.repeat(data[:1], len(data), axis=0)
        routing_defects = data != routing_ideal
        route_policy = "normal"
    elif policy == "repair_all":
        route_policy = "normal"
    paint, repaired, _, stats = _route(routing_defects, parsed["entries"], route_policy)
    if policy == "repair_all":
        repaired = defects.copy()
    out = a.copy()
    region = out[2:-2:2, 1:-1]
    region[repaired] = routing_ideal[repaired]
    out[1:-1:2, 1:-1][paint] = guide
    return np.rot90(out, parsed["rotation"]).copy(), stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _recognize(input_grid)
    frame, _, ink, _ = parsed["roles"]
    ideal, _ = _recover(parsed["data"], frame, ink)
    return _answer(parsed, ideal)[0]


def _add_run(mask, occupied, row, left, right, period, limit):
    width = mask.shape[1]
    if not (0 <= left <= right < width):
        return False
    if int(mask.sum()) + right - left + 1 > limit:
        return False
    if np.any(occupied[row % period, left:right + 1]):
        return False
    if np.any(mask[row, max(0, left - 1):min(width, right + 2)]):
        return False
    mask[row, left:right + 1] = True
    occupied[row % period, left:right + 1] = True
    return True


def _mask(rng, n, width, period, style):
    mask = np.zeros((n, width), dtype=bool)
    occupied = np.zeros((period, width), dtype=bool)
    limit = mask.size // 8
    if style == "fork":
        a = int(rng.integers(1, 5))
        length = int(rng.integers(2, 5))
        b = a + length - 1
        r1 = int(rng.integers(0, n - 3))
        r2 = int(rng.integers(r1 + 1, n - 1))
        r3 = int(rng.integers(r2 + 1, n))
        entries = {int(rng.integers(a, b + 1))}
        if length >= 3 and rng.random() < .45:
            entries = {a, b}
        plans = ((r1, a, b), (r2, 0, a - 1), (r3, b + 1, b + 1))
        if not all(_add_run(mask, occupied, r, x, y, period, limit - 1)
                   for r, x, y in plans):
            return None
    elif style == "braid":
        left_length = int(rng.integers(2, 4))
        right_length = int(rng.integers(1, 4))
        a = int(rng.integers(1, width - left_length - right_length - 1))
        b = a + left_length - 1
        c = b + 2
        d = c + right_length - 1
        # Two clear samples before the first braid collision teach that a
        # legitimate change of periodic phase is transparent to the beam.
        r1 = int(rng.integers(2, n - 3))
        choices = [r for r in range(r1 + 1, n) if (r - r1) % period]
        if not choices:
            return None
        r2 = int(rng.choice(choices))
        entries = {int(rng.integers(a, b + 1)), int(rng.integers(c, d + 1))}
        plans = ((r1, a, b), (r1, c, d), (r2, a - 1, b + 1))
        if not all(_add_run(mask, occupied, r, x, y, period, limit - 1)
                   for r, x, y in plans):
            return None
    else:
        entries = set()
        quota = int(rng.integers(1, min(4, (width + 1) // 2) + 1))
        for c in map(int, rng.permutation(width)):
            if all(abs(c - x) >= 2 for x in entries):
                entries.add(c)
            if len(entries) == quota:
                break
        # A separated subset also permits entries at either transverse wall.
        if not entries:
            entries = {int(rng.integers(width))}
        rows = set(map(int, rng.choice(n, size=int(rng.integers(1, min(7, n) + 1)),
                                      replace=False)))
        active = entries.copy()
        for r in range(n):
            if r in rows:
                pivots = list(sorted(active))
                rng.shuffle(pivots)
                for pivot in pivots[:int(rng.integers(1, 4))]:
                    for _ in range(12):
                        max_length = min(width, max(1, limit - 1 - int(mask.sum())))
                        if max_length < 1:
                            break
                        if rng.random() < .25:
                            length = 1
                        elif rng.random() < .2:
                            length = max_length
                        else:
                            length = int(rng.integers(1, min(8, max_length) + 1))
                        lo = max(0, pivot - length + 1)
                        hi = min(pivot, width - length)
                        left = int(rng.integers(lo, hi + 1))
                        if _add_run(mask, occupied, r, left, left + length - 1,
                                    period, limit - 1):
                            break
            hit = [(a, b) for a, b in _runs(mask[r])
                   if any(a <= c <= b for c in active)]
            active = {c for c in active if not mask[r, c]}
            for a, b in hit:
                active.update(c for c in (a - 1, b + 1) if 0 <= c < width)
        if not np.any(mask):
            return None
    # Place genuine damaged but unreachable runs from the same discrepancy
    # alphabet. A separating clean cell prevents changing any reached run.
    _, _, incoming, _ = _route(mask, entries)
    added = 0
    desired = int(rng.integers(1, 4))
    candidates = rng.permutation(n * width)
    for index in candidates:
        r, c = divmod(int(index), width)
        length = int(rng.integers(1, min(4, width - c) + 1))
        if any(x in incoming[r] for x in range(c, c + length)):
            continue
        if _add_run(mask, occupied, r, c, c + length - 1, period, limit):
            added += 1
            if added == desired:
                break
    if not added:
        return None
    return mask, entries


def _assemble(data, entries, palette):
    frame, padding, ink, guide = palette
    n, width = data.shape
    a = np.full((2 * n + 3, width + 2), padding, dtype=np.int8)
    a[:, 0] = a[:, -1] = frame
    a[0] = a[-1] = frame
    a[2:-2:2, 1:-1] = data
    for c in entries:
        a[:2, c + 1] = guide
    return a


def _template(rng, period, width, deterministic=False):
    if deterministic:
        if period == 1:
            return (np.arange(width)[None, :] % 2).astype(np.int8)
        return (np.arange(period)[:, None] == np.arange(width)[None, :] % period).astype(np.int8)
    bits = rng.integers(0, 2, size=(period, width), dtype=np.int8)
    return bits


def _fallback_mask(n, width, period, style):
    """Small algebraic witness; width is distinct for every puzzle serial."""
    mask = np.zeros((n, width), dtype=bool)
    if style == "braid":
        mask[0, 3:5] = mask[0, 6:8] = True
        mask[1, 2:6] = True
        entries = {3, 6}
    else:
        mask[0, 2:5] = True
        mask[1, :2] = True
        mask[2, 5] = True
        entries = {3}
    mask[0, width - 1] = True
    return mask, entries


def _make_pair(rng, period, depth, style, palette, rotation, serial, seen):
    frame, padding, ink, guide = palette
    for attempt in range(100):
        if attempt < 96:
            low, high = (max(8, 4 * period), 13)
            if depth == "short":
                low, high = 8, 10
            elif depth == "long":
                low, high = max(12, 4 * period), 13
            n = int(rng.integers(low, high + 1))
            width = int(rng.integers(12 if style in ("fork", "braid") else 7, 29))
            sampled = _mask(rng, n, width, period, style)
            if sampled is None:
                continue
            mask, entries = sampled
            bits = _template(rng, period, width)
            if style in ("fork", "braid"):
                # Make a reached virtual bar visibly unlike a solid ink bar:
                # its endpoints demonstrate opposite binary flip directions.
                _, targeted, _, _ = _route(mask, entries)
                wide = next((r, a, b) for r in range(n) for a, b in _runs(mask[r])
                            if b > a and np.any(targeted[r, a:b + 1]))
                r, left, right = wide
                bits[r % period, left] = int(rng.integers(2))
                bits[r % period, right] = 1 - bits[r % period, left]
                if style == "braid":
                    # This entry belongs to the right-hand run, so these
                    # constraints cannot undo the mixed left-hand error run.
                    source = max(entries)
                    bits[0, source] = 0
                    bits[1, source] = 1
        else:
            n = max(12 if depth == "long" else 8, 4 * period)
            width = 12 + serial
            mask, entries = _fallback_mask(n, width, period, style)
            bits = _template(rng, period, width, deterministic=True)
        clean_bits = bits[np.arange(n) % period]
        data = np.where(clean_bits ^ mask, ink, frame).astype(np.int8)
        ideal = np.where(clean_bits, ink, frame).astype(np.int8)
        try:
            recovered, p = _recover(data, frame, ink)
        except ValueError:
            continue
        if p != period or not np.array_equal(recovered, ideal):
            continue
        a = _assemble(data, entries, palette)
        if attempt < 96 and rng.random() < .5:
            a = np.fliplr(a).copy()
        a = np.rot90(a, rotation).copy()
        signature = (a.shape, a.tobytes())
        if signature in seen:
            continue
        parsed = _recognize(a)
        ideal, _ = _recover(parsed["data"], frame, ink)
        out, stats = _answer(parsed, ideal)
        if not stats["reached_runs"] or not stats["inert_runs"]:
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded pair construction exhausted")


_SHORTCUTS = ("repair_all", "straight", "pixelwise", "raw_ink", "one_run",
              "ignore_singletons", "copy_first")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = "depth" if rng.random() < .45 else "turn"
    periods = (1, 2) if shift == "depth" else ((1, 2), (1, 3), (2, 3))[int(rng.integers(3))]
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    mixed_palette = rng.random() < .55
    train_rotation = int(rng.integers(4))
    test_rotation = (train_rotation + int(rng.choice([1, 3]))) % 4
    styles = ["fork", "braid"]
    order = list(map(int, rng.permutation(2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for serial in range(nt + nq):
        split = "train" if serial < nt else "test"
        i = serial if split == "train" else serial - nt
        if split == "train" and i < 2:
            witness = order[i]
            period, style = periods[witness], styles[witness]
        else:
            period = int(rng.choice(periods))
            style = "general"
        local_palette = tuple(map(int, rng.choice(10, 4, replace=False))) if mixed_palette else palette
        depth = ("short" if split == "train" else "long") if shift == "depth" else "any"
        rotation = train_rotation if shift == "turn" and split == "train" else (
            test_rotation if shift == "turn" else int(rng.integers(4)))
        pair = _make_pair(rng, period, depth, style, local_palette, rotation, serial, seen)
        seen.add((pair["input"].shape, pair["input"].tobytes()))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantic masks, causal witnesses, shortcuts and the visible shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen = set()
    rejected = set()
    flip_directions = set()
    mixed_error_run = False
    descriptions = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "shape changed")
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            parsed = _recognize(a)
            frame, padding, ink, guide = parsed["roles"]
            ideal, period = _recover(parsed["data"], frame, ink)
            expected, stats = _answer(parsed, ideal)
            _require(np.array_equal(out, expected), "incorrect mending beam output")
            defects = parsed["data"] != ideal
            paint, repaired, _, _ = _route(defects, parsed["entries"])
            actual = np.rot90(out, -parsed["rotation"])
            data_out = actual[2:-2:2, 1:-1]
            _require(np.array_equal(data_out != parsed["data"], repaired), "wrong repaired cells")
            _require(np.all(repaired <= defects), "an undamaged cell was mended")
            _require(np.array_equal(data_out[~repaired], parsed["data"][~repaired]),
                     "unreached damage or clean data changed")
            _require(np.array_equal(actual[[0, -1]], parsed["a"][[0, -1]])
                     and np.array_equal(actual[:, [0, -1]], parsed["a"][:, [0, -1]]),
                     "frame/ports changed")
            _require(np.all(np.isin(actual[1:-1:2, 1:-1], [padding, guide])),
                     "illegal spacer paint")
            _require(stats["reached_runs"] > 0 and stats["inert_runs"] > 0,
                     "pair needs both reached and untouched damaged runs")
            n, width = defects.shape
            _require(n >= 4 * period, "insufficient repeated evidence")
            descriptions[split].append((n, period, parsed["rotation"], stats))
            if split == "train" and index < 2:
                flip_directions.update((int(x == ink), int(y == ink)) for x, y in
                                       zip(parsed["data"][repaired], ideal[repaired]))
                mixed_error_run |= any(len(np.unique(parsed["data"][r, a:b + 1])) == 2
                                       for r in range(n) for a, b in _runs(defects[r])
                                       if np.any(repaired[r, a:b + 1]))
                for shortcut in _SHORTCUTS:
                    wrong, _ = _answer(parsed, ideal, shortcut)
                    if not np.array_equal(wrong, out):
                        rejected.add(shortcut)
    _require(rejected == set(_SHORTCUTS), "training admits a selected simpler rule")
    _require(flip_directions == {(0, 1), (1, 0)} and mixed_error_run,
             "training must expose both flips within a virtual barrier")
    train = descriptions["train"]
    test = descriptions["test"]
    _require(len({x[1] for x in train}) >= 2, "training needs two genuine periods")
    _require({x[1] for x in test} <= {x[1] for x in train}, "test introduces an undemonstrated period")
    for feature in ("simultaneous", "multi_hit", "shared_flanks", "clipped", "singleton"):
        _require(any(x[3][feature] for x in train[:2]), f"first two examples lack {feature}")
    depth_shift = min(x[0] for x in test) > max(x[0] for x in train)
    turn_shift = not ({x[2] for x in train} & {x[2] for x in test})
    _require(depth_shift or turn_shift, "missing train/test depth or direction shift")


def _digest(puzzle):
    value = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
             for s in ("train", "test")}
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 streamed seeds, bounded summaries, metamorphic and causal assertions."""
    signatures, shapes, structures = set(), set(), set()
    periods, row_counts, widths, entry_counts, run_lengths = set(), set(), set(), set(), set()
    roles = [set() for _ in range(4)]
    rotations = set()
    totals = {name: 0 for name in ("simultaneous", "multi_hit", "shared_flanks",
                                  "clipped", "singleton", "extinguished")}
    pair_count = partial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = pair["input"].copy()
                _require(np.array_equal(solve(original), pair["output"]), "solve disagreement")
                _require(np.array_equal(original, pair["input"]), "solve mutated its input")
                parsed = _recognize(original)
                ideal, period = _recover(parsed["data"], parsed["roles"][0], parsed["roles"][2])
                _, stats = _answer(parsed, ideal)
                n, width = parsed["data"].shape
                shapes.add(original.shape)
                periods.add(period)
                row_counts.add(n)
                widths.add(width)
                entry_counts.add(len(parsed["entries"]))
                run_lengths.update(stats["reached_lengths"])
                rotations.add(parsed["rotation"])
                partial += n % period != 0
                pair_count += 1
                for i, color in enumerate(parsed["roles"]):
                    roles[i].add(color)
                normalized = np.empty_like(parsed["a"])
                for i, color in enumerate(parsed["roles"]):
                    normalized[parsed["a"] == color] = i
                structures.add((normalized.shape, normalized.tobytes()))
                for name in totals:
                    totals[name] += stats[name]
    _require(len(signatures) == 42 and len(structures) >= 190, "structural variation collapsed")
    _require(len(shapes) >= 40 and len(widths) >= 15 and len(row_counts) >= 5, "geometry variation collapsed")
    _require(periods == {1, 2, 3} and rotations == {0, 1, 2, 3}, "period/direction variation collapsed")
    _require(all(x == set(range(10)) for x in roles), "palette role variation collapsed")
    _require(entry_counts == {1, 2, 3, 4}, "entry variation collapsed")
    _require(1 in run_lengths and max(run_lengths) >= 7 and partial > 0, "run/phase variation collapsed")
    # A fixed semantic witness: the run 4..6 is reached, then its left
    # descendant reaches a singleton at (3,3); errors (2,10),(5,0) stay.
    n, width = 8, 12
    bits = _template(np.random.default_rng(101), 2, width, deterministic=True)
    ideal = np.where(bits[np.arange(n) % 2], 2, 0).astype(np.int8)
    mask = np.zeros((n, width), dtype=bool)
    mask[0, 4:7] = True
    mask[3, 3] = mask[2, 10] = mask[5, 0] = True
    data = np.where(mask, 2 - ideal, ideal)
    a = _assemble(data, {5}, (0, 8, 2, 6))
    output = solve(a)
    actual_data = output[2:-2:2, 1:-1]
    _require(np.array_equal(actual_data[0, 4:7], ideal[0, 4:7])
             and actual_data[3, 3] == ideal[3, 3], "causal restoration witness failed")
    _require(actual_data[2, 10] == data[2, 10] and actual_data[5, 0] == data[5, 0],
             "unreached defects were restored")
    _require(np.all(output[1, 4:9] == 6), "upstream crosspiece endpoints wrong")
    for rotation in range(4):
        for reflect in (False, True):
            transform = lambda x: np.rot90(np.fliplr(x) if reflect else x, rotation)
            _require(np.array_equal(solve(transform(a)), transform(output)), "dihedral covariance failed")
    permutation = np.random.default_rng(17).permutation(10)
    _require(np.array_equal(solve(permutation[a]), permutation[output]), "color covariance failed")
    # Full-width virtual obstruction extinguishes every beam; later errors
    # remain, and the final gutter is blank. This boundary case is legal.
    n = 12
    ideal = np.where(bits[np.arange(n) % 2], 2, 0).astype(np.int8)
    mask = np.zeros((n, width), dtype=bool)
    mask[1] = True
    mask[4, 0] = True
    a = _assemble(np.where(mask, 2 - ideal, ideal), {5}, (0, 8, 2, 6))
    output = solve(a)
    _require(np.all(output[-2, 1:-1] == 8) and output[10, 1] != ideal[4, 0],
             "extinction boundary failed")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        first = generate_puzzle(113825090, nt, nq)
        second = generate_puzzle(113825090, nt, nq)
        _require(_digest(first) == _digest(second), "nondeterministic whole puzzle")
    # Minimum-count regression: the first two witnesses alone must reject
    # copying the first sample, including when its errors happen to be inert.
    generate_puzzle(434, 2, 1)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "canonical_structures": len(structures), "shapes": len(shapes),
            "periods": sorted(periods), "data_row_counts": sorted(row_counts),
            "data_widths": sorted(widths), "entry_counts": sorted(entry_counts),
            "reached_run_lengths": sorted(run_lengths), "partial_cycles": partial,
            "events": totals, "shortcut_rules_rejected_in_every_puzzle": list(_SHORTCUTS),
            "verification_scope": "generator-owned checks; no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
