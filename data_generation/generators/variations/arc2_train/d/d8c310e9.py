"""Generate horizontal, full-column periodic completion puzzles."""

import numpy as np


_SOURCE_PAIRS = (
    (("000000000000000", "000000000000000", "002000200000000", "002000200000000", "332133210000000"),
     ("000000000000000", "000000000000000", "002000200020002", "002000200020002", "332133213321332")),
    (("000000000000000", "000000000000000", "000000000000000", "022022000000000", "122122100000000"),
     ("000000000000000", "000000000000000", "000000000000000", "022022022022022", "122122122122122")),
    (("000000000000000", "000000000000000", "400004400000000", "430034430000000", "432234432200000"),
     ("000000000000000", "000000000000000", "400004400004400", "430034430034430", "432234432234432")),
    (("000000000000000", "000000000000000", "002000200000000", "622062206200000", "662366236600000"),
     ("000000000000000", "000000000000000", "002000200020002", "622062206220622", "662366236623662")),
)


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), grid.astype(np.int16, copy=False).tobytes())


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _source_copy(inp, out):
    for a, b in _SOURCE_PAIRS:
        if (inp.shape == (len(a), len(a[0])) and
            np.array_equal(inp, np.array([[int(x) for x in row] for row in a])) and
            np.array_equal(out, np.array([[int(x) for x in row] for row in b]))):
            return True
    return False


def _overlap(period):
    # At least two repeated vectors, and a longer fragment for wide periods.
    return max(2, (period + 2) // 3)


def _candidate_periods(prefix):
    width = prefix.shape[1]
    return [p for p in range(1, min(10, width - 1) + 1)
            if width >= p + _overlap(p)
            and np.array_equal(prefix[:, p:], prefix[:, :-p])]


def _period_from_input(inp, background):
    active = np.any(inp != background, axis=0)
    if not np.any(active):
        return None
    visible = int(np.flatnonzero(active)[-1]) + 1
    prefix = inp[:, :visible]
    candidates = _candidate_periods(prefix)
    if not candidates:
        return None
    period = candidates[0]
    # Multiples are redundant descriptions of the same continuation.
    if any(p % period != 0 for p in candidates[1:]):
        return None
    return period, visible


def _sample_block(rng, height, period, background, mode, want_blank):
    for _ in range(250):
        active_height = int(rng.integers(2, height + 1))
        top = int(rng.integers(0, height - active_height + 1))
        colors = rng.choice([c for c in range(10) if c != background],
                            size=int(rng.integers(2, 5)), replace=False)
        tile = np.full((active_height, period), background, dtype=np.int8)
        if mode == "supports":
            # A base, aligned uprights, and optional caps or interior layers.
            tile[-1, :] = colors[0]
            supports = rng.choice(period, size=int(rng.integers(1, min(period, 4) + 1)), replace=False)
            for col in supports:
                rise = int(rng.integers(2, active_height + 1))
                tile[-rise:, col] = colors[1]
                if rise > 2 and len(colors) > 2:
                    tile[-rise, col] = colors[2]
            if active_height > 2 and len(colors) > 2 and rng.random() < .6:
                row = int(rng.integers(1, active_height - 1))
                tile[row, :] = np.where(tile[row, :] == background, colors[2], tile[row, :])
        elif mode == "bands":
            rows = rng.choice(active_height, size=int(rng.integers(2, min(active_height, 4) + 1)), replace=False)
            for i, row in enumerate(sorted(rows)):
                left = int(rng.integers(0, max(1, period // 3 + 1)))
                right = int(rng.integers(max(left + 1, period * 2 // 3), period + 1))
                tile[row, left:right] = colors[i % len(colors)]
                if right - left > 2 and rng.random() < .5:
                    tile[row, int(rng.integers(left + 1, right - 1))] = background
            # Vertical stitches connect some bands.
            for col in rng.choice(period, size=int(rng.integers(1, min(period, 3) + 1)), replace=False):
                tile[:, col] = np.where(tile[:, col] == background, colors[1], tile[:, col])
        elif mode == "steps":
            level = int(rng.integers(1, active_height + 1))
            for col in range(period):
                level = int(np.clip(level + int(rng.integers(-1, 2)), 1, active_height))
                tile[-level:, col] = colors[0]
                tile[-level, col] = colors[1]
                if len(colors) > 2 and level > 2:
                    tile[-1, col] = colors[2]
        else:  # Deliberately retain some sparse, irregular textures.
            mask = rng.random(tile.shape) < float(rng.uniform(.25, .68))
            tile[mask] = colors[rng.integers(0, len(colors), size=int(mask.sum()))]

        if want_blank:
            if period < 4:
                continue
            blank_col = int(rng.integers(1, period - 1))
            tile[:, blank_col] = background
        else:
            # Texture mode may contain a gap as well; only intentional gap
            # cases are counted separately in the coverage measurements.
            pass

        # Preserve coherent geometry while ensuring at least two motif colors.
        if len(set(tile.flat) - {background}) < 2:
            occupied = np.argwhere(tile != background)
            if len(occupied) < 2:
                continue
            row, col = occupied[int(rng.integers(0, len(occupied)))]
            tile[row, col] = colors[1] if tile[row, col] != colors[1] else colors[0]
        block = np.full((height, period), background, dtype=np.int8)
        block[top:top + active_height, :] = tile
        if np.sum(np.any(block != background, axis=1)) < 2:
            continue
        if not np.any(block == background):
            continue
        if np.any(np.all(block == background, axis=0)) != want_blank:
            continue
        if _candidate_periods(np.tile(block, (1, 2)))[0] != period:
            continue
        return block
    raise RuntimeError("could not construct a primitive block")


def _sample_pair(rng, regime, preferred_period=None, desired_cutoff=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    for _ in range(500):
        period = int(preferred_period if preferred_period is not None else rng.integers(2, 11))
        min_width = period + _overlap(period) + 2
        if min_width > 30:
            continue
        if rng.random() < .38:
            width = int(rng.integers(min_width, min(18, 30) + 1)) if min_width <= 18 else int(rng.integers(min_width, 31))
        else:
            width = int(rng.integers(min_width, 31))
        height = int(rng.integers(4, 13))
        background = int(rng.integers(0, 10))
        mode = str(rng.choice(("supports", "bands", "steps", "irregular"), p=(.35, .23, .23, .19)))
        want_blank = period >= 4 and rng.random() < .17
        block = _sample_block(rng, height, period, background, mode, want_blank)
        out = np.tile(block, (1, (width + period - 1) // period))[:, :width].copy()
        choices = [v for v in range(period + _overlap(period), width - 1)
                   if np.any(block[:, (v - 1) % period] != background)
                   and np.any(out[:, v:] != background)
                   and set(np.unique(out[:, :v])) == set(np.unique(out))
                   and (not want_blank or
                        all(v > period + j for j in np.flatnonzero(np.all(block == background, axis=0))))]
        if not choices:
            continue
        # Vary complete/partial cutoffs and how much of the suffix is drawn.
        complete = [v for v in choices if v % period == 0]
        partial = [v for v in choices if v % period != 0]
        if desired_cutoff == "complete":
            pool = complete
        elif desired_cutoff == "partial":
            pool = partial
        elif complete and partial:
            pool = complete if rng.random() < .35 else partial
        else:
            pool = choices
        if not pool:
            continue
        visible = int(rng.choice(pool))
        inp = out.copy()
        inp[:, visible:] = background
        inferred = _period_from_input(inp, background)
        if inferred != (period, visible):
            continue
        if _source_copy(inp, out):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct an unambiguous pair")


def _check_count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return value


def _counts(seed, rng, num_train, num_test):
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = seed % 4 + 1
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(isinstance(inp, np.ndarray) and isinstance(out, np.ndarray), "arrays required")
    _require(inp.ndim == out.ndim == 2 and inp.shape == out.shape, "shape changed")
    h, w = inp.shape
    _require(1 <= h <= 30 and 1 <= w <= 30, "grid size")
    _require(np.issubdtype(inp.dtype, np.integer) and np.issubdtype(out.dtype, np.integer), "integer grids")
    _require(np.all((0 <= inp) & (inp <= 9)) and np.all((0 <= out) & (out <= 9)), "ARC colors")
    bg = int(inp[0, -1])
    _require(np.all(inp[:, -1] == bg), "suffix background")
    result = _period_from_input(inp, bg)
    _require(result is not None, "period not recoverable")
    period, visible = result
    _require(2 <= period <= 10 and period + _overlap(period) <= visible < w - 1, "invalid recurrence")
    _require(np.all(inp[:, visible:] == bg), "erasure must be contiguous")
    expected = inp[:, np.arange(w) % period]
    _require(np.array_equal(out, expected), "incorrect horizontal continuation")
    _require(not np.array_equal(inp, out), "no completion")
    _require(set(np.unique(inp)) == set(np.unique(out)), "new color in output")
    _require(not _source_copy(inp, out), "source copy")
    return {"height": h, "width": w, "background": bg, "period": period,
            "visible": visible, "cutoff": "complete" if visible % period == 0 else "partial",
            "short": visible < 2 * period,
            "blank": bool(np.any(np.all(inp[:, :period] == bg, axis=0))),
            "motif_colors": len(set(np.unique(out).tolist()) - {bg}),
            "active_height": int(np.sum(np.any(out != bg, axis=1)))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle structure")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "pair counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair structure")
            info = _inspect_pair(pair)
            infos[split].append(info)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
    # The only puzzle convention is left-to-right continuation, demonstrated
    # by every training pair. Dimensions, palette, and period are local cues.
    _require(infos["train"][0]["period"] != infos["train"][1]["period"], "training period variation")
    _require(infos["train"][0]["cutoff"] != infos["train"][1]["cutoff"], "training cutoff variation")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_periods = []
    training_cutoffs = ["complete", "partial"]
    rng.shuffle(training_cutoffs)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(50):
                # New test widths occur naturally; some tests revisit a
                # demonstrated width without requiring that cue for solving.
                familiar = split == "test" and rng.random() < .4
                preference = int(rng.choice(train_periods)) if familiar else None
                if split == "train" and index == 1:
                    preference = int(rng.choice([p for p in range(2, 11) if p != train_periods[0]]))
                cutoff = training_cutoffs[index] if split == "train" and index < 2 else None
                pair = _sample_pair(rng, "source_train" if split == "train" else "target_test",
                                    preference, cutoff)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate unique pairs")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_periods.append(_inspect_pair(pair)["period"])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [_inspect_pair(p) for task in puzzles for split in ("train", "test") for p in task[split]]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "puzzle diversity")
    _require(len({(i["height"], i["width"]) for i in infos}) >= 30, "size diversity")
    _require(len({i["background"] for i in infos}) >= 8, "background diversity")
    _require(len({i["period"] for i in infos}) >= 8, "period diversity")
    _require(any(i["short"] for i in infos), "missing one-block fragments")
    _require(any(i["blank"] for i in infos), "missing blank phases")
    _require({i["cutoff"] for i in infos} == {"complete", "partial"}, "cutoff diversity")
    _require(generate_puzzle(314159, 7, 3) and
             _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3)),
             "determinism")
    for args in ((2, 1), (10, 4), (None, 4), (2, None)):
        task = generate_puzzle(101, *args)
        if args[0] is not None:
            _require(len(task["train"]) == args[0], "requested train count")
        if args[1] is not None:
            _require(len(task["test"]) == args[1], "requested test count")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2)):
        try:
            generate_puzzle(101, *args)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
