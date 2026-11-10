"""Bottom-anchored wave followed by literal adjacent-vector contraction.

No parent code, files, seed, or generation metadata is used by solve().
Generation keeps at most one puzzle and a bounded number of candidates in RAM;
neither generation nor self_test writes files.
"""

from itertools import groupby, product
import hashlib

import numpy as np


WAVE = (0, -1, 0, 1)
HOST_SEEDS = (0, 824881618, 375912685, 985254852, 845760518, 188559598,
              470954221, 706624183, 541951719, 11965792, 567279772,
              745125669, 994524793, 515058846, 653755542, 177078612,
              942500692, 1691984, 590230400, 220400025, 910326182,
              209983446, 421646892, 839140278, 706917472, 990725427,
              409369342, 732833191, 328709319, 228577334, 274960678,
              439274176, 848178097, 570029614, 32040720, 35970044,
              880805306, 177815284, 10550961, 702332514, 491249900,
              509712495)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return array.astype(np.int8, copy=False)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _wave(grid, table=WAVE, anchor="bottom"):
    """Translate occupied row masks; never wrap or clip."""
    background = int(grid[0, 0])
    rows, columns = np.nonzero(grid != background)
    if not len(rows):
        raise ValueError("empty scaffold")
    if anchor == "bottom":
        distance = int(rows.max()) - rows
    elif anchor == "top":
        distance = rows - int(rows.min())
    elif anchor == "canvas":
        distance = grid.shape[0] - 1 - rows
    else:
        raise ValueError("unknown anchor")
    translated = columns + np.asarray(table, dtype=int)[distance % 4]
    if np.any(translated < 0) or np.any(translated >= grid.shape[1]):
        raise ValueError("wave would clip the scaffold")
    result = np.full_like(grid, background)
    result[rows, translated] = grid[rows, columns]
    return result


def _contract(grid):
    # Both masks refer to the complete waved grid. Keeping duplicate row
    # representatives cannot erase a witness that two columns differ.
    row_keep = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    column_keep = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[np.ix_(row_keep, column_keep)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed bottom wave, THEN contract both equality-run axes."""
    grid = _grid(input_grid)
    if len(np.unique(grid)) != 2:
        raise ValueError("family inputs have exactly two colors")
    return _contract(_wave(grid))


def _run_lengths(vectors):
    return tuple(sum(1 for _ in run)
                 for _, run in groupby(tuple(int(x) for x in row)
                                       for row in vectors))


def _literal_contract(grid):
    """Audit implementation using literal tuples instead of solver masks."""
    rows = [value for value, _ in groupby(map(tuple, grid.tolist()))]
    columns = [value for value, _ in groupby(zip(*rows))]
    return np.asarray(columns, dtype=np.int8).T.copy()


def _input_info(grid):
    background = int(grid[0, 0])
    colors = np.unique(grid)
    _require(len(colors) == 2, "one background and one scaffold color required")
    foreground = int(colors[colors != background][0])
    mask = grid == foreground
    occupied = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(occupied[0]), int(occupied[-1])
    height = bottom - top + 1
    _require(height >= 5 and len(occupied) == height, "noncontiguous scaffold")
    lefts, rights = [], []
    for row in range(top, bottom + 1):
        columns = np.flatnonzero(mask[row])
        lefts.append(int(columns[0]))
        rights.append(int(columns[-1]))
    width = rights[0] - lefts[0] + 1
    _require(all(b - a + 1 == width for a, b in zip(lefts, rights)),
             "row spans must have one common width")
    _require(max(lefts) - min(lefts) <= 1, "jogs exceed one cell")
    _require(min(lefts) >= 1 and max(rights) <= grid.shape[1] - 2,
             "input must have blank lateral borders")
    core = np.asarray([mask[row, left:left + width]
                       for row, left in zip(range(top, bottom + 1), lefts)])
    _require(core[0].all() and core[-1].all(), "end bars must span all supports")
    rails = np.all(core, axis=0)
    _require(rails[0] and rails[-1], "outer supports missing")
    starts = np.flatnonzero(rails & ~np.r_[False, rails[:-1]])
    ends = np.flatnonzero(rails & ~np.r_[rails[1:], False])
    supports = tuple((int(a), int(b)) for a, b in zip(starts, ends))
    _require(2 <= len(supports) <= 5, "support inventory outside family")
    bare = sum(np.array_equal(row, rails) for row in core)
    _require(bare >= 2, "at least two bare support rows required")
    gaps = tuple(supports[i + 1][0] - supports[i][1] - 1
                 for i in range(len(supports) - 1))
    bar_rows, partial = [], 0
    for index, row in enumerate(core):
        _require(np.all(row[rails]), "support interrupted")
        filled = []
        for i in range(len(supports) - 1):
            segment = row[supports[i][1] + 1:supports[i + 1][0]]
            _require(segment.all() or not segment.any(), "fragmented crossbar")
            filled.append(bool(segment.all()))
        if any(filled):
            selected = np.flatnonzero(filled)
            _require(selected[-1] - selected[0] + 1 == len(selected),
                     "one crossbar segment per row")
            bar_rows.append(index)
            partial += not all(filled)

    # Check the promised single connected object, including jogged supports.
    start = (top, lefts[0])
    seen, stack = {start}, [start]
    while stack:
        row, column = stack.pop()
        for r, c in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1)):
            if (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                    and mask[r, c] and (r, c) not in seen):
                seen.add((r, c))
                stack.append((r, c))
    _require(len(seen) == int(mask.sum()), "disconnected scaffold")
    return dict(background=background, foreground=foreground, top=top,
                bottom=bottom, height=height, width=width, lefts=tuple(lefts),
                supports=supports, gaps=gaps, bars=tuple(bar_rows),
                partial=int(partial), bottom_padding=grid.shape[0] - bottom - 1,
                jogged=len(set(lefts)) > 1)


def _audit_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have exactly input/output")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    info = _input_info(grid)
    background, foreground = info["background"], info["foreground"]
    # Construct the intermediate cell by cell, checking physical conservation.
    waved = np.full_like(grid, background)
    for row in range(info["top"], info["bottom"] + 1):
        distance = info["bottom"] - row
        shift = 0 if distance % 2 == 0 else (-1 if distance % 4 == 1 else 1)
        for column in np.flatnonzero(grid[row] != background):
            target = int(column) + shift
            _require(0 <= target < grid.shape[1], "clipped wave")
            waved[row, target] = grid[row, column]
        _require(np.count_nonzero(waved[row] == foreground)
                 == np.count_nonzero(grid[row] == foreground), "nonrigid wave")
    _require(np.array_equal(output, _literal_contract(waved)), "incorrect answer")
    _require(np.array_equal(output, solve(grid)), "solve disagrees")
    row_runs, column_runs = _run_lengths(waved), _run_lengths(waved.T)
    expanded = np.repeat(np.repeat(output, row_runs, axis=0), column_runs, axis=1)
    _require(np.array_equal(expanded, waved), "run expansion lost cell values")
    _require(not np.any(np.all(output[1:] == output[:-1], axis=1)),
             "adjacent output rows still identical")
    _require(not np.any(np.all(output[:, 1:] == output[:, :-1], axis=0)),
             "adjacent output columns still identical")
    _require(set(np.unique(output)) == set(np.unique(grid)), "colors changed")
    _require(not np.array_equal(_contract(grid), output), "A is dispensable")
    _require(waved.shape != output.shape, "B is dispensable")
    foreground_merges = sum(np.array_equal(waved[r], waved[r - 1])
                            and np.any(waved[r] != background)
                            for r in range(1, waved.shape[0]))
    destroyed = sum(np.array_equal(grid[r], grid[r - 1])
                    and not np.array_equal(waved[r], waved[r - 1])
                    for r in range(1, grid.shape[0]))
    info.update(row_runs=row_runs, column_runs=column_runs,
                foreground_merges=int(foreground_merges), destroyed=int(destroyed))
    return info, waved


def _tables_fitting(train, anchor):
    result = []
    for table in product((-1, 0, 1), repeat=4):
        for pair in train:
            try:
                prediction = _contract(_wave(np.asarray(pair["input"]), table, anchor))
            except ValueError:
                break
            if not np.array_equal(prediction, pair["output"]):
                break
        else:
            result.append(table)
    return result


def _global_unique(grid, axis):
    vectors = grid if axis == 0 else grid.T
    seen, keep = set(), []
    for index, vector in enumerate(vectors):
        key = tuple(int(x) for x in vector)
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return grid[keep] if axis == 0 else grid[:, keep]


def _training_checks(train, infos):
    _require(any(not a["jogged"] and a["destroyed"] > 0 for a in infos),
             "training must show the wave splitting original equal rows")
    _require(any(a["jogged"] and a["foreground_merges"] > 0 for a in infos),
             "training must show the wave creating occupied equal rows")
    _require(any(max(a["row_runs"]) > 1 and max(a["column_runs"]) > 1
                 for a in infos), "training needs contraction on both axes")
    _require(len({a["background"] for a in infos}) >= 2,
             "training must expose local background roles")
    _require(_tables_fitting(train, "bottom") == [WAVE],
             "training does not uniquely determine the four-phase table")
    _require(not _tables_fitting(train, "top"), "top anchor still fits training")
    _require(not _tables_fitting(train, "canvas"), "canvas anchor still fits training")
    failures = dict(reverse_order=False, reverse_order_recontracted=False,
                    global_rows=False, global_columns=False,
                    rows_only=False, columns_only=False, crop=False)
    for pair in train:
        grid, answer = np.asarray(pair["input"]), np.asarray(pair["output"])
        waved = _wave(grid)
        backwards = _wave(_contract(grid))
        failures["reverse_order"] |= not np.array_equal(backwards, answer)
        failures["reverse_order_recontracted"] |= not np.array_equal(_contract(backwards), answer)
        failures["global_rows"] |= not np.array_equal(_contract(_global_unique(waved, 0)), answer)
        failures["global_columns"] |= not np.array_equal(_contract(_global_unique(waved, 1)), answer)
        row_keep = np.r_[True, np.any(waved[1:] != waved[:-1], axis=1)]
        col_keep = np.r_[True, np.any(waved[:, 1:] != waved[:, :-1], axis=0)]
        failures["rows_only"] |= not np.array_equal(waved[row_keep], answer)
        failures["columns_only"] |= not np.array_equal(waved[:, col_keep], answer)
        rr, cc = np.nonzero(waved != int(grid[0, 0]))
        cropped = waved[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        failures["crop"] |= not np.array_equal(_contract(cropped), answer)
    _require(all(failures.values()), "a simpler alternative still fits: " + str(failures))


def _palette(rng, previous=None):
    background = int(rng.integers(10))
    if previous is not None:
        background = (previous[0] + int(rng.integers(1, 10))) % 10
    candidates = [c for c in range(10) if c != background
                  and (previous is None or c != previous[1])]
    return background, int(rng.choice(candidates))


def _sample_input(rng, height, jogged, palette=None, top=None, bottom=None,
                  show_alignment=False, hollow=False, support_count=None):
    compact = rng.random() < 0.28
    count = (int(support_count) if support_count is not None else
             int(rng.integers(2, 4 if compact else 6)))
    widths = np.full(count, 2 if jogged else 1, dtype=int)
    gaps = np.ones(count - 1, dtype=int)
    # Force an internal column equality somewhere, in foreground, a gap, or
    # both; how much survives is determined by the wave, not by a target size.
    wide_mode = int(rng.integers(3))
    if wide_mode != 0:
        widths[int(rng.integers(count))] = 5 if jogged else 4
    if wide_mode != 1:
        gaps[int(rng.integers(count - 1))] = 5 if jogged else 4
    base_width = int(widths.sum() + gaps.sum())
    extra_limit = min(26 - base_width, 3) if compact else 26 - base_width
    extra = int(rng.integers(0, extra_limit + 1))
    allocation = rng.multinomial(extra, rng.dirichlet(np.full(2 * count - 1, 0.7)))
    widths += allocation[::2]
    gaps += allocation[1::2]
    supports, cursor = [], 0
    for index, width in enumerate(widths):
        supports.append((cursor, cursor + int(width) - 1))
        cursor += int(width) + (int(gaps[index]) if index < count - 1 else 0)
    base_width = cursor

    distances = np.arange(height - 1, -1, -1)
    if not jogged:
        jogs = np.zeros(height, dtype=int)
    else:
        mode = int(rng.integers(4))
        if mode == 0:
            jogs = distances % 2
        elif mode == 1:
            jogs = (distances % 4 >= 2).astype(int)
        elif mode == 2:
            jogs = rng.integers(0, 2, size=height)
        else:
            jogs = (np.arange(height) >= int(rng.integers(1, height))).astype(int)
        if show_alignment:
            for distance, jog in ((2, 0), (3, 1), (4, 0), (5, 1), (6, 0)):
                jogs[height - 1 - distance] = jog
        if not np.any(jogs != jogs[0]):
            jogs[height // 2] = 1 - jogs[0]

    available = 30 - height
    if top is None:
        top = int(rng.integers(0, min(6, available) + 1))
    if bottom is None:
        bottom = int(rng.integers(0, min(6, available - top) + 1))
    _require(top + bottom <= available, "invalid vertical padding")
    right_min = 1 + int(jogs.max())
    slack = 30 - base_width - 1 - right_min
    left_limit = min(slack, 2) if compact else slack
    left = 1 + int(rng.integers(0, left_limit + 1))
    right_limit = slack - (left - 1)
    if compact:
        right_limit = min(right_limit, 2)
    right = right_min + int(rng.integers(0, right_limit + 1))
    background, foreground = _palette(rng) if palette is None else palette
    grid = np.full((top + height + bottom, left + base_width + right),
                   background, dtype=np.int8)

    # Bare witnesses make the visible support inventory unambiguous.
    protected = {height - 1 - d for d in (2, 4, 5, 6)} if show_alignment else set()
    candidates = [r for r in range(1, height - 1) if r not in protected]
    if not show_alignment:
        protected.update(int(x) for x in rng.choice(candidates, size=2, replace=False))
    density = float(rng.uniform(0.08, 0.55))
    for row in range(height):
        origin = left + int(jogs[row])
        for a, b in supports:
            grid[top + row, origin + a:origin + b + 1] = foreground
        if row in (0, height - 1):
            grid[top + row, origin:origin + base_width] = foreground
        elif not hollow and row not in protected and rng.random() < density:
            first = int(rng.integers(0, count - 1))
            last = int(rng.integers(first + 1, count))
            grid[top + row, origin + supports[first][1]:
                 origin + supports[last][0] + 1] = foreground
    return grid


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    residue = int(rng.integers(4))
    held_out = (residue + 1) % 4
    first_height = int(rng.choice([h for h in range(8, 20) if h % 4 == residue]))
    second_height = int(rng.choice([h for h in range(10, 22)
                                    if h % 4 == (residue + 2) % 4]))
    top = int(rng.integers(2, 5))
    bottom = int(rng.integers(4))
    first_palette = _palette(rng)
    second_palette = _palette(rng, first_palette)

    # Two witnesses already suffice even when num_train=2. Bounded retries
    # only reject ambiguous demonstrations, not exceptions to the rule.
    for _ in range(32):
        first = _sample_input(rng, first_height, False, first_palette, top, bottom,
                              hollow=True, support_count=int(rng.integers(2, 4)))
        second = _sample_input(rng, second_height, True, second_palette,
                               top, (bottom + 1) % 4, show_alignment=True)
        train = [{"input": g, "output": solve(g)} for g in (first, second)]
        try:
            infos = [_audit_pair(p)[0] for p in train]
            _training_checks(train, infos)
        except AssertionError:
            continue
        break
    else:
        raise RuntimeError("could not construct unambiguous training witnesses")

    puzzle = {"train": train, "test": []}
    used = {_key(p["input"]) for p in train}
    train_heights = [h for h in range(5, 26) if h % 4 != held_out]
    test_heights = [int(h) for h in rng.permutation([h for h in range(5, 28)
                                                   if h % 4 == held_out])][:num_test]
    for split, number in (("train", num_train - 2), ("test", num_test)):
        for index in range(number):
            height = int(rng.choice(train_heights)) if split == "train" else test_heights[index]
            for _ in range(48):
                grid = _sample_input(rng, height, bool(rng.integers(2)),
                                     hollow=rng.random() < 0.25)
                if _key(grid) in used:
                    continue
                pair = {"input": grid, "output": solve(grid)}
                try:
                    _audit_pair(pair)
                except AssertionError:
                    continue
                used.add(_key(grid))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a distinct nondegenerate pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle schema invalid")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info, _ = _audit_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _training_checks(puzzle["train"], infos["train"])
    trained_residues = {a["height"] % 4 for a in infos["train"]}
    _require(all(a["height"] % 4 not in trained_residues for a in infos["test"]),
             "test must hold out a top phase (object height residue)")
    _require(len({a["height"] for a in infos["test"]}) == len(puzzle["test"]),
             "test structures must differ")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 supplied seeds, all explicit counts, finite alternative-rule checks.

    Only hashes, shape sets, and scalar counts accumulate. No puzzle archive,
    full execution trace, subprocess, or log is written.
    """
    digests, shapes, output_shapes, geometries = set(), set(), set(), set()
    heights, widths, supports, gaps, palettes = set(), set(), set(), set(), set()
    foregrounds, backgrounds, residues = set(), set(), set()
    merges = destroyed = partial = jogged = straight = foreground_columns = 0
    both = one_axis = boundary = small = large = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, waved = _audit_pair(pair)
                grid = pair["input"]
                shapes.add(grid.shape)
                output_shapes.add(pair["output"].shape)
                heights.add(info["height"])
                widths.add(info["width"])
                supports.add(len(info["supports"]))
                gaps.update(info["gaps"])
                palettes.add((info["background"], info["foreground"]))
                foregrounds.add(info["foreground"])
                backgrounds.add(info["background"])
                if split == "test":
                    residues.add(info["height"] % 4)
                geometries.add((info["height"], info["supports"], info["bars"],
                                tuple(x - min(info["lefts"]) for x in info["lefts"])))
                merges += info["foreground_merges"] > 0
                destroyed += info["destroyed"] > 0
                partial += info["partial"] > 0
                jogged += info["jogged"]
                straight += not info["jogged"]
                foreground_columns += any(np.array_equal(waved[:, c], waved[:, c - 1])
                                          and np.any(waved[:, c] != info["background"])
                                          for c in range(1, waved.shape[1]))
                row_shrink = pair["output"].shape[0] < grid.shape[0]
                col_shrink = pair["output"].shape[1] < grid.shape[1]
                both += row_shrink and col_shrink
                one_axis += row_shrink != col_shrink
                boundary += info["top"] == 0 or info["bottom_padding"] == 0
                small += max(grid.shape) <= 15
                large += max(grid.shape) >= 29
                # Literal compression commutes and is idempotent.
                _require(np.array_equal(_literal_contract(waved),
                                        _literal_contract(waved.T).T), "axis order matters")
                _require(np.array_equal(_contract(pair["output"]), pair["output"]),
                         "contraction not idempotent")
                # Color permutation must commute with the full rule.
                permutation = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(permutation[grid]), permutation[pair["output"]]),
                         "literal color shortcut")
    _require(len(digests) == 42, "puzzle diversity missing")
    _require(len(shapes) >= 50 and len(output_shapes) >= 35, "shape diversity missing")
    _require(len(geometries) >= 180, "geometry diversity missing")
    _require(len(heights) >= 20 and len(widths) >= 15, "scale diversity missing")
    _require(supports == {2, 3, 4, 5} and len(gaps) >= 7, "inventory diversity missing")
    _require(len(palettes) >= 50 and len(backgrounds) == len(foregrounds) == 10,
             "palette role diversity missing")
    _require(residues == {0, 1, 2, 3}, "held-out top phases missing")
    _require(merges >= 42 and destroyed >= 42 and partial >= 15, "causal witnesses missing")
    _require(jogged >= 60 and straight >= 60 and foreground_columns >= 20,
             "scaffold variety missing")
    _require(both >= 100 and boundary >= 15 and large >= 30 and small >= 2,
             "canvas/compression regimes missing")

    # Every legal explicit count combination, including the 14-pair maximum.
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            puzzle = generate_puzzle(7301 + train_count * 5 + test_count,
                                     train_count, test_count)
            _require(len(puzzle["train"]) == train_count
                     and len(puzzle["test"]) == test_count, "counts not respected")
    for train_count, test_count in ((1, 1), (11, 1), (4, 0), (6, 5),
                                     (2, 2), (3, 4), (True, 1), (4.0, 1),
                                     (4, False), (4, None)):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for bad_seed in (-1, True, 0.0):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "scaffold_geometries": len(geometries), "height_values": len(heights),
            "width_values": len(widths), "palettes": len(palettes),
            "pairs_with_created_foreground_equalities": merges,
            "pairs_with_destroyed_equalities": destroyed,
            "pairs_with_foreground_column_equalities": foreground_columns,
            "partial_bar_pairs": partial, "jogged_pairs": jogged,
            "straight_pairs": straight, "both_axes_shrink": both,
            "one_axis_shrinks": one_axis, "edge_contact_pairs": boundary,
            "small_pairs": small, "near_limit_pairs": large}
