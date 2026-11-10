"""Recover a stretched self-indexing lattice, then complete its blank tiles.

Only order two is compatible with *exact maximal whole-vector collapse*:
higher-order one-hole tiles already contain adjacent duplicate latent rows.
See composition.md for the recoverability proof. No parent module is imported.
Generation and tests retain no files, traces, or global random state.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np


HOST_SEEDS = (
    0, 520555606, 368747797, 479917241, 389902635, 397772081,
    111222744, 770975437, 586083282, 387841775, 634885260, 468447826,
    532098703, 663858149, 142070300, 801853745, 599092721, 858042804,
    322413136, 251841441, 921918601, 235412291, 776728835, 819063593,
    408391713, 363130584, 689859946, 222354393, 485097165, 128903031,
    301580183, 715930649, 234008137, 125014955, 496072120, 924398389,
    189989190, 964226897, 515924238, 458369128, 659584073, 389417943,
)


def _grid(value) -> np.ndarray:
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= s <= 30 for s in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid in colors 0..9")
    return grid.astype(np.int8, copy=False)


def _runs(grid: np.ndarray, axis: int) -> tuple[list[int], tuple[int, ...]]:
    lines = grid if axis == 0 else grid.T
    starts = [0]
    for i in range(1, len(lines)):
        if not np.array_equal(lines[i - 1], lines[i]):
            starts.append(i)
    edges = starts + [len(lines)]
    return starts, tuple(b - a for a, b in zip(edges, edges[1:]))


def _analyse(input_grid):
    """Read roles, boundaries, and clues using the input alone."""
    grid = _grid(input_grid)
    rs, heights = _runs(grid, 0)
    cs, widths = _runs(grid, 1)
    compact = grid[np.ix_(rs, cs)]
    if compact.shape != (5, 5):
        raise ValueError("the recoverable order-two lattice must collapse to 5 by 5")
    background = int(compact[2, 2])
    colors = [int(x) for x in np.unique(compact)]
    if (len(colors) != 2 or not np.all(compact[2] == background)
            or not np.all(compact[:, 2] == background)):
        raise ValueError("expected two colors and intact central separator lines")
    foreground = next(c for c in colors if c != background)
    mask = np.zeros((2, 2), dtype=bool)
    for i, j in itertools.product(range(2), repeat=2):
        tile = compact[3 * i:3 * i + 2, 3 * j:3 * j + 2]
        if np.all(tile == background):
            continue
        expected = np.full((2, 2), foreground, dtype=np.int8)
        expected[i, j] = background
        if not np.array_equal(tile, expected):
            raise ValueError("a clue must be complete, with its own coordinate hole")
        mask[i, j] = True
    if (not mask.any(axis=0).all() or not mask.any(axis=1).all()
            or mask.all()):
        raise ValueError("cover every macro row and column and leave a blank tile")
    return compact, background, foreground, mask, heights, widths


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse whole-line repetitions, then fill only the blank compartments."""
    compact, background, foreground, mask, _, _ = _analyse(input_grid)
    output = compact.copy()
    for i, j in itertools.product(range(2), repeat=2):
        if not mask[i, j]:
            output[3 * i:3 * i + 2, 3 * j:3 * j + 2] = foreground
            output[3 * i + i, 3 * j + j] = background
    return output


def _complete(background: int, foreground: int) -> np.ndarray:
    grid = np.full((5, 5), background, dtype=np.int8)
    for i, j in itertools.product(range(2), repeat=2):
        grid[3 * i:3 * i + 2, 3 * j:3 * j + 2] = foreground
        grid[4 * i, 4 * j] = background
    return grid


def _latent(mask: np.ndarray, background: int, foreground: int) -> np.ndarray:
    grid = _complete(background, foreground)
    for i, j in itertools.product(range(2), repeat=2):
        if not mask[i, j]:
            grid[3 * i:3 * i + 2, 3 * j:3 * j + 2] = background
    return grid


def _expand(grid, heights, widths):
    return np.repeat(np.repeat(grid, heights, axis=0), widths, axis=1)


def _valid_masks() -> tuple[np.ndarray, ...]:
    masks = []
    for bits in itertools.product((False, True), repeat=4):
        mask = np.array(bits, dtype=bool).reshape(2, 2)
        if mask.any(axis=0).all() and mask.any(axis=1).all() and not mask.all():
            masks.append(mask)
    return tuple(masks)


def _training_runs(rng, nonuniform=False) -> tuple[int, ...]:
    mode = int(rng.integers(5))
    values = [int(v) for v in rng.integers(1, 6, size=5)]
    if mode == 0 and not nonuniform:
        values = [int(rng.integers(2, 6))] * 5
    elif mode == 1:
        values.sort()
    elif mode == 2:
        values.sort(reverse=True)
    if len(set(values)) == 1 and (nonuniform or values[0] == 1):
        values[0] = 1 if values[0] != 1 else 2
    return tuple(values)


def _long_runs(rng) -> tuple[int, ...]:
    values = [int(v) for v in rng.integers(1, 4, size=5)]
    index = int(rng.integers(5))
    remaining = sum(values) - values[index]
    values[index] = int(rng.integers(8, min(20, 30 - remaining) + 1))
    return tuple(values)


def _schedules(rng, split: str, index: int):
    if split == "test":
        # At least one unseen run width (>=8, versus training's <=5).
        heights, widths = _long_runs(rng), _long_runs(rng)
        if rng.random() < 0.45:
            other = ((1,) * 5 if rng.random() < 0.3 else _training_runs(rng))
            if rng.random() < 0.5:
                heights = other
            else:
                widths = other
        return heights, widths
    heights = _training_runs(rng, nonuniform=index == 0)
    widths = _training_runs(rng, nonuniform=True)
    if index > 0 and rng.random() < 0.30:
        if rng.random() < 0.5:
            heights = (1,) * 5
        else:
            widths, heights = (1,) * 5, _training_runs(rng, nonuniform=True)
    if index == 0 and heights == widths:
        widths = tuple(1 if v != 1 else 2 for v in widths)
    return heights, widths


def _integer(value, name: str, low: int, high: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return int(value)


def _key(grid) -> tuple:
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shared_background = int(rng.integers(10)) if rng.random() < 0.5 else None
    # One diagonal and one off-diagonal target distinguish coordinate transforms.
    missing = [int(rng.choice((0, 3))), int(rng.choice((1, 2)))]
    if rng.random() < 0.5:
        missing.reverse()
    masks = _valid_masks()
    sparse_masks = [m for m in masks if m.sum() == 2]
    seen = set()
    puzzle = {"train": [], "test": []}
    previous_foreground = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            background = (int(rng.integers(10)) if shared_background is None
                          else shared_background)
            choices = [c for c in range(10) if c != background
                       and not (split == "train" and index == 1
                                and c == previous_foreground)]
            foreground = int(rng.choice(choices))
            if split == "train" and index < 2:
                mask = np.ones((2, 2), dtype=bool)
                mask.flat[missing[index]] = False
            elif split == "test" and index == 0:
                mask = sparse_masks[int(rng.integers(len(sparse_masks)))].copy()
            else:
                mask = masks[int(rng.integers(len(masks)))].copy()
            latent = _latent(mask, background, foreground)
            for _ in range(64):
                heights, widths = _schedules(rng, split, index)
                input_grid = _expand(latent, heights, widths)
                if (split == "train" and index == 1
                        and input_grid.shape == puzzle["train"][0]["input"].shape):
                    continue
                fingerprint = _key(input_grid)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                puzzle[split].append({"input": input_grid,
                                      "output": _complete(background, foreground)})
                break
            else:
                raise RuntimeError("bounded search could not find a distinct expansion")
            previous_foreground = foreground
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _coordinate_candidates(infos):
    """Alternatives evaluated only at tiles newly filled by demonstrations."""
    fits = []
    for swap, flip_r, flip_c in itertools.product((False, True), repeat=3):
        good = True
        for _, _, _, mask, _, _ in infos:
            for i, j in np.argwhere(~mask):
                a, b = (int(j), int(i)) if swap else (int(i), int(j))
                a, b = (1 - a if flip_r else a), (1 - b if flip_c else b)
                good = good and (a, b) == (i, j)
        if good:
            fits.append((swap, flip_r, flip_c))
    return fits


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input/output")
            large, output = _grid(pair["input"]), _grid(pair["output"])
            info = _analyse(large)
            compact, bg, fg, mask, heights, widths = info
            _require(_key(large) not in seen, "duplicate input")
            seen.add(_key(large))
            _require(output.shape == (5, 5), "output must be the compact lattice")
            # Cellwise specification, separate from solve's blank-tile filling.
            for r, c in itertools.product(range(5), repeat=2):
                separator = r == 2 or c == 2
                hole = r % 3 == r // 3 and c % 3 == c // 3
                expected = bg if separator or hole else fg
                _require(int(output[r, c]) == expected, "wrong separator or coordinate hole")
            _require(np.array_equal(_expand(compact, heights, widths), large),
                     "compression did not losslessly recover the expanded input")
            _require(np.all(np.any(compact[1:] != compact[:-1], axis=1))
                     and np.all(np.any(compact[:, 1:] != compact[:, :-1], axis=0)),
                     "an adjacent latent boundary is hidden")
            _require(np.count_nonzero(output != compact) == 3 * int((~mask).sum()),
                     "must add exactly three foreground cells per missing tile")
            for i, j in itertools.product(range(2), repeat=2):
                if mask[i, j]:
                    _require(np.array_equal(compact[3*i:3*i+2, 3*j:3*j+2],
                                            output[3*i:3*i+2, 3*j:3*j+2]),
                             "completed clue changed")
            _require(np.array_equal(solve(large), output), "input-only solve disagrees")
            _require(max(heights) > 1 or max(widths) > 1, "no compression witness")
            _require(len(set(heights)) > 1 or len(set(widths)) > 1,
                     "each generated enlargement must be nonuniform")
            # B without A gives compact, and A without B retains the wrong scale.
            _require(not np.array_equal(compact, output) and large.shape != output.shape,
                     "both parent concepts must change the answer")
            infos[split].append(info)
    train = infos["train"]
    _require(_coordinate_candidates(train) == [(False, False, False)],
             "training fails to distinguish swapped/reflected coordinate rules")
    targets = {tuple(int(x) for x in p) for info in train for p in np.argwhere(~info[3])}
    _require(len(targets) >= 2, "a shared missing-hole position still explains training")
    _require(any(len(set(i[4])) > 1 and len(set(i[5])) > 1
                 and max(i[4]) > 1 and max(i[5]) > 1 and i[4] != i[5] for i in train),
             "training must expose independently nonuniform expansion on both axes")
    _require(len({np.asarray(p["input"]).shape for p in puzzle["train"]}) >= 2,
             "training needs different physical shapes")
    train_max = max(max(i[4] + i[5]) for i in train)
    _require(train_max <= 5, "generated training runs must be at most five")
    _require(all(max(i[4] + i[5]) >= 8 for i in infos["test"]),
             "test must generalize to an unseen stretch magnitude")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _expect_error(call, error=(ValueError, AssertionError)):
    try:
        call()
    except error:
        return
    raise AssertionError("invalid case was accepted")


def self_test() -> dict:
    """42 seeds, finite boundary cases, and semantic/metamorphic checks; no I/O."""
    shapes, layouts, masks, palettes, row_runs, col_runs, fingerprints = (
        set() for _ in range(7))
    events = {name: 0 for name in ("one_axis", "uniform_axis", "nonmonotonic",
                                  "wide_thin", "foreground_zero", "background_zero")}
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)),
                 "seed does not reproduce the entire puzzle")
        fingerprints.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                large, output = pair["input"], pair["output"]
                before = _key(large)
                compact, bg, fg, mask, heights, widths = _analyse(large)
                # Different line-comparison expression verifies exact B stage.
                kr = np.r_[True, np.any(large[1:] != large[:-1], axis=1)]
                kc = np.r_[True, np.any(large[:, 1:] != large[:, :-1], axis=0)]
                _require(np.array_equal(large[kr][:, kc], compact), "wrong maximal runs")
                _require(np.array_equal(solve(large.T), output.T), "axis symmetry failed")
                remap = np.array([(c + 3) % 10 for c in range(10)], dtype=np.int8)
                _require(np.array_equal(solve(remap[large]), remap[output]),
                         "color roles depend on literal colors")
                _require(_key(large) == before, "solver mutated its input")
                pairs += 1
                shapes.add(large.shape)
                layouts.add(_key((large == fg).astype(np.int8)))
                masks.add(mask.tobytes())
                palettes.add((bg, fg))
                row_runs.add(heights)
                col_runs.add(widths)
                events["one_axis"] += (max(heights) == 1) != (max(widths) == 1)
                events["uniform_axis"] += len(set(heights)) == 1 or len(set(widths)) == 1
                events["nonmonotonic"] += any(
                    any(a < b for a, b in zip(v, v[1:]))
                    and any(a > b for a, b in zip(v, v[1:])) for v in (heights, widths))
                events["wide_thin"] += any(
                    min(a, b) == 1 and max(a, b) >= 8
                    for v in (heights, widths) for a, b in zip(v, v[1:]))
                events["foreground_zero"] += fg == 0
                events["background_zero"] += bg == 0
    _require(len(fingerprints) == 42 and len(masks) == 6, "seed/mask variety failed")
    for observed, minimum in ((len(shapes), 40), (len(layouts), 120),
                               (len(palettes), 30), (len(row_runs), 80),
                               (len(col_runs), 80)):
        _require(observed >= minimum, "insufficient structural variation")
    _require(all(v > 0 for v in events.values()), "a compatible variation axis is absent")
    schedules = ((1, 1, 1, 1, 1), (2, 2, 2, 2, 2), (1, 2, 3, 4, 5),
                 (20, 1, 2, 3, 4), (1, 4, 1, 4, 1), (5, 4, 3, 2, 1))
    # All six recoverable masks, palette roles, and edge schedules, including
    # unexpanded and uniform cases supported by solve beyond the sampling policy.
    for mask, (bg, fg), heights in itertools.product(
            _valid_masks(), ((0, 1), (3, 0), (9, 4)), schedules):
        widths = tuple(reversed(heights))
        large = _expand(_latent(mask, bg, fg), heights, widths)
        _require(np.array_equal(solve(large), _complete(bg, fg)),
                 "a valid mask or boundary schedule cannot be solved")
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(17000 + 10*a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b),
                     "explicit counts not honored")
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        _expect_error(lambda: generate_puzzle(0, a, b), ValueError)
    for seed in (-1, True, 0.5):
        _expect_error(lambda: generate_puzzle(seed), ValueError)
    damaged = generate_puzzle(0)
    damaged["train"][0]["output"][2, 2] = int(damaged["train"][0]["output"][1, 1])
    _expect_error(lambda: validate_generated_puzzle(damaged))
    wrong_clue = _latent(_valid_masks()[0], 0, 1)
    position = np.argwhere(wrong_clue == 1)[0]
    wrong_clue[tuple(position)] = 0
    _expect_error(lambda: solve(wrong_clue), ValueError)
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "color_normalized_inputs": len(layouts),
            "masks": len(masks), "ordered_role_palettes": len(palettes),
            "row_schedules": len(row_runs), "column_schedules": len(col_runs),
            "legal_count_combinations": legal, "events": events}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
