"""Parity-selected corners of a categorical equality tensor.

Only standard-library modules and NumPy are used. All conventions needed by
solve are fixed here: square side 2..5, categories 1..9, mismatch blank 0.
Generation and validation do not supply state to solve.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


# Original and selected-variation input hashes, restricted to supported sizes.
# These exclusions prevent copied evidence; they are not solver exceptions.
_EVIDENCE_INPUT_HASHES = frozenset((
    "17a1d49b6c4fd3b87d92b52b62973aa58c7c6746970259a8bbeca92fdc929012",
    "1a1e005d38a4688b52d5011b1e42295cc379da62667a9378553bcf7e5218c2ea",
    "36a296066a7548bbe81569a9fb9c5170c614f833d9fbe8fc0bd52268558aa4ab",
    "3af052b64ac4ee246f022ed369dc6f010ae8de79158e97757eaa1806341a36e9",
    "3ec67279deb29b6600f659384359db8e2b904d18c08fcaf9fbdf89651066a47d",
    "59d5966c96af7ecad5c9d2918d6582d102b2c67f6b765ea28ac24371ab4f93be",
    "5e770b4509873ec5f1cc657e8ae2140c282e6465346b90395d8b3d11939588c2",
    "6d25bd3c6ea391cac9a7f89e49603b159fb08fdd9248131082fb7d46b077579a",
    "8b59dc3648a3abfb97e6409eaac591d291762ed1daf9f25ac4370f9525216504",
    "add9c3826c14a0843ae1bdc538124feb60897d5fdadafadeb4279c36916a4ff1",
    "ae44a30ee566f39b9e2da31917efbf2cf2993abd611927d87e89c2123415cad7",
    "b0290509e0af54a56be45f1be20b186d7b2940a4fa1cad6aff83a3893b4bf598",
    "d11f3021cd22f6b4f5f4f1d84c91dcf63d42544cb86a3833e719bd03ee3cb321",
    "f4ee533e8fbc8c777bbfc4813e21ae1752b3d259fcb7394a7b278e8301b24296",
    "ffc1e3e60df2707df7e1ff077d45987ee2e615864ac761c7f1b1cbd50eee139a",
))


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _input(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.shape[0] != grid.shape[1]
            or not 2 <= grid.shape[0] <= 5
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((1 <= grid) & (grid <= 9))):
        raise ValueError("input must be a 2..5 square of categories 1..9")
    return grid.astype(np.int8, copy=False)


def _tensor(grid):
    """E[r*n+u,c*n+v] compares G[r,c] with G[u,v]."""
    n = grid.shape[0]
    # Axis order is macro row, within row, macro column, within column.
    selector = grid[:, None, :, None]
    reference = grid[None, :, None, :]
    return np.where(selector == reference, selector, 0).reshape(n*n, n*n)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Build the specified canvas, then preserve its parity-selected 2x2 crop."""
    grid = _input(input_grid)
    expanded = _tensor(grid)
    return (expanded[:2, :2] if grid.shape[0] % 2
            else expanded[:2, -2:]).copy()


def _scalar_crop(grid, left):
    """Four scalar tensor-coordinate evaluations, separate from solve's code."""
    n = grid.shape[0]
    start = 0 if left else n*n - 2
    result = np.empty((2, 2), dtype=np.int8)
    for row in range(2):
        for col in range(2):
            macro_r, within_r = divmod(row, n)
            macro_c, within_c = divmod(start + col, n)
            category = grid[macro_r, macro_c]
            result[row, col] = (category if category == grid[within_r, within_c]
                                else 0)
    return result


def _raw_crop(grid):
    return grid[:2, :2] if grid.shape[0] % 2 else grid[:2, -2:]


def _profile(mask):
    return sum(int(v) << i for i, v in enumerate(np.asarray(mask).flat))


def _mask(profile):
    return np.array([(profile >> i) & 1 for i in range(4)], dtype=bool).reshape(2, 2)


def _profiles(parity, partial=False):
    anchor = 0 if parity else 1
    return [p for p in range(16) if p & (1 << anchor)
            and (not partial or p.bit_count() in (2, 3))]


def _key(grid):
    return grid.shape[0], grid.tobytes()


def _is_evidence(grid):
    return hashlib.sha256(grid.tobytes()).hexdigest() in _EVIDENCE_INPUT_HASHES


def _texture(rng, n, palette):
    """Small categorical fields; no texture mode changes the transformation."""
    k = len(palette)
    rows, cols = np.indices((n, n))
    mode = int(rng.integers(8))
    if mode == 0:
        labels = rng.integers(k, size=(n, n))
    elif mode == 1:
        labels = rng.choice(k, size=(n, n), p=rng.dirichlet(np.full(k, .6)))
    elif mode == 2:
        h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        tile = rng.integers(k, size=(h, w))
        labels = tile[rows % h, cols % w]
    elif mode == 3:
        width = int(rng.integers(1, 3))
        labels = ((rows if rng.random() < .5 else cols) // width) % k
    elif mode == 4:
        centers = rng.random((k, 2)) * (n - 1)
        distance = ((rows[None] - centers[:, 0, None, None]) ** 2
                    + (cols[None] - centers[:, 1, None, None]) ** 2)
        labels = np.argmin(distance, axis=0)
    elif mode == 5:
        labels = rng.integers(k, size=(n, n))
        labels[:, n//2:] = labels[:, :n-n//2][:, ::-1]
    elif mode == 6:
        labels = np.zeros((n, n), dtype=int)
        positions = rng.choice(n*n, size=int(rng.integers(1, n+1)), replace=False)
        labels.flat[positions] = rng.integers(k, size=len(positions))
    else:
        a, b = int(rng.integers(1, k+1)), int(rng.integers(1, k+1))
        labels = (a*rows + b*cols + int(rng.integers(k))) % k
    return palette[labels].copy()


def _make_input(rng, n, profile, *, witness=False, minority=False,
                nonunit=False, test=False):
    maximum = min(9 if test else 6, n*n)
    k = int(rng.integers(2, maximum+1))
    palette = rng.choice(np.arange(1, 10, dtype=np.int8), size=k, replace=False)
    if nonunit and palette[0] == 1:
        palette[[0, 1]] = palette[[1, 0]]
    category = palette[0]
    grid = _texture(rng, n, palette)
    if minority:
        # A visible counterexample to selecting the globally dominant category.
        grid[:] = palette[1]
    elif not witness and profile == 15 and rng.random() < .25:
        grid[:] = category
    start = 0 if n % 2 else n-2
    mask = _mask(profile)
    patch = rng.choice(palette[1:], size=(2, 2))
    patch[mask] = category
    grid[:2, start:start+2] = patch
    if witness:
        # Both witness sizes exceed two, so the opposite anchor is outside patch.
        grid[0, n-1 if n % 2 else 0] = palette[1]
    return grid.astype(np.int8, copy=False)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Reserve a nondegenerate local support arrangement separately for each
    # parity. This controls the content shift only; it is not a solving key.
    reserved = {p: int(rng.choice(_profiles(p, partial=True))) for p in (0, 1)}
    odd_side = int(rng.choice([3, 5]))
    train_specs = [(odd_side, True, False), (4, True, True)]
    if num_train >= 3:
        train_specs.append((8-odd_side, False, False))
    if num_train >= 4:
        train_specs.append((2, False, False))
    while len(train_specs) < num_train:
        train_specs.append((int(rng.integers(2, 6)), False, False))
    test_parities = ([int(rng.integers(2))] if num_test == 1 else [0, 1])
    while len(test_parities) < num_test:
        test_parities.append(int(rng.integers(2)))
    rng.shuffle(test_parities)
    used = set()
    puzzle = {"train": [], "test": []}

    def add(split, n, profile, witness=False, minority=False, nonunit=False):
        for _ in range(64):
            grid = _make_input(rng, n, profile, witness=witness,
                               minority=minority, nonunit=nonunit,
                               test=split == "test")
            if _key(grid) not in used and not _is_evidence(grid):
                used.add(_key(grid))
                puzzle[split].append({"input": grid, "output": solve(grid)})
                return
        raise RuntimeError("bounded sampler exhausted distinct inputs")

    for n, witness, minority in train_specs:
        candidates = [p for p in _profiles(n % 2, partial=witness)
                      if p != reserved[n % 2]]
        if witness and n % 2:
            candidates = [p for p in candidates
                          if not np.array_equal(_mask(p), _mask(p).T)]
        profile = int(rng.choice(candidates))
        add("train", n, profile, witness, minority, witness and n % 2 == 1)
    # Train order carries no information. Every count still has both witnesses.
    rng.shuffle(puzzle["train"])
    for parity in test_parities:
        n = int(rng.choice([3, 5] if parity else [2, 4]))
        add("test", n, reserved[parity])
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def validate_generated_puzzle(puzzle) -> None:
    """Check ARC schema, scalar semantics, witnesses, and the content shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "invalid counts")
    seen, witnesses = set(), set()
    profiles = {0: set(), 1: set()}
    dominant_counterexample = nonunit_witness = transpose_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input/output")
            grid = _input(pair["input"])
            out = np.asarray(pair["output"])
            _require(out.shape == (2, 2)
                     and np.issubdtype(out.dtype, np.integer)
                     and np.all((0 <= out) & (out <= 9)), "invalid output")
            n, parity = grid.shape[0], grid.shape[0] % 2
            expected = _scalar_crop(grid, bool(parity))
            _require(np.array_equal(out, expected), "incorrect tensor corner")
            _require(np.array_equal(solve(grid), expected), "solve disagrees")
            category = int(grid[0, 0 if parity else n-1])
            _require(set(int(x) for x in out.flat) <= {0, category},
                     "output is not one category's support")
            _require(out[0, 0 if parity else 1] == category,
                     "selected anchor must survive")
            _require(_key(grid) not in seen and not _is_evidence(grid),
                     "duplicate or evidence input")
            seen.add(_key(grid))
            profile = _profile(out != 0)
            if split == "train":
                profiles[parity].add(profile)
                if (not np.array_equal(out, _scalar_crop(grid, not bool(parity)))
                        and not np.array_equal(out, _raw_crop(grid))
                        and 0 < np.count_nonzero(out) < 4):
                    witnesses.add(parity)
                counts = np.bincount(grid.flat, minlength=10)
                dominant_counterexample |= counts[category] < counts.max()
                nonunit_witness |= category != 1
                transpose_witness |= not np.array_equal(out, out.T)
            else:
                _require(profile not in profiles[parity],
                         "test support profile was not withheld")
    _require(witnesses == {0, 1}, "training must expose both parent mechanisms")
    _require(dominant_counterexample, "missing category-frequency counterexample")
    _require(nonunit_witness and transpose_witness,
             "training must show literal color and coordinate preservation")
    if tr >= 4:
        _require({np.asarray(p["input"]).shape[0] for p in puzzle["train"]}
                 == {2, 3, 4, 5}, "four demonstrations must cover all sizes")
    if te >= 2:
        _require({np.asarray(p["input"]).shape[0] % 2 for p in puzzle["test"]}
                 == {0, 1}, "multiple tests must exercise both parities")


def _signature(puzzle):
    return hashlib.sha256(json.dumps(
        {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
         for s in ("train", "test")}, separators=(",", ":")).encode()).hexdigest()


def _normalized_key(grid):
    roles = {}
    values = bytes(roles.setdefault(int(x), len(roles)) for x in grid.flat)
    return grid.shape[0], values


def self_test() -> dict:
    """Bounded streaming checks over 42 seeds, all counts, and all crop masks.

    No files, logs, journals, indexes, backups, or traces are written. Retain
    only tiny signatures and diversity counters, not a corpus of puzzles.
    """
    signatures, structures, palettes, color_counts = set(), set(), set(), set()
    sizes = {"train": set(), "test": set()}
    masks = {0: set(), 1: set()}
    match_counts = set()
    pair_count = uniform_inputs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _signature(puzzle)
        _require(signature == _signature(generate_puzzle(seed)), "nondeterminism")
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                n = grid.shape[0]
                pair_count += 1
                sizes[split].add(n)
                structures.add(_normalized_key(grid))
                palettes.add(tuple(int(x) for x in np.unique(grid)))
                color_counts.add(len(np.unique(grid)))
                masks[n % 2].add(_profile(out != 0))
                match_counts.add(int(np.count_nonzero(out)))
                uniform_inputs += len(np.unique(grid)) == 1
                # Scalar checks cover every tile, including discarded ones.
                expanded = _tensor(grid)
                for r in range(n):
                    for c in range(n):
                        for u in range(n):
                            for v in range(n):
                                expected = grid[r, c] if grid[r, c] == grid[u, v] else 0
                                _require(expanded[r*n+u, c*n+v] == expected,
                                         "full equality tensor is wrong")
    _require(len(signatures) == 42 and len(structures) >= 100, "poor structure variation")
    _require(len(palettes) >= 35 and len(color_counts) >= 5, "poor palette variation")
    _require(all(sizes[s] == {2, 3, 4, 5} for s in sizes), "size coverage missing")
    _require(all(len(masks[p]) == 8 for p in masks), "not all corner supports exercised")
    _require(match_counts == {1, 2, 3, 4}, "support cardinality variation missing")
    # Directly exercise all eight supported masks on each size, not only samples.
    for n in range(2, 6):
        for profile in _profiles(n % 2):
            grid = np.full((n, n), 7, dtype=np.int8)
            patch = np.where(_mask(profile), 3, 7).astype(np.int8)
            start = 0 if n % 2 else n-2
            grid[:2, start:start+2] = patch
            _require(np.array_equal(solve(grid), np.where(_mask(profile), 3, 0)),
                     "solver failed a valid support mask")
        _require(np.array_equal(solve(np.full((n, n), 9)), np.full((2, 2), 9)),
                 "uniform boundary failed")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(1000+a*10+b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b),
                     "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    for bad in (np.ones((1, 1), dtype=int), np.ones((6, 6), dtype=int),
                np.ones((3, 4), dtype=int), np.zeros((3, 3), dtype=int),
                np.ones((3, 3), dtype=float), np.full((3, 3), 10)):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-family input accepted")
    # Validator must detect a corrupted answer, rather than simply check shapes.
    bad = generate_puzzle(123)
    bad["test"][0]["output"][0, 0] = 9 - bad["test"][0]["output"][0, 0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "normalized_input_structures": len(structures), "palettes": len(palettes),
            "category_counts": sorted(color_counts),
            "train_sides": sorted(sizes["train"]), "test_sides": sorted(sizes["test"]),
            "profiles_per_parity": {str(p): len(masks[p]) for p in (0, 1)},
            "support_counts": sorted(match_counts), "uniform_inputs": uniform_inputs,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
