"""Southeast corner-state growth followed by a bottom-phased row wave.

The input contains one solid square seed in a distinct exterior background.
Its reserved canvas is large enough for doubling and one-cell row translations.
Only NumPy and the Python standard library are used; nothing is read or saved.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import hashlib

import numpy as np


SIZES = tuple(range(5, 15))
OFFSETS = (0, -1, 0, 1)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= s <= 30 for s in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=False)


def _masks(n, profile):
    patch, reach, half = profile
    core = 1 if n % 2 else 2
    a = (n - core) // 2
    corner = np.zeros((n, n), dtype=bool)
    for r in (0, n - patch):
        for c in (0, n - patch):
            corner[r:r + patch, c:c + patch] = True
    cross = np.zeros_like(corner)
    cross[a - half:a + core + half, a - reach:a + core + reach] = True
    cross[a - reach:a + core + reach, a - half:a + core + half] = True
    return corner, cross


@lru_cache(maxsize=10)
def _profiles(n):
    """Feasible equal corner patches and centered orthogonal crosses."""
    a = (n - (1 if n % 2 else 2)) // 2
    result = []
    for patch in range(1, a + 1):
        for reach in range(1, a + 1):
            for half in range(min(2, reach - 1) + 1):
                profile = (patch, reach, half)
                cm, xm = _masks(n, profile)
                if not np.any(cm & xm) and np.any(~(cm | xm)):
                    result.append(profile)
    return tuple(result)


def _seed(n, profile, roles):
    cm, xm = _masks(n, profile)
    result = np.full((n, n), roles[2], dtype=np.int8)
    result[cm], result[xm] = roles[0], roles[1]
    return result


def _analyze(input_grid):
    """Parse all solving information from visible geometry, never generation state."""
    grid = _array(input_grid)
    bg = int(grid[0, 0])  # The input's entire leftmost column is exterior.
    rr, cc = np.where(grid != bg)
    if len(rr) == 0:
        raise ValueError("missing seed")
    top, left = int(rr.min()), int(cc.min())
    bottom, right = int(rr.max()), int(cc.max())
    n = bottom - top + 1
    if n not in SIZES or right - left + 1 != n or left < 1:
        raise ValueError("seed must be a square of side 5..14 with left margin")
    crop = grid[top:bottom + 1, left:right + 1]
    if np.any(crop == bg) or np.count_nonzero(grid != bg) != n * n:
        raise ValueError("the sole seed must be solid and distinct from exterior")
    colors = set(map(int, np.unique(crop)))
    corner, cross = int(crop[0, 0]), int(crop[n // 2, n // 2])
    if len(colors) != 3 or corner == cross:
        raise ValueError("three distinct seed roles are required")
    filler = next(iter(colors - {corner, cross}))
    roles = (corner, cross, filler)
    core = 1 if n % 2 else 2
    patch = next((i for i, color in enumerate(crop[0]) if color != corner), n)
    cross_mask = crop == cross
    reach = (int(np.count_nonzero(cross_mask[n // 2])) - core) // 2
    first = int(np.where(cross_mask)[0].min())
    half = (int(np.count_nonzero(cross_mask[first])) - core) // 2
    profile = (patch, reach, half)
    if profile not in _profiles(n) or not np.array_equal(crop, _seed(n, profile, roles)):
        raise ValueError("seed is not a disjoint corner-patch / centered-cross motif")
    side = 2 * n
    if top + side > grid.shape[0] or left + side >= grid.shape[1]:
        raise ValueError("insufficient reserved southeast space or wave margin")
    return dict(grid=grid, bg=bg, seed=crop, n=n, top=top, left=left,
                bottom=top + side - 1, side=side, roles=roles, profile=profile,
                bottom_pad=grid.shape[0] - top - side)


def _grow(info):
    """B: each layer reads the preceding southeast corner before advancing it."""
    n, roles = info["n"], info["roles"]
    grown = np.empty((2 * n, 2 * n), dtype=np.int8)
    grown[:n, :n] = info["seed"]
    successor = dict(zip(roles, roles[1:] + roles[:1]))
    state = int(grown[n - 1, n - 1])
    for k in range(n, 2 * n):
        grown[:k, k] = state
        grown[k, :k] = state
        state = successor[state]
        grown[k, k] = state
    return grown


def _place(info, block, offsets=None):
    output = np.full_like(info["grid"], info["bg"])
    side = block.shape[0]
    for r in range(side):
        shift = OFFSETS[(side - 1 - r) % 4] if offsets is None else offsets[r]
        start = info["left"] + shift
        if start < 0 or start + side > output.shape[1]:
            raise ValueError("row translation would clip")
        output[info["top"] + r, start:start + side] = block[r]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """B constructs the whole square; A moves its complete multicolor rows."""
    info = _analyze(input_grid)
    return _place(info, _grow(info))


def _key(grid):
    return grid.shape, grid.tobytes()


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _make_input(rng, n, profile, roles, bg, top=None, bottom_pad=None):
    spare_h = 30 - 2 * n
    if top is None:
        top = int(rng.integers(0, min(3, spare_h) + 1))
    if bottom_pad is None:
        bottom_pad = int(rng.integers(0, min(5, spare_h - top) + 1))
    spare_w = 28 - 2 * n  # Two mandatory lateral cells, in addition to expansion.
    extra_left = int(rng.integers(0, min(6, spare_w) + 1))
    extra_right = int(rng.integers(0, min(6, spare_w - extra_left) + 1))
    left, right = 1 + extra_left, 1 + extra_right
    grid = np.full((top + 2 * n + bottom_pad, left + 2 * n + right), bg, dtype=np.int8)
    grid[top:top + n, left:left + n] = _seed(n, profile, roles)
    return grid


def _pick_profile(rng, n, kind="any"):
    candidates = list(_profiles(n))
    if kind == "base":
        return (1, 1, 0)
    if kind == "rich":
        candidates = [p for p in candidates if p[0] >= 2 and p[1] >= 2 and p[2] >= 1
                      and np.count_nonzero(_seed(n, p, (1, 2, 3)) == 2)
                      >= np.count_nonzero(_seed(n, p, (1, 2, 3)) == 3)]
    return candidates[int(rng.integers(len(candidates)))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    shared_top = int(rng.integers(0, 4))
    # Scale extrapolation may go down, up, or in both directions.
    regime = int(rng.integers(0, 3))
    test_pool = ((11, 12, 13, 14), (5, 6), (5, 6, 11, 12, 13, 14))[regime]
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            top = bottom_pad = None
            if split == "train" and i < 2:
                n = int(rng.choice((7, 9) if i == 0 else (8, 10)))
                profile = _pick_profile(rng, n, "base" if i == 0 else "rich")
                roles = first_roles if i == 0 else first_roles[1:] + first_roles[:1]
                top, bottom_pad = shared_top, i
            else:
                n = int(rng.choice((7, 8, 9, 10) if split == "train" else test_pool))
                profile = _pick_profile(rng, n)
                roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
            bg = int(rng.choice([c for c in range(10) if c not in roles]))
            inp = _make_input(rng, n, profile, roles, bg, top, bottom_pad)
            if _key(inp) in seen:
                # At most 13 inputs are already present; 14 distinct palette
                # candidates guarantee a free one without unbounded retries.
                for candidate in permutations([c for c in range(10) if c != bg], 3):
                    info = _analyze(inp)
                    altered = inp.copy()
                    altered[info["top"]:info["top"] + n,
                            info["left"]:info["left"] + n] = _seed(n, profile, candidate)
                    if _key(altered) not in seen:
                        inp = altered
                        break
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": solve(inp)})
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    inp, out = _array(pair["input"]), _array(pair["output"])
    info = _analyze(inp)
    _require(out.shape == inp.shape, "canvas dimensions changed")
    side, top, left = info["side"], info["top"], info["left"]
    recovered = np.empty((side, side), dtype=np.int8)
    offsets = []
    for r in range(out.shape[0]):
        columns = np.flatnonzero(out[r] != info["bg"])
        if not top <= r < top + side:
            _require(len(columns) == 0, "extra occupied row")
            continue
        _require(len(columns) == side and np.all(np.diff(columns) == 1),
                 "each completed row must remain solid and retain its full length")
        delta = int(columns[0]) - left
        _require(delta == OFFSETS[(info["bottom"] - r) % 4], "wrong bottom phase")
        offsets.append(delta)
        recovered[r - top] = out[r, columns]
    n, roles = info["n"], info["roles"]
    _require(np.array_equal(recovered[:n, :n], info["seed"]), "unwaved seed changed")
    # Check all arms and intersections directly, without calling solve/_grow.
    for k in range(n, side):
        arm = roles[(k - n) % 3]
        corner = roles[(k - n + 1) % 3]
        _require(np.all(recovered[:k, k] == arm), "right arm did not inherit old corner")
        _require(np.all(recovered[k, :k] == arm), "bottom arm did not inherit old corner")
        _require(recovered[k, k] == corner, "new corner did not advance")
    _require(set(map(int, np.unique(out))) == {info["bg"], *roles}, "palette changed")
    # Row translation preserves the complete ordered color word, including the seed.
    for r in range(side):
        _require(np.count_nonzero(out[top + r] != info["bg"]) == side, "row lost cells")
    no_a = _place(info, recovered, [0] * side)
    no_b = _place(info, info["seed"])
    _require(not np.array_equal(out, no_a), "A is dispensable")
    _require(not np.array_equal(out, no_b), "B is dispensable")
    info.update(recovered=recovered, observed=offsets)
    return info


def validate_generated_puzzle(puzzle):
    """Validate the rule, geometry, necessity, and distinguishing demonstrations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _require({i["n"] % 2 for i in train} == {0, 1}, "training needs both size parities")
    _require(len({i["bottom_pad"] % 4 for i in train}) >= 2, "canvas anchor ambiguity")
    _require(len({i["bottom"] % 4 for i in train}) >= 2, "absolute phase ambiguity")
    _require(any(i["profile"] == (1, 1, 0) for i in train), "missing simple motif")
    _require(any(i["profile"][0] >= 2 and i["profile"][2] >= 1 for i in train),
             "missing patch and thick-cross demonstration")
    _require(any(c in j["roles"] and j["roles"].index(c) != pos
                 for i in train for j in train for pos, c in enumerate(i["roles"])),
             "literal colors have not changed roles in training")
    for anchor in ("top", "canvas", "seed"):
        matches = True
        for i in train:
            for r, actual in enumerate(i["observed"]):
                distance = (r if anchor == "top" else
                            i["grid"].shape[0] - 1 - i["top"] - r if anchor == "canvas"
                            else i["n"] - 1 - r)
                matches &= OFFSETS[distance % 4] == actual
        _require(not matches, f"training permits {anchor}-anchored wave")
    # Infer the wave and role successor from returned demonstrations, then verify
    # that tests combine those same observed local relations at another scale.
    learned_wave, learned_cycle = {}, {}
    for i in train:
        for r, delta in enumerate(i["observed"]):
            phase = (i["side"] - 1 - r) % 4
            _require(phase not in learned_wave or learned_wave[phase] == delta, "wave conflict")
            learned_wave[phase] = delta
        roles, state = i["roles"], 0
        for k in range(i["n"], i["side"]):
            nxt = roles.index(int(i["recovered"][k, k]))
            _require(state not in learned_cycle or learned_cycle[state] == nxt, "cycle conflict")
            learned_cycle[state] = nxt
            state = nxt
    _require(learned_wave == dict(enumerate(OFFSETS)), "training omits wave phases")
    _require(learned_cycle == {0: 1, 1: 2, 2: 0}, "training omits state transitions")
    train_sizes = {i["n"] for i in train}
    for i in infos["test"]:
        _require(i["n"] not in train_sizes, "test must exercise an unseen seed size")
        for r, actual in enumerate(i["observed"]):
            _require(actual == learned_wave[(i["side"] - 1 - r) % 4], "test phase mismatch")
        state = 0
        for k in range(i["n"], i["side"]):
            _require(np.all(i["recovered"][k, :k] == i["roles"][state]), "test arm mismatch")
            state = learned_cycle[state]
            _require(i["recovered"][k, k] == i["roles"][state], "test corner mismatch")


def self_test():
    """Bounded 42-seed sweep, all 30 legal count combinations, and mutations.

    Retain only hashes/statistics and one puzzle at a time; no files or traces.
    The returned measurements can be saved by a caller as compact JSON.
    """
    sizes, profiles, shapes, palettes, digests = set(), set(), set(), set(), set()
    colors = [set() for _ in range(4)]
    regimes, pairs = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        digests.add(_puzzle_digest(puzzle))
        train_sizes = {_analyze(p["input"])["n"] for p in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
                pairs += 1
                sizes.add(i["n"])
                profiles.add((i["n"], *i["profile"]))
                shapes.add(i["grid"].shape)
                palettes.add((i["bg"], *i["roles"]))
                for bucket, color in zip(colors, (i["bg"], *i["roles"])):
                    bucket.add(color)
                if split == "test":
                    regimes.add("smaller" if i["n"] < min(train_sizes) else "larger")
    _require(len(digests) == 42 and sizes == set(SIZES), "seed/size diversity missing")
    _require(len(profiles) >= 45 and len(shapes) >= 40, "structural variation too narrow")
    _require(all(s == set(range(10)) for s in colors), "palette role coverage missing")
    _require(regimes == {"smaller", "larger"}, "scale shifts missing")
    legal_count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "wrong explicit counts")
            legal_count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0)
    # A wrong phase, a missing B expansion, and a changed corner are each rejected.
    pair = p["train"][0]
    i = _analyze(pair["input"])
    block = _grow(i)
    wrong_corner = block.copy()
    wrong_corner[i["n"], i["n"]] = i["roles"][0]
    bad_outputs = (_place(i, block, [0] * i["side"]),
                   _place(i, i["seed"]), _place(i, wrong_corner))
    for bad in bad_outputs:
        try:
            _check_pair({"input": pair["input"], "output": bad})
        except AssertionError:
            pass
        else:
            raise AssertionError("family-specific mutation was accepted")
    # Solver coverage outside the generator's split schedule: every valid profile
    # at every size is parsed and checked at minimal vertical margins, in memory.
    profile_probes = 0
    rng = np.random.default_rng(9981)
    for n in SIZES:
        for profile in _profiles(n):
            inp = _make_input(rng, n, profile, (0, 4, 8), 2, 0, 0)
            _check_pair({"input": inp, "output": solve(inp)})
            profile_probes += 1
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "seed_sides": sorted(sizes), "size_motif_profiles": len(profiles),
            "canvas_shapes": len(shapes), "palettes": len(palettes),
            "test_scale_regimes": sorted(regimes), "legal_count_calls": legal_count_calls,
            "all_profile_probes": profile_probes, "mutation_checks": len(bad_outputs),
            "independent_semantic_review": False}
