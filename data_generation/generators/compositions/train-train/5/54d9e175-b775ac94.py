"""Mirrored keys and locks: reflection and code-conditioned chamber filling.

The input contains its own codebook, in the centers and surrounds of the
already-filled source chambers. No generation parameters are used by solve.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np


QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
VERIFICATION_SEEDS = (
    0, 668853959, 920337073, 808123512, 328925198, 115098355,
    181362577, 188172500, 802193002, 14162255, 102435452, 898551856,
    842853127, 725243409, 352417463, 882515029, 439876697, 168277557,
    418622480, 911082639, 109425412, 282332591, 133563011, 37166380,
    838313123, 635280495, 962629111, 967840542, 993180366, 44078121,
    654453270, 463493664, 132986179, 941184790, 473139399, 902741246,
    454939378, 623469738, 46205135, 870066030, 765780050, 939952968,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _qcell(m, n, quadrant, dr, dc):
    """Room coordinates from outward distances to the central junction."""
    return (m - 1 - dr if quadrant[0] == 0 else m + dr,
            n - 1 - dc if quadrant[1] == 0 else n + dc)


def _others(source):
    y, x = source
    return ((y, 1 - x), (1 - y, x), (1 - y, 1 - x))


def _connected(cells):
    remaining = set(cells)
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = r + dy, c + dx
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
    return not remaining


def _runs(length, lines):
    result, start = [], 0
    for stop in (*lines, length):
        side = stop - start
        if side not in (3, 5, 7):
            return None
        result.append((start, stop))
        start = stop + 1
    return tuple(result)


def _pointer(row_runs, col_runs, m, n, quadrant):
    r, c = _qcell(m, n, quadrant, 0, 0)
    r0, r1 = row_runs[r]
    c0, c1 = col_runs[c]
    return (r1 - 1 if quadrant[0] == 0 else r0,
            c1 - 1 if quadrant[1] == 0 else c0)


def _read(input_grid):
    """Parse only visible input: dividers, source swatches, locks and seeds."""
    grid = _grid(input_grid)
    candidates = []
    for color in np.unique(grid):
        rows = tuple(int(r) for r in np.flatnonzero(np.all(grid == color, axis=1)))
        cols = tuple(int(c) for c in np.flatnonzero(np.all(grid == color, axis=0)))
        if len(rows) not in (3, 5) or len(cols) not in (3, 5):
            continue
        rr, cr = _runs(grid.shape[0], rows), _runs(grid.shape[1], cols)
        if rr is not None and cr is not None:
            candidates.append((int(color), rr, cr))
    _require(len(candidates) == 1, "unique full divider grid required")
    separator, rr, cr = candidates[0]
    m, n = len(rr) // 2, len(cr) // 2
    # At most one quadrant contains swatches. Thus background has a strict
    # majority among all noncentral room pixels, regardless of color numbers.
    peripheral = []
    blocks = {}
    for r, (r0, r1) in enumerate(rr):
        for c, (c0, c1) in enumerate(cr):
            block = grid[r0:r1, c0:c1]
            mask = np.ones(block.shape, dtype=bool)
            mask[block.shape[0] // 2, block.shape[1] // 2] = False
            values = block[mask]
            peripheral.extend(int(v) for v in values)
            blocks[r, c] = (block, mask)
    background, frequency = Counter(peripheral).most_common(1)[0]
    _require(frequency * 2 > len(peripheral) and background != separator,
             "background must be the peripheral majority")
    centers = np.empty((2 * m, 2 * n), dtype=int)
    donors, mapping, extras = {}, {}, []
    for (r, c), (block, mask) in blocks.items():
        code = int(block[block.shape[0] // 2, block.shape[1] // 2])
        centers[r, c] = code
        values = block[mask]
        if np.all(values == values[0]) and int(values[0]) != background:
            fill = int(values[0])
            _require(code != fill, "source swatches must retain a contrasting code")
            _require(code not in mapping or mapping[code] == fill,
                     "contradictory source codebook")
            mapping[code] = fill
            donors[r, c] = code
        else:
            locations = np.argwhere(mask & (block != background))
            _require(len(locations) <= 1, "ordinary room has stray ink")
            for dy, dx in locations:
                extras.append((rr[r][0] + int(dy), cr[c][0] + int(dx),
                               int(block[dy, dx])))
    _require(2 <= len(mapping) <= 3, "two or three visible code mappings required")
    _require(len(set(mapping.values())) == len(mapping), "fills must be distinct")
    _require(not (set(mapping) & set(mapping.values())), "codes and fills overlap")
    _require(not ({background, separator} & (set(mapping) | set(mapping.values()))),
             "neutral colors overlap codebook")
    _require(set(int(v) for v in centers.flat) <= set(mapping), "unknown room code")
    source_quadrants = {(int(r >= m), int(c >= n)) for r, c in donors}
    _require(len(source_quadrants) == 1, "all swatches must lie in one quadrant")
    source = next(iter(source_quadrants))
    keys = {}
    for (r, c), code in donors.items():
        dr = m - 1 - r if source[0] == 0 else r - m
        dc = n - 1 - c if source[1] == 0 else c - n
        keys[dr, dc] = code
    _require(3 <= len(keys) < m * n and (0, 0) in keys,
             "source must include its corner and leave at least one room unused")
    _require(_connected(keys), "source rooms must be eight-connected")
    _require(max(r for r, c in keys) == m - 1 and max(c for r, c in keys) == n - 1,
             "source must span its quadrant in both directions")
    _require(1 <= len(extras) <= 3, "one to three corner seeds required")
    control_colors = {v for _, _, v in extras}
    _require(len(control_colors) == 1, "seeds must share one control color")
    control = next(iter(control_colors))
    _require(control not in {background, separator} | set(mapping) | set(mapping.values()),
             "control color overlaps another role")
    seeds = {}
    for row, col, color in extras:
        found = [q for q in _others(source) if _pointer(rr, cr, m, n, q) == (row, col)]
        _require(len(found) == 1 and found[0] not in seeds,
                 "seed must touch the central junction in a target quadrant")
        seeds[found[0]] = (row, col)
    return dict(grid=grid, separator=separator, background=background, control=control,
                row_runs=rr, col_runs=cr, m=m, n=n, source=source, keys=keys,
                mapping=mapping, centers=centers, seeds=seeds, donors=donors)


def _destinations(info, mode="match"):
    actions = {}
    m, n = info["m"], info["n"]
    if mode == "all_codes":
        for r in range(2 * m):
            for c in range(2 * n):
                if (r, c) not in info["donors"]:
                    actions[r, c] = info["mapping"][int(info["centers"][r, c])]
        return actions
    quadrants = _others(info["source"]) if mode == "all_quadrants" else info["seeds"]
    for q in quadrants:
        for (dr, dc), code in info["keys"].items():
            if mode == "translate":
                sr, sc = _qcell(m, n, info["source"], dr, dc)
                r = sr + (q[0] - info["source"][0]) * m
                c = sc + (q[1] - info["source"][1]) * n
            else:
                r, c = _qcell(m, n, q, dr, dc)
            if mode == "ignore_locks" or int(info["centers"][r, c]) == code:
                actions[r, c] = info["mapping"][code]
    return actions


def _paint(info, actions):
    result = info["grid"].copy()
    for (r, c), fill in actions.items():
        r0, r1 = info["row_runs"][r]
        c0, c1 = info["col_runs"][c]
        result[r0:r1, c0:c1] = fill
    # Corner control seeds are overlays, like dividers; they are never erased.
    for row, col in info["seeds"].values():
        result[row, col] = info["control"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    return _paint(info, _destinations(info))


@lru_cache(maxsize=8)
def _motif_options(m, n):
    other = [(r, c) for r in range(m) for c in range(n) if (r, c) != (0, 0)]
    options = {}
    for count in range(3, m * n):
        shapes = []
        for rest in combinations(other, count - 1):
            cells = ((0, 0), *rest)
            if (max(r for r, c in cells) == m - 1
                    and max(c for r, c in cells) == n - 1 and _connected(cells)):
                shapes.append(cells)
        if shapes:
            options[count] = tuple(shapes)
    return options


def _lengths(rng, count, unequal=False):
    sides = [3] * count
    budget = (30 - (count - 1) - 3 * count) // 2
    steps = int(rng.integers(0, budget + 1))
    for _ in range(steps):
        candidates = [i for i, side in enumerate(sides) if side < 7]
        sides[int(rng.choice(candidates))] += 2
    if unequal and len(set(sides)) == 1:
        if sides[0] > 3:
            sides[0] -= 2
        else:
            sides[0] += 2
    return tuple(sides)


def _intervals(sides):
    result, start = [], 0
    for side in sides:
        result.append((start, start + side))
        start += side + 1
    return tuple(result)


def _make_pair(rng, palette, source, mask, index, is_train):
    codes, fills, neutral = palette
    k = len(codes)
    if is_train and index == 0:
        m, n = 2, int(rng.integers(2, 4))
    else:
        m, n = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    options = _motif_options(m, n)
    count = int(rng.choice(list(options)))
    shapes = options[count]
    motif = shapes[int(rng.integers(len(shapes)))]
    roles = list(int(v) for v in rng.integers(0, k, size=count))
    for role, pos in enumerate(rng.permutation(count)[:k]):
        roles[int(pos)] = role
    keys = dict(zip(motif, roles))
    seeds = tuple(q for bit, q in enumerate(_others(source)) if mask & (1 << bit))
    # Sample lock equality and inequality deliberately; do not hope that the
    # random palettes accidentally provide instructive counterexamples.
    for _ in range(200):
        locks = rng.integers(0, k, size=(2 * m, 2 * n))
        matched_roles = set()
        for q in _others(source):
            if q in seeds:
                matches = set(int(v) for v in rng.choice(
                    count, size=int(rng.integers(1, count)), replace=False))
            else:
                matches = {int(rng.integers(count))}
            for j, ((dr, dc), role) in enumerate(keys.items()):
                r, c = _qcell(m, n, q, dr, dc)
                if j in matches:
                    locks[r, c] = role
                    if q in seeds:
                        matched_roles.add(role)
                else:
                    choices = [v for v in range(k) if v != role]
                    locks[r, c] = int(rng.choice(choices))
        if not (is_train and index == 0) or matched_roles == set(range(k)):
            break
    else:
        raise RuntimeError("could not expose every lookup color")
    row_sides = _lengths(rng, 2 * m, unequal=is_train and index == 1)
    col_sides = _lengths(rng, 2 * n, unequal=is_train and index == 1)
    rr, cr = _intervals(row_sides), _intervals(col_sides)
    background, separator, control = neutral
    inp = np.full((rr[-1][1], cr[-1][1]), separator, dtype=np.int8)
    for r, (r0, r1) in enumerate(rr):
        for c, (c0, c1) in enumerate(cr):
            inp[r0:r1, c0:c1] = background
            inp[(r0 + r1) // 2, (c0 + c1) // 2] = codes[int(locks[r, c])]
    for (dr, dc), role in keys.items():
        r, c = _qcell(m, n, source, dr, dc)
        r0, r1 = rr[r]
        c0, c1 = cr[c]
        inp[r0:r1, c0:c1] = fills[role]
        inp[(r0 + r1) // 2, (c0 + c1) // 2] = codes[role]
    output = inp.copy()
    # Ground truth is constructed directly from sampled keys and locks.
    # It does not call the input parser, solve, or _destinations.
    for q in seeds:
        for (dr, dc), role in keys.items():
            r, c = _qcell(m, n, q, dr, dc)
            if int(locks[r, c]) == role:
                r0, r1 = rr[r]
                c0, c1 = cr[c]
                output[r0:r1, c0:c1] = fills[role]
        row, col = _pointer(rr, cr, m, n, q)
        inp[row, col] = output[row, col] = control
    return {"input": inp, "output": output}


def _checked_integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside permitted range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    _require(int(seed) >= 0, "seed must be nonnegative")
    rng = np.random.default_rng(int(seed))
    colors = tuple(int(v) for v in rng.permutation(10))
    k = int(rng.integers(2, 4))
    codes, fills, pool = colors[:k], colors[k:2 * k], colors[2 * k:]
    variable_neutral = bool(rng.integers(2))
    held_mask = int(rng.integers(1, 7))
    train_masks = [mask for mask in range(1, 8) if mask != held_mask]
    orientation_order = [QUADRANTS[int(i)] for i in rng.permutation(4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            is_train = section == "train"
            mask = (7 if index == 0 else int(rng.choice(train_masks))) if is_train else held_mask
            if is_train and index == 1:
                mask = int(rng.choice([x for x in train_masks if x != 7]))
            source = orientation_order[index] if is_train and index < 2 else QUADRANTS[int(rng.integers(4))]
            if variable_neutral:
                neutral = tuple(int(v) for v in rng.choice(pool, size=3, replace=False))
            else:
                neutral = pool[:3]
            for _ in range(100):
                pair = _make_pair(rng, (codes, fills, neutral), source, mask, index, is_train)
                if _key(pair["input"]) in seen:
                    continue
                info = _read(pair["input"])
                # One rich demonstration rules out moving the stencil by
                # translation while keeping its raster orientation.
                if is_train and index == 0 and np.array_equal(
                        _paint(info, _destinations(info, "translate")), pair["output"]):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("distinct instructive pair sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, codebook, infos = set(), None, {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "changed dimensions")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            infos[section].append(info)
            if codebook is None:
                codebook = info["mapping"]
            _require(info["mapping"] == codebook, "codebook must be puzzle-wide")
            actions = _destinations(info)
            _require(actions, "each pair must open at least one lock")
            _require(np.array_equal(_paint(info, actions), out), "wrong conditional reflection")
            for q in info["seeds"]:
                flags = [int(info["centers"][_qcell(info["m"], info["n"], q, dr, dc)]) == code
                         for (dr, dc), code in info["keys"].items()]
                _require(any(flags) and not all(flags), "seed needs both fitting and unfitting keys")
            for mode in ("ignore_locks", "all_codes"):
                _require(not np.array_equal(_paint(info, _destinations(info, mode)), out),
                         f"pair admits shortcut: {mode}")
            # Room-by-room preservation, independent of the painting helper.
            for r, (r0, r1) in enumerate(info["row_runs"]):
                for c, (c0, c1) in enumerate(info["col_runs"]):
                    block = out[r0:r1, c0:c1]
                    if (r, c) not in actions:
                        _require(np.array_equal(block, inp[r0:r1, c0:c1]), "inactive room changed")
                    else:
                        mask = np.ones(block.shape, dtype=bool)
                        for row, col in info["seeds"].values():
                            if r0 <= row < r1 and c0 <= col < c1:
                                mask[row - r0, col - c0] = False
                                _require(out[row, col] == inp[row, col], "control seed changed")
                        _require(np.all(block[mask] == actions[r, c]), "opened room not fully filled")
    train_infos = infos["train"]
    _require(any(len(i["seeds"]) == 3 for i in train_infos), "training must expose all three maps")
    _require(any(len(i["seeds"]) < 3 for i in train_infos), "partial seeding training evidence")
    _require(len({i["source"] for i in train_infos}) >= 2, "source orientation evidence")
    opened_codes = {int(i["centers"][r, c]) for i in train_infos for r, c in _destinations(i)}
    _require(opened_codes == set(codebook), "every lookup must be demonstrated in action")
    for mode in ("all_quadrants", "translate"):
        _require(any(not np.array_equal(_paint(i, _destinations(i, mode)), p["output"])
                     for i, p in zip(train_infos, puzzle["train"])),
                 f"training admits shortcut: {mode}")
    return None


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _counterfactual_checks(pair):
    """Check changes with known local effects, beyond sampled match ratios."""
    inp, out = pair["input"], pair["output"]
    info = _read(inp)
    # Geometry and palette are roles, never fixed literals or compass sides.
    transforms = (lambda a: a.T, lambda a: np.flip(a, axis=0),
                  lambda a: np.flip(a, axis=1))
    for transform in transforms:
        _require(np.array_equal(solve(transform(inp)), transform(out)),
                 "solver is not geometrically covariant")
    permutation = np.array([7, 0, 8, 2, 9, 4, 1, 6, 3, 5])
    _require(np.array_equal(solve(permutation[inp]), permutation[out]),
             "solver depends on literal palette values")
    # Source samples remain intact. Change every target key to fit, or to
    # fail. These local rule members need not meet the sampler's requirement
    # that every quadrant demonstrate both outcomes.
    for fits in (False, True):
        changed = inp.copy()
        for q in info["seeds"]:
            for (dr, dc), code in info["keys"].items():
                r, c = _qcell(info["m"], info["n"], q, dr, dc)
                r0, r1 = info["row_runs"][r]
                c0, c1 = info["col_runs"][c]
                lock = code if fits else next(k for k in info["mapping"] if k != code)
                changed[(r0 + r1) // 2, (c0 + c1) // 2] = lock
        expected = changed.copy()
        if fits:
            for q in info["seeds"]:
                for (dr, dc), code in info["keys"].items():
                    r, c = _qcell(info["m"], info["n"], q, dr, dc)
                    r0, r1 = info["row_runs"][r]
                    c0, c1 = info["col_runs"][c]
                    expected[r0:r1, c0:c1] = info["mapping"][code]
            for row, col in info["seeds"].values():
                expected[row, col] = info["control"]
        _require(np.array_equal(solve(changed), expected), "all-fit/all-fail rule failed")
    # Removing one request must suppress exactly that target quadrant.
    q = next(iter(info["seeds"]))
    row, col = info["seeds"][q]
    changed = inp.copy()
    changed[row, col] = info["background"]
    expected = out.copy()
    for r in range(q[0] * info["m"], (q[0] + 1) * info["m"]):
        for c in range(q[1] * info["n"], (q[1] + 1) * info["n"]):
            r0, r1 = info["row_runs"][r]
            c0, c1 = info["col_runs"][c]
            expected[r0:r1, c0:c1] = changed[r0:r1, c0:c1]
    _require(np.array_equal(solve(changed), expected), "seed removal is not local")


def self_test():
    shapes, normalized, motifs, counts, orientations = set(), set(), set(), set(), set()
    code_counts, backgrounds, controls, masks, room_sides = set(), set(), set(), set(), set()
    unequal = sparse = diagonal = holes = repeated_keys = 0
    for seed in VERIFICATION_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_masks = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _read(pair["input"])
                shapes.add(pair["input"].shape)
                roles = {}
                normalized.add((pair["input"].shape, tuple(
                    roles.setdefault(int(v), len(roles)) for v in pair["input"].flat)))
                code_counts.add(len(info["mapping"]))
                backgrounds.add(info["background"])
                controls.add(info["control"])
                counts.add((info["m"], info["n"]))
                orientations.add(info["source"])
                motif = tuple(sorted(info["keys"]))
                motifs.add((info["m"], info["n"], motif))
                sparse += len(motif) * 2 <= info["m"] * info["n"]
                diagonal += not _four_connected(motif)
                holes += info["m"] == info["n"] == 3 and len(motif) == 8 and (1, 1) not in motif
                repeated_keys += len(info["keys"]) > len(info["mapping"])
                lengths = [b - a for a, b in (*info["row_runs"], *info["col_runs"])]
                room_sides.update(lengths)
                unequal += len(set(lengths)) > 1
                mask = sum(1 << j for j, q in enumerate(_others(info["source"])) if q in info["seeds"])
                masks.add(mask)
                if section == "train":
                    train_masks.add(mask)
                else:
                    _require(mask not in train_masks, "test subset was not held out")
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "solver disagrees")
                _require(np.array_equal(before, pair["input"]), "solve mutated input")
        _require(_same(puzzle, generate_puzzle(seed)), "nondeterministic puzzle")
        if seed == 0:
            _counterfactual_checks(puzzle["train"][0])
    _require(code_counts == {2, 3} and len(backgrounds) >= 8 and len(controls) >= 8,
             "palette variation missing")
    _require(counts == {(2, 2), (2, 3), (3, 2), (3, 3)}, "lattice variation missing")
    _require(orientations == set(QUADRANTS) and masks == set(range(1, 8)), "reflection variation missing")
    _require(room_sides == {3, 5, 7} and unequal >= 100, "chamber geometry variation missing")
    _require(len(shapes) >= 30 and len(normalized) >= 200 and len(motifs) >= 30,
             "structural variation insufficient")
    _require(sparse > 0 and diagonal > 0 and holes > 0 and repeated_keys > 0,
             "sparse, diagonal, hollow or repeated-code stencils missing")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(812 + nt * 10 + ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "explicit counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "normalized_inputs": len(normalized), "stencils": len(motifs),
            "unequal_geometry": unequal, "sparse": sparse, "diagonal_only_links": diagonal,
            "hollow": holes, "repeated_codes": repeated_keys}


def _four_connected(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = r + dy, c + dx
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
