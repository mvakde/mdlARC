"""Glyph-stencil programs initialize a three-state southeast square growth.

The visible, monochrome legend supplies the complete glyph-to-stencil mapping.
No solver state is learned or retained between calls. Only NumPy and the Python
standard library are used; generating and testing never writes files.
"""
from __future__ import annotations

import itertools
import json

import numpy as np


HOST_SEEDS = (0, 994055201, 862842446, 904372673, 863853680, 419633605,
              100000680, 473353526, 518272803, 160251403, 485299924,
              873898076, 600989433, 941632692, 394542176, 440066370,
              418220272, 542292412, 130065512, 775988116, 162114571,
              793666598, 47504282, 278804007, 467757170, 68173707,
              755697569, 299535551, 17690688, 850364820, 56523161,
              942228478, 69915276, 359239354, 305775916, 365035921,
              523854752, 541223981, 175212696, 793565781, 282397790,
              851761021)
_SIDES = (5, 7, 9)


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(name, value, low, high=None):
    _need(not isinstance(value, (bool, np.bool_))
          and isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    reached = {min(cells)}
    pending = list(reached)
    while pending:
        r, c = pending.pop()
        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            q = (r + dr, c + dc)
            if q in cells and q not in reached:
                reached.add(q)
                pending.append(q)
    return reached == cells


def _glyphs(rng):
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
    result, used = [], set()

    def consider(bits):
        mask = np.array([(bits >> k) & 1 for k in range(h * w)],
                        dtype=bool).reshape(h, w)
        if (bits not in used and 2 <= int(mask.sum()) < h * w
                and mask.any(axis=0).all() and mask.any(axis=1).all()
                and _connected(mask)):
            result.append(mask)
            used.add(bits)

    for _ in range(128):
        consider(int(rng.integers(1, 1 << (h * w))))
        if len(result) == 3:
            return tuple(result)
    # At most 4095 candidates, and terminate immediately on the third mask.
    for bits in range(1, 1 << (h * w)):
        consider(bits)
        if len(result) == 3:
            return tuple(result)
    raise AssertionError("insufficient glyph masks")


def _stencils(n, profile):
    patch, reach, half = profile
    corner = np.zeros((n, n), dtype=bool)
    for r, c in itertools.product((0, n - patch), repeat=2):
        corner[r:r + patch, c:c + patch] = True
    cross = np.zeros_like(corner)
    m = n // 2
    cross[m - half:m + half + 1, m - reach:m + reach + 1] = True
    cross[m - reach:m + reach + 1, m - half:m + half + 1] = True
    return corner, cross, ~(corner | cross)


def _profiles(n):
    result = []
    for patch in range(1, n // 2 + 1):
        for reach in range(1, n // 2 + 1):
            for half in range(min(2, reach - 1) + 1):
                masks = _stencils(n, (patch, reach, half))
                if not np.any(masks[0] & masks[1]) and masks[2].any():
                    result.append((patch, reach, half))
    return result


def _orders(bands):
    for major, reverse_lanes, reverse_bands in itertools.product((0, 1), repeat=3):
        lanes = range(2, -1, -1) if reverse_lanes else range(3)
        rows = range(bands - 1, -1, -1) if reverse_bands else range(bands)
        yield ([c * bands + r for c in lanes for r in rows] if major == 0 else
               [c * bands + r for r in rows for c in lanes])


def _last_indices(program, order):
    result = [-1, -1, -1]
    for index in order:
        result[program[index]] = index
    return tuple(result)


def _good_program(program, bands, all_orders):
    if any(program.count(role) < 2 for role in range(3)):
        return False
    if any(len(set(program[c * bands:(c + 1) * bands])) < 2 for c in range(3)):
        return False
    if len(set(program[bands - 1::bands])) == 3:
        return False  # Reading only the bottom band must miss a role.
    last = _last_indices(program, range(len(program)))
    if len({i // bands for i in last}) < 2:
        return False  # Earlier lanes must contribute to the completed seed.
    alternatives = list(_orders(bands))[1:] if all_orders else [
        [c * bands + r for r in range(bands) for c in range(3)]]
    return all(_last_indices(program, order) != last for order in alternatives)


def _program(rng, bands, all_orders=False):
    # Two occurrences per role, plus variable repetitions, then shuffle.
    for _ in range(256):
        program = [0, 0, 1, 1, 2, 2]
        program += [int(v) for v in rng.integers(0, 3, 3 * bands - 6)]
        rng.shuffle(program)
        if _good_program(program, bands, all_orders):
            return tuple(program)
    # Finite constructive completion: these schedules satisfy all needed tests.
    bases = {2: (0, 1, 0, 2, 1, 2),
             3: (1, 2, 1, 1, 0, 1, 0, 2, 2),
             4: (2, 0, 0, 0, 0, 1, 1, 2, 1, 0, 1, 2),
             5: (0, 0, 1, 2, 0, 2, 0, 2, 1, 1, 1, 1, 0, 0, 1)}
    permutation = [int(v) for v in rng.permutation(3)]
    program = tuple(permutation[k] for k in bases[bands])
    assert _good_program(program, bands, all_orders)
    return program


def _render(glyphs, masks, legend_order, program, colors, structural):
    bg, divider, ink = structural
    n, h, w = masks[0].shape[0], *glyphs[0].shape
    bands = len(program) // 3
    first = h + 1
    second = first + n + 1
    height = second + 1 + bands * h + bands - 1
    a = np.full((height, 3 * n + 2), bg, dtype=np.int8)
    a[first, :] = divider
    a[second, :] = divider
    offset = (n - w) // 2
    for c, role in enumerate(legend_order):
        left = c * (n + 1)
        a[1:first, left + offset:left + offset + w][glyphs[role]] = ink
        a[first + 1:second, left:left + n][masks[role]] = ink
    for index, (role, color) in enumerate(zip(program, colors)):
        lane, band = divmod(index, bands)
        top, left = second + 1 + band * (h + 1), lane * (n + 1) + offset
        a[top:top + h, left:left + w][glyphs[role]] = color
    return a


def _extract_glyph(block, bg):
    occupancy = block != bg
    columns = np.flatnonzero(occupancy.any(axis=0))
    _need(len(columns) >= 2, "missing glyph")
    left, right = int(columns[0]), int(columns[-1]) + 1
    mask = occupancy[:, left:right]
    _need(2 <= mask.shape[1] <= 4 and left == (block.shape[1] - mask.shape[1]) // 2,
          "glyph must be centered in its tile")
    _need(mask.any(axis=0).all() and mask.any(axis=1).all()
          and 2 <= int(mask.sum()) < mask.size and _connected(mask), "invalid glyph")
    ink = np.unique(block[occupancy])
    _need(len(ink) == 1, "glyph must be monochrome")
    return mask, int(ink[0])


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _need(np.all(a[0] == bg), "top row must show the background")
    n, rem = divmod(a.shape[1] - 2, 3)
    _need(rem == 0 and n in _SIDES, "three equal stencil tiles required")
    separators = [r for r in range(1, a.shape[0])
                  if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    _need(len(separators) == 2, "two horizontal dividers required")
    first, second = separators
    h = first - 1
    divider = int(a[first, 0])
    _need(h in (2, 3) and second == first + n + 1
          and np.all(a[second] == divider), "invalid legend dimensions")
    for c in (n, 2 * n + 1):
        _need(np.all(np.delete(a[:, c], separators) == bg), "nonblank tile gap")
    neutral = set(int(v) for v in np.unique(a[1:second])) - {bg, divider}
    _need(len(neutral) == 1, "legend needs one neutral ink color")
    ink = next(iter(neutral))
    records = []
    for c in range(3):
        left = c * (n + 1)
        glyph, color = _extract_glyph(a[1:first, left:left + n], bg)
        _need(color == ink, "legend glyph has wrong ink")
        block = a[first + 1:second, left:left + n]
        _need(set(int(v) for v in np.unique(block)) <= {bg, ink}, "invalid stencil ink")
        records.append((glyph, block == ink))
    _need(len({_key(g) for g, _ in records}) == 3
          and len({g.shape for g, _ in records}) == 1, "ambiguous glyph vocabulary")
    _need(np.all(sum(m.astype(int) for _, m in records) == 1),
          "stencils must partition a square")
    corners = [i for i, (_, m) in enumerate(records) if m[0, 0]]
    crosses = [i for i, (_, m) in enumerate(records) if m[n // 2, n // 2]]
    _need(len(corners) == len(crosses) == 1 and corners[0] != crosses[0],
          "corner and cross stencils must differ")
    role_to_slot = [corners[0], crosses[0], next(
        i for i in range(3) if i not in (corners[0], crosses[0]))]
    masks = tuple(records[i][1] for i in role_to_slot)
    matches = [p for p in _profiles(n) if all(np.array_equal(x, y)
               for x, y in zip(masks, _stencils(n, p)))]
    _need(len(matches) == 1, "stencils do not define a corner/cross/filler seed")
    glyphs = tuple(records[i][0] for i in role_to_slot)
    mapping = {_key(g): role for role, g in enumerate(glyphs)}
    remaining = a.shape[0] - second - 1
    bands, rem = divmod(remaining + 1, h + 1)
    _need(bands >= 1 and rem == 0, "incomplete program band")
    program, colors = [], []
    for lane in range(3):
        left = lane * (n + 1)
        for band in range(bands):
            top = second + 1 + band * (h + 1)
            g, color = _extract_glyph(a[top:top + h, left:left + n], bg)
            _need(_key(g) in mapping and color not in (bg, divider, ink),
                  "unknown or structural-colored instruction")
            if band + 1 < bands:
                _need(np.all(a[top + h, left:left + n] == bg), "nonblank band gap")
            program.append(mapping[_key(g)])
            colors.append(color)
    _need(set(program) == {0, 1, 2}, "program omits a seed role")
    final = [colors[i] for i in _last_indices(program, range(len(program)))]
    _need(len(set(final)) == 3, "completed seed must have three distinct colors")
    return {"n": n, "bands": bands, "glyphs": glyphs, "masks": masks,
            "profile": matches[0], "program": tuple(program), "colors": tuple(colors),
            "roles": tuple(final), "structural": (bg, divider, ink),
            "legend_order": tuple(role_to_slot.index(i) for i in range(3))}


def _paint(masks, program, colors):
    seed = np.full(masks[0].shape, -1, dtype=np.int8)
    for role, color in zip(program, colors):
        seed[masks[role]] = color
    _need(np.all(seed >= 0), "incomplete seed")
    return seed


def _grow(seed, roles):
    n = seed.shape[0]
    result = np.empty((2 * n, 2 * n), dtype=np.int8)
    result[:n, :n] = seed
    successor = dict(zip(roles, roles[1:] + roles[:1]))
    state = int(seed[-1, -1])
    for edge in range(n, 2 * n):
        result[:edge, edge] = state
        result[edge, :edge] = state
        state = successor[state]
        result[edge, edge] = state
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible legend, execute the program, and grow its seed."""
    info = _decode(input_grid)
    seed = _paint(info["masks"], info["program"], info["colors"])
    return _grow(seed, info["roles"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    _need(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    glyphs = _glyphs(rng)
    h = glyphs[0].shape[0]
    train_sides = ((5, 7), (7, 9), (5, 9))[int(rng.integers(3))]
    test_side = next(n for n in _SIDES if n not in train_sides)
    first_order = tuple(int(v) for v in rng.permutation(3))
    first_roles = None
    used_structural = set()
    result = {"train": [], "test": []}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            n = (train_sides[index] if index < 2 else int(rng.choice(train_sides))) \
                if split == "train" else test_side
            if split == "train":
                bands = 2 if index == 0 else 3 if index == 1 else int(rng.integers(2, 4))
            else:
                bands = int(rng.integers(4, 6)) if h == 2 else 4
            profiles = _profiles(n)
            if split == "train" and index == 0:
                profile = (1, 1, 0)
            elif split == "train" and index == 1:
                rich = []
                for p in profiles:
                    populations = [int(m.sum()) for m in _stencils(n, p)]
                    if (p[0] >= 2 and p[1] >= 2 and p[2] >= 1
                            and not populations[0] < populations[1] < populations[2]):
                        rich.append(p)
                profile = rich[int(rng.integers(len(rich)))]
            else:
                profile = profiles[int(rng.integers(len(profiles)))]
            masks = _stencils(n, profile)
            excluded = set(first_roles) if split == "train" and index == 1 else set()
            structural_pool = [p for p in itertools.permutations(range(10), 3)
                               if p not in used_structural and not (set(p) & excluded)]
            structural = structural_pool[int(rng.integers(len(structural_pool)))]
            used_structural.add(structural)  # Makes every input provably distinct.
            available = [c for c in range(10) if c not in structural]
            if split == "train" and index == 1:
                roles = first_roles[1:] + first_roles[:1]
            else:
                roles = tuple(int(v) for v in rng.choice(available, 3, replace=False))
            if split == "train" and index == 0:
                first_roles = roles
            if split == "train" and index < 2:
                legend_order = first_order if index == 0 else first_order[1:] + first_order[:1]
            else:
                legend_order = tuple(int(v) for v in rng.permutation(3))
            program = _program(rng, bands, all_orders=(split == "train" and index == 1))
            last = _last_indices(program, range(len(program)))
            colors = []
            for i, role in enumerate(program):
                choices = [c for c in available if c != roles[role]]
                colors.append(roles[role] if i == last[role] else int(rng.choice(choices)))
            # Each role's first appearance is another role's final color.
            for role in range(3):
                colors[program.index(role)] = roles[(role + 1) % 3]
            inp = _render(glyphs, masks, legend_order, program, colors, structural)
            square = _paint(masks, program, colors)
            result[split].append({"input": inp, "output": _grow(square, roles)})
    validate_generated_puzzle(result)
    return result


def _replayed_roles(info, order):
    result = [None, None, None]
    for i in order:
        result[info["program"][i]] = info["colors"][i]
    return tuple(result)


def validate_generated_puzzle(puzzle):
    """Check schema, visible semantics, every output cell, and demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos = {"train": [], "test": []}
    seen = set()
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            n, roles = info["n"], info["roles"]
            assert out.shape == (2 * n, 2 * n), "must double the decoded side"
            # Direct cell formula checks iteration, corner/arm distinction,
            # all three transitions, and exact upper-left seed preservation.
            for r in range(2 * n):
                for c in range(2 * n):
                    if r < n and c < n:
                        role = next(k for k, m in enumerate(info["masks"]) if m[r, c])
                        expected = roles[role]
                    elif r == c:
                        expected = roles[(r - n + 1) % 3]
                    else:
                        expected = roles[(max(r, c) - n) % 3]
                    assert int(out[r, c]) == expected, "seed or L-state violation"
            assert set(int(v) for v in np.unique(out)) == set(roles)
            program, colors, bands = info["program"], info["colors"], info["bands"]
            assert _good_program(program, bands, False), "shortcut-prone program"
            for role in range(3):
                final = roles[role]
                last = max(i for i, k in enumerate(program) if k == role)
                assert all(colors[i] != final for i, k in enumerate(program)
                           if k == role and i != last), "overwrite not exposed"
            row_order = [lane * bands + band for band in range(bands) for lane in range(3)]
            assert _replayed_roles(info, row_order) != roles, "row-major shortcut"
            # B ablation: making the first arms equal to their intersection fails.
            assert out[0, n] != out[n, n]
            assert out[0, n] != out[0, n + 1], "state must propagate recursively"
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    vocabularies = {tuple(_key(g) for g in info["glyphs"]) for info in all_infos}
    assert len(vocabularies) == 1, "glyph-role dictionary must be puzzle-wide"
    train = infos["train"]
    assert len({i["n"] for i in train}) == 2, "two training sizes required"
    assert min(i["bands"] for i in infos["test"]) > max(i["bands"] for i in train)
    assert {i["n"] for i in infos["test"]}.isdisjoint({i["n"] for i in train})
    assert train[0]["profile"] == (1, 1, 0)
    assert all(v >= t for v, t in zip(train[1]["profile"], (2, 2, 1)))
    populations = [int(m.sum()) for m in train[1]["masks"]]
    assert not populations[0] < populations[1] < populations[2], "frequency-rank shortcut"
    assert train[0]["legend_order"] != train[1]["legend_order"]
    assert train[1]["roles"] == train[0]["roles"][1:] + train[0]["roles"][:1]
    # All seven alternate raster traversals fail the second demonstration.
    assert all(_replayed_roles(train[1], o) != train[1]["roles"]
               for o in list(_orders(train[1]["bands"]))[1:])


def puzzle_features(puzzle):
    infos = [_decode(p["input"]) for p in puzzle["train"] + puzzle["test"]]
    return {"glyph_size": list(infos[0]["glyphs"][0].shape),
            "sides": sorted({i["n"] for i in infos}),
            "profiles": sorted({(i["n"], *i["profile"]) for i in infos}),
            "bands": sorted({i["bands"] for i in infos}),
            "legend_orders": sorted({i["legend_order"] for i in infos}),
            "output_roles": sorted({i["roles"] for i in infos}),
            "input_shapes": sorted({tuple(np.asarray(p["input"]).shape)
                                     for p in puzzle["train"] + puzzle["test"]})}


def _input_only_checks():
    """Exercise valid inputs outside the sampled training/test schedule."""
    cases = 0
    roles, structural = (2, 5, 7), (0, 8, 9)
    for n in _SIDES:
        for profile in _profiles(n):
            # Cover every compatible seed geometry, cycling glyph dimensions
            # and using the largest program that fits this particular input.
            h, w = 2 + cases % 2, 2 + (cases // 2) % 3
            glyphs = []
            for r, c in ((0, 0), (0, w - 1), (h - 1, 0)):
                glyph = np.ones((h, w), dtype=bool)
                glyph[r, c] = False
                glyphs.append(glyph)
            bands = (30 - n - h - 2) // (h + 1)
            program = tuple(range(3)) * bands
            last = _last_indices(program, range(len(program)))
            colors = tuple(roles[role] if i == last[role] else roles[(role + 1) % 3]
                           for i, role in enumerate(program))
            masks = _stencils(n, profile)
            inp = _render(glyphs, masks, (2, 0, 1), program, colors, structural)
            out = solve(inp)
            expected_seed = np.zeros((n, n), dtype=np.int8)
            for mask, color in zip(masks, roles):
                expected_seed[mask] = color
            assert np.array_equal(out[:n, :n], expected_seed)
            for edge in range(n, 2 * n):
                assert np.all(out[:edge, edge] == roles[(edge - n) % 3])
                assert np.all(out[edge, :edge] == roles[(edge - n) % 3])
                assert out[edge, edge] == roles[(edge - n + 1) % 3]
            # One instruction per role is also a valid input-only program,
            # although the generator supplies repetitions to teach overwriting.
            simple = _render(glyphs, masks, (1, 2, 0), (0, 1, 2), roles, structural)
            assert np.array_equal(solve(simple), out)
            cases += 1

    pair = generate_puzzle(0, 2, 1)["train"][1]
    inp, out = pair["input"], pair["output"]
    info = _decode(inp)
    h, n = info["glyphs"][0].shape[0], info["n"]
    # Neutral legend cards may move without altering glyph meanings.
    moved = inp.copy()
    for destination, source in enumerate((2, 0, 1)):
        left, old_left = destination * (n + 1), source * (n + 1)
        moved[1:h + n + 2, left:left + n] = inp[1:h + n + 2, old_left:old_left + n]
    assert np.array_equal(solve(moved), out)
    # Reassign just the corner and cross glyph headers, leaving stencils and
    # program in place: A now decodes another seed, which must also change B.
    swapped = inp.copy()
    first, second = (info["legend_order"].index(role) for role in (0, 1))
    x, y = first * (n + 1), second * (n + 1)
    swapped[1:h + 1, x:x + n] = inp[1:h + 1, y:y + n]
    swapped[1:h + 1, y:y + n] = inp[1:h + 1, x:x + n]
    alternate = solve(swapped)
    old, new = _decode(inp), _decode(swapped)
    assert new["roles"] == (old["roles"][1], old["roles"][0], old["roles"][2])
    assert not np.array_equal(out[:n, :n], alternate[:n, :n])
    assert not np.array_equal(out[:, n:], alternate[:, n:])
    # Global color renaming must commute with the complete transformation.
    permutation = np.array((7, 3, 0, 8, 2, 9, 4, 1, 6, 5), dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[out])
    return {"all_motif_profiles": cases, "programs_per_profile": 2,
            "legend_card_reordering": True, "glyph_dependency_in_growth": True,
            "palette_equivariance": True}


def self_test():
    """Bounded, in-memory checks of the 42 specified seeds and count contract."""
    sizes, glyph_sizes, profiles, shapes, orders, vocabulary, signatures = (
        set() for _ in range(7))
    test_regimes, bands, role_colors = set(), set(), [set(), set(), set()]
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        key = tuple(_key(np.asarray(p[k])) for s in ("train", "test")
                    for p in puzzle[s] for k in ("input", "output"))
        assert key == tuple(_key(np.asarray(p[k])) for s in ("train", "test")
                            for p in again[s] for k in ("input", "output"))
        signatures.add(key)
        train_sizes = {info["n"] for info in map(lambda p: _decode(p["input"]), puzzle["train"])}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                info = _decode(before)
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated input"
                sizes.add(info["n"])
                glyph_sizes.add(info["glyphs"][0].shape)
                vocabulary.add(tuple(_key(g) for g in info["glyphs"]))
                profiles.add((info["n"], *info["profile"]))
                shapes.add(pair["input"].shape)
                orders.add(info["legend_order"])
                bands.add(info["bands"])
                for role, color in enumerate(info["roles"]):
                    role_colors[role].add(color)
                if split == "test":
                    n = info["n"]
                    test_regimes.add("smaller" if n < min(train_sizes) else
                                     "larger" if n > max(train_sizes) else "intermediate")
                pair_count += 1
    assert len(signatures) == 42 and sizes == set(_SIDES)
    assert len(glyph_sizes) >= 5 and len(vocabulary) >= 35
    assert len(profiles) >= 20 and len(shapes) >= 10
    assert orders == set(itertools.permutations(range(3)))
    assert bands == {2, 3, 4, 5} and test_regimes == {"smaller", "larger", "intermediate"}
    assert all(cs == set(range(10)) for cs in role_colors)
    legal_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(0, nt, nq)
            legal_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A altered final command, a corrupt stencil, and B errors at different
    # layers must all be caught. Copies are bounded to one puzzle at a time.
    for damage in ("arm", "corner", "seed", "stencil", "command"):
        p = generate_puzzle(0, 2, 1)
        pair = p["test"][0]
        info = _decode(pair["input"])
        n, gh = info["n"], info["glyphs"][0].shape[0]
        if damage in ("arm", "corner", "seed"):
            r, c = {"arm": (0, n + 2), "corner": (n + 3, n + 3),
                    "seed": (n // 2, n // 2)}[damage]
            pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        elif damage == "stencil":
            pair["input"][gh + 2, 0] = info["structural"][1]
        else:
            role = info["program"][-1]
            gw = info["glyphs"][0].shape[1]
            top = gh + n + 3 + (info["bands"] - 1) * (gh + 1)
            left = 2 * (n + 1) + (n - gw) // 2
            block = pair["input"][top:top + gh, left:left + gw]
            block[info["glyphs"][role]] = info["roles"][(role + 1) % 3]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("semantic mutation accepted: " + damage)
    input_only_checks = _input_only_checks()
    return {"seeds": len(HOST_SEEDS), "default_pairs": pair_count,
            "legal_count_calls": legal_calls, "distinct_puzzles": len(signatures),
            "glyph_sizes": sorted(glyph_sizes), "distinct_vocabularies": len(vocabulary),
            "seed_sides": sorted(sizes), "size_motif_profiles": len(profiles),
            "input_shapes": sorted(shapes), "legend_orders": len(orders),
            "program_bands": sorted(bands), "test_size_regimes": sorted(test_regimes),
            "all_colors_in_every_output_role": True, "mutation_checks": 5,
            "input_only_checks": input_only_checks,
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
