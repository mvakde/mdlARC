"""Reflection-repaired seeds drive an alternating ABA-center recurrence.

No parent generator is imported. All random choices use the supplied Generator.
The module writes no artifacts; its bounded self-test processes one puzzle at a time.
"""
from collections import Counter
from hashlib import sha256

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _next(mask):
    following = np.zeros(len(mask), dtype=bool)
    following[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return following


def _generations(mask):
    result = []
    for _ in range(len(mask)):
        mask = _next(mask)
        if not mask.any():
            return result
        result.append(mask)
    raise ValueError("recurrence failed to contract")


def _orbits(height, width, row_sum, col_sum):
    seen = set()
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            rows = {r, row_sum - r} & set(range(height))
            cols = {c, col_sum - c} & set(range(width))
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            yield orbit


def _axis_candidates(field, erase, dimension):
    a = field if dimension == 0 else field.T
    result = []
    for total in range(1, 2 * len(a) - 2):
        lo, hi = max(0, total - len(a) + 1), (total + 1) // 2
        indices = np.arange(lo, hi)
        first, second = a[indices], a[total - indices]
        visible = (first != erase) & (second != erase)
        # Even a vacuously agreeing rival is retained: weak evidence must fail closed.
        if np.all(first[visible] == second[visible]):
            result.append(total)
    return result


def _axes(field, erase):
    rows = _axis_candidates(field, erase, 0)
    cols = _axis_candidates(field, erase, 1)
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError("survivors do not identify unique perpendicular folds")
    return rows[0], cols[0]


def _restore(field, erase, axes):
    restored = field.copy()
    for orbit in _orbits(*field.shape, *axes):
        witnesses = {int(field[p]) for p in orbit if field[p] != erase}
        if len(witnesses) != 1:
            raise ValueError("reflection orbit is empty or contradictory")
        color = witnesses.pop()
        for p in orbit:
            restored[p] = color
    return restored


def _decode(input_grid):
    inp = _grid(input_grid)
    if inp.shape[1] < 3 or inp.shape[0] < 5:
        raise ValueError("missing ink key or working field")
    erase, derived, background = int(inp[0, 0]), int(inp[0, 1]), int(inp[-1, 0])
    if (len({erase, derived, background}) != 3
            or not np.all(inp[0, 1:] == derived)
            or not np.all(inp[-1] == background)):
        raise ValueError("invalid ink key or background tail")
    nonblank = np.flatnonzero(np.any(inp[1:] != background, axis=1))
    if not len(nonblank):
        raise ValueError("missing reflection field")
    seed_row = int(nonblank[-1]) + 1
    field = inp[1:seed_row + 1]
    if np.any(field == derived) or not np.any(field == erase):
        raise ValueError("reserved ink in field, or no damage")
    visible_seed = set(map(int, field[-1])) - {erase, background}
    if len(visible_seed) != 1:
        raise ValueError("seed ink must have a visible witness")
    seed = visible_seed.pop()
    axes = _axes(field, erase)
    restored = _restore(field, erase, axes)
    if set(map(int, restored[-1])) != {background, seed}:
        raise ValueError("repaired seed must be binary")
    if len(np.unique(restored)) < 3:
        raise ValueError("reflection evidence must be multicolor")
    generations = _generations(restored[-1] == seed)
    if not generations or seed_row + 2 * len(generations) >= len(inp) - 1:
        raise ValueError("missing recurrence or insufficient blank tail")
    return inp, field, restored, (erase, background, seed, derived), seed_row, axes, generations


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the ink key, visible folds, seed row, and blank lower field."""
    inp, field, restored, roles, seed_row, _, generations = _decode(input_grid)
    erase, background, seed, derived = roles
    result = inp.copy()
    result[1:seed_row + 1] = restored
    for depth, mask in enumerate(generations, 1):
        result[seed_row + 2 * depth, mask] = derived if depth % 2 else seed
    return result


def _alternating(depth):
    a = np.zeros(2 * depth + 1, dtype=bool)
    a[::2] = True
    return a


def _seed_pattern(rng, kind, max_depth, target=None, allow_offset=True):
    """Make a symmetric core plus an optional one-sided, independently valued fringe."""
    for _ in range(128):
        depth = int(target if target is not None else rng.integers(1, max_depth + 1))
        if kind in ("chain", "mixed"):
            core = _alternating(depth)
            if kind == "mixed":
                solid = np.ones(int(rng.integers(3, 5)), dtype=bool)
                gap = np.zeros(int(rng.integers(2, 4)), dtype=bool)
                core = np.concatenate((solid, gap, core, gap, solid))
            else:
                pad = np.zeros(int(rng.integers(0, 4)), dtype=bool)
                core = np.concatenate((pad, core, pad))
        elif kind == "twins":
            part = _alternating(min(depth, 5))
            core = np.concatenate((part, np.zeros(int(rng.integers(2, 5)), bool), part[::-1]))
        elif kind == "multi":
            side = _alternating(int(rng.integers(1, min(depth, 3) + 1)))
            gap = np.zeros(int(rng.integers(2, 4)), bool)
            core = np.concatenate((side, gap, _alternating(depth), gap, side[::-1]))
        elif kind == "dense":
            # One or two symmetric one-cell holes in otherwise solid runs.
            length = int(rng.integers(7, 23))
            core = np.ones(length, bool)
            c = int(rng.integers(1, (length - 1) // 2 + 1))
            if abs(length - 1 - 2 * c) == 1:
                continue
            core[c] = core[length - 1 - c] = False
        elif kind == "random":
            length = int(rng.integers(9, 28))
            half = rng.random((length + 1) // 2) < rng.uniform(0.32, 0.72)
            core = np.concatenate((half, half[:length // 2][::-1]))
        else:
            raise ValueError("unknown seed construction")
        if not 7 <= len(core) <= 30:
            continue
        fringe = int(rng.integers(0, min(3, 30 - len(core)) + 1)) if allow_offset else 0
        extra = rng.random(fringe) < rng.uniform(0.2, 0.8)
        if rng.random() < 0.5:
            mask = np.concatenate((extra, core))
            col_sum = 2 * fringe + len(core) - 1
        else:
            mask = np.concatenate((core, extra))
            col_sum = len(core) - 1
        actual = len(_generations(mask))
        if not 1 <= actual <= max_depth or (target is not None and actual != target):
            continue
        centers = _next(mask)
        candidates = [c for c in range(len(mask)) if mask[c]
                      and 0 <= col_sum - c < len(mask) and col_sum - c != c
                      and ((c > 0 and centers[c - 1]) or (c + 1 < len(mask) and centers[c + 1]))]
        if candidates:
            return mask, col_sum, candidates
    raise RuntimeError("seed construction exhausted")


def _clean_field(rng, mask, col_sum, height, row_sum, roles):
    erase, background, seed, derived = roles
    choices = [c for c in range(10) if c not in roles]
    texture = list(map(int, rng.choice(choices, int(rng.integers(1, 5)), replace=False)))
    palette = [background, seed] + texture
    field = np.empty((height, len(mask)), dtype=np.int8)
    style = int(rng.integers(4))
    stripe = int(rng.integers(1, 4))
    nonseed = []
    for orbit in _orbits(height, len(mask), row_sum, col_sum):
        r, c = orbit[0]
        if any(y == height - 1 for y, x in orbit):
            color = seed if mask[c] else background
        else:
            nonseed.append(orbit)
            if style == 0 or rng.random() < 0.24:
                index = int(rng.integers(len(palette)))
            elif style == 1:
                index = (r + c // stripe) % len(palette)
            elif style == 2:
                index = (r // 2 + c // stripe) % len(palette)
            else:
                index = max(r, c // stripe) % len(palette)
            color = palette[index]
        for p in orbit:
            field[p] = color
    # A nonbinary payload witness is guaranteed even in compact fields.
    for p in nonseed[int(rng.integers(len(nonseed)))]:
        field[p] = texture[0]
    return field


def _raw_mask(rng, shape, topology):
    h, w = shape
    missing = np.zeros(shape, bool)
    if topology in ("scatter", "mixture"):
        missing |= rng.random(shape) < rng.uniform(0.04, 0.14)
    if topology in ("blocks", "mixture"):
        for _ in range(int(rng.integers(1, 5))):
            rh, cw = int(rng.integers(1, min(4, h) + 1)), int(rng.integers(1, min(6, w) + 1))
            r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - cw + 1))
            missing[r:r + rh, c:c + cw] = True
    if topology == "strip":
        if rng.random() < 0.5:
            missing[int(rng.integers(h)), :] = True
        else:
            missing[:, int(rng.integers(w))] = True
    if topology == "steps":
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for k in range(int(rng.integers(2, h + 1))):
            missing[(r + k) % h, c:min(w, c + 1 + k)] = True
    return missing


def _damage(rng, field, axes, roles, candidates, hard, topology):
    erase, background, seed, derived = roles
    h, w = field.shape
    row_sum, col_sum = axes
    c = int(rng.choice(candidates))
    donor, mirror = row_sum - (h - 1), col_sum - c
    forced = {(h - 1, c)}
    if hard and int(np.count_nonzero(field[-1] == seed)) <= 2:
        return None
    if hard:
        forced.update(((h - 1, mirror), (donor, c)))
    witness = (donor, mirror)
    for _ in range(16):
        missing = _raw_mask(rng, field.shape, topology)
        for p in forced:
            missing[p] = True
        missing[witness] = False
        for orbit in _orbits(h, w, *axes):
            if all(missing[p] for p in orbit):
                choices = [p for p in orbit if p not in forced]
                if not choices:
                    raise RuntimeError("forced mask erased a complete orbit")
                missing[choices[int(rng.integers(len(choices)))]] = False
        visible_seed = [int(x) for x in np.flatnonzero(field[-1] == seed)
                        if (h - 1, int(x)) not in forced]
        if not visible_seed:
            continue
        missing[-1, int(rng.choice(visible_seed))] = False
        inp = field.copy()
        inp[missing] = erase
        if (_axis_candidates(inp, erase, 0) == [row_sum]
                and _axis_candidates(inp, erase, 1) == [col_sum]
                and not np.array_equal(_next(inp[-1] == seed), _next(field[-1] == seed))):
            return inp
    return None


def _make_pair(rng, roles, kind, max_depth, target, hard, topologies):
    for _ in range(160):
        try:
            mask, col_sum, candidates = _seed_pattern(rng, kind, max_depth, target)
        except RuntimeError:
            continue
        generations = _generations(mask)
        depth = len(generations)
        height = int(rng.integers(6, min(12, 28 - 2 * depth) + 1))
        offset = int(rng.integers(0, min(3, height - 4) + 1))
        row_sum = height - 1 + offset
        axes = row_sum, col_sum
        field = _clean_field(rng, mask, col_sum, height, row_sum, roles)
        erase, background, seed, derived = roles
        if (_axis_candidates(field, erase, 0) != [row_sum]
                or _axis_candidates(field, erase, 1) != [col_sum]):
            continue
        damaged = _damage(rng, field, axes, roles, candidates, hard, str(rng.choice(topologies)))
        if damaged is None:
            continue
        tail = int(rng.integers(1, min(8, 29 - height - 2 * depth) + 1))
        inp = np.full((height + 2 * depth + 1 + tail, len(mask)), background, np.int8)
        inp[0] = derived
        inp[0, 0] = erase
        inp[1:height + 1] = damaged
        out = inp.copy()
        out[1:height + 1] = field
        for k, generation in enumerate(generations, 1):
            out[height + 2 * k, generation] = derived if k % 2 else seed
        return {"input": inp, "output": out}
    raise RuntimeError("could not build an unambiguous damaged field")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    shift = str(rng.choice(("depth", "mask")))
    order = rng.permutation(nt)
    demonstration, contrast = int(order[0]), int(order[1])
    shifted_test = int(rng.integers(ns))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            target = None
            hard = bool(rng.random() < (0.8 if split == "test" else 0.45))
            kinds = ("chain", "dense", "twins", "multi", "mixed", "random")
            kind = str(rng.choice(kinds))
            max_depth = 4 if split == "train" or shift == "mask" else 8
            topologies = ("scatter", "blocks", "strip", "steps", "mixture")
            if split == "train" and index == demonstration:
                kind, target, hard = "mixed", int(rng.integers(2, 4)), True
            elif split == "train" and index == contrast:
                kind, target = "dense", 1
            elif split == "test" and shift == "depth" and index == shifted_test:
                kind, target, max_depth, hard = "chain", int(rng.integers(5, 10)), 9, True
            if shift == "mask":
                topologies = ("scatter", "blocks", "strip") if split == "train" else ("steps", "mixture")
                if split == "test":
                    hard = True
            for _ in range(32):
                pair = _make_pair(rng, roles, kind, max_depth, target, hard, topologies)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input attempts exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, field, restored, roles, seed_row, axes, generations = _decode(pair["input"])
    erase, background, seed, derived = roles
    h, w = field.shape
    a, b = axes
    missing = field == erase
    diagonal = 0
    for c in np.flatnonzero(missing[-1] & (restored[-1] == seed)):
        rr, cc = a - h + 1, b - int(c)
        if (0 <= cc < w and cc != c and missing[-1, cc] and missing[rr, c]
                and field[rr, cc] != erase):
            diagonal += 1
    seed_mask = restored[-1] == seed
    solid = bool(np.any(seed_mask[:-2] & seed_mask[1:-1] & seed_mask[2:]))
    return {"shape": inp.shape, "field_shape": field.shape, "axes": axes,
            "offset": (a - h + 1, b - w + 1), "parity": (a % 2, b % 2),
            "depth": len(generations), "first_count": int(generations[0].sum()),
            "diagonal_seed": diagonal, "solid": solid, "erased": int(missing.sum()),
            "tail": len(inp) - 1 - seed_row - 2 * len(generations),
            "seed_signature": (w, seed_mask.tobytes()),
            "palette_size": len(np.unique(restored))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, role_sets, demos = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            out = _grid(pair["output"])
            inp, field, clean, roles, seed_row, axes, generations = _decode(pair["input"])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            role_sets.add(roles)
            erase, background, seed, derived = roles
            assert np.array_equal(inp[0], out[0]), "ink key changed"
            assert np.array_equal(out[1:seed_row + 1], clean), "incorrect reflection repair"
            assert np.array_equal(field != clean, field == erase), "visible payload changed"
            # Replay the local law by explicit scalar column tests, including AAA rejection.
            active = set(map(int, np.flatnonzero(clean[-1] == seed)))
            expected_rows = {}
            k = 0
            while active:
                following = {c for c in range(1, inp.shape[1] - 1)
                             if c not in active and c - 1 in active and c + 1 in active}
                if not following:
                    break
                k += 1
                expected_rows[seed_row + 2 * k] = (following, derived if k % 2 else seed)
                active = following
            for r in range(seed_row + 1, len(inp)):
                expected = np.full(inp.shape[1], background, np.int8)
                if r in expected_rows:
                    columns, ink = expected_rows[r]
                    expected[list(columns)] = ink
                assert np.array_equal(out[r], expected), "wrong recurrence, spacing, parity, or stop"
            assert np.array_equal(solve(inp), out), "input-only solve disagreement"
            # A missing active endpoint must change the recurrence, not just the repaired panel.
            assert not np.array_equal(_next(field[-1] == seed), generations[0]), "reflection is causally idle"
            assert np.any(out[seed_row + 1:] != background), "stencil is causally idle"
            if split == "train":
                demos.append(_features(pair))
    assert len(role_sets) == 1, "palette roles vary within a generated puzzle"
    assert any(f["depth"] >= 2 and f["solid"] and f["diagonal_seed"] for f in demos)
    assert any(f["depth"] == 1 and f["solid"] for f in demos)


def self_test():
    """42 seeds, 210 pairs, bounded memory; no files, logs, or execution traces."""
    shapes, seeds_seen, seed_masks, offsets, parities, palettes, roles_seen = set(), set(), set(), set(), set(), set(), set()
    depths, counts = set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                seed_masks.add(f["seed_signature"])
                offsets.add(tuple(int(np.sign(x)) for x in f["offset"]))
                parities.add(f["parity"])
                palettes.add(f["palette_size"])
                depths.add(f["depth"])
                counts["diagonal"] += f["diagonal_seed"] > 0
                counts["one_center"] += f["first_count"] == 1
                counts["long_tail"] += f["tail"] >= 5
                counts["pairs"] += 1
                roles_seen.add(_decode(pair["input"])[3])
                for role in ("input", "output"):
                    digest.update(repr(np.asarray(pair[role]).shape).encode())
                    digest.update(np.asarray(pair[role], np.int8).tobytes())
        seeds_seen.add(digest.digest())
    assert len(seeds_seen) == 42 and counts["pairs"] == 210
    assert len(shapes) >= 45 and len(seed_masks) >= 100 and len(roles_seen) >= 35
    assert len(offsets) >= 5 and len(parities) == 4 and len(palettes) >= 3
    assert min(depths) == 1 and max(depths) >= 8
    assert counts["diagonal"] >= 42 and counts["one_center"] >= 1 and counts["long_tail"] >= 20
    first, repeat = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(p[role], q[role]) for s in first
               for p, q in zip(first[s], repeat[s]) for role in ("input", "output"))
    # Every legal count combination; only the current puzzle is retained.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutations must fail validation even if they preserve all generic ARC bounds.
    for kind in ("repair", "generation", "spacer", "visible"):
        bad = {s: [{k: np.asarray(a).copy() for k, a in p.items()} for p in first[s]] for s in first}
        pair = bad["test"][0]
        inp, field, clean, roles, seed_row, axes, generations = _decode(pair["input"])
        if kind == "repair":
            r, c = map(int, np.argwhere(field == roles[0])[0])
            point = r + 1, c
        elif kind == "visible":
            r, c = map(int, np.argwhere(field != roles[0])[0])
            point = r + 1, c
        elif kind == "generation":
            point = seed_row + 2, int(np.flatnonzero(generations[0])[0])
        else:
            point = seed_row + 1, 0
        pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output accepted")
