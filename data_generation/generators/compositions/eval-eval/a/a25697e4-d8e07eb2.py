"""Color-matched locks whose keys must collectively complete one bank line.

Only NumPy and the Python standard library are used. All inference is input-only.
No parent module, saved examples, RNG state, or generation metadata is consulted.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 train, 1..4 test, with train > test")
    return int(a), int(b)


def _orient(a, op):
    return np.rot90(np.fliplr(a) if op >= 4 else a, op % 4).copy()


def _components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(frozenset(group))
    return groups


def _points(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _trim(a, bg):
    r0, r1, c0, c1 = _bbox(_points(a != bg))
    return a[r0:r1 + 1, c0:c1 + 1]


def _center(a, bg, size):
    crop = _trim(a, bg)
    result = np.full((size, size), bg, dtype=np.int8)
    r, c = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    result[r:r + crop.shape[0], c:c + crop.shape[1]] = crop
    return result


def _shape_signature(mask):
    signatures = []
    for op in range(8):
        crop = _trim(_orient(mask, op), False)
        signatures.append((crop.shape, crop.tobytes()))
    return min(signatures)


def _lock_info(tile, bg):
    inks = [int(c) for c in np.unique(tile) if c != bg]
    if len(inks) != 1:
        raise ValueError("a lock needs one ink")
    occupied = _points(tile != bg)
    if len(_components(occupied)) != 2:
        raise ValueError("a lock needs exactly two connected fragments")
    r0, r1, c0, c1 = _bbox(occupied)
    rect = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    slot = rect - occupied
    if not slot or len(_components(slot)) != 1:
        raise ValueError("a socket must be connected and nonempty")
    return {"tile": tile, "ink": inks[0], "rect": rect, "slot": slot,
            "bbox": (r0, r1, c0, c1)}


def _key_options(tile, bg):
    colors = tuple(int(c) for c in np.unique(tile) if c != bg)
    if len(colors) != 2:
        raise ValueError("a key needs two inks")
    if any(len(_components(_points(tile == c))) != 1 for c in colors):
        raise ValueError("each key region must be connected")
    if len(_components(_points(tile != bg))) != 1:
        raise ValueError("key regions must touch")
    options = []
    for op in range(8):
        a = _orient(tile, op)
        regions = tuple(_points(a == c) for c in colors)
        options.append((op, colors, regions))
    return options


def _fits(options, lock, size, bg, policy=(False, True), require_ink=True):
    """Distinct final colored chamber placements, with parity witnesses."""
    sr, _, sc, _ = _bbox(lock["slot"])
    results = {}
    for op, colors, regions in options:
        new_colors = colors[::-1] if policy[op >= 4] else colors
        for plug_index in (0, 1):
            if require_ink and new_colors[plug_index] != lock["ink"]:
                continue
            pr, _, pc, _ = _bbox(regions[plug_index])
            dr, dc = sr - pr, sc - pc
            plug = frozenset((r + dr, c + dc) for r, c in regions[plug_index])
            if plug != lock["slot"]:
                continue
            handle = frozenset((r + dr, c + dc) for r, c in regions[1 - plug_index])
            if handle & lock["rect"] or any(not (0 <= r < size and 0 <= c < size) for r, c in handle):
                continue
            if any(lock["tile"][p] != bg for p in handle):
                continue
            out = lock["tile"].copy()
            for i, region in enumerate((plug, handle) if plug_index == 0 else (handle, plug)):
                for p in region:
                    out[p] = new_colors[i]
            signature = out.tobytes()
            if signature not in results:
                results[signature] = {"tile": out, "parities": set(), "plug_indices": set(),
                                      "handle_size": len(handle), "plug_size": len(plug)}
            results[signature]["parities"].add(int(op >= 4))
            results[signature]["plug_indices"].add(plug_index)
    return list(results.values())


def _grid(a):
    a = np.asarray(a)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _slice(r, c, s):
    return (slice(1 + r * (s + 1), 1 + r * (s + 1) + s),
            slice(1 + c * (s + 1), 1 + c * (s + 1) + s))


def _parse_canonical(g):
    wall = int(g[0, 0])
    if not (np.all(g[[0, -1]] == wall) and np.all(g[:, [0, -1]] == wall)):
        raise ValueError("missing lattice border")
    rows = np.flatnonzero(np.all(g == wall, axis=1)).tolist()
    cols = np.flatnonzero(np.all(g == wall, axis=0)).tolist()
    n = len(cols) - 1
    if n not in (2, 3) or len(rows) != n + 2:
        raise ValueError("need a square bank and one rack row")
    pitch = cols[1] - cols[0]
    s = pitch - 1
    if s < 3 or rows != [i * pitch for i in range(n + 2)] or cols != [i * pitch for i in range(n + 1)]:
        raise ValueError("irregular chamber lattice")
    interior = np.concatenate([g[_slice(r, c, s)].ravel() for r in range(n + 1) for c in range(n)])
    if np.any(interior == wall):
        raise ValueError("wall ink inside a chamber")
    colors, counts = np.unique(interior, return_counts=True)
    order = np.argsort(counts)
    if len(colors) < 3 or counts[order[-1]] == counts[order[-2]]:
        raise ValueError("background must be uniquely dominant inside chambers")
    bg = int(colors[order[-1]])
    locks = [_lock_info(g[_slice(r, c, s)], bg) for r in range(n) for c in range(n)]
    keys = [g[_slice(n, c, s)].copy() for c in range(n)]
    options = [_key_options(k, bg) for k in keys]
    identities = [(_trim(k, bg).shape, _trim(k, bg).tobytes()) for k in keys]
    if len(set(identities)) != n:
        raise ValueError("rack keys must be distinct")
    return {"grid": g, "n": n, "s": s, "bg": bg, "wall": wall,
            "locks": locks, "keys": keys, "options": options}


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(a, rotation).copy())
        except ValueError:
            continue
        info["rotation"] = rotation
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("bank and rack must be uniquely identifiable")
    return candidates[0]


def _compatibility(info, policy=(False, True), require_ink=True):
    return [[_fits(opts, lock, info["s"], info["bg"], policy, require_ink)
             for opts in info["options"]] for lock in info["locks"]]


def _line_solutions(info, fits):
    """Bounded search: two different answers suffice to prove ambiguity."""
    n = info["n"]
    lines = [tuple(r * n + c for c in range(n)) for r in range(n)]
    lines += [tuple(r * n + c for r in range(n)) for c in range(n)]
    found = {}

    def visit(line, depth, used, moves):
        if len(found) >= 2:
            return
        if depth == n:
            signature = tuple(sorted((p, fit["tile"].tobytes()) for p, _, fit in moves))
            found.setdefault(signature, tuple(moves))
            return
        pos = line[depth]
        for key in range(n):
            if key in used:
                continue
            for fit in fits[pos][key]:
                visit(line, depth + 1, used | {key}, moves + [(pos, key, fit)])
                if len(found) >= 2:
                    return

    for line in lines:
        visit(line, 0, set(), [])
        if len(found) >= 2:
            break
    return found


def _solve_details(input_grid):
    info = _analyze(input_grid)
    fits = _compatibility(info)
    solutions = _line_solutions(info, fits)
    if len(solutions) > 1:
        raise ValueError("competing complete colored line assemblies")
    out = info["grid"].copy()
    n, s = info["n"], info["s"]
    for c in range(n):
        out[_slice(n, c, s)] = info["bg"]
    moves = next(iter(solutions.values())) if solutions else ()
    for pos, _, fit in moves:
        out[_slice(*divmod(pos, n), s)] = fit["tile"]
    return np.rot90(out, -info["rotation"]).copy(), info, fits, solutions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique output from pixels alone; reject ambiguous inputs."""
    return _solve_details(input_grid)[0]


def _sample_geometry(rng, s, max_span, large=False):
    """Sample a separator and grow a whole exterior region, in one chamber."""
    limit = min(s - 1, max_span)
    for _ in range(160):
        h, w = map(int, rng.integers(3, limit + 1, size=2))
        if large:
            w = limit
        y = int(rng.integers(1, h - 1))
        slot = {(y, 0)}
        for c in range(1, w):
            ny = int(np.clip(y + rng.integers(-1, 2), 0, h - 1))
            slot.update((r, c) for r in range(min(y, ny), max(y, ny) + 1))
            y = ny
        rect = {(r, c) for r in range(h) for c in range(w)}
        for _ in range(int(rng.integers(0, 3))):
            frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in _DIRS} & rect - slot)
            if frontier:
                slot.add(frontier[int(rng.integers(len(frontier)))])
        target = rect - slot
        parts = _components(target)
        if len(parts) != 2 or min(map(len, parts)) < 2 or _bbox(target) != (0, h - 1, 0, w - 1):
            continue
        r0, c0 = int(rng.integers(s - h + 1)), int(rng.integers(s - w + 1))
        slot = {(r + r0, c + c0) for r, c in slot}
        rect = {(r + r0, c + c0) for r, c in rect}
        target = {(r + r0, c + c0) for r, c in target}
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in _DIRS
                         if 0 <= r + dr < s and 0 <= c + dc < s} - rect)
        if not starts:
            continue
        handle = {starts[int(rng.integers(len(starts)))]}
        size = int(rng.integers(1, min(11, s * s - len(rect) + 1)))
        style = int(rng.integers(3))
        tip = next(iter(handle))
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in _DIRS
                               if 0 <= r + dr < s and 0 <= c + dc < s} - rect - handle)
            if not frontier:
                break
            weights = []
            for r, c in frontier:
                degree = sum((r + dr, c + dc) in handle for dr, dc in _DIRS)
                weights.append(degree ** 2 if style == 0 else
                               (8 if abs(r - tip[0]) + abs(c - tip[1]) == 1 else .2)
                               if style == 1 else 1 / degree)
            weights = np.asarray(weights, dtype=float)
            tip = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
            handle.add(tip)
        labels = np.zeros((s, s), np.int8)
        mask = np.zeros((s, s), bool)
        for p in slot:
            labels[p] = 1
        for p in handle:
            labels[p] = 2
        for p in target:
            mask[p] = True
        # Either sorted key color can be the plug; do not infer it from size.
        if _shape_signature(labels == 1) == _shape_signature(labels == 2):
            continue
        return mask, labels
    raise ValueError("separator sampling budget exhausted")


def _motif(rng, s, max_span, bg, wall, large=False, forbidden=()):
    available = [c for c in range(10) if c not in (bg, wall)]
    for _ in range(100):
        target, labels = _sample_geometry(rng, s, max_span, large)
        signatures = {_shape_signature(labels == i) for i in (1, 2)}
        if signatures & set(forbidden):
            continue
        inks = tuple(map(int, rng.choice(available, 2, replace=False)))
        source_op = int(rng.integers(8))
        src_labels = _center(_orient(labels, source_op), 0, s)
        key = np.full((s, s), bg, np.int8)
        key[src_labels == 1], key[src_labels == 2] = inks
        opts = _key_options(key, bg)
        variants = {0: [], 1: []}
        for op in range(8):
            parity = int((source_op >= 4) != (op >= 4))
            color = inks[parity]
            tile = np.full((s, s), bg, np.int8)
            tile[_orient(target, op)] = color
            fits = _fits(opts, _lock_info(tile, bg), s, bg)
            bad = tile.copy()
            bad[bad == color] = inks[1 - parity]
            # This wrong-ink copy must really be incompatible; a symmetric
            # slot is allowed when its handle and walls remove the other fit.
            bad_fits = _fits(opts, _lock_info(bad, bg), s, bg)
            if len(fits) == 1 and fits[0]["parities"] == {parity} and not bad_fits:
                variants[parity].append((tile, fits[0], bad))
        if variants[0] and variants[1]:
            return {"key": key, "variants": variants, "shapes": signatures,
                    "target": target, "labels": labels}
    raise ValueError("unambiguous key construction budget exhausted")


def _inventory(rng, n, s, span, bg, wall, large=False):
    result, forbidden = [], set()
    for i in range(n):
        m = _motif(rng, s, span, bg, wall, large=large and i == 0, forbidden=forbidden)
        result.append(m)
        forbidden.update(m["shapes"])
    return result


def _variant(rng, motif, parity):
    choices = motif["variants"][parity]
    return choices[int(rng.integers(len(choices)))]


def _distractor(rng, s, span, bg, wall, inventory):
    forbidden = set().union(*(m["shapes"] for m in inventory))
    for _ in range(120):
        target, labels = _sample_geometry(rng, s, span)
        if _shape_signature(labels == 1) in forbidden:
            continue
        ink = int(rng.choice([c for c in range(10) if c not in (bg, wall)]))
        tile = np.full((s, s), bg, np.int8)
        tile[_orient(target, int(rng.integers(8)))] = ink
        return tile
    raise ValueError("distractor sampling budget exhausted")


def _build_pair(rng, inventory, s, span, bg, wall, axis, success, orientation, mixed_witness=False):
    n = len(inventory)
    # A partial bank line plus an off-line copy is a genuine nonwinner even
    # though every key still has a compatible destination.
    index = int(rng.integers(n))
    line = [index * n + c for c in range(n)] if axis == "row" else [r * n + index for r in range(n)]
    order = list(map(int, rng.permutation(n)))
    if not success:
        # Exactly one wrong-ink reflected socket counterfeits a full line
        # under the no-swap shortcut; all other keys preserve their colors.
        parities = [1] + [0] * (n - 1)
    elif mixed_witness:
        parities = [0, 1] + [int(rng.integers(2)) for _ in range(n - 2)]
    else:
        regime = int(rng.integers(3))
        parities = ([regime] * n if regime < 2 else
                    [0, 1] + [int(rng.integers(2)) for _ in range(n - 2)])
    rng.shuffle(parities)
    board, installed = {}, {}
    for pos, key, parity in zip(line, order, parities):
        tile, fit, bad = _variant(rng, inventory[key], parity)
        board[pos] = tile.copy()
        installed[pos] = (key, fit)
    off_line = [p for p in range(n * n) if p not in line]
    if success:
        # Duplicate the key at the line's intersection with this chamber's
        # perpendicular line. Thus a single extra copy cannot complete it.
        count = 1 if n == 2 else int(rng.integers(1, 4))
        rng.shuffle(off_line)
        for pos in off_line[:count]:
            r, c = divmod(pos, n)
            anchor = index * n + c if axis == "row" else r * n + index
            key = installed[anchor][0]
            board[pos] = _variant(rng, inventory[key], int(rng.integers(2)))[0].copy()
    else:
        broken = parities.index(1)
        pos, key = line[broken], order[broken]
        # A reflected geometric fit with the unswapped ink would counterfeit
        # this line. Move its genuine compatible lock off the line.
        tile, fit, bad = _variant(rng, inventory[key], 1)
        board[pos] = bad.copy()
        r, c = divmod(pos, n)
        alternatives = [j * n + c for j in range(n) if j != r] if axis == "row" else [r * n + j for j in range(n) if j != c]
        new_pos = int(rng.choice(alternatives))
        board[new_pos] = tile.copy()
    for pos in range(n * n):
        if pos not in board:
            board[pos] = _distractor(rng, s, span, bg, wall, inventory)
    shape = ((n + 1) * (s + 1) + 1, n * (s + 1) + 1)
    inp = np.full(shape, wall, np.int8)
    for pos, tile in board.items():
        inp[_slice(*divmod(pos, n), s)] = tile
    rack_order = list(map(int, rng.permutation(n)))
    for c, key in enumerate(rack_order):
        inp[_slice(n, c, s)] = inventory[key]["key"]
    out = inp.copy()
    for c in range(n):
        out[_slice(n, c, s)] = bg
    if success:
        for pos, (_, fit) in installed.items():
            out[_slice(*divmod(pos, n), s)] = fit["tile"]
    pair = {"input": _orient(inp, orientation), "output": _orient(out, orientation)}
    solved, info, fits, solutions = _solve_details(pair["input"])
    if not np.array_equal(solved, pair["output"]) or bool(solutions) != success:
        raise ValueError("constructed bank has an unintended line")
    if not all(any(fits[p][k] for p in range(n * n)) for k in range(n)):
        raise ValueError("each key must have some genuine compatible destination")
    if not success:
        # Both counterfactuals must actually produce a complete line; this is
        # measured from the input, not assumed from the sampled parity bit.
        if not _line_solutions(info, _compatibility(info, require_ink=False)):
            raise ValueError("missing geometric false-line lesson")
        if not _line_solutions(info, _compatibility(info, policy=(False, False))):
            raise ValueError("missing unswapped false-line lesson")
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n = int(rng.choice([2, 3]))
    palette = list(map(int, rng.permutation(10)))
    bg0, wall0 = palette[:2]
    varying_palette = bool(rng.random() < .3)
    orientation0 = int(rng.integers(4))
    first_axis = str(rng.choice(["row", "col"]))
    other_axis = "col" if first_axis == "row" else "row"
    lessons = [(True, first_axis), (False, first_axis)]
    if num_train >= 3:
        lessons.append((True, other_axis))
    while len(lessons) < num_train:
        lessons.append((bool(rng.integers(2)), str(rng.choice(["row", "col"]))))
    tests = [(bool(rng.integers(2)), str(rng.choice(["row", "col"])) if num_train >= 3 else first_axis)
             for _ in range(num_test)]
    puzzle, seen, shared = {"train": [], "test": []}, set(), None
    for i, (success, axis) in enumerate(lessons + tests):
        training = i < num_train
        s = 5 if training else 6 if n == 3 else int(rng.integers(6, 9))
        span = 4 if training else min(s - 1, 6)
        bg, wall = bg0, wall0
        if varying_palette and i >= 2:
            bg, wall = map(int, rng.choice(10, 2, replace=False))
        for attempt in range(48):
            try:
                # The first two demonstrations reuse exactly the same keys;
                # only the bank's constraints decide the opposite outcomes.
                inventory = shared if i == 1 else _inventory(rng, n, s, span, bg, wall, large=not training)
                orientation = orientation0 if i < 2 else int(rng.integers(4))
                pair = _build_pair(rng, inventory, s, span, bg, wall, axis, success, orientation,
                                   mixed_witness=i == 0)
                signature = _digest(pair["input"])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle["train" if training else "test"].append(pair)
                if i == 0:
                    shared = inventory
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("bounded construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _features(pair):
    out, info, fits, solutions = _solve_details(pair["input"])
    n = info["n"]
    moves = next(iter(solutions.values())) if solutions else ()
    candidate_counts = [sum(bool(fits[p][k]) for p in range(n * n)) for k in range(n)]
    no_ink = _line_solutions(info, _compatibility(info, require_ink=False))
    no_swap = _line_solutions(info, _compatibility(info, policy=(False, False)))
    spans = []
    for lock in info["locks"]:
        r0, r1, c0, c1 = _bbox(lock["slot"])
        spans.append(max(r1 - r0 + 1, c1 - c0 + 1))
    return {"info": info, "fits": fits, "solutions": solutions,
            "success": bool(solutions), "candidates": candidate_counts,
            "parities": set().union(*(fit["parities"] for _, _, fit in moves)) if moves else set(),
            "no_ink_changed": set(no_ink) != set(solutions),
            "no_swap_changed": set(no_swap) != set(solutions),
            "geometric_false_line": not solutions and bool(no_ink),
            "unswapped_false_line": not solutions and bool(no_swap),
            "max_span": max(spans), "prediction": out}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("exactly train/test keys required")
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _digest(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            f = _features(pair)
            assert np.array_equal(out, f["prediction"]), "output disagrees with input-only fitting"
            assert len(f["solutions"]) <= 1
            assert all(c >= 1 for c in f["candidates"]), "a key has no compatible lock"
            info, n, s = f["info"], f["info"]["n"], f["info"]["s"]
            canonical_out = np.rot90(out, info["rotation"])
            for c in range(n):
                assert np.all(canonical_out[_slice(n, c, s)] == info["bg"])
            assert np.array_equal(canonical_out[info["grid"] == info["wall"]],
                                  info["grid"][info["grid"] == info["wall"]])
            for pos, lock in enumerate(info["locks"]):
                block = canonical_out[_slice(*divmod(pos, n), s)]
                mask = lock["tile"] != info["bg"]
                assert np.array_equal(block[mask], lock["tile"][mask]), "fixed fragment changed"
            features[split].append(f)
    train, test = features["train"], features["test"]
    assert {f["success"] for f in train} == {False, True}
    assert any(f["success"] and max(f["candidates"]) > 1 for f in train), "teach global disambiguation"
    assert set().union(*(f["parities"] for f in train)) == {0, 1}, "teach both parity rules"
    assert any(f["geometric_false_line"] and f["unswapped_false_line"] for f in train)
    assert any(f["no_ink_changed"] for f in train) and any(f["no_swap_changed"] for f in train)
    assert max(f["max_span"] for f in train) <= 4
    assert all(f["max_span"] >= 5 for f in test), "test must extend socket span"
    train_axes = set()
    for f in train:
        for moves in f["solutions"].values():
            positions = [p for p, _, _ in moves]
            train_axes.add("row" if len({p // f["info"]["n"] for p in positions}) == 1 else "col")
    for f in test:
        for moves in f["solutions"].values():
            positions = [p for p, _, _ in moves]
            axis = "row" if len({p // f["info"]["n"] for p in positions}) == 1 else "col"
            assert axis in train_axes, "successful test axis must be taught"


def self_test():
    """42 seeds, bounded summaries only; no saved grids or execution traces."""
    grids, puzzle_ids, shapes, dimensions, banks, tests, inks = set(), set(), set(), set(), set(), set(), set()
    operations, relationships, spans, rack_sides, candidate_counts = set(), set(), set(), set(), set()
    operation_regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        ids = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                info = f["info"]
                digest = _digest(pair["input"])
                grids.add(digest)
                ids.append(digest)
                dimensions.add(pair["input"].shape)
                banks.add(info["n"])
                spans.add(f["max_span"])
                rack_sides.add(info["rotation"])
                candidate_counts.update(f["candidates"])
                operations.update(f["parities"])
                if f["success"]:
                    operation_regimes.add(tuple(sorted(f["parities"])))
                inks.add((info["bg"], info["wall"]))
                if split == "test":
                    tests.add(f["success"])
                for key in info["keys"]:
                    colors = [c for c in np.unique(key) if c != info["bg"]]
                    for color in colors:
                        shapes.add(_shape_signature(key == color))
                for moves in f["solutions"].values():
                    for _, _, fit in moves:
                        relationships.add(int(np.sign(fit["plug_size"] - fit["handle_size"])))
        puzzle_ids.add(tuple(ids))
    assert len(puzzle_ids) == 42 and len(grids) >= 200
    assert banks == {2, 3} and tests == {False, True} and operations == {0, 1}
    assert len(shapes) >= 100 and len(dimensions) >= 8 and len(inks) >= 25
    assert relationships == {-1, 0, 1}, relationships
    assert operation_regimes == {(0,), (1,), (0, 1)}, operation_regimes
    assert rack_sides == {0, 1, 2, 3} and max(spans) >= 6 and max(candidate_counts) >= 2
    # Exact requested counts and deterministic whole-puzzle reproduction.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split]) for k in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Conjugating both source and destination by any scene symmetry preserves
    # the relative orientation parity and therefore the rule.
    for op in range(8):
        for pair in p["train"] + p["test"]:
            assert np.array_equal(solve(_orient(pair["input"], op)), _orient(pair["output"], op))
    # Rack permutation carries no ordering information; output is identical
    # because the entire consumed rack is empty afterwards.
    for pair in p["train"] + p["test"]:
        info = _analyze(pair["input"])
        g = info["grid"].copy()
        for c in range(info["n"]):
            g[_slice(info["n"], c, info["s"])] = info["keys"][info["n"] - 1 - c]
        g = np.rot90(g, -info["rotation"]).copy()
        assert np.array_equal(solve(g), pair["output"])
    # Copying a winning row/column to another position creates two genuinely
    # different destinations. The solver must reject, rather than select one.
    positive = next(pair for pair in p["train"] if _features(pair)["success"])
    _, info, _, solutions = _solve_details(positive["input"])
    positions = [pos for pos, _, _ in next(iter(solutions.values()))]
    n, s = info["n"], info["s"]
    g = info["grid"].copy()
    if len({pos // n for pos in positions}) == 1:
        source = positions[0] // n
        destination = (source + 1) % n
        for c in range(n):
            g[_slice(destination, c, s)] = info["grid"][_slice(source, c, s)]
    else:
        source = positions[0] % n
        destination = (source + 1) % n
        for r in range(n):
            g[_slice(r, destination, s)] = info["grid"][_slice(r, source, s)]
    try:
        solve(np.rot90(g, -info["rotation"]).copy())
    except ValueError as error:
        assert "competing" in str(error)
    else:
        raise AssertionError("competing winning lines accepted")
    damaged = {split: [{k: v.copy() for k, v in pair.items()} for pair in p[split]] for split in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzle_ids), "distinct_inputs": len(grids),
            "region_shapes": len(shapes), "grid_shapes": len(dimensions),
            "bank_sizes": sorted(banks), "test_outcomes": sorted(tests),
            "plug_handle_relations": sorted(relationships),
            "operation_regimes": sorted(operation_regimes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
