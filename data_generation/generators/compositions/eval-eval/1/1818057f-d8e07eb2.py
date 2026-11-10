"""Index-certified crosses. Standard library + NumPy; no external state or I/O."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

_CORE = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_ARMS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONALS = ((0, 0), (0, 2), (2, 0), (2, 2))
_CORE_CODE = sum(1 << (3 * r + c) for r, c in _CORE)
_PLUS_CODES = tuple(_CORE_CODE + sum(1 << (3 * r + c)
                    for i, (r, c) in enumerate(_DIAGONALS) if bits & (1 << i))
                    for bits in range(16))


def _bits(code):
    return np.array([(code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)


def _canonical_code(code):
    mask = _bits(code)
    rr, cc = np.where(mask)
    crop = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    result = np.zeros((3, 3), dtype=bool)
    r, c = (3 - len(crop)) // 2, (3 - crop.shape[1]) // 2
    result[r:r + len(crop), c:c + crop.shape[1]] = crop
    return sum(int(v) << i for i, v in enumerate(result.flat))


_NONPLUS_CODES = tuple(sorted({_canonical_code(c) for c in range(1, 512)
                             if (c & _CORE_CODE) != _CORE_CODE}))
_FALSE_CODES = (sum(1 << i for i in (0, 1, 2, 4, 7)),  # T
                sum(1 << i for i in (0, 2, 4, 6, 8)),  # X
                511 - (1 << 4))                     # ring


def _glyph(ink, code, bg):
    result = np.full((3, 3), bg, dtype=np.int8)
    result[_bits(code)] = ink
    return result


def _key(array):
    return tuple(map(int, array.flat))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _orient(grid, orientation):
    return np.rot90(np.fliplr(grid) if orientation >= 4 else grid,
                    orientation % 4).copy()


def _occurrences(field, bg):
    """Production scan: equal-ink cardinal centers, with untouched patch keys."""
    middle = field[1:-1, 1:-1]
    mask = ((middle != bg) & (middle == field[:-2, 1:-1])
            & (middle == field[2:, 1:-1]) & (middle == field[1:-1, :-2])
            & (middle == field[1:-1, 2:]))
    records = []
    for r, c in np.argwhere(mask):
        r, c = int(r) + 1, int(c) + 1
        records.append((r, c, _key(field[r - 1:r + 2, c - 1:c + 2])))
    return records


def _qualified(selected, nr, nc):
    selected = set(selected)
    rows = [r for r in range(nr) if all((r, c) in selected for c in range(nc))]
    cols = [c for c in range(nc) if all((r, c) in selected for r in range(nr))]
    qualified = {(r, c) for r, c in selected if r in rows or c in cols}
    return qualified, rows, cols


def _parse_canonical(grid):
    h, w = grid.shape
    frame = int(grid[0, 0])
    assert np.all(grid[0] == frame) and (w - 1) % 4 == 0
    nc = (w - 1) // 4
    assert 3 <= nc <= 5
    lines = np.flatnonzero(np.all(grid == frame, axis=1)).tolist()
    assert len(lines) in (4, 5) and lines == list(range(0, 4 * len(lines), 4))
    nr = len(lines) - 1
    table_h, start = 4 * nr + 1, 4 * nr + 2
    assert 9 <= h - start <= 30 - start
    bg = int(grid[table_h, 0])
    assert bg != frame and np.all(grid[table_h] == bg)
    assert not np.any(grid[start:] == frame)
    structural = np.zeros((table_h, w), dtype=bool)
    structural[::4, :] = True
    structural[:, ::4] = True
    assert np.all(grid[:table_h][structural] == frame)
    board, lookup, inks, trimmed = {}, {}, set(), set()
    for r, c in itertools.product(range(nr), range(nc)):
        patch = grid[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4]
        colors = set(map(int, np.unique(patch))) - {bg}
        assert len(colors) == 1 and frame not in colors
        ink = next(iter(colors))
        inks.add(ink)
        mask = patch != bg
        rr, cc = np.where(mask)
        assert abs(int(rr.min()) - (2 - int(rr.max()))) <= 1
        assert abs(int(cc.min()) - (2 - int(cc.max()))) <= 1
        crop = patch[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        trimkey = crop.shape, _key(crop)
        assert trimkey not in trimmed, "offset-only aliases are excluded"
        trimmed.add(trimkey)
        key = _key(patch)
        assert key not in lookup, "unique exact index identities required"
        lookup[key] = (r, c)
        board[r, c] = patch.copy()
    assert 2 <= len(inks) <= 3
    assert set(map(int, np.unique(grid[start:]))) <= inks | {bg}
    occurrences = _occurrences(grid[start:], bg)
    selected = {lookup[key] for _, _, key in occurrences if key in lookup}
    qualified, rows, cols = _qualified(selected, nr, nc)
    return dict(grid=grid, nr=nr, nc=nc, bg=bg, frame=frame, start=start,
                board=board, lookup=lookup, occurrences=occurrences,
                selected=selected, qualified=qualified, rows=rows, cols=cols)


def _analyze(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(grid, rotation))
        except (AssertionError, IndexError, ValueError):
            continue
        info["rotation"] = rotation
        candidates.append(info)
    assert len(candidates) == 1, "one unambiguous framed index and field required"
    return candidates[0]


def _paint(info, qualified=None, full_patch=False):
    out = info["grid"].copy()
    mask = np.zeros(out.shape, dtype=bool)
    qualified = info["qualified"] if qualified is None else qualified
    for r, c, key in info["occurrences"]:
        if info["lookup"].get(key) in qualified:
            r += info["start"]
            if full_patch:
                mask[r - 1:r + 2, c - 1:c + 2] = True
            else:
                for dr, dc in _ARMS:
                    mask[r + dr, c + dc] = True
    out[mask] = info["frame"]
    return out, mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention and every decision solely from this input."""
    info = _analyze(input_grid)
    out, _ = _paint(info)
    return np.rot90(out, -info["rotation"]).copy()


def _plan(rng, nr, nc, mode):
    if mode == "col":
        active, false_pos = _plan(rng, nc, nr, "row")
        return {(c, r) for r, c in active}, (false_pos[1], false_pos[0])
    r, other = map(int, rng.choice(nr, 2, replace=False))
    missing = int(rng.integers(nc))
    active = {(r, c) for c in range(nc)}
    active |= {(other, c) for c in range(nc) if c != missing}
    if mode == "none":
        active.remove((r, int(rng.integers(nc))))
    elif mode == "multi":
        col = int(rng.choice([c for c in range(nc) if c != missing]))
        active |= {(a, col) for a in range(nr)}
    else:
        assert mode == "row"
    return active, (other, missing)


def _pack(rng, shapes, h, w):
    """Bounded random packing, followed by a guaranteed 3-row shelf packing."""
    for _ in range(12):
        occupied = np.zeros((h, w), dtype=bool)
        positions = [None] * len(shapes)
        order = [0] + list(map(int, rng.permutation(np.arange(1, len(shapes)))))
        for i in order:
            a, b = shapes[i]
            for _ in range(70):
                r, c = int(rng.integers(h - a + 1)), int(rng.integers(w - b + 1))
                if not occupied[r:r + a, c:c + b].any():
                    occupied[r:r + a, c:c + b] = True
                    positions[i] = (r, c)
                    break
            else:
                break
        if all(p is not None for p in positions):
            return positions
    rows, current, used = [], [], 0
    order = [0] + list(map(int, rng.permutation(np.arange(1, len(shapes)))))
    for i in order:
        assert shapes[i][0] == 3
        if used + shapes[i][1] > w:
            rows.append(current)
            current, used = [], 0
        current.append(i)
        used += shapes[i][1]
    rows.append(current)
    assert 3 * len(rows) <= h
    vertical = rng.multinomial(h - 3 * len(rows), [1 / (len(rows) + 1)] * (len(rows) + 1))
    positions, r = [None] * len(shapes), int(vertical[0])
    for j, row in enumerate(rows):
        gaps = rng.multinomial(w - sum(shapes[i][1] for i in row), [1 / (len(row) + 1)] * (len(row) + 1))
        c = int(gaps[0])
        for k, i in enumerate(row):
            positions[i] = r, c
            c += shapes[i][1] + int(gaps[k + 1])
        r += 3 + int(vertical[j + 1])
    return positions


def _field(rng, symbols, bg, inks, h, w, style, testing):
    patches = [_glyph(ink, code, bg) for ink, code in symbols]
    ink, code = symbols[0]
    if style == "adjacent":
        assert code == 511
        patches[0] = np.full((3, 4), ink, dtype=np.int8)
    elif style == "shared":
        assert code == _CORE_CODE
        p = np.full((3, 5), bg, dtype=np.int8)
        p[1, :] = ink
        p[0, [1, 3]] = p[2, [1, 3]] = ink
        patches[0] = p
    else:
        assert style == "extended"
        p = np.full((3, 5), bg, dtype=np.int8)
        p[:, :3] = patches[0]
        p[1, 3:] = ink
        patches[0] = p
    positions = _pack(rng, [p.shape for p in patches], h, w)
    field, protected = np.full((h, w), bg, dtype=np.int8), np.zeros((h, w), dtype=bool)
    for patch, (r, c) in zip(patches, positions):
        a, b = patch.shape
        field[r:r + a, c:c + b] = patch
        protected[r:r + a, c:c + b] = True
    # Extend arms outside protected patches, retaining each original exact key.
    for (r, c), (ink, _) in zip(positions[:-1], symbols[:-1]):
        if rng.random() < .7:
            dr, dc = _ARMS[1 + int(rng.integers(4))]
            for k in range(2, int(rng.integers(3, 7))):
                a, b = r + 1 + dr * k, c + 1 + dc * k
                if not (0 <= a < h and 0 <= b < w) or protected[a, b]:
                    break
                field[a, b] = ink
    density = float(rng.uniform(.075, .14) if testing else rng.uniform(.015, .065))
    free = np.argwhere(~protected)
    # A bounded budget of short walks combines bends, lines and isolated ink.
    for _ in range(max(1, int(len(free) * density))):
        if not len(free):
            break
        r, c = map(int, free[int(rng.integers(len(free)))])
        ink = int(rng.choice(inks))
        direction = int(rng.integers(4))
        for step in range(int(rng.integers(1, 5))):
            if 0 <= r < h and 0 <= c < w and not protected[r, c]:
                field[r, c] = ink
            if step and rng.random() < .4:
                direction = (direction + int(rng.choice((-1, 1)))) % 4
            dr, dc = _ARMS[1 + direction]
            r, c = r + dr, c + dc
    return field


def _make_pair(rng, nr, nc, mode, style, bg, frame, inks, orientation, testing):
    active, false_pos = _plan(rng, nr, nc, mode)
    qualified, _, _ = _qualified(active, nr, nc)
    code = _CORE_CODE if style == "shared" else 511 if style == "adjacent" else int(rng.choice(_PLUS_CODES))
    first = sorted(qualified or active)[int(rng.integers(len(qualified or active)))]
    outside = sorted(active - qualified - {first})
    assert outside
    twin = outside[int(rng.integers(len(outside)))]
    assignments = {first: (inks[0], code), twin: (inks[1], code)}
    # A same-ink/equal-area diagonal contrast makes shape identity visible,
    # independently of the same-shape/different-ink contrast above.
    if qualified and len(outside) >= 2:
        other_good = sorted(qualified - {first})[0]
        other_bad = next(p for p in outside if p != twin)
        diagonal_codes = [c for c in (_PLUS_CODES[1], _PLUS_CODES[2], _PLUS_CODES[4], _PLUS_CODES[8]) if c != code]
        rng.shuffle(diagonal_codes)
        assignments[other_good] = (inks[0], diagonal_codes[0])
        assignments[other_bad] = (inks[0], diagonal_codes[1])
    pool = [(ink, c) for ink in inks for c in _PLUS_CODES if (ink, c) not in assignments.values()]
    rng.shuffle(pool)
    rest = sorted(active - set(assignments))
    rng.shuffle(rest)
    for j, position in enumerate(rest):
        if j == 0 and len(inks) == 3:
            symbol = next(s for s in reversed(pool) if s[0] == inks[2])
            pool.remove(symbol)
            assignments[position] = symbol
        else:
            assignments[position] = pool.pop()
    symbols = [assignments[first]] + [assignments[p] for p in sorted(active - {first})]
    false_symbol = (int(rng.choice(inks)), int(rng.choice(_FALSE_CODES)))
    symbols.append(false_symbol)
    start = 4 * nr + 2
    h, w = int(rng.integers(9, 31 - start)), 4 * nc + 1
    field = _field(rng, symbols, bg, inks, h, w, style, testing)
    observed = {key for _, _, key in _occurrences(field, bg)}
    assert all(_key(_glyph(ink, c, bg)) in observed for ink, c in symbols[:-1])
    board = {p: _glyph(*symbol, bg) for p, symbol in assignments.items()}
    board[false_pos] = _glyph(*false_symbol, bg)
    used = {_key(p) for p in board.values()}
    # Filler keys cannot accidentally select an unintended table entry.
    choices = [(ink, c) for ink in inks for c in _PLUS_CODES + _NONPLUS_CODES]
    rng.shuffle(choices)
    for p in itertools.product(range(nr), range(nc)):
        if p in board:
            continue
        while choices:
            ink, c = choices.pop()
            patch = _glyph(ink, c, bg)
            key = _key(patch)
            if key not in used and key not in observed:
                board[p] = patch
                used.add(key)
                break
        else:
            raise AssertionError("finite glyph bank unexpectedly exhausted")
    inp = np.full((start + h, w), bg, dtype=np.int8)
    inp[:start - 1:4, :] = frame
    inp[:start - 1, ::4] = frame
    for (r, c), patch in board.items():
        inp[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4] = patch
    inp[start:] = field
    # Construction labels use intended certified identities and a direct scan,
    # independently of solve's scaffold parser and matched-coordinate inference.
    allowed = {_key(board[p]) for p in qualified}
    out = inp.copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = field[r - 1:r + 2, c - 1:c + 2]
            if _key(patch) in allowed and all(field[r + dr, c + dc] == field[r, c] for dr, dc in _ARMS):
                for dr, dc in _ARMS:
                    out[start + r + dr, c + dc] = frame
    return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible family member; counts and sampling never enter solve."""
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    bg0, frame, inks0 = colors[0], colors[1], colors[2:2 + int(rng.integers(2, 4))]
    nr0, nc0 = int(rng.integers(3, 5)), int(rng.integers(3, 6))
    variable_size = nt >= 4 and rng.random() < .5
    variable_palette = nt >= 3 and rng.random() < .5
    mixed_orientation = rng.random() < .5
    orientation0 = int(rng.integers(8))
    lessons = ["row", "col", "none", "multi"][:nt]
    choices = ["row", "col"] + (["none"] if nt >= 3 else []) + (["multi"] if nt >= 4 else [])
    while len(lessons) < nt:
        lessons.append(str(rng.choice(choices)))
    modes = lessons + [str(rng.choice(choices)) for _ in range(nq)]
    pairs, seen = [], set()
    for i, mode in enumerate(modes):
        style = "adjacent" if i == 0 else "shared" if i == 1 else str(rng.choice(("adjacent", "shared", "extended")))
        nr, nc = nr0, nc0
        if variable_size and i > 0:
            nr = 7 - nr0 if i == 1 else int(rng.integers(3, 5))
            nc = (nc0 - 2) % 3 + 3 if i == 1 else int(rng.integers(3, 6))
        bg, inks = bg0, list(inks0)
        if variable_palette and i > 0:
            palette = [c for c in range(10) if c != frame]
            local = list(map(int, rng.choice(palette, 4, replace=False)))
            bg, inks = local[0], local[1:1 + int(rng.integers(2, 4))]
        orientation = int(rng.integers(8)) if mixed_orientation else orientation0
        for _ in range(12):
            pair = _make_pair(rng, nr, nc, mode, style, bg, frame, inks, orientation, i >= nt)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
                break
        else:
            raise AssertionError("bounded uniqueness retries exhausted")
    puzzle = {"train": [pairs[int(i)] for i in rng.permutation(nt)],
              "test": [pairs[nt + int(i)] for i in rng.permutation(nq)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit(info):
    """Direct pixel audit and meaningful counterfactuals; no generated metadata."""
    grid, bg, start = info["grid"], info["bg"], info["start"]
    field = grid[start:]
    eligible, all_matches = [], set()
    for r in range(1, len(field) - 1):
        for c in range(1, field.shape[1] - 1):
            key = tuple(int(field[a, b]) for a in range(r - 1, r + 2) for b in range(c - 1, c + 2))
            p = info["lookup"].get(key)
            if p is not None:
                all_matches.add(p)
            ink = int(field[r, c])
            if ink != bg and all(int(field[r + dr, c + dc]) == ink for dr, dc in _ARMS):
                eligible.append((r, c, key, p))
    selected = {p for _, _, _, p in eligible if p is not None}
    good = set()
    for p in selected:
        if all((p[0], c) in selected for c in range(info["nc"])) or all((r, p[1]) in selected for r in range(info["nr"])):
            good.add(p)
    assert selected == info["selected"] and good == info["qualified"]
    mask, no_b, full, no_a = [np.zeros(field.shape, dtype=bool) for _ in range(4)]
    wrong_good, _, _ = _qualified(all_matches, info["nr"], info["nc"])
    remaining, sequential = field.copy(), np.zeros(field.shape, dtype=bool)
    certified_count, certified_keys = 0, set()
    for r, c, key, p in eligible:
        for dr, dc in _ARMS:
            no_b[r + dr, c + dc] = True
            if p in good:
                mask[r + dr, c + dc] = True
            if p in wrong_good:
                no_a[r + dr, c + dc] = True
        if p in good:
            full[r - 1:r + 2, c - 1:c + 2] = True
            certified_count += 1
            certified_keys.add(key)
            ink = int(remaining[r, c])
            if ink != bg and ink != info["frame"] and all(remaining[r + dr, c + dc] == ink for dr, dc in _ARMS):
                for dr, dc in _ARMS:
                    remaining[r + dr, c + dc] = info["frame"]
                    sequential[r + dr, c + dc] = True
    expected = grid.copy()
    expected[start:][mask] = info["frame"]
    shapes = {p: _key(patch != bg) for p, patch in info["board"].items()}
    inks = {p: int(patch[patch != bg][0]) for p, patch in info["board"].items()}
    color_witness = any(shapes[p] == shapes[q] for p in good for q in selected - good)
    diagonal_witness = any(inks[p] == inks[q] and shapes[p] != shapes[q]
                           and sum(shapes[p]) == sum(shapes[q])
                           for p in good for q in selected - good)
    alternatives = {}
    for name, signatures in (("ignore_ink", shapes), ("ignore_shape", inks)):
        occurrence_signatures = [(_key(field[r - 1:r + 2, c - 1:c + 2] != bg)
                                  if name == "ignore_ink" else int(field[r, c]))
                                 for r, c, _, _ in eligible]
        present = set(occurrence_signatures)
        alt_selected = {p for p, sig in signatures.items() if sig in present}
        alt_good, _, _ = _qualified(alt_selected, info["nr"], info["nc"])
        approved = {signatures[p] for p in alt_good}
        alt_mask = np.zeros(field.shape, dtype=bool)
        for (r, c, _, _), sig in zip(eligible, occurrence_signatures):
            if sig in approved:
                for dr, dc in _ARMS:
                    alt_mask[r + dr, c + dc] = True
        alternatives[name] = bool(np.any(alt_mask != mask))
    return expected, dict(centers=len(eligible), matched=len(selected), certified=certified_count,
                         rows=len(info["rows"]), cols=len(info["cols"]),
                         no_a=bool(np.any(mask != no_a)), no_b=bool(np.any(mask != no_b)),
                         full_patch=bool(np.any(mask != full)),
                         sequential=bool(np.any(mask != sequential)), color_witness=color_witness,
                         diagonal_witness=diagonal_witness, **alternatives,
                         overlap=5 * certified_count > int(mask.sum()),
                         repeated=certified_count > len(certified_keys),
                         changed=int(mask.sum()), field_shape=field.shape,
                         board_shape=(info["nr"], info["nc"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records, frames = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen, "inputs must be distinct within the puzzle"
            seen.add(key)
            info = _analyze(inp)
            expected, record = _audit(info)
            assert np.array_equal(expected, np.rot90(out, info["rotation"])), "incorrect core certification or preservation"
            assert np.array_equal(solve(inp), out)
            frames.add(info["frame"])
            records[split].append(record)
    assert len(frames) == 1
    train, tests = records["train"], records["test"]
    assert any(r["rows"] for r in train) and any(r["cols"] for r in train)
    for witness in ("no_a", "no_b", "full_patch", "sequential", "color_witness", "diagonal_witness", "ignore_ink", "ignore_shape", "overlap", "repeated"):
        assert any(r[witness] for r in train), f"missing training witness: {witness}"
    if any(r["rows"] and r["cols"] for r in tests):
        assert any(r["rows"] and r["cols"] for r in train)
    if any(not r["rows"] and not r["cols"] for r in tests):
        assert any(not r["rows"] and not r["cols"] for r in train)
    if len({r["board_shape"] for r in train + tests}) > 1:
        assert len({r["board_shape"] for r in train}) > 1


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds streamed in memory; no disk writes, logs, or saved grid corpus."""
    fingerprints, shapes, boards, colors, center_counts = set(), set(), set(), set(), set()
    outcomes, witnesses, pairs = set(), {k: 0 for k in ("no_a", "no_b", "overlap", "sequential", "repeated", "color_witness", "diagonal_witness", "ignore_ink", "ignore_shape")}, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                _, record = _audit(info)
                pairs += 1
                shapes.add(pair["input"].shape)
                boards.add(record["board_shape"])
                colors.add((info["bg"], info["frame"]))
                center_counts.add(record["centers"])
                outcomes.add((bool(record["rows"]), bool(record["cols"])))
                for k in witnesses:
                    witnesses[k] += int(record[k])
    assert len(fingerprints) == 42 and len(shapes) >= 20
    assert boards == set(itertools.product((3, 4), (3, 4, 5)))
    assert len(colors) >= 25 and len(center_counts) >= 8
    assert outcomes == {(False, False), (True, False), (False, True), (True, True)}
    assert all(v >= 42 for v in witnesses.values())
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(230 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Local semantics: adjacent and shared-arm motifs, diagonal invariance,
    # mixed inks, complete edge-touching motifs, and no boundary wrapping.
    block = np.ones((3, 4), dtype=np.int8)
    assert len(_occurrences(block, 0)) == 2
    for code in _PLUS_CODES:
        assert len(_occurrences(_glyph(2, code, 0), 0)) == 1
    clipped = np.zeros((4, 5), dtype=np.int8)
    clipped[0, 1:4] = 2
    clipped[1, 2] = clipped[-1, 2] = 2
    assert not _occurrences(clipped, 0)
    mixed = _glyph(2, _CORE_CODE, 0)
    mixed[0, 1] = 3
    assert not _occurrences(mixed, 0)
    # Hand-constructed members beyond the sampler's outcome templates.
    # A visible 3x3 index with nine distinct plus tokens; no hidden metadata.
    scene = np.zeros((26, 13), dtype=np.int8)
    scene[:13:4, :] = 9
    scene[:13, ::4] = 9
    cards = [_glyph(1 if i < 5 else 2, _PLUS_CODES[i], 0) for i in range(9)]
    for i, card in enumerate(cards):
        r, c = divmod(i, 3)
        scene[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4] = card
    assert np.array_equal(solve(scene), scene), "an empty field selects nothing"
    single_line = scene.copy()
    expected = scene.copy()
    for i in range(3):
        single_line[14:17, 4 * i:4 * i + 3] = cards[i]
        expected[14:17, 4 * i:4 * i + 3] = cards[i]
        for dr, dc in _ARMS:
            expected[15 + dr, 4 * i + 1 + dc] = 9
    assert np.array_equal(solve(single_line), expected)
    mixed_diagonal = single_line.copy()
    mixed_diagonal[14, 8] = 2  # core still valid, but exact colored patch absent
    assert np.array_equal(solve(mixed_diagonal), mixed_diagonal)
    unknown = single_line.copy()
    unknown[14:17, 8:11] = _glyph(2, 511, 0)
    assert np.array_equal(solve(unknown), unknown), "an unregistered plus cannot finish a line"
    all_lines, expected = scene.copy(), scene.copy()
    for i, card in enumerate(cards):
        r, c = divmod(i, 3)
        all_lines[14 + 4 * r:17 + 4 * r, 4 * c:4 * c + 3] = card
        expected[14 + 4 * r:17 + 4 * r, 4 * c:4 * c + 3] = card
        for dr, dc in _ARMS:
            expected[15 + 4 * r + dr, 4 * c + 1 + dc] = 9
    assert np.array_equal(solve(all_lines), expected), "all six completed lines use one union rule"
    baseline = generate_puzzle(0)
    for orientation in range(8):
        transformed = {s: [{k: _orient(v, orientation) for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
        validate_generated_puzzle(transformed)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                board_shapes=sorted(boards), palettes=len(colors), center_counts=sorted(center_counts),
                outcomes=sorted(outcomes), witnesses=witnesses)
