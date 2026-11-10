"""A glyph loom: local completion-driven expansion beneath native fine print.

No parent module or evidence file is imported. The solver reads a single input.
"""

from bisect import bisect_right
from itertools import product

import numpy as np


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(name + " outside its permitted range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer grid required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _center(mask):
    r, c = np.nonzero(mask)
    crop = mask[r.min():r.max() + 1, c.min():c.max() + 1]
    result = np.zeros(mask.shape, dtype=bool)
    y, x = ((s - t) // 2 for s, t in zip(mask.shape, crop.shape))
    result[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return result


def _mask_bank(gh, gw):
    bank = {}
    for code in range(1, 1 << (gh * gw)):
        m = np.array([(code >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        m = _center(m)
        bank[_key(m)] = m
    return tuple(bank.values())


_BANKS = {(gh, gw): _mask_bank(gh, gw) for gh, gw in product((2, 3), repeat=2)}
_LAYOUTS = tuple((n, gh, gw) for n in range(2, 6) for gh, gw in product((2, 3), repeat=2)
                 if (n + 2) * (gh + 1) + 4 <= 30
                 and n * (gw + 1) + 3 <= 30
                 and 2 * (n + 1) * max(gh, gw) <= 30)


def _frame(layout, bg, divider):
    n, gh, gw = layout
    sh, sw = gh + 1, gw + 1
    inner = np.full(((n + 2) * sh + 2, n * sw + 1), bg, dtype=np.int64)
    inner[:, ::sw] = divider
    rows = [0, sh, 2 * sh] + [j * sh + 1 for j in range(2, n + 3)]
    inner[rows, :] = divider
    a = np.full((inner.shape[0] + 2, inner.shape[1] + 2), bg, dtype=np.int64)
    a[1:-1, 1:-1] = inner
    return a


def _cell(layout, region, r, c):
    _, gh, gw = layout
    y = 2 + r * (gh + 1) if region == "query" else 2 * (gh + 1) + 3 + r * (gh + 1)
    x = 2 + c * (gw + 1)
    return slice(y, y + gh), slice(x, x + gw)


def _parse(input_grid):
    a = _grid(input_grid)
    bg, divider = int(a[0, 0]), int(a[1, 1])
    assert bg != divider, "distinct frame/background roles"
    assert np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)
    inner = a[1:-1, 1:-1]
    cols = np.flatnonzero(np.all(inner == divider, axis=0))
    rows = np.flatnonzero(np.all(inner == divider, axis=1))
    assert len(cols) >= 3 and len(rows) >= 6
    assert cols[0] == rows[0] == 0
    sw = int(cols[1])
    n, gw = len(cols) - 1, sw - 1
    sh = int(rows[1])
    gh = sh - 1
    layout = n, gh, gw
    assert layout in _LAYOUTS, "unsupported cell geometry"
    expected = _frame(layout, bg, divider)
    assert a.shape == expected.shape and np.array_equal(cols, np.arange(n + 1) * sw)
    expected_rows = np.array([0, sh, 2 * sh] + [j * sh + 1 for j in range(2, n + 3)])
    assert np.array_equal(rows, expected_rows), "double seam and regular row spacing"
    tokens = np.empty((n, n, gh, gw), dtype=np.int64)
    inks = np.empty((n, n), dtype=np.int64)
    lookup, trimmed = {}, set()
    for r, c in np.ndindex(n, n):
        sl = _cell(layout, "board", r, c)
        t = a[sl]
        mask = t != bg
        values = np.unique(t[mask])
        assert len(values) == 1 and int(values[0]) != divider, "monochrome nonempty board key"
        assert np.array_equal(mask, _center(mask)), "consistently centered glyph"
        key = _key(t)
        rr, cc = np.nonzero(mask)
        crop_key = _key(t[rr.min():rr.max() + 1, cc.min():cc.max() + 1])
        assert key not in lookup and crop_key not in trimmed, "unique exact and cropped keys"
        lookup[key] = r, c
        trimmed.add(crop_key)
        tokens[r, c], inks[r, c] = t, int(values[0])
        expected[sl] = t
    selected = np.zeros((n, n), dtype=bool)
    order, query_keys = [], []
    for r, c in np.ndindex(2, n):
        sl = _cell(layout, "query", r, c)
        t = a[sl]
        if np.any(t != bg):
            key = _key(t)
            assert key in lookup, "every request has an exact board copy"
            p = lookup[key]
            assert not selected[p], "requests may not repeat keys"
            selected[p] = True
            order.append(p)
            query_keys.append(key)
        expected[sl] = t
    assert np.array_equal(expected, a), "ink outside glyph interiors or malformed frame"
    lengths_r = 2 * gh * (1 + selected.all(axis=1).astype(int))
    lengths_c = 2 * gw * (1 + selected.all(axis=0).astype(int))
    assert max(int(lengths_r.sum()), int(lengths_c.sum())) <= 30, "oversized textile"
    return dict(grid=a, layout=layout, bg=bg, divider=divider, tokens=tokens, inks=inks,
                selected=selected, order=order, query_keys=query_keys)


def _parts(tokens, bg, selected):
    n, _, gh, gw = tokens.shape
    inks = np.array([[int(t[t != bg][0]) for t in line] for line in tokens])
    lengths_r = 2 * gh * (1 + selected.all(axis=1).astype(int))
    lengths_c = 2 * gw * (1 + selected.all(axis=0).astype(int))
    carrier = np.repeat(np.repeat(inks, lengths_r, axis=0), lengths_c, axis=1)
    stencil = np.where(selected[:, :, None, None], tokens, bg).transpose(0, 2, 1, 3).reshape(n * gh, n * gw)
    return carrier, stencil, lengths_r, lengths_c


def _construct_output(tokens, bg, selected):
    carrier, stencil, _, _ = _parts(tokens, bg, selected)
    h, w = carrier.shape
    tiled = stencil[np.arange(h)[:, None] % stencil.shape[0],
                    np.arange(w)[None, :] % stencil.shape[1]]
    return np.where(tiled == bg, carrier, tiled)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode one loom and reconstruct each output pixel without generation state."""
    info = _parse(input_grid)
    n, gh, gw = info["layout"]
    selected, tokens, inks, bg = (info[k] for k in ("selected", "tokens", "inks", "bg"))
    # A deliberately separate pixel path: cumulative band boundaries, not repeat.
    rows = [0]
    cols = [0]
    for r in range(n):
        rows.append(rows[-1] + (4 if all(selected[r]) else 2) * gh)
    for c in range(n):
        cols.append(cols[-1] + (4 if all(selected[:, c]) else 2) * gw)
    result = np.empty((rows[-1], cols[-1]), dtype=np.int64)
    for y, x in np.ndindex(result.shape):
        r, c = bisect_right(rows, y) - 1, bisect_right(cols, x) - 1
        sr, sy = divmod(y % (n * gh), gh)
        sc, sx = divmod(x % (n * gw), gw)
        ink = int(tokens[sr, sc, sy, sx])
        result[y, x] = ink if selected[sr, sc] and ink != bg else inks[r, c]
    return result


def _completion(selected):
    return int(selected.all(axis=1).sum()), int(selected.all(axis=0).sum())


def _selection(rng, n, kind, count=None):
    s = np.zeros((n, n), dtype=bool)
    if kind in ("row", "col"):
        axis = 0 if kind == "row" else 1
        line = int(rng.integers(n))
        count = n if count is None else count
        for _ in range(256):
            s[:] = False
            if axis == 0:
                s[line] = True
            else:
                s[:, line] = True
            outside = np.flatnonzero(~s)
            if count > n:
                s.flat[rng.choice(outside, count - n, replace=False)] = True
            if _completion(s) == ((1, 0) if axis == 0 else (0, 1)):
                return s
        # A constructive near-full witness for n >= 3 and count == 2n-1.
        assert n >= 3 and count == 2 * n - 1
        s[:] = False
        s[0] = True
        s[1:n - 1, 0] = True
        s[n - 1, 1] = True
        return s if axis == 0 else s.T.copy()
    if kind == "cross":
        s[int(rng.integers(n)), :] = True
        s[:, int(rng.integers(n))] = True
        if count == 2 * n:
            outside = np.flatnonzero(~s)
            s.flat[int(rng.choice(outside))] = True
        return s
    if kind in ("two_rows", "two_cols"):
        lines = rng.choice(n, 2, replace=False)
        if kind == "two_rows":
            s[lines, :] = True
        else:
            s[:, lines] = True
        return s
    count = int(rng.integers(n)) if count is None else count
    for _ in range(256):
        s[:] = False
        s.flat[rng.choice(n * n, count, replace=False)] = True
        if _completion(s) == (0, 0):
            return s
    # Explicit nonwinning arrangement at the principal lesson count.
    assert n >= 3 and count == 2 * n - 1
    s[:] = False
    s[0, :n - 1] = True
    s[1, :n - 1] = True
    s[2, n - 1] = True
    return s


def _board(rng, layout, bg, palette, selected):
    n, gh, gw = layout
    bank = _BANKS[gh, gw]
    minimum = max(2, (n * n + len(bank) - 1) // len(bank))
    k = int(rng.integers(minimum, len(palette) + 1))
    inks = list(map(int, rng.choice(palette, k, replace=False)))
    combos = list(product(range(len(bank)), inks))
    rng.shuffle(combos)
    # Deliberately select one shape/color and leave both types of aliases outside.
    inside, outside = np.flatnonzero(selected), np.flatnonzero(~selected)
    assignments = {}
    if len(inside) and len(outside) >= 2:
        m, other = map(int, rng.choice(len(bank), 2, replace=False))
        p = int(rng.choice(inside))
        q, z = map(int, rng.choice(outside, 2, replace=False))
        assignments = {p: (m, inks[0]), q: (m, inks[1]), z: (other, inks[0])}
    reserved = set(assignments.values())
    combos = [t for t in combos if t not in reserved]
    tokens = np.full((n, n, gh, gw), bg, dtype=np.int64)
    for p in range(n * n):
        m, ink = assignments[p] if p in assignments else combos.pop()
        tokens[p // n, p % n][bank[m]] = ink
    return tokens


def _input(rng, layout, tokens, selected, bg, divider, query_tokens=None):
    n = layout[0]
    a = _frame(layout, bg, divider)
    for r, c in np.ndindex(n, n):
        a[_cell(layout, "board", r, c)] = tokens[r, c]
    if query_tokens is None:
        query_tokens = [tokens[r, c].copy() for r, c in np.argwhere(selected)]
    else:
        query_tokens = [t.copy() for t in query_tokens]
    slots = list(map(int, rng.choice(2 * n, len(query_tokens), replace=False)))
    if rng.random() < .25:
        # Deliberately include already board-ordered requests, even at high counts.
        positions = {_key(t): p for p, t in enumerate(tokens.reshape(-1, *layout[1:]))}
        query_tokens.sort(key=lambda t: positions[_key(t)])
        slots.sort()
    else:
        rng.shuffle(query_tokens)
    for slot, t in zip(slots, query_tokens):
        a[_cell(layout, "query", slot // n, slot % n)] = t
    return a


def _pair(rng, layout, tokens, selected, bg, divider, query_tokens=None):
    return {"input": _input(rng, layout, tokens, selected, bg, divider, query_tokens),
            "output": _construct_output(tokens, bg, selected)}


def _relocate(rng, tokens, old, new):
    result = np.empty_like(tokens)
    selected_tokens = tokens[old].copy()
    other_tokens = tokens[~old].copy()
    rng.shuffle(selected_tokens)
    rng.shuffle(other_tokens)
    result[new], result[~new] = selected_tokens, other_tokens
    return result


_REQUIRED = frozenset(("shape_identity", "color_identity", "completion_gate", "partial_line_gate", "local_rows",
                       "local_columns", "carrier_replication", "native_stencil", "selected_stencil",
                       "transparent", "overwrite", "phase_rows", "phase_columns", "continuous_phase"))


def _evidence(pair):
    i = _parse(pair["input"])
    n, gh, gw = i["layout"]
    s, tokens, bg = i["selected"], i["tokens"], i["bg"]
    out = np.asarray(pair["output"])
    base, stencil, lr, lc = _parts(tokens, bg, s)
    rr, cc = np.arange(len(base))[:, None], np.arange(base.shape[1])[None, :]
    tiled = stencil[rr % (n * gh), cc % (n * gw)]
    evidence = set()

    def different(name, alt):
        if not np.array_equal(out, alt):
            evidence.add(name)

    def overlay(carrier, print_layer):
        return np.where(print_layer == bg, carrier, print_layer)

    shape_s, ink_s = np.zeros_like(s), np.zeros_like(s)
    query_masks = {_key(tokens[r, c] != bg) for r, c in np.argwhere(s)}
    query_inks = set(map(int, i["inks"][s]))
    for r, c in np.ndindex(n, n):
        shape_s[r, c] = _key(tokens[r, c] != bg) in query_masks
        ink_s[r, c] = int(i["inks"][r, c]) in query_inks
    different("shape_identity", _construct_output(tokens, bg, ink_s))
    different("color_identity", _construct_output(tokens, bg, shape_s))
    # Ignore the completed-line operation while retaining exact selected print.
    ordinary = np.repeat(np.repeat(i["inks"], 2 * gh, 0), 2 * gw, 1)
    ordinary_print = stencil[np.arange(len(ordinary))[:, None] % (n * gh),
                             np.arange(ordinary.shape[1])[None, :] % (n * gw)]
    different("completion_gate", overlay(ordinary, ordinary_print))
    touched = np.repeat(np.repeat(i["inks"], 2 * gh * (1 + s.any(axis=1).astype(int)), 0),
                        2 * gw * (1 + s.any(axis=0).astype(int)), 1)
    touched_print = stencil[np.arange(len(touched))[:, None] % (n * gh),
                            np.arange(touched.shape[1])[None, :] % (n * gw)]
    different("partial_line_gate", overlay(touched, touched_print))
    # Uniform global resizing, separately on each axis, preserves output shape.
    sr = np.arange(len(base)) * n // len(base)
    sc = np.arange(base.shape[1]) * n // base.shape[1]
    actual_r = np.repeat(np.arange(n), lr)
    actual_c = np.repeat(np.arange(n), lc)
    different("local_rows", overlay(i["inks"][sr[:, None], actual_c[None, :]], tiled))
    different("local_columns", overlay(i["inks"][actual_r[:, None], sc[None, :]], tiled))
    different("carrier_replication", overlay(i["inks"][(rr // gh) % n, (cc // gw) % n], tiled))
    stretched = np.repeat(np.repeat(stencil, np.repeat(lr // gh, gh), 0),
                          np.repeat(lc // gw, gw), 1)
    different("native_stencil", overlay(base, stretched))
    all_print = tokens.transpose(0, 2, 1, 3).reshape(n * gh, n * gw)[rr % (n * gh), cc % (n * gw)]
    different("selected_stencil", overlay(base, all_print))
    different("transparent", tiled)
    different("overwrite", base)
    for name, layer in (("phase_rows", stencil[(rr + 1) % (n * gh), cc % (n * gw)]),
                        ("phase_columns", stencil[rr % (n * gh), (cc + 1) % (n * gw)])):
        different(name, overlay(base, layer))
    # Restart each selected key inside its own expanded carrier rectangle.
    reset = np.full(base.shape, bg, dtype=np.int64)
    starts_r, starts_c = np.r_[0, np.cumsum(lr)], np.r_[0, np.cumsum(lc)]
    for r, c in np.argwhere(s):
        block = tokens[r, c][np.arange(lr[r])[:, None] % gh, np.arange(lc[c])[None, :] % gw]
        reset[starts_r[r]:starts_r[r + 1], starts_c[c]:starts_c[c + 1]] = block
    different("continuous_phase", overlay(base, reset))
    return evidence


def _anchor(rng, layout, bg, divider, palette):
    n = layout[0]
    for _ in range(96):
        a = _selection(rng, n, "row", 2 * n - 1)
        b = _selection(rng, n, "col", 2 * n - 1)
        tokens = _board(rng, layout, bg, palette, a)
        moved = _relocate(rng, tokens, a, b)
        queries = [t.copy() for t in tokens[a]]
        pairs = [_pair(rng, layout, tokens, a, bg, divider, queries),
                 _pair(rng, layout, moved, b, bg, divider, queries)]
        # The relocation contrast keeps the entire visible tray unchanged.
        for qr, qc in np.ndindex(2, n):
            sl = _cell(layout, "query", qr, qc)
            pairs[1]["input"][sl] = pairs[0]["input"][sl]
        if _REQUIRED <= (_evidence(pairs[0]) | _evidence(pairs[1])):
            return pairs, layout
    return _fallback_anchor(rng, bg, divider, palette), (3, 2, 2)


def _fallback_anchor(rng, bg, divider, palette):
    # Finite constructive fallback uses a verified witness in the same family.
    layout = (3, 2, 2)
    a = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], bool)
    b = a.T.copy()
    codes = (1, 3, 5, 7, 15, 1, 3, 5, 7)
    colors = (0, 1, 2, 0, 1, 1, 2, 0, 2)
    tokens = np.full((3, 3, 2, 2), bg, dtype=np.int64)
    for p, (code, color) in enumerate(zip(codes, colors)):
        m = _center(np.array([(code >> j) & 1 for j in range(4)], bool).reshape(2, 2))
        tokens[p // 3, p % 3][m] = palette[color]
    moved = tokens.transpose(1, 0, 2, 3).copy()
    queries = [t.copy() for t in tokens[a]]
    pairs = [_pair(rng, layout, tokens, a, bg, divider, queries),
             _pair(rng, layout, moved, b, bg, divider, queries)]
    for qr, qc in np.ndindex(2, 3):
        sl = _cell(layout, "query", qr, qc)
        pairs[1]["input"][sl] = pairs[0]["input"][sl]
    assert _REQUIRED <= (_evidence(pairs[0]) | _evidence(pairs[1])), "fallback evidence"
    return pairs


def _fresh_pair(rng, layout, kind, bg, divider, palette, count=None):
    s = _selection(rng, layout[0], kind, count)
    tokens = _board(rng, layout, bg, palette, s)
    return _pair(rng, layout, tokens, s, bg, divider)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.permutation(10)))
    bg, divider = roles[:2]
    palette = roles[2:2 + int(rng.integers(3, 9))]
    layout = _LAYOUTS[int(rng.choice([j for j, v in enumerate(_LAYOUTS) if v[0] >= 3]))]
    train, layout = _anchor(rng, layout, bg, divider, palette)
    mixed = bool(rng.integers(2)) and nt >= 4
    taught_layouts = [layout]
    seen = {_key(p["input"]) for p in train}
    for index in range(2, nt):
        local_layout = layout
        if mixed:
            choices = [v for v in _LAYOUTS if v[0] >= 3] if index == 2 else list(_LAYOUTS)
            local_layout = choices[int(rng.integers(len(choices)))]
        n = local_layout[0]
        kind = "none" if index == 2 else "cross" if index == 3 else str(rng.choice(["row", "col", "none", "cross"]))
        count = 2 * n - 1 if index == 2 else None
        for _ in range(32):
            pair = _fresh_pair(rng, local_layout, kind, bg, divider, palette, count)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("fresh input uniqueness exhausted")
        train.append(pair)
        taught_layouts.append(local_layout)
        seen.add(_key(pair["input"]))
    test = []
    for index in range(nq):
        local_layout = layout if index == 0 else taught_layouts[int(rng.integers(len(taught_layouts)))]
        n, gh, gw = local_layout
        count = None
        if index == 0:
            if nt < 4:
                kind = "cross"
            else:
                kinds = ["cross"]
                if 2 * (n + 2) * gh <= 30:
                    kinds.append("two_rows")
                if 2 * (n + 2) * gw <= 30:
                    kinds.append("two_cols")
                kind, count = str(rng.choice(kinds)), 2 * n
        else:
            kinds = ["row", "col", "cross"] + (["none"] if nt >= 3 else [])
            kind = str(rng.choice(kinds))
        for _ in range(32):
            pair = _fresh_pair(rng, local_layout, kind, bg, divider, palette, count)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("fresh input uniqueness exhausted")
        test.append(pair)
        seen.add(_key(pair["input"]))
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos, seen, roles, evidence = [], set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            roles.add((info["bg"], info["divider"]))
            infos.append(info)
            assert np.array_equal(solve(inp), out), "incorrect textile"
            assert set(map(int, np.unique(out))) <= set(map(int, info["inks"].flat))
            assert info["bg"] not in out and info["divider"] not in out
            if split == "train":
                evidence |= _evidence(pair)
    assert len(roles) == 1, "puzzle palette roles must stay consistent"
    assert _REQUIRED <= evidence, "missing training evidence: " + str(_REQUIRED - evidence)
    ti = infos[:nt]
    assert any(_completion(i["selected"]) == (1, 0) for i in ti)
    assert any(_completion(i["selected"]) == (0, 1) for i in ti)
    if nt >= 3:
        assert any(_completion(i["selected"]) == (0, 0) for i in ti)
    if nt >= 4:
        assert any(_completion(i["selected"]) == (1, 1) for i in ti)
    relocated = False
    for a, b in product(ti, repeat=2):
        if a["layout"] != b["layout"]:
            continue
        inventory_a = sorted(_key(t) for t in a["tokens"].reshape(-1, *a["layout"][1:]))
        inventory_b = sorted(_key(t) for t in b["tokens"].reshape(-1, *b["layout"][1:]))
        if (inventory_a == inventory_b and sorted(a["query_keys"]) == sorted(b["query_keys"])
                and _completion(a["selected"]) != _completion(b["selected"])
                and all(np.array_equal(a["grid"][_cell(a["layout"], "query", r, c)],
                                       b["grid"][_cell(b["layout"], "query", r, c)])
                        for r, c in np.ndindex(2, a["layout"][0]))):
            relocated = True
    assert relocated, "same requested keys must visibly relocate between completed axes"
    assert {i["layout"] for i in infos[nt:]} <= {i["layout"] for i in ti}, "test geometry must be witnessed"
    if nt == 2:
        assert any(_completion(i["selected"]) == (1, 1) for i in infos[nt:])
    if nt >= 4:
        assert any(int(i["selected"].sum()) == 2 * i["layout"][0] for i in infos[nt:])


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(_key(x[k]) == _key(y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_parse(p["input"]) for task in puzzles for s in task for p in task[s]]
    assert len({_key(task["test"][0]["input"]) for task in puzzles}) == 42
    assert len({_normalized(task["test"][0]["output"]) for task in puzzles}) >= 35
    assert {i["layout"][0] for i in records} >= {3, 4, 5}
    assert len({i["layout"] for i in records}) >= 8
    assert len({(i["bg"], i["divider"]) for i in records}) >= 25
    assert len({_key(i["selected"]) for i in records}) >= 70
    assert len({_normalized(i["tokens"].transpose(0, 2, 1, 3).reshape(
        i["layout"][0] * i["layout"][1], -1)) for i in records}) >= 120
    assert len({p["output"].shape for task in puzzles for s in task for p in task[s]}) >= 15
    assert any(2 in _completion(i["selected"]) for i in records)
    assert any(i["layout"][1] != i["layout"][2] for i in records)
    assert any(i["bg"] != 0 and 0 in i["inks"] for i in records)
    assert {i["order"] == sorted(i["order"]) for i in records} == {False, True}
    assert any(len({i["layout"] for i in map(_parse, (p["input"] for p in task["train"]))}) > 1 for task in puzzles)
    for seed in (0, 7, 41):
        assert _equal(puzzles[seed], generate_puzzle(seed))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2026 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Input-only metamorphisms verify order invariance and visible palette roles.
    rng = np.random.default_rng(4407)
    base_pair = puzzles[0]["train"][0]
    i = _parse(base_pair["input"])
    permuted = _input(rng, i["layout"], i["tokens"], i["selected"], i["bg"], i["divider"])
    assert np.array_equal(solve(permuted), base_pair["output"])
    palette = rng.permutation(10)
    assert np.array_equal(solve(palette[base_pair["input"]]), palette[base_pair["output"]])
    empty = np.zeros_like(i["selected"])
    empty_input = _input(rng, i["layout"], i["tokens"], empty, i["bg"], i["divider"])
    expected = np.repeat(np.repeat(i["inks"], 2 * i["layout"][1], 0), 2 * i["layout"][2], 1)
    assert np.array_equal(solve(empty_input), expected)
    fallback = _fallback_anchor(rng, 0, 9, [1, 2, 3])
    assert _REQUIRED <= set().union(*(_evidence(p) for p in fallback))
    assert all(np.array_equal(solve(p["input"]), p["output"]) for p in fallback)
    # Exercise side 2, simultaneous full-board completion, and input purity.
    small_layout = (2, 3, 2)
    full = np.ones((2, 2), bool)
    tokens = _board(rng, small_layout, 0, [1, 2, 3], full)
    small = _pair(rng, small_layout, tokens, full, 0, 9)
    original = small["input"].copy()
    assert np.array_equal(solve(small["input"]), small["output"])
    assert np.array_equal(original, small["input"])
    for defect in ("label", "duplicate", "query"):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        if defect == "label":
            damaged["test"][0]["output"][0, 0] = i["bg"]
        elif defect == "duplicate":
            damaged["train"][1] = damaged["train"][0]
        else:
            target = damaged["test"][0]["input"]
            j = _parse(target)
            target[_cell(j["layout"], "query", 0, 0)] = j["divider"]
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, IndexError):
            pass
        else:
            raise AssertionError("validator accepted " + defect)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal counts)")
