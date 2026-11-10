"""Consensus-relative edit interference: bitmap differences become XOR stencils.

All semantic conventions are visible or fixed. Construction paints labels from
its own records; solve reparses only the supplied input. No parent code is used.
"""
from collections import Counter
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= g) & (g <= 9)), "invalid colors"
    return g.astype(np.int8, copy=True)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _parse(input_grid):
    """Recover the register, color roles, master, differences and coordinates."""
    g = _grid(input_grid)
    h, w = g.shape
    frame = int(g[0, 0])
    # Only the separating carrier row is uniform in a non-frame color.
    gaps = [r for r in range(h) if np.all(g[r] == g[r, 0]) and g[r, 0] != frame]
    assert len(gaps) == 1, "one visible carrier separator required"
    gap = gaps[0]
    carrier = int(g[gap, 0])
    bank = g[:gap]
    rs = np.flatnonzero(np.all(bank == frame, axis=1))
    assert len(rs) >= 2 and rs[0] == 0 and rs[-1] == gap - 1
    assert np.all(np.diff(rs) == rs[1]), "irregular register bands"
    ph = int(rs[1] - 1)
    # Outer vertical borders contain tags; internal separators are unbroken.
    cs = [0] + [int(c) for c in np.flatnonzero(np.all(bank == frame, axis=0))
                if c not in (0, w - 1)] + [w - 1]
    assert len(cs) >= 4 and np.all(np.diff(cs) == cs[1])
    pw = cs[1] - 1
    bands, copies = len(rs) - 1, len(cs) - 1
    assert ph >= 3 and pw >= 3 and ph % 2 == pw % 2 == 1
    assert (ph - 2) * (pw - 2) >= 3 and 1 <= bands <= 4 and 3 <= copies <= 7
    panels, keys = [], []
    for r, end in zip(rs, rs[1:]):
        mid = int(r + 1 + ph // 2)
        tag = int(bank[mid, 0])
        assert tag == int(bank[mid, -1]) and tag not in (frame, carrier)
        edge = bank[r + 1:end, [0, w - 1]].copy()
        edge[ph // 2] = frame
        assert np.all(edge == frame), "tag must occupy the band midpoint only"
        keys.append(tag)
        row = [bank[r + 1:end, c + 1:d].copy() for c, d in zip(cs, cs[1:])]
        assert all(p.shape == (ph, pw) for p in row)
        panels.append(row)
    assert len(set(keys)) == bands, "band keys must be distinct"
    interior_colors = set(int(v) for row in panels for p in row for v in np.unique(p))
    assert carrier in interior_colors and len(interior_colors) == 2
    ink = next(iter(interior_colors - {carrier}))
    assert len({frame, carrier, ink, *keys}) == 3 + bands
    for row in panels:
        for p in row:
            assert np.all(p[[0, -1], :] == carrier) and np.all(p[:, [0, -1]] == carrier)
    frequencies = Counter(_key(p) for row in panels for p in row)
    maximum = max(frequencies.values())
    winners = [k for k, n in frequencies.items() if n == maximum]
    assert len(winners) == 1, "master frequency tie"
    master = next(p for row in panels for p in row if _key(p) == winners[0])
    deltas, columns = [], []
    for row in panels:
        exceptional = [c for c, p in enumerate(row) if not np.array_equal(p, master)]
        assert len(exceptional) == 1, "one visible exceptional panel per band"
        columns.append(exceptional[0])
        deltas.append(row[exceptional[0]] != master)
    assert maximum == bands * (copies - 1)
    top = gap + 1
    board = g[top:]
    bh = board.shape[0]
    assert bh >= ph + 2 and w >= pw + 2
    assert np.all(board[np.array([0, 0, bh - 1, bh - 1]),
                        np.array([0, w - 1, 0, w - 1])] == frame)
    assert np.array_equal(board[0], board[-1])
    assert np.array_equal(board[:, 0], board[:, -1])
    border = np.concatenate((board[0], board[-1], board[:, 0], board[:, -1]))
    assert set(map(int, border)) <= {frame, *keys}, "unknown coordinate label"
    assert set(map(int, np.unique(board[1:-1, 1:-1]))) <= {carrier, ink}
    rows, cols = [], []
    for tag in keys:
        rr = list(map(int, np.flatnonzero(board[:, 0] == tag)))
        cc = list(map(int, np.flatnonzero(board[0] == tag)))
        assert rr and cc, "every band needs both coordinate axes"
        assert all(1 + ph // 2 <= r < bh - 1 - ph // 2 for r in rr)
        assert all(1 + pw // 2 <= c < w - 1 - pw // 2 for c in cc)
        rows.append(rr)
        cols.append(cc)
    return dict(grid=g, frame=frame, carrier=carrier, ink=ink, keys=keys,
                ph=ph, pw=pw, bands=bands, copies=copies, panels=panels,
                master=master, deltas=deltas, columns=columns, top=top,
                bh=bh, rows=rows, cols=cols)


def _hits(p, mode="correct"):
    """Count translations; alternative modes are diagnostics, never latent rules."""
    hits = np.zeros((p["bh"], p["grid"].shape[1]), dtype=np.int16)
    for b, delta in enumerate(p["deltas"]):
        band_hits = np.zeros_like(hits)
        mask = delta
        if mode == "whole":
            mask = p["panels"][b][p["columns"][b]] == p["ink"]
        elif mode == "master":
            mask = p["master"] == p["ink"]
        elif mode == "first_column" or mode.startswith("column:"):
            col = 0 if mode == "first_column" else int(mode.split(":")[1])
            mask = p["panels"][b][col] != p["master"]
        cells = np.argwhere(mask)
        anchor = np.array([p["ph"] // 2, p["pw"] // 2])
        if mode == "recenter" and len(cells):
            anchor = (cells.min(axis=0) + cells.max(axis=0)) // 2
        offsets = cells - anchor
        centers = [(r, c) for r in p["rows"][b] for c in p["cols"][b]]
        if mode == "single":
            centers = centers[:1]
        elif mode == "zip":
            centers = list(zip(p["rows"][b], p["cols"][b]))
        for r, c in centers:
            for dr, dc in offsets:
                y, x = r + int(dr), c + int(dc)
                if 1 <= y < p["bh"] - 1 and 1 <= x < hits.shape[1] - 1:
                    band_hits[y, x] += 1
        hits += (band_hits > 0) if mode == "band_union" else band_hits
    return hits


def _prediction(p, mode="correct"):
    hits = _hits(p, mode)
    active = hits > 0 if mode == "union" else hits % 2 == 1
    out = p["grid"].copy()
    view = out[p["top"]:]
    if mode == "paint":
        view[active] = p["ink"]
    else:
        view[active] = np.where(view[active] == p["carrier"], p["ink"], p["carrier"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output solely from this visible input."""
    return _prediction(_parse(input_grid))


def _master_mask(rng, ph, pw):
    h, w = ph - 2, pw - 2
    yy, xx = np.indices((h, w))
    style = int(rng.integers(6))
    if style == 0:
        core = rng.random((h, w)) < float(rng.uniform(.2, .8))
    elif style == 1:
        core = (yy + xx) % int(rng.integers(2, 4)) == 0
    elif style == 2:
        core = (yy == h // 2) | (xx == w // 2)
    elif style == 3:
        core = (yy == 0) | (yy == h - 1) | (xx == 0) | (xx == w - 1)
    elif style == 4:
        core = np.abs(yy - h // 2) + np.abs(xx - w // 2) <= max(h, w) // 3
    else:
        core = rng.random((h, w)) < .35
        if h > 1:
            core[1:] |= core[:-1] & (rng.random((h - 1, w)) < .6)
    if rng.random() < .3:
        core |= core[::-1, ::-1]
    area = h * w
    minimum = 2 if area >= 5 else 1
    if not minimum <= int(core.sum()) <= area - minimum:
        core[:] = False
        core.flat[rng.choice(area, int(rng.integers(minimum, area - minimum + 1)),
                             replace=False)] = True
    mask = np.zeros((ph, pw), dtype=bool)
    mask[1:-1, 1:-1] = core
    return mask


def _difference(rng, master, mode=None):
    core = master[1:-1, 1:-1]
    ink, blank = np.flatnonzero(core), np.flatnonzero(~core)
    if mode is None:
        mode = str(rng.choice(["add", "remove", "exchange", "scatter", "line"]))
    diff = np.zeros(core.shape, dtype=bool)
    if mode in ("add", "remove"):
        candidates = blank if mode == "add" else ink
        n = int(rng.integers(1, min(len(candidates), max(1, core.size // 3)) + 1))
        diff.flat[rng.choice(candidates, n, replace=False)] = True
    elif mode == "exchange":
        n = int(rng.integers(1, min(len(ink), len(blank), max(1, core.size // 4)) + 1))
        diff.flat[rng.choice(ink, n, replace=False)] = True
        diff.flat[rng.choice(blank, n, replace=False)] = True
    elif mode == "line":
        axis = int(rng.integers(2))
        index = int(rng.integers(core.shape[axis]))
        if axis == 0:
            diff[index, :] = rng.random(core.shape[1]) < .75
        else:
            diff[:, index] = rng.random(core.shape[0]) < .75
    else:
        n = int(rng.integers(1, max(2, core.size // 2 + 1)))
        diff.flat[rng.choice(core.size, n, replace=False)] = True
    if not diff.any():
        diff.flat[int(rng.integers(core.size))] = True
    result = np.zeros(master.shape, dtype=bool)
    result[1:-1, 1:-1] = diff
    return result


def _canvas(rng, h, w):
    yy, xx = np.indices((h, w))
    style = int(rng.integers(6))
    if style == 0:
        return np.zeros((h, w), bool)
    if style == 1:
        return (xx + int(rng.integers(3))) % int(rng.integers(2, 5)) == 0
    if style == 2:
        return (yy + xx) % int(rng.integers(2, 5)) == 0
    if style == 3:
        cy, cx = h // 2, w // 2
        return np.abs(yy - cy) + np.abs(xx - cx) == int(rng.integers(1, min(h, w)))
    if style == 4:
        return rng.random((h, w)) < float(rng.uniform(.15, .7))
    return (yy == h // 2) | ((xx == w // 2) & (yy % 2 == 0))


def _coordinate_sets(rng, length, panel_size, counts):
    pool = np.arange(1 + panel_size // 2, length - 1 - panel_size // 2)
    n = sum(counts)
    assert len(pool) >= n
    if rng.random() < .5:
        start = int(rng.integers(len(pool) - n + 1))
        chosen = pool[start:start + n].copy()
    else:
        chosen = rng.choice(pool, n, replace=False)
    rng.shuffle(chosen)
    result, start = [], 0
    for count in counts:
        result.append(sorted(map(int, chosen[start:start + count])))
        start += count
    return result


_SHAPES = tuple((ph, pw) for ph in (3, 5, 7) for pw in (3, 5, 7)
                if (ph - 2) * (pw - 2) >= 3)


def _geometry(rng, kind, split, shift):
    if kind == "compare":
        ph, pw = [(3, 7), (5, 5), (5, 7)][int(rng.integers(3))]
        bands = 3
    elif kind == "interference":
        ph, pw, bands = 5, int(rng.choice([5, 7])), 2
    else:
        choices = []
        for ph, pw in _SHAPES:
            for bands in range(1, 5):
                extra = 2 if split == "test" and shift == "products" else 0
                min_bh = ph + bands + extra + 1
                total = bands * (ph + 1) + 2 + min_bh
                if total > 30:
                    continue
                if split == "train" and (ph > 5 or bands > 3):
                    continue
                if split == "test" and shift == "taller" and ph != 7:
                    continue
                if split == "test" and shift == "bands" and bands != 4:
                    continue
                choices.append((ph, pw, bands))
        ph, pw, bands = choices[int(rng.integers(len(choices)))]
    copies = int(rng.integers(3, 29 // (pw + 1) + 1))
    return ph, pw, bands, copies


def _assemble(rng, ph, pw, bands, copies, master, deltas, rows, cols, bh, kind):
    """Construct input and ground truth without calling parse, hits or solve."""
    palette = list(map(int, rng.permutation(10)))
    frame, carrier, ink = palette[:3]
    keys = palette[3:3 + bands]
    width = copies * (pw + 1) + 1
    gap = bands * (ph + 1) + 1
    top = gap + 1
    inp = np.full((top + bh, width), frame, dtype=np.int8)
    inp[gap] = carrier
    columns = list(map(int, rng.integers(copies, size=bands)))
    if kind == "compare" and len(set(columns)) == 1:
        columns[-1] = (columns[-1] + 1) % copies
    for b in range(bands):
        r0 = 1 + b * (ph + 1)
        inp[r0 + ph // 2, [0, width - 1]] = keys[b]
        for c in range(copies):
            mask = master ^ deltas[b] if c == columns[b] else master
            c0 = 1 + c * (pw + 1)
            inp[r0:r0 + ph, c0:c0 + pw] = np.where(mask, ink, carrier)
    canvas = _canvas(rng, bh - 2, width - 2)
    # Count directly from construction records, independently of the parser.
    covered = Counter()
    for b in range(bands):
        for r in rows[b]:
            for c in cols[b]:
                for y, x in np.argwhere(deltas[b]):
                    covered[(r + int(y) - ph // 2, c + int(x) - pw // 2)] += 1
    odd = [point for point, count in covered.items() if count % 2]
    assert odd, "construction must visibly change the canvas"
    if kind in ("compare", "interference"):
        assert len(odd) >= 2
        # A witness flips at least one cell of each initial canvas color.
        a, b = odd[:2]
        canvas[a[0] - 1, a[1] - 1] = False
        canvas[b[0] - 1, b[1] - 1] = True
    inp[top + 1:-1, 1:-1] = np.where(canvas, ink, carrier)
    for b, tag in enumerate(keys):
        for r in rows[b]:
            inp[top + r, [0, width - 1]] = tag
        for c in cols[b]:
            inp[[top, inp.shape[0] - 1], c] = tag
    out = inp.copy()
    # Replay each edit individually. Repeated edits naturally cancel.
    for b in range(bands):
        for r in rows[b]:
            for c in cols[b]:
                for y, x in np.argwhere(deltas[b]):
                    rr, cc = top + r + int(y) - ph // 2, c + int(x) - pw // 2
                    out[rr, cc] = ink if out[rr, cc] == carrier else carrier
    return {"input": inp, "output": out}


def _make_pair(rng, kind, split, shift):
    ph, pw, bands, copies = _geometry(rng, kind, split, shift)
    width = copies * (pw + 1) + 1
    max_bh = 30 - (bands * (ph + 1) + 2)
    master = _master_mask(rng, ph, pw)
    if kind == "compare":
        modes = list(rng.permutation(["add", "remove", "exchange"]))
        deltas = [_difference(rng, master, str(m)) for m in modes]
        row_counts = col_counts = [1] * bands
    elif kind == "interference":
        deltas = []
        dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        for b in range(2):
            d = np.zeros((ph, pw), dtype=bool)
            offsets = [(dr, dc), (0, dc)] if b == 0 else [(0, 0), (-dr, -dc)]
            for r, c in offsets:
                d[r + ph // 2, c + pw // 2] = True
            # Extra cells vary the actual stencils while keeping a guaranteed
            # even collision at the second band's primary center.
            forbidden = {(dr, 2 * dc), (2 * dr, dc), (2 * dr, 2 * dc)}
            if b == 0:
                # Keep the common cell of two first-band copies and one
                # second-band copy at exactly three hits, teaching odd overlap.
                forbidden |= {(0, 0), (dr, 0)}
            available = [(r, c) for r in range(-1, 2)
                         for c in range(1 - pw // 2, pw // 2)
                         if (r, c) not in forbidden and (r, c) not in offsets]
            n = int(rng.integers(min(4, len(available)) + 1))
            for i in rng.choice(len(available), n, replace=False):
                r, c = available[int(i)]
                d[r + ph // 2, c + pw // 2] = True
            deltas.append(d)
        row_counts, col_counts = [2, 1], [2, 1]
    else:
        deltas = [_difference(rng, master) for _ in range(bands)]
        row_counts, col_counts = [1] * bands, [1] * bands
        required_axis = int(rng.integers(2)) if split == "test" and shift == "products" else None
        if required_axis == 0:
            row_counts[0] = 3
        elif required_axis == 1:
            col_counts[0] = 3
        for counts, budget in ((row_counts, max_bh - ph - 1),
                               (col_counts, width - pw - 1)):
            for b in rng.permutation(bands):
                if counts[int(b)] == 1 and sum(counts) < budget and rng.random() < .48:
                    counts[int(b)] = 2
    min_bh = ph + sum(row_counts) + 1
    bh = int(rng.integers(min_bh, max_bh + 1))
    if kind == "interference":
        rpool = list(range(1 + ph // 2, bh - 1 - ph // 2))
        cpool = list(range(1 + pw // 2, width - 1 - pw // 2))
        r = int(rng.choice(rpool[1:-1]))
        c = int(rng.choice(cpool[1:-1]))
        rows, cols = [sorted([r, r - dr]), [r + dr]], [sorted([c, c - dc]), [c + dc]]
    else:
        rows = _coordinate_sets(rng, bh, ph, row_counts)
        cols = _coordinate_sets(rng, width, pw, col_counts)
    # A zero parity result can occur in general cases. Change one local edit;
    # its distinct Cartesian centers ensure its added incidence is nonzero.
    counts = Counter((r + int(y) - ph // 2, c + int(x) - pw // 2)
                     for b in range(bands) for r in rows[b] for c in cols[b]
                     for y, x in np.argwhere(deltas[b]))
    if kind == "compare" and sum(n % 2 for n in counts.values()) < 2:
        return None  # Resample this witness without altering its edit directions.
    if not any(n % 2 for n in counts.values()):
        cells = [(r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)
                 if not deltas[0][r, c] or deltas[0].sum() > 1]
        r, c = cells[int(rng.integers(len(cells)))]
        deltas[0][r, c] = not deltas[0][r, c]
    return _assemble(rng, ph, pw, bands, copies, master, deltas, rows, cols, bh, kind)


def _fallback_witness(rng, kind):
    """Guaranteed evidence after bounded search, with the same visible rule."""
    master = np.zeros((5, 5), bool)
    for r, c in ((1, 1), (1, 2), (2, 2), (3, 3)):
        master[r, c] = True
    if kind == "compare":
        edits = [[(2, 3)], [(1, 1)], [(1, 2), (3, 1)]]
        rows, cols = [[3], [4], [5]], [[4], [8], [12]]
    else:
        edits = [[(2, 3), (3, 3)], [(2, 2), (1, 1)]]
        rows, cols = [[3, 4], [5]], [[6, 7], [8]]
    deltas = []
    for cells in edits:
        d = np.zeros((5, 5), bool)
        for cell in cells:
            d[cell] = True
        deltas.append(d)
    return _assemble(rng, 5, 5, len(edits), 3, master, deltas, rows, cols, 10, kind)


def _unique_palette(pair, seen):
    """Twenty distinct bijections exceed the maximum thirteen earlier inputs."""
    if _key(pair["input"]) not in seen:
        return pair
    p = _parse(pair["input"])
    for attempt in range(20):
        lookup = np.arange(10, dtype=np.int8)
        if attempt >= 10:
            a, b = p["frame"], p["carrier"]
            lookup[a], lookup[b] = b, a
        lookup = (lookup + attempt % 10) % 10
        candidate = {k: lookup[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("unreachable palette exhaustion")


def _features(inp):
    p = _parse(inp)
    hits = _hits(p)
    correct = _prediction(p)
    additions, removals = [], []
    for d in p["deltas"]:
        additions.append(int(np.count_nonzero(d & (p["master"] == p["carrier"]))))
        removals.append(int(np.count_nonzero(d & (p["master"] == p["ink"]))))
    return dict(ph=p["ph"], pw=p["pw"], bands=p["bands"], copies=p["copies"],
                bh=p["bh"], palette=(p["frame"], p["carrier"], p["ink"], *p["keys"]),
                master=_key(p["master"] == p["ink"]),
                deltas=tuple(_key(d) for d in p["deltas"]), columns=tuple(p["columns"]),
                products=tuple((len(r), len(c)) for r, c in zip(p["rows"], p["cols"])),
                additions=tuple(additions), removals=tuple(removals),
                even=int(np.count_nonzero((hits > 0) & (hits % 2 == 0))),
                multiple=int(np.count_nonzero(hits > 1)),
                max_hits=int(hits.max()),
                odd_multiple=int(np.count_nonzero((hits >= 3) & (hits % 2 == 1))),
                empty_center=any(not d[p["ph"] // 2, p["pw"] // 2] for d in p["deltas"]),
                erase=int(np.count_nonzero((hits % 2 == 1) & (inp[p["top"]:] == p["ink"]))),
                create=int(np.count_nonzero((hits % 2 == 1) & (inp[p["top"]:] == p["carrier"]))),
                fixed_columns=tuple(int(np.count_nonzero(correct != _prediction(p, f"column:{c}")))
                                    for c in range(p["copies"])),
                shortcuts={mode: int(np.count_nonzero(correct != _prediction(p, mode)))
                           for mode in ("whole", "master", "first_column", "recenter",
                                        "single", "zip", "union", "band_union", "paint")})


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["taller", "bands", "products"]))
    puzzle, seen = {"train": [], "test": []}, set()
    kinds = ["compare", "interference"] + ["ordinary"] * (nt - 2)
    rng.shuffle(kinds)
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            kind = kinds[i] if split == "train" else "ordinary"
            candidate = None
            for _ in range(64):
                pair = _make_pair(rng, kind, split, shift)
                if pair is None:
                    continue
                f = _features(pair["input"])
                if kind == "compare" and (not all(f["shortcuts"][m] for m in ("whole", "master", "first_column"))
                                          or not all(f["fixed_columns"])):
                    continue
                if kind == "interference" and not all(f["shortcuts"][m] for m in (
                        "single", "zip", "union", "band_union", "paint", "recenter")):
                    continue
                candidate = pair
                if _key(pair["input"]) not in seen:
                    break
            else:
                pair = candidate if candidate is not None else _fallback_witness(rng, kind)
                pair = _unique_palette(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            assert np.array_equal(_prediction(p), out), "incorrect stencil interference"
            assert not np.array_equal(inp, out)
            # The entire register, spacer, all four border edges, and all cells
            # with an even number of edits must retain their input values.
            assert np.array_equal(inp[:p["top"]], out[:p["top"]])
            hit = _hits(p)
            unchanged = hit % 2 == 0
            assert np.array_equal(inp[p["top"]:][unchanged], out[p["top"]:][unchanged])
            if split == "train":
                train.append(_features(inp))
            else:
                test.append(_features(inp))
    changes = [(a, r) for f in train for a, r in zip(f["additions"], f["removals"])]
    assert any(a > 0 and r == 0 for a, r in changes), "teach added source edits"
    assert any(r > 0 and a == 0 for a, r in changes), "teach removed source edits"
    assert any(a == r > 0 for a, r in changes), "teach count-preserving anomalies"
    assert any(len(set(f["columns"])) > 1 for f in train), "teach position-independent selection"
    assert any(all(f["fixed_columns"]) for f in train), "exclude every fixed panel column"
    assert any(f["erase"] and f["create"] for f in train), "teach two-way canvas flipping"
    assert any(f["odd_multiple"] for f in train), "teach three-hit inversion"
    assert any(a >= 2 and b >= 2 for f in train for a, b in f["products"])
    for mode in ("whole", "master", "first_column", "recenter", "single", "zip", "union", "band_union", "paint"):
        assert any(f["shortcuts"][mode] > 0 for f in train), f"undistinguished shortcut: {mode}"
    max_ph, max_bands = max(f["ph"] for f in train), max(f["bands"] for f in train)
    max_multiplicity = max(max(a, b) for f in train for a, b in f["products"])
    assert all(f["ph"] > max_ph or f["bands"] > max_bands or
               max(max(a, b) for a, b in f["products"]) > max_multiplicity
               for f in test), "test must carry a demonstrated structural extension"


def _hand_case():
    """Hand-authored answer: four specified odd cells, two even cancellations."""
    inp = np.full((24, 19), 9, dtype=np.int8)
    inp[13] = 0
    master = np.zeros((5, 5), dtype=np.int8)
    master[1, 1] = master[1, 2] = master[2, 2] = master[3, 3] = 2
    for b, tag in enumerate((5, 7)):
        start = 1 + 6 * b
        inp[start + 2, [0, 18]] = tag
        for c in range(3):
            panel = master.copy()
            if (b, c) == (0, 2):
                panel[2, 3], panel[3, 3] = 2, 0
            if (b, c) == (1, 0):
                panel[2, 2], panel[1, 1] = 0, 0
            inp[start:start + 5, 1 + 6 * c:6 + 6 * c] = panel
    inp[15:23, 1:18] = 0
    for r in (3, 4):
        inp[14 + r, [0, 18]] = 5
    for c in (6, 7):
        inp[[14, 23], c] = 5
    inp[19, [0, 18]] = 7
    inp[[14, 23], 8] = 7
    for r, c in ((3, 7), (4, 8), (5, 8), (7, 13)):
        inp[14 + r, c] = 2
    expected = inp.copy()
    # These four cells were enumerated by hand from the two offset sets.
    expected[17, 7] = 0
    expected[17, 8] = expected[18, 7] = expected[19, 7] = 2
    return inp, expected


def self_test():
    """Bounded 42-seed variation, all legal counts, algebra and mutation checks."""
    hand_in, hand_out = _hand_case()
    assert np.array_equal(solve(hand_in), hand_out)
    hand = _parse(hand_in)
    hand_hits = _hits(hand)
    assert hand_hits[4, 7] == 3 and hand_hits[4, 8] == hand_hits[5, 8] == 2
    # Move bands with their visible tags: program order has no priority effect.
    swapped = hand_in.copy()
    swapped[1:6], swapped[7:12] = hand_in[7:12], hand_in[1:6]
    expected = hand_out.copy()
    expected[:14] = swapped[:14]
    assert np.array_equal(solve(swapped), expected)
    # Moving an exception horizontally also leaves the canvas result unchanged.
    swapped = hand_in.copy()
    swapped[1:6, 1:6], swapped[1:6, 13:18] = hand_in[1:6, 13:18], hand_in[1:6, 1:6]
    expected[:14] = swapped[:14]
    assert np.array_equal(solve(swapped), expected)
    for mode in ("recenter", "single", "zip", "union", "band_union", "paint"):
        assert not np.array_equal(_prediction(hand, mode), hand_out)
    for kind in ("compare", "interference"):
        f = _features(_fallback_witness(np.random.default_rng(19), kind)["input"])
        required = ("whole", "master", "first_column") if kind == "compare" else (
            "single", "zip", "union", "band_union", "paint", "recenter")
        assert all(f["shortcuts"][mode] for mode in required)
        if kind == "compare":
            assert all(f["fixed_columns"])
    records, signatures, structures, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(pair["input"]), _key(pair["output"]))
                          for split in ("train", "test") for pair in puzzle[split])
        signatures.add(signature)
        if seed in (0, 41):
            repeated = generate_puzzle(seed)
            assert signature == tuple((_key(pair["input"]), _key(pair["output"]))
                                      for s in ("train", "test") for pair in repeated[s])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                f = _features(inp)
                records.append((split, f))
                shapes.add(inp.shape)
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat))
                assert np.array_equal(solve(inp.tolist()), pair["output"])
                # Applying the same edit mask twice is the identity.
                p = _parse(inp)
                active = _hits(p) % 2 == 1
                twice = pair["output"].copy()
                view = twice[p["top"]:]
                view[active] = np.where(view[active] == p["carrier"], p["ink"], p["carrier"])
                assert np.array_equal(twice, inp)
                # Palette bijections, including black in every semantic role.
                if seed < 4:
                    lookup = (np.arange(10, dtype=np.int8) + seed + 3) % 10
                    assert np.array_equal(solve(lookup[inp]), lookup[pair["output"]])
    assert len(signatures) == 42 and len(structures) >= 190 and len(shapes) >= 30
    fs = [f for _, f in records]
    assert len({f["master"] for f in fs}) >= 50
    assert len({d for f in fs for d in f["deltas"]}) >= 60
    assert {1, 2, 3, 4} <= {f["bands"] for f in fs}
    assert {3, 5, 7} <= {f["ph"] for f in fs}
    assert {3, 5, 7} <= {f["pw"] for f in fs}
    assert len({f["copies"] for f in fs}) >= 4
    assert {f["palette"][0] for f in fs} == set(range(10))
    assert {f["palette"][1] for f in fs} == set(range(10))
    assert {f["palette"][2] for f in fs} == set(range(10))
    assert any(f["max_hits"] >= 3 for f in fs)
    assert any(f["multiple"] == 0 for f in fs)
    assert any(f["empty_center"] for f in fs)
    assert any(a >= 3 or b >= 3 for s, f in records if s == "test" for a, b in f["products"])
    # Every legal explicit count is exercised at least once.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    base = generate_puzzle(0, 2, 1)
    for defect in ("missing_exception", "output", "coordinate", "union"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]]
               for s in ("train", "test")}
        pair = bad["train"][0]
        p = _parse(pair["input"])
        if defect == "missing_exception":
            c0 = 1 + p["columns"][0] * (p["pw"] + 1)
            pair["input"][1:1 + p["ph"], c0:c0 + p["pw"]] = p["master"]
        elif defect == "output":
            pair["output"][p["top"] + 1, 1] = p["frame"]
        elif defect == "coordinate":
            pair["input"][p["top"], p["cols"][0][0]] = p["frame"]
        else:
            pair = next(x for x in bad["train"] if _features(x["input"])["even"])
            pair["output"] = _prediction(_parse(pair["input"]), "union")
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"validator accepted mutation: {defect}")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
