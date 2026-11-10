"""Ranked switchboard: length-addressed columns and a matched scale clue.

Only numpy and the standard library are used. Generation has no filesystem I/O.
The parser and solver receive only the input pixels. See task_spec.md.
"""
from functools import lru_cache

import numpy as np


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and
                        mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _rank_map(lengths):
    """For each physical channel, return its zero-based length address."""
    return tuple(sum(other < length for other in lengths) for length in lengths)


def _parse(input_grid):
    """Extract a canonical (horizontal-rail) device, with no generator metadata."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1, "background must be unique"
    rows = np.flatnonzero(np.all(inp != bg, axis=1))
    cols = np.flatnonzero(np.all(inp != bg, axis=0))
    assert len(rows) + len(cols) == 1, "one complete rail is required"
    vertical = bool(len(cols))
    a = inp.T if vertical else inp
    rail_row = int(cols[0] if vertical else rows[0])
    rail = int(a[rail_row, 0])
    assert rail != bg and a[rail_row, -1] == rail
    assert np.count_nonzero(a == rail) == np.count_nonzero(a[rail_row] == rail)
    rebuilt = np.full(a.shape, bg, dtype=np.int8)
    rebuilt[rail_row] = rail
    records = []
    x = 0
    while x < a.shape[1]:
        color = int(a[rail_row, x])
        end = x + 1
        while end < a.shape[1] and a[rail_row, end] == color:
            end += 1
        if color != rail:
            assert x > 0 and end < a.shape[1]
            top = rail_row
            while top > 0 and np.all(a[top - 1, x:end] == color):
                top -= 1
            assert top > 0 and np.all(a[top - 1, x:end] == bg)
            rebuilt[top:rail_row + 1, x:end] = color
            records.append((x, end - x, color, rail_row - top + 1))
        x = end
    n = len(records)
    assert 2 <= n <= 5
    lengths = [r[3] for r in records]
    assert len(set(lengths)) == n, "tied addresses are not defined"
    residual = (a != bg) & (rebuilt == bg)
    points, squares = [], []
    for cells in _components(residual):
        if len(cells) == 1:
            points.append(cells[0])
            continue
        ys, xs = zip(*cells)
        y, x, bottom, right = min(ys), min(xs), max(ys) + 1, max(xs) + 1
        side = bottom - y
        assert side >= 2 and right - x == side and len(cells) == side * side
        color = int(a[y, x])
        assert np.all(a[y:bottom, x:right] == color)
        assert y >= rail_row + 2
        squares.append((y, x, side, color))
    assert len(points) >= 3 and len(squares) >= 2
    p = np.array(points)
    mr, mc = map(int, p.min(axis=0))
    er, ec = map(int, p.max(axis=0) + 1)
    assert er <= rail_row and np.all((p - (mr, mc)) % 2 == 0)
    assert all(np.any(np.diff(np.unique(p[:, d])) == 2) for d in (0, 1))
    plan = a[mr:er:2, mc:ec:2].copy()
    h, w = plan.shape
    assert 3 <= h <= 6 and w == n
    # Miniature must be wholly on one side of the key bank, including widths.
    assert ec < records[0][0] or records[-1][0] + records[-1][1] < mc
    rebuilt[mr:er:2, mc:ec:2] = plan
    squares.sort()
    side = squares[0][2]
    assert all(s[1] == squares[0][1] and s[2] == side for s in squares)
    stride = squares[1][0] - squares[0][0]
    assert stride > side and all(b[0] - t[0] == stride for t, b in zip(squares, squares[1:]))
    assert 2 <= len(squares) < h
    physical = [i for i, r in enumerate(records) if r[0] == squares[0][1]]
    assert len(physical) == 1, "seed must be left aligned with one key"
    physical = physical[0]
    rank = _rank_map(lengths)
    column = rank[physical]
    seq = tuple(s[3] for s in squares)
    starts = [r for r in range(h - len(seq) + 1)
              if tuple(int(v) for v in plan[r:r + len(seq), column]) == seq]
    assert len(starts) == 1, "sequence must match once in its addressed column"
    start = starts[0]
    base = squares[0][0] - start * stride
    assert base >= rail_row + 2 and base + (h - 1) * stride + side <= a.shape[0]
    assert all(1 <= r[1] <= side and r[0] + side <= a.shape[1] for r in records)
    assert all(r[0] + side < s[0] for r, s in zip(records, records[1:]))
    for y, x, _, color in squares:
        rebuilt[y:y + side, x:x + side] = color
    assert np.array_equal(rebuilt, a), "input has unexplained pixels"
    for i, record in enumerate(records):
        for r in range(h):
            y, x = base + r * stride, record[0]
            value = plan[r, rank[i]] if i == physical and start <= r < start + len(seq) else bg
            assert np.all(a[y:y + side, x:x + side] == value)
    return dict(a=a, vertical=vertical, background=bg, rail=rail, rail_row=rail_row,
                records=tuple(records), ranks=rank, plan=plan, miniature=(mr, mc),
                side=side, stride=stride, base=base, anchor=physical, match=start,
                seed_length=len(seq))


def _completion(meta, mapping=None, repeated_column=None):
    """Solver rendering, also used to construct explicit shortcut predictions."""
    out = meta["a"].copy()
    mapping = meta["ranks"] if mapping is None else mapping
    for i, record in enumerate(meta["records"]):
        col = mapping[i] if repeated_column is None else repeated_column
        for r in range(meta["plan"].shape[0]):
            y, x = meta["base"] + r * meta["stride"], record[0]
            out[y:y + meta["side"], x:x + meta["side"]] = meta["plan"][r, col]
    return out.T.copy() if meta["vertical"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete all channels using only the visibly encoded addresses and clue."""
    return _completion(_parse(input_grid))


def _good_plan(plan, bg, col, start, length, witness):
    h, n = plan.shape
    if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
        return False
    points = np.argwhere(plan != bg)
    if any(not np.any(np.diff(np.unique(points[:, d])) == 1) for d in (0, 1)):
        return False
    seq = tuple(int(v) for v in plan[start:start + length, col])
    if bg in seq:
        return False
    matches = [r for r in range(h - length + 1) if tuple(plan[r:r + length, col]) == seq]
    if matches != [start]:
        return False
    rest = plan.copy()
    rest[start:start + length, col] = bg
    if len(set(map(int, rest.flat)) - {bg}) < 2:
        return False
    if witness:
        if len({tuple(plan[:, c]) for c in range(n)}) != n:
            return False
        if not any(c != col and r != start and tuple(plan[r:r + length, c]) == seq
                   for c in range(n) for r in range(h - length + 1)):
            return False
        if witness == 2 and not np.any(plan == bg):
            return False
    return True


def _sample_plan(rng, h, n, bg, palette, col, start, length, witness):
    # Actual selected-column uniqueness matters; other columns may repeat clues.
    for _ in range(96):
        plan = rng.choice(palette, (h, n)).astype(np.int8)
        probability = float(rng.choice([0.0, .10, .22, .32]))
        plan[rng.random((h, n)) < probability] = bg
        seq = rng.choice(palette, length)
        plan[start:start + length, col] = seq
        if witness:
            other = int(rng.choice([c for c in range(n) if c != col]))
            offset = int(rng.choice([r for r in range(h - length + 1) if r != start]))
            plan[offset:offset + length, other] = seq
        if _good_plan(plan, bg, col, start, length, witness):
            return plan
    # Constructive bounded fallback: a/b code occurs once amid c in its column;
    # other columns have different b/c codes. One displaced copy is deliberate.
    a, b, c = map(int, palette[:3])
    plan = np.full((h, n), c, dtype=np.int8)
    others = [j for j in range(n) if j != col]
    for j, other in enumerate(others):
        if j < h:
            plan[j, other] = b
    seq = [a, b] if length == 2 else [a, b, a]
    plan[start:start + length, col] = seq
    other = others[0]
    offset = next(r for r in range(h - length + 1) if r != start)
    plan[offset:offset + length, other] = seq
    if witness == 2:
        plan[1, others[1]] = bg
    assert _good_plan(plan, bg, col, start, length, witness)
    return plan


@lru_cache(maxsize=12)
def _options(mode, split, witness):
    if mode == "rows":
        hs = (4,) if witness else range(3, 5) if split == "train" else range(5, 7)
        ns = (3,) if witness == 1 else (4,) if witness == 2 else range(2, 6)
        sides = (2,) if witness == 1 else (3,) if witness == 2 else range(2, 5)
    elif mode == "channels":
        hs = range(4, 6) if witness else range(3, 6)
        ns = (3,) if witness == 1 else (4,) if witness == 2 else range(2, 5) if split == "train" else (5,)
        sides = (2,) if witness == 1 else (3,) if witness == 2 else range(2, 5)
    else:  # Bigger square scale, with already demonstrated side inference.
        hs = (4,) if witness else range(3, 5)
        ns = (3,) if witness else range(2, 4)
        sides = (2,) if witness == 1 else (3,) if witness == 2 else (2, 3) if split == "train" else (4,)
    result = []
    for h in hs:
        for n in ns:
            top = max(2 * h - 1, n + 2)
            for side in sides:
                for gutter in range(1, 4):
                    bh = (h - 1) * (side + gutter) + side
                    bw = (n - 1) * (side + 1) + side
                    if top + 2 + bh <= 30 and 2 * n - 1 + 3 + bw <= 30:
                        result.append((h, n, side, gutter, top))
    assert result
    return tuple(result)


def _draw_pair(plan, bg, rail, shape, rail_row, mini, records, ranks,
               side, stride, base, anchor, match, seed_length, vertical):
    # Ground truth comes from the sampled plan, not solve(), _parse() or
    # _completion(). Thus parser and alignment errors can be detected.
    inp = np.full(shape, bg, dtype=np.int8)
    inp[rail_row] = rail
    for x, width, color, length in records:
        inp[rail_row - length + 1:rail_row + 1, x:x + width] = color
    my, mx = mini
    inp[my:my + 2 * plan.shape[0]:2, mx:mx + 2 * plan.shape[1]:2] = plan
    for r in range(match, match + seed_length):
        y, x = base + r * stride, records[anchor][0]
        inp[y:y + side, x:x + side] = plan[r, ranks[anchor]]
    out = inp.copy()
    for rank in range(plan.shape[1]):
        i = ranks.index(rank)
        for row in range(plan.shape[0]):
            for dy in range(side):
                for dx in range(side):
                    out[base + row * stride + dy, records[i][0] + dx] = plan[row, rank]
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _make_pair(rng, mode, split, witness, vertical):
    opts = _options(mode, split, witness)
    h, n, side, gutter, top = opts[int(rng.integers(len(opts)))]
    colors = [int(x) for x in rng.permutation(10)]
    bg, rail = colors[:2]
    palette = colors[2:2 + int(rng.integers(3, 7))]
    cap = min(9, top)
    if witness:
        lengths = [2, 3] + [int(v) for v in rng.choice(range(4, cap + 1), n - 2, replace=False)]
    else:
        lengths = [int(v) for v in rng.choice(range(1, cap + 1), n, replace=False)]
    rng.shuffle(lengths)
    ranks = _rank_map(lengths)
    if witness and (ranks == tuple(range(n)) or ranks == tuple(reversed(range(n)))):
        ordered = sorted(lengths)
        lengths = ordered[1:] + ordered[:1]
        ranks = _rank_map(lengths)
    widths = [int(x) for x in rng.integers(1, side + 1, n)]
    if witness:
        widths[lengths.index(2)], widths[lengths.index(3)] = side, 1
    # Horizontal channel spacing is deliberately independent of vertical pitch.
    min_bank = (n - 1) * (side + 1) + side
    spare = 30 - (2 * n - 1 + 3 + min_bank)
    extra = int(rng.integers(min(spare, 7) + 1))
    additions = rng.multinomial(extra, [1 / (n - 1)] * (n - 1))
    gaps = [1 + int(v) for v in additions]
    bank_width = n * side + sum(gaps)
    used_width = 2 * n - 1 + 3 + bank_width
    width = int(rng.integers(used_width, min(30, used_width + 4) + 1))
    mini_left = bool(rng.integers(2))
    margin = int(rng.integers(width - used_width + 1))
    if mini_left:
        mini_x, bank_x = margin + 1, margin + 2 * n + 1
    else:
        bank_x, mini_x = margin + 1, margin + bank_width + 2
    records = []
    x = bank_x
    for i in range(n):
        records.append((x, widths[i], int(rng.choice(colors[2:])), lengths[i]))
        if i < n - 1:
            x += side + gaps[i]
    possible = [i for i in range(n) if not witness or ranks[i] != i]
    anchor = int(rng.choice(possible))
    length = 2 if witness else int(rng.integers(2, min(3, h - 1) + 1))
    if witness == 1:
        match = int(rng.choice([0, h - length]))
    elif witness == 2 or (split == "test" and h >= length + 2 and rng.random() < .8):
        match = int(rng.integers(1, h - length))
    else:
        match = int(rng.integers(h - length + 1))
    plan = _sample_plan(rng, h, n, bg, palette, ranks[anchor], match, length, witness)
    bottom_height = (h - 1) * (side + gutter) + side
    extra = int(rng.integers(min(4, 30 - top - 2 - bottom_height) + 1))
    margins = rng.multinomial(extra, [1 / 3] * 3)
    rail_row = top + int(margins[0])
    base = rail_row + 2 + int(margins[1])
    height = base + bottom_height + int(margins[2])
    mini_y = int(rng.integers(rail_row - (2 * h - 1) + 1))
    return _draw_pair(plan, bg, rail, (height, width), rail_row, (mini_y, mini_x),
                      records, ranks, side, side + gutter, base, anchor, match, length, vertical)


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # Ninety distinct injective role recolorings exceed the fourteen-pair bound.
    active = sorted(set(map(int, pair["input"].flat)))
    domain = active + [c for c in range(10) if c not in active]
    for first in range(10):
        for second in range(10):
            if first == second:
                continue
            mapping = np.empty(10, dtype=np.int8)
            mapping[domain] = [first, second] + [c for c in range(10) if c not in (first, second)]
            result = {k: mapping[v] for k, v in pair.items()}
            if _key(result["input"]) not in seen:
                return result
    raise AssertionError("finite uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["rows", "channels", "scale"]))
    axis_mode = int(rng.integers(3))
    axes = [False] if axis_mode == 0 else [True] if axis_mode == 1 else [False, True]
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    rng.shuffle(train_axes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            vertical = train_axes[i] if split == "train" else bool(rng.choice(axes))
            pair = _unique(_make_pair(rng, mode, split, witness, vertical), seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(meta):
    a = meta
    plan, rs = a["plan"], a["records"]
    n, h = len(rs), plan.shape[0]
    seq = tuple(plan[a["match"]:a["match"] + a["seed_length"], a["ranks"][a["anchor"]]])
    wrong_offsets = sum(c != a["ranks"][a["anchor"]] and r != a["match"] and
                        tuple(plan[r:r + len(seq), c]) == seq
                        for c in range(n) for r in range(h - len(seq) + 1))
    correct = _completion(a)
    physical = _completion(a, mapping=tuple(range(n)))
    area_order = tuple(sorted(range(n), key=lambda i: (rs[i][1] * rs[i][3], i)))
    area_map = tuple(area_order.index(i) for i in range(n))
    areas = _completion(a, mapping=area_map)
    repeated = _completion(a, repeated_column=a["ranks"][a["anchor"]])
    return dict(h=h, n=n, side=a["side"], gutter=a["stride"] - a["side"],
                vertical=a["vertical"], background=a["background"], rail=a["rail"],
                holes=int(np.count_nonzero(plan == a["background"])), match=a["match"],
                seed_length=a["seed_length"], anchor=a["anchor"],
                ranks=a["ranks"], lengths=tuple(r[3] for r in rs), widths=tuple(r[1] for r in rs),
                wrong_offset_matches=wrong_offsets, miniature=a["miniature"],
                physical_fails=not np.array_equal(correct, physical),
                area_fails=not np.array_equal(correct, areas),
                repeat_fails=not np.array_equal(correct, repeated),
                interior=0 < a["match"] < h - a["seed_length"],
                structural=(a["vertical"], a["a"].shape, tuple(r[0] for r in rs),
                            tuple(r[1] for r in rs), tuple(r[3] for r in rs),
                            a["base"], a["stride"], a["side"],
                            tuple(tuple(row) for row in (plan == a["background"]).tolist())))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            m = _parse(inp)
            canonical_out = out.T if m["vertical"] else out
            changed_allowed = np.zeros(m["a"].shape, dtype=bool)
            additions = 0
            # Relational block checks: no calls to solve() or _completion().
            for i, (x, _, _, length) in enumerate(m["records"]):
                column = sum(r[3] < length for r in m["records"])
                for r in range(m["plan"].shape[0]):
                    y = m["base"] + r * m["stride"]
                    block = canonical_out[y:y + m["side"], x:x + m["side"]]
                    assert np.all(block == m["plan"][r, column]), "wrong addressed block or blank"
                    changed_allowed[y:y + m["side"], x:x + m["side"]] = True
                    additions += int(np.any(block != m["a"][y:y + m["side"], x:x + m["side"]]))
            assert additions > 0
            assert np.array_equal(canonical_out[~changed_allowed], m["a"][~changed_allowed])
            # Even the supplied seed is preserved, as required by the match.
            mask = m["a"] != m["background"]
            assert np.array_equal(canonical_out[mask], m["a"][mask])
            features[split].append(_features(m))
    train = features["train"]
    assert sum(f["physical_fails"] and f["repeat_fails"] and f["wrong_offset_matches"] > 0 for f in train) >= 2
    assert any(f["area_fails"] for f in train)
    assert any(f["interior"] and f["holes"] > 0 for f in train)
    assert {f["vertical"] for f in features["test"]} <= {f["vertical"] for f in train}
    assert len({f["side"] for f in train}) >= 2


def self_test():
    """42 streaming seeds, explicit counts, negative tests; no artifacts or logs."""
    shapes, structures, signatures = set(), set(), set()
    measured = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple((_key(pair["input"]), _key(pair["output"]))
                          for s in ("train", "test") for pair in p[s])
        signatures.add(signature)
        if seed == 0:
            q = generate_puzzle(seed)
            assert signature == tuple((_key(pair["input"]), _key(pair["output"]))
                                      for s in ("train", "test") for pair in q[s])
        for split in ("train", "test"):
            for pair in p[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"])
                m = _parse(original)
                f = _features(m)
                measured.append((split, f))
                shapes.add(original.shape)
                structures.add(f["structural"])
    assert len(signatures) == 42 and len(shapes) >= 25 and len(structures) >= 180
    assert {f["background"] for _, f in measured} == set(range(10))
    assert {f["rail"] for _, f in measured} == set(range(10))
    for split in ("train", "test"):
        fs = [f for s, f in measured if s == split]
        assert {f["vertical"] for f in fs} == {False, True}
        assert {f["n"] for f in fs} >= {2, 3, 4, 5}
        assert {f["side"] for f in fs} >= ({2, 3} if split == "train" else {2, 4})
        assert len({f["ranks"] for f in fs}) >= 8
        assert any(f["holes"] >= 3 for f in fs) and any(f["holes"] == 0 for f in fs)
        assert any(f["interior"] for f in fs) and any(not f["interior"] for f in fs)
        assert sum(f["physical_fails"] for f in fs) > len(fs) // 2
        assert sum(f["repeat_fails"] for f in fs) > len(fs) // 2
    # All legal explicit counts, retaining only one puzzle at a time.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(420000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, False), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A corruption in a preserved area must be rejected.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted cell")
    # Reject concrete shortcuts, rather than merely recording their differences.
    p = generate_puzzle(0, 2, 1)
    index = next(i for i, pair in enumerate(p["train"])
                 if _features(_parse(pair["input"]))["area_fails"])
    m = _parse(p["train"][index]["input"])
    area_order = sorted(range(len(m["records"])),
                        key=lambda i: (m["records"][i][1] * m["records"][i][3], i))
    wrong_outputs = [
        _completion(m, mapping=tuple(range(len(m["records"])))),
        _completion(m, mapping=tuple(area_order.index(i) for i in range(len(area_order)))),
        _completion(m, repeated_column=m["ranks"][m["anchor"]]),
    ]
    for wrong in wrong_outputs:
        bad = {s: [dict(pair) for pair in p[s]] for s in ("train", "test")}
        bad["train"][index]["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a shortcut output")
    # Explicit malformed inputs: ties and multiple in-column registrations.
    plan = np.array([[1, 2, 3], [2, 2, 1], [1, 2, 3], [3, 2, 1]], dtype=np.int8)
    for lens, error_text in (((2, 3, 4), "sequence"), ((2, 2, 4), "tied")):
        records = [(7, 2, 4, lens[0]), (11, 1, 5, lens[1]), (15, 1, 6, lens[2])]
        bad = _draw_pair(plan, 0, 9, (22, 18), 8, (0, 0), records, (0, 1, 2),
                         2, 3, 10, 1, 0, 2, False)
        try:
            solve(bad["input"])
        except AssertionError as error:
            assert error_text in str(error)
        else:
            raise AssertionError("solver accepted ambiguous input")
    # Exercise constructive plan fallbacks directly, including length-three clues.
    for h in range(3, 7):
        for n in range(2, 6):
            for length in range(2, min(3, h - 1) + 1):
                for col in range(n):
                    for start in range(h - length + 1):
                        plan = _sample_plan(_FallbackRng(), h, n, 0, [1, 2, 3], col, start, length, 0)
                        assert _good_plan(plan, 0, col, start, length, 0)
    for witness in (1, 2):
        for h in range(4, 7):
            for n in range(3, 6):
                for col in range(n):
                    start = 0 if witness == 1 else 1
                    plan = _sample_plan(_FallbackRng(), h, n, 0, [1, 2, 3], col, start, 2, witness)
                    assert _good_plan(plan, 0, col, start, 2, witness)


class _FallbackRng:
    """Adversarial deterministic draws used solely to reach the bounded fallback."""
    def choice(self, values, size=None):
        return np.full(size, values[0], dtype=np.int8) if size is not None else values[0]

    def random(self, size):
        return np.zeros(size)
