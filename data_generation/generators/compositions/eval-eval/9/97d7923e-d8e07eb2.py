"""Ranked stitches: query-defined anchors create the bars that unary keys rank.

No file access, parent imports, retained examples, or generation state in solve.
All experiments stream through memory; self_test retains only small feature sets.
"""
from itertools import combinations
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a


def _signature(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _spans(labels, absent=0):
    """Intervals between consecutive anchors, including both endpoints."""
    result = []
    for r, row in enumerate(labels):
        occupied = list(map(int, np.flatnonzero(row != absent)))
        for a, b in zip(occupied, occupied[1:]):
            if row[a] == row[b]:
                _require(b - a >= 2, "a bar needs at least one intervening slot")
                result.append((int(row[a]), b - a + 1, r, a, b))
    return result


def _competitors(labels, ranks, absent=0):
    spans = _spans(labels, absent)
    groups = {}
    for family, rank in ranks.items():
        bars = sorted((s for s in spans if s[0] == family), key=lambda s: -s[1])
        _require(len(bars) >= 2, "each family needs competing brackets")
        _require(len({s[1] for s in bars}) == len(bars), "length ties are excluded")
        _require(1 <= rank <= len(bars), "rank must exist")
        groups[family] = bars
    _require(set(int(x) for x in np.unique(labels) if x != absent) == set(ranks), "family keys")
    return groups


def _parse_canonical(a):
    h, w = a.shape
    _require(h >= 12 and w >= 8 and (w - 2) % 3 == 0, "board width")
    bg, frame = int(a[1, 0]), int(a[-1, 0])
    _require(bg != frame and np.all(a[-1] == frame), "bottom frame")
    full_frame = np.flatnonzero(np.all(a == frame, axis=1))
    _require(len(full_frame) == 2 and full_frame[-1] == h - 1, "two frame edges")
    b0 = int(full_frame[0])
    _require(b0 >= 5 and (b0 - 1) % 4 == 0 and (h - b0 - 2) % 3 == 0, "tray lattice")
    rows, cols, qrows = (h - b0 - 2) // 3, (w - 2) // 3, (b0 - 1) // 4
    _require(rows >= 1 and qrows >= 1, "nonempty board")
    _require(np.all(a[b0:, 0] == frame) and np.all(a[b0:, -1] == frame), "side frame")
    key = a[0]
    _require(int(key[0]) == bg and int(key[-1]) == bg, "key margin")
    families = sorted(int(x) for x in np.unique(key) if x != bg)
    _require(1 <= len(families) <= 3 and frame not in families, "key families")
    ranks = {f: int(np.count_nonzero(key == f)) for f in families}
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[0] = key
    expected[b0, :] = expected[-1, :] = frame
    expected[b0:, 0] = expected[b0:, -1] = frame
    board, lookup = [], {}
    inks = set()
    for r in range(rows):
        for c in range(cols):
            y, x = b0 + 1 + 3 * r, 1 + 3 * c
            token = a[y:y + 2, x:x + 2]
            colors = set(map(int, np.unique(token))) - {bg}
            _require(len(colors) == 1 and not colors.intersection(families + [frame]), "glyph ink roles")
            inks.update(colors)
            k = tuple(map(int, token.flat))
            _require(k not in lookup, "board glyphs must have unique colored identities")
            lookup[k] = (r, c)
            board.append(token.copy())
            expected[y:y + 2, x:x + 2] = token
    labels = np.full((rows, cols), -1, dtype=np.int8)
    query_order = []
    for qr in range(qrows):
        for qc in range(cols):
            y, x = 2 + 4 * qr, 1 + 3 * qc
            token = a[y:y + 2, x:x + 2]
            marker = a[y + 2, x:x + 2]
            if np.all(token == bg):
                _require(np.all(marker == bg), "blank query card")
                continue
            _require(int(marker[0]) in ranks and np.all(marker == marker[0]), "query underline")
            k = tuple(map(int, token.flat))
            _require(k in lookup, "query needs one exact shape-and-color copy")
            r, c = lookup[k]
            _require(labels[r, c] == -1, "queries must be distinct")
            labels[r, c] = int(marker[0])
            query_order.append((r, c))
            expected[y:y + 2, x:x + 2] = token
            expected[y + 2, x:x + 2] = marker
    _require(np.array_equal(expected, a), "unexpected pixels outside glyphs, keys, or frame")
    groups = _competitors(labels, ranks, absent=-1)
    return {"grid": a, "bg": bg, "frame": frame, "b0": b0, "rows": rows,
            "cols": cols, "ranks": ranks, "labels": labels, "groups": groups,
            "board": board, "order": query_order, "ink_count": len(inks)}


def _decode(value):
    a = _grid(value)
    candidates = []
    for reflected in (False, True):
        for rotation in range(4):
            try:
                info = _parse_canonical(np.rot90(np.fliplr(a) if reflected else a, rotation))
            except (AssertionError, IndexError):
                continue
            info["rotation"], info["reflected"] = rotation, reflected
            candidates.append(info)
    _require(len(candidates) == 1, "unique tray-to-board orientation required")
    return candidates[0]


def _selected_slots(info):
    colors = info["labels"].copy()
    for f, bars in info["groups"].items():
        _, _, r, a, b = bars[info["ranks"][f] - 1]
        colors[r, a:b + 1] = f
    return colors


def _paint(a, b0, bg, colors):
    out = a.copy()
    for r, c in np.argwhere(colors >= 0):
        y, x = b0 + 1 + 3 * int(r), 1 + 3 * int(c)
        block = out[y:y + 3, x:x + 3]
        block[block == bg] = int(colors[r, c])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read every convention from the new input, then highlight ranked stitches."""
    info = _decode(input_grid)
    out = _paint(info["grid"], info["b0"], info["bg"], _selected_slots(info))
    out = np.rot90(out, -info["rotation"])
    return np.ascontiguousarray(np.fliplr(out) if info["reflected"] else out)


def _mask_bank():
    bank = {}
    for bits in range(1, 16):
        m = np.array([(bits >> i) & 1 for i in range(4)], dtype=bool).reshape(2, 2)
        rr, cc = np.where(m)
        crop = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        n = np.zeros((2, 2), dtype=bool)
        n[:crop.shape[0], :crop.shape[1]] = crop
        bank[n.tobytes()] = n
    return tuple(bank.values())


_MASKS = _mask_bank()


def _plan(rng, families, cols, rich=False, lesson=False):
    # Main rows supply two different lengths with a shared endpoint. One
    # additional row gives family 1 a third competitor (a non-extreme rank 2).
    minimum = families + (2 if families > 1 else 1)
    rows = minimum if lesson else 5 if rich else int(rng.integers(max(3, minimum), 6))
    labels = np.zeros((rows, cols), dtype=np.int8)
    triples = [p for p in combinations(range(cols), 3)
               if p[1] - p[0] >= 2 and p[2] - p[1] >= 2
               and p[1] - p[0] != p[2] - p[1]]
    for f in range(1, families + 1):
        pos = triples[int(rng.integers(len(triples)))]
        labels[f - 1, list(pos)] = f
    cursor = families
    if families > 1:
        f = int(rng.integers(2, families + 1)) if lesson else int(rng.integers(1, families + 1))
        g = int(rng.choice([j for j in range(1, families + 1) if j != f]))
        used = {s[1] for s in _spans(labels) if s[0] == g}
        inner_lengths = [n for n in range(3, cols - 1) if n not in used]
        if lesson:
            labels[cursor, [0, int(rng.integers(1, cols - 1)), cols - 1]] = [f, g, f]
        elif inner_lengths and rng.random() < .55:
            length = int(rng.choice(inner_lengths))
            start = int(rng.integers(0, cols - length - 1))
            labels[cursor, [start, start + 1, start + length, start + length + 1]] = [f, g, g, f]
        else:
            pos = triples[int(rng.integers(len(triples)))]
            labels[cursor, list(pos)] = [f, g, f]
        cursor += 1
    while cursor < rows:
        if cursor > minimum - 1 and families > 1 and rng.random() < .55:
            used = {f: {s[1] for s in _spans(labels) if s[0] == f}
                    for f in range(1, families + 1)}
            options = [(p, f, g) for p in combinations(range(cols), 4)
                       for f in used for g in used if f != g
                       and p[1] - p[0] >= 2 and p[3] - p[2] >= 2
                       and p[1] - p[0] + 1 not in used[f]
                       and p[3] - p[2] + 1 not in used[g]]
            if options:
                pos, f, g = options[int(rng.integers(len(options)))]
                labels[cursor, list(pos)] = [f, f, g, g]
                cursor += 1
                continue
        f = 1 if cursor == minimum - 1 or rich else int(rng.integers(1, families + 1))
        used = {s[1] for s in _spans(labels) if s[0] == f}
        options = [n for n in range(3, cols + 1) if n not in used]
        if not options:
            cursor += 1
            continue
        length = int(rng.choice(options))
        start = int(rng.integers(0, cols - length + 1))
        labels[cursor, [start, start + length - 1]] = f
        cursor += 1
    labels = labels[rng.permutation(rows)]
    if bool(rng.integers(2)):
        labels = labels[:, ::-1]
    return labels.copy()


def _board(rng, labels, bg, inks, ranks):
    rows, cols = labels.shape
    # Reserve an anchor and two untouched distractors. One shares shape only,
    # the other ink only. They are not offset aliases of the same colored token.
    covered = labels != 0
    for _, _, r, a, b in _spans(labels):
        covered[r, a:b + 1] = True
    spare = list(map(tuple, np.argwhere(~covered)))
    if len(spare) < 2:
        covered = labels != 0
        for f, rank in ranks.items():
            bars = sorted((s for s in _spans(labels) if s[0] == f), key=lambda s: -s[1])
            _, _, r, a, b = bars[rank - 1]
            covered[r, a:b + 1] = True
        spare = list(map(tuple, np.argwhere(~covered)))
    _require(len(spare) >= 2, "two untouched lookup distractors")
    p = tuple(np.argwhere(labels != 0)[int(rng.integers(np.count_nonzero(labels)))])
    rng.shuffle(spare)
    m, n = map(int, rng.choice(len(_MASKS), 2, replace=False))
    i, j = map(int, rng.choice(len(inks), 2, replace=False))
    assignments = {p: (m, i), spare[0]: (m, j), spare[1]: (n, i)}
    pool = [(s, k) for s in range(len(_MASKS)) for k in range(len(inks))
            if (s, k) not in assignments.values()]
    rng.shuffle(pool)
    tokens = np.full((rows, cols, 2, 2), bg, dtype=np.int8)
    for r, c in np.ndindex(rows, cols):
        s, k = assignments[(r, c)] if (r, c) in assignments else pool.pop()
        tokens[r, c][_MASKS[s]] = inks[k]
    return tokens


def _render(rng, labels, ranks, colors, tokens=None, query_layout=None):
    bg, frame, caps, inks = colors
    rows, cols = labels.shape
    if tokens is None:
        tokens = _board(rng, labels, bg, inks, ranks)
    anchors = list(map(tuple, np.argwhere(labels != 0)))
    qrows = (len(anchors) + cols - 1) // cols
    b0 = 1 + 4 * qrows
    a = np.full((b0 + 3 * rows + 2, 3 * cols + 2), bg, dtype=np.int8)
    a[b0] = a[-1] = frame
    a[b0:, 0] = a[b0:, -1] = frame
    for r, c in np.ndindex(rows, cols):
        y, x = b0 + 1 + 3 * r, 1 + 3 * c
        a[y:y + 2, x:x + 2] = tokens[r, c]
    if query_layout is None:
        rng.shuffle(anchors)
        slots = list(map(int, rng.choice(qrows * cols, len(anchors), replace=False)))
        query_layout = list(zip(slots, anchors))
    for slot, (r, c) in query_layout:
        qr, qc = divmod(slot, cols)
        y, x = 2 + 4 * qr, 1 + 3 * qc
        a[y:y + 2, x:x + 2] = tokens[r, c]
        a[y + 2, x:x + 2] = caps[int(labels[r, c]) - 1]
    order = list(map(int, rng.permutation(len(ranks))))
    key_width = sum(ranks.values()) + len(ranks) - 1
    x = int(rng.integers(1, a.shape[1] - key_width))
    for index, f in enumerate(order):
        family = f + 1
        a[0, x:x + ranks[family]] = caps[f]
        x += ranks[family] + 1
    slot_colors = np.full(labels.shape, -1, dtype=np.int8)
    for f, rank in ranks.items():
        slot_colors[labels == f] = caps[f - 1]
        bars = sorted([s for s in _spans(labels) if s[0] == f], reverse=True)
        _, _, r, left, right = bars[rank - 1]
        slot_colors[r, left:right + 1] = caps[f - 1]
    return {"input": a, "output": _paint(a, b0, bg, slot_colors)}, tokens, query_layout


def _orient(pair, rotation, reflected):
    return {k: np.ascontiguousarray(np.rot90(np.fliplr(v) if reflected else v, rotation))
            for k, v in pair.items()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside permitted range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    caps = palette[:3]
    rest = [c for c in palette if c not in caps]
    bg, frame = rest[:2]
    inks = rest[2:6]
    colors = bg, frame, caps, inks
    rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
    mode = int(rng.integers(3))  # width, family count, or competitor population
    base_f = 2 if mode == 1 else int(rng.integers(2, 4))
    base_c = 8 if mode == 2 else int(rng.integers(6, 8))
    base_labels = _plan(rng, base_f, base_c, lesson=True)
    puzzle = {"train": [], "test": []}
    tokens = queries = None
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index < 2:
                labels = base_labels
                ranks = {f: index + 1 for f in range(1, base_f + 1)}
                pair, tokens, queries = _render(rng, labels, ranks, colors, tokens, queries)
            else:
                f = (3 if split == "test" and mode == 1 else
                     1 if split == "test" and mode == 2 else
                     int(rng.integers(2 if mode == 2 else 1, base_f + 1)))
                c = 8 if mode == 2 or (split == "test" and mode == 0) else int(rng.integers(6, 8))
                labels = _plan(rng, f, c, rich=split == "test" and mode == 2)
                bars = _competitors(labels, {j: 1 for j in range(1, f + 1)})
                ranks = {j: int(rng.integers(1, min(len(bars[j]), 3 if nt >= 3 else 2) + 1))
                         for j in bars}
                if split == "train" and index == 2:
                    ranks[1] = 3  # family 1 always has at least three spans
                pair, _, _ = _render(rng, labels, ranks, colors)
            pair = _orient(pair, rotation, reflected)
            sig = _signature(pair["input"])
            # At most 13 earlier examples; a new board gives enormous identity
            # capacity. Retry only duplicates, never change rule or conventions.
            for _ in range(16):
                if sig not in seen:
                    break
                pair, _, _ = _render(rng, labels, ranks, colors)
                pair = _orient(pair, rotation, reflected)
                sig = _signature(pair["input"])
            _require(sig not in seen, "bounded uniqueness attempts exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    # The teaching contrast remains present, without a fixed returned position.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    labels, groups = info["labels"], info["groups"]
    shape_keys = [tuple((t != info["bg"]).flat) for t in info["board"]]
    ink_keys = [int(t[t != info["bg"]][0]) for t in info["board"]]
    chosen = {r * info["cols"] + c for r, c in np.argwhere(labels >= 0)}
    colored = _selected_slots(info).ravel()
    shape_needed = any(shape_keys[p] == shape_keys[q] and ink_keys[p] != ink_keys[q]
                       and colored[q] < 0 for p in chosen for q in range(len(shape_keys)))
    ink_needed = any(ink_keys[p] == ink_keys[q] and shape_keys[p] != shape_keys[q]
                     and colored[q] < 0 for p in chosen for q in range(len(shape_keys)))
    blocked = False
    shared = False
    for row in labels:
        xs = list(map(int, np.flatnonzero(row >= 0)))
        blocked |= any(row[xs[i]] == row[xs[j]] and any(row[xs[k]] != row[xs[i]]
                       for k in range(i + 1, j))
                       for i in range(len(xs)) for j in range(i + 2, len(xs)))
        shared |= any(row[a] == row[b] == row[c] for a, b, c in zip(xs, xs[1:], xs[2:]))
    return {"families": len(groups), "rows": info["rows"], "cols": info["cols"],
            "ranks": tuple(info["ranks"].values()), "populations": tuple(len(b) for b in groups.values()),
            "lengths": tuple(s[1] for b in groups.values() for s in b),
            "blocked": bool(blocked), "shared": bool(shared),
            "shape_needed": bool(shape_needed), "ink_needed": bool(ink_needed),
            "intermediate": any(1 < info["ranks"][f] < len(b) for f, b in groups.items()),
            "shortest": any(info["ranks"][f] == len(b) for f, b in groups.items())}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "same output shape")
            sig = _signature(inp)
            _require(sig not in seen, "distinct inputs")
            seen.add(sig)
            info = _decode(inp)
            expected = solve(inp)
            _require(np.array_equal(expected, out), "wrong ranked stitch output")
            canonical_out = np.rot90(np.fliplr(out) if info["reflected"] else out, info["rotation"])
            changed = canonical_out != info["grid"]
            _require(np.any(changed), "nontrivial example")
            _require(np.all(info["grid"][changed] == info["bg"]), "only background may change")
            # Recover painted slot colors from their guaranteed blank lower
            # right corners, then check every span independently by counting
            # how many same-family spans are strictly longer.
            actual = np.full(info["labels"].shape, -1, dtype=np.int8)
            wanted = info["labels"].copy()
            for f, length, r, a, b in _spans(info["labels"], absent=-1):
                higher = sum(s[1] > length for s in info["groups"][f])
                if higher + 1 == info["ranks"][f]:
                    wanted[r, a:b + 1] = f
            for r, c in np.ndindex(actual.shape):
                value = int(canonical_out[info["b0"] + 3 + 3 * r, 3 + 3 * c])
                actual[r, c] = -1 if value == info["bg"] else value
            _require(np.array_equal(actual, wanted), "slot membership/rank invariant")
            infos[split].append(info)
    feats = [_features(i) for i in infos["train"]]
    ranks = {r for i in infos["train"] for r in i["ranks"].values()}
    _require({1, 2} <= ranks, "unary count contrast must be taught")
    _require(any(f["intermediate"] for f in feats), "intermediate rank demonstration")
    _require(any(f["blocked"] for f in feats), "foreign anchor blocking demonstration")
    _require(any(f["shape_needed"] and f["ink_needed"] for f in feats), "exact glyph identity demonstration")
    _require(all(set(i["ranks"].values()) <= ranks for i in infos["test"]), "test ordinal demonstrated")
    _require(any(np.array_equal(a["grid"][1:], b["grid"][1:])
                 and any(a["ranks"].get(f) == 1 and b["ranks"].get(f) == 2
                         for f in a["ranks"])
                 for a in infos["train"] for b in infos["train"]), "same-board rank contrast")


def self_test():
    """42 seeds, meaningful semantic coverage, metamorphisms and negative cases.

    At most two puzzles (28 pairs of <=900 bytes per grid) are retained at once.
    No files, traces, logs, RNG globals, subprocesses, or worker pools are used.
    """
    features, signatures, shapes, canonical = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        sig = tuple(_signature(p[k]) for s in ("train", "test") for p in puzzle[s]
                    for k in ("input", "output"))
        _require(sig == tuple(_signature(p[k]) for s in ("train", "test") for p in again[s]
                             for k in ("input", "output")), "seed determinism")
        signatures.add(hashlib.sha256(repr(sig).encode()).digest())
        for s in ("train", "test"):
            for pair in puzzle[s]:
                info = _decode(pair["input"])
                features.append(_features(info))
                shapes.add(pair["input"].shape)
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in info["grid"].flat)
                canonical.add(hashlib.sha256(normalized).digest())
        # Palette changes and all eight spatial symmetries preserve the rule.
        if seed < 3:
            for rotation in range(4):
                for reflected in (False, True):
                    p = _orient(puzzle["test"][0], rotation, reflected)
                    _require(np.array_equal(solve(p["input"]), p["output"]), "spatial equivariance")
            mapping = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
            p = puzzle["test"][0]
            _require(np.array_equal(solve(mapping[p["input"]]), mapping[p["output"]]), "palette equivariance")
    _require(len(signatures) == 42 and len(canonical) >= 100, "structural diversity")
    _require(len(shapes) >= 12, "shape diversity")
    _require({f["families"] for f in features} == {1, 2, 3}, "family counts")
    _require({f["rows"] for f in features} == {3, 4, 5}, "board row counts")
    _require({f["cols"] for f in features} == {6, 7, 8}, "board column counts")
    _require({1, 2, 3} <= {r for f in features for r in f["ranks"]}, "ordinal range")
    _require(len({f["populations"] for f in features}) >= 10, "competitor populations")
    for field in ("blocked", "shared", "intermediate", "shortest", "shape_needed", "ink_needed"):
        _require(sum(f[field] for f in features) >= 20, field + " coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(101 + nt * 5 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    # Reject a missing stitch, altered glyph, changed key, and duplicate input.
    for defect in ("stitch", "ink", "key", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        p = bad["test"][0]
        info = _decode(p["input"])
        out = np.rot90(np.fliplr(p["output"]) if info["reflected"] else p["output"], info["rotation"])
        if defect == "stitch":
            r, c = np.argwhere((out != info["grid"]))[0]
            out[r, c] = info["grid"][r, c]
        elif defect == "ink":
            r, c = np.argwhere(info["board"][0] != info["bg"])[0]
            out[info["b0"] + 1 + r, 1 + c] = info["bg"]
        elif defect == "key":
            r, c = np.argwhere(info["grid"][0:1] != info["bg"])[0]
            out[r, c] = info["bg"]
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("validator accepted " + defect)


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, semantic coverage, symmetries, all legal counts")
